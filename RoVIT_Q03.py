import os
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# =============================================================================
# 1. HARDWARE & EXPERIMENT CONFIGURATION
# =============================================================================
HF_TOKEN = ""
DATA_CACHE_DIR = "./imagenet_data"  # Thư mục lưu 50,000 ảnh nội bộ trên máy

BATCH_SIZE = 64  # Tối ưu hóa cho RTX 5080
NUM_WORKERS = 8  # Đọc dữ liệu đa luồng từ ổ cứng
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENTS = {
    'W6A6': {'weight_bits': 6, 'act_bits': 6},
    'W4A4': {'weight_bits': 4, 'act_bits': 4}
}

MODELS = [
    'facebook/deit-small-patch16-224',
    'google/vit-base-patch16-224'
]


# =============================================================================
# 2. QUANTIZATION ALGORITHMS
# =============================================================================
def fake_quantize_tensor(tensor, bits, symmetric=True, dim=None):
    if bits >= 32: return tensor

    if symmetric:
        qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
        if dim is not None:
            scale = tensor.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
        else:
            scale = tensor.abs().max().clamp(min=1e-8) / qmax
        return torch.round(tensor / scale).clamp(qmin, qmax) * scale
    else:
        qmin, qmax = 0, (2 ** bits) - 1
        if dim is not None:
            min_val = tensor.amin(dim=dim, keepdim=True)
            max_val = tensor.amax(dim=dim, keepdim=True)
        else:
            min_val, max_val = tensor.min(), tensor.max()

        scale = (max_val - min_val).clamp(min=1e-8) / qmax
        zero_point = torch.round(-min_val / scale)
        q_tensor = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        return (q_tensor - zero_point) * scale


def standard_ptq_hook(mod, inp, b):
    return (fake_quantize_tensor(inp[0], b, symmetric=True),)


def apply_standard_ptq(model, w_bits, a_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits, symmetric=True)
            module.register_forward_pre_hook(lambda mod, inp, b=a_bits: standard_ptq_hook(mod, inp, b))
    return model


def advanced_ptq_hook(mod, inp, b):
    return (fake_quantize_tensor(inp[0], b, symmetric=False),)


def apply_advanced_sota_ptq(model, w_bits, a_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits, symmetric=True, dim=1)
            module.register_forward_pre_hook(lambda mod, inp, b=a_bits: advanced_ptq_hook(mod, inp, b))
    return model


def rovit_hook(mod, inp, b, R_matrix):
    X_rotated = torch.matmul(inp[0], R_matrix)
    return (fake_quantize_tensor(X_rotated, b, symmetric=True),)


def apply_rovit_ptq(model, w_bits, a_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            Q, _ = torch.linalg.qr(torch.randn(module.weight.shape[1], module.weight.shape[1], device=device))
            module.weight.data = fake_quantize_tensor(torch.matmul(module.weight.data, Q), w_bits, symmetric=True)
            module.register_forward_pre_hook(lambda mod, inp, b=a_bits, r=Q: rovit_hook(mod, inp, b, r))
    return model


# =============================================================================
# 3. HIGH-PERFORMANCE DATALOADER
# =============================================================================

class CustomCollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [example['image'].convert("RGB") for example in examples]
        labels = torch.tensor([example['label'] for example in examples])
        inputs = self.processor(images=images, return_tensors="pt")
        inputs['labels'] = labels
        return inputs


def evaluate_imagenet_full(model, processor, dataset, device):
    model.eval()
    correct = 0
    total = 0

    collate_fn = CustomCollateFn(processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS,
                            pin_memory=True)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="      [Inferring 50K Images]", leave=False)
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Cập nhật hiển thị Acc Real-time trên TQDM
            progress_bar.set_postfix({'Acc': f"{(correct / total) * 100:.2f}%"})

    return (correct / total) * 100.0


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
def main():
    print(f"[*] STARTING FULL 50K SOTA BENCHMARK on {DEVICE}\n")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print("[System] Loading ImageNet-1K Validation Set (50,000 images)...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", token=HF_TOKEN, streaming=False,
                           cache_dir=DATA_CACHE_DIR, trust_remote_code=True)
    print(f"[System] Dataset ready. Total samples: {len(dataset)}\n")

    results_log = []

    for model_name in MODELS:
        print(f"{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        processor = AutoImageProcessor.from_pretrained(model_name)

        # 1. FP32 BASELINE
        print(f"---> Evaluating Baseline FP32...")
        model_fp32 = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
        acc_fp32 = evaluate_imagenet_full(model_fp32, processor, dataset, DEVICE)
        print(f"     => Baseline Top-1 Acc: {acc_fp32:.2f}%\n")
        del model_fp32;
        torch.cuda.empty_cache()

        for exp_name, config in EXPERIMENTS.items():
            w_bits, a_bits = config['weight_bits'], config['act_bits']
            print(f"--- BATTLE: {exp_name} (W{w_bits}A{a_bits}) ---")

            # 2. STANDARD PTQ
            print("      [System] Competitor 1: Standard PTQ...")
            model_ptq = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
            acc_ptq = evaluate_imagenet_full(apply_standard_ptq(model_ptq, w_bits, a_bits), processor, dataset, DEVICE)
            print(f"     => Standard PTQ Acc: {acc_ptq:.2f}%")
            del model_ptq;
            torch.cuda.empty_cache()

            # 3. ADVANCED SOTA PTQ
            print("      [System] Competitor 2: Advanced SOTA PTQ (Per-Channel)...")
            model_sota = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
            acc_sota = evaluate_imagenet_full(apply_advanced_sota_ptq(model_sota, w_bits, a_bits), processor, dataset,
                                              DEVICE)
            print(f"     => Advanced SOTA Acc: {acc_sota:.2f}%")
            del model_sota;
            torch.cuda.empty_cache()

            # 4. ROVIT (OURS)
            print("      [System] Our Method: RoViT...")
            model_rovit = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
            acc_rovit = evaluate_imagenet_full(apply_rovit_ptq(model_rovit, w_bits, a_bits), processor, dataset, DEVICE)
            print(f"     => RoViT Acc: {acc_rovit:.2f}%\n")
            del model_rovit;
            torch.cuda.empty_cache()

            results_log.append({
                'Model': model_name.split('/')[-1],
                'Bit_Width': exp_name,
                'FP32_Acc(%)': round(acc_fp32, 2),
                'Standard_PTQ(%)': round(acc_ptq, 2),
                'Advanced_SOTA(%)': round(acc_sota, 2),
                'RoViT_Ours(%)': round(acc_rovit, 2)
            })

    # =============================================================================
    # 5. EXPORT FINAL BENCHMARK
    # =============================================================================
    df_results = pd.DataFrame(results_log)
    output_filename = 'Experiment_Results_03.csv'
    df_results.to_csv(output_filename, index=False)
    print("\n" + "=" * 60 + f"\n[*] BENCHMARK COMPLETED!\n[*] Results saved to: {output_filename}\n")
    print(df_results.to_string(index=False))


if __name__ == '__main__':
    main()