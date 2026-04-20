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
DATA_CACHE_DIR = "./imagenet_data"

BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'google/vit-base-patch16-224'

# Sử dụng dải W6A6 để khuếch đại sự khác biệt giữa các cấu hình (dễ thấy sai số nhất)
W_BITS = 6
A_BITS = 6


# =============================================================================
# 2. MATRIX GENERATORS (ABLATION PART A)
# =============================================================================
def get_rotation_matrix(matrix_type, d, device):

    if matrix_type == 'Identity':
        return torch.eye(d, device=device)

    elif matrix_type == 'Gaussian':
        return torch.randn(d, d, device=device) / (d ** 0.5)

    elif matrix_type == 'Householder':
        v = torch.randn(d, 1, device=device)
        v = v / torch.norm(v)
        return torch.eye(d, device=device) - 2 * torch.matmul(v, v.T)

    elif matrix_type == 'QR_Decomposition':
        Q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        return Q
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


# =============================================================================
# 3. ROVIT QUANTIZATION ALGORITHM WITH LAYER TARGETING (ABLATION PART B)
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32: return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def rovit_hook(mod, inp, b, R_matrix):
    X_rotated = torch.matmul(inp[0], R_matrix)
    return (fake_quantize_tensor(X_rotated, b),)


def apply_ablation_rovit(model, w_bits, a_bits, matrix_type='QR_Decomposition', target_modules='all'):

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Xác định vị trí layer
            is_attention = 'attention' in name
            is_mlp = ('intermediate' in name) or ('output.dense' in name and 'attention' not in name)

            # Cấu hình bỏ qua
            if target_modules == 'attention_only' and not is_attention:
                module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
                module.register_forward_pre_hook(lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
                continue

            if target_modules == 'mlp_only' and not is_mlp:
                module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
                module.register_forward_pre_hook(lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
                continue

            device = module.weight.device
            d = module.weight.shape[1]
            R = get_rotation_matrix(matrix_type, d, device)

            module.weight.data = fake_quantize_tensor(torch.matmul(module.weight.data, R), w_bits)
            module.register_forward_pre_hook(lambda mod, inp, b=a_bits, r=R: rovit_hook(mod, inp, b, r))

    return model


# =============================================================================
# 4. HIGH-PERFORMANCE DATALOADER & EVALUATION PIPELINE
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

            progress_bar.set_postfix({'Acc': f"{(correct / total) * 100:.2f}%"})

    return (correct / total) * 100.0


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def main():
    print(f"[*] STARTING FULL 50K ABLATION STUDIES (GROUP 4) on {DEVICE}\n")

    print("[System] Loading ImageNet-1K Validation Set from local SSD...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", token=HF_TOKEN, streaming=False,
                           cache_dir=DATA_CACHE_DIR, trust_remote_code=True)
    print(f"[System] Dataset ready. Total samples: {len(dataset)}\n")

    results_log = []
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # 0. FP32 BASELINE
    print(f"---> Evaluating Baseline FP32 Model...")
    model_fp32 = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    acc_fp32 = evaluate_imagenet_full(model_fp32, processor, dataset, DEVICE)
    print(f"     => Baseline Top-1 Acc: {acc_fp32:.2f}%\n")
    del model_fp32;
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PART A: ABLATION ON MATRIX TYPES (target_modules='all')
    # -------------------------------------------------------------------------
    print(f"{'#' * 60}\nPART A: MATRIX TYPE ABLATION (W6A6)\n{'#' * 60}")
    matrix_types = ['Identity', 'Gaussian', 'Householder', 'QR_Decomposition']

    for m_type in matrix_types:
        print(f"--- Matrix: {m_type} ---")
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        model = apply_ablation_rovit(model, W_BITS, A_BITS, matrix_type=m_type, target_modules='all')
        acc = evaluate_imagenet_full(model, processor, dataset, DEVICE)
        print(f"     => Top-1 Acc: {acc:.2f}%\n")
        del model;
        torch.cuda.empty_cache()

        results_log.append({
            'Study_Type': 'Matrix Type',
            'Configuration': m_type,
            'Top1_Acc(%)': round(acc, 2),
            'Note': 'Baseline PTQ' if m_type == 'Identity' else (
                'Destroys Math' if m_type == 'Gaussian' else 'Orthogonal')
        })

    # -------------------------------------------------------------------------
    # PART B: ABLATION ON LAYER LOCATIONS (matrix='QR_Decomposition')
    # -------------------------------------------------------------------------
    print(f"{'#' * 60}\nPART B: LAYER-WISE ABLATION (W6A6)\n{'#' * 60}")
    layer_targets = ['attention_only', 'mlp_only', 'all']

    for target in layer_targets:
        print(f"--- Layers: {target} ---")
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        model = apply_ablation_rovit(model, W_BITS, A_BITS, matrix_type='QR_Decomposition', target_modules=target)
        acc = evaluate_imagenet_full(model, processor, dataset, DEVICE)
        print(f"     => Top-1 Acc: {acc:.2f}%\n")
        del model;
        torch.cuda.empty_cache()

        results_log.append({
            'Study_Type': 'Layer Location',
            'Configuration': target,
            'Top1_Acc(%)': round(acc, 2),
            'Note': 'Full RoViT' if target == 'all' else 'Partial RoViT'
        })

    # =============================================================================
    # EXPORT RESULTS
    # =============================================================================
    df_results = pd.DataFrame(results_log)
    output_filename = 'Experiment_Results_04.csv'
    df_results.to_csv(output_filename, index=False)
    print("\n" + "=" * 60 + f"\n[*] ABLATION STUDIES COMPLETED!\n[*] Results saved to: {output_filename}\n")
    print(df_results.to_string(index=False))


if __name__ == '__main__':
    main()