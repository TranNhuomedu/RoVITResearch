import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import pandas as pd

# Import Hugging Face datasets
from datasets import load_dataset

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

HF_TOKEN = ""
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENTS = {
    'W8A8': {'weight_bits': 8, 'act_bits': 8, 'models': ['deit_base_patch16_224']},
    'W6A6': {'weight_bits': 6, 'act_bits': 6, 'models': ['deit_base_patch16_224', 'vit_small_patch16_224']},
    'W4A4': {'weight_bits': 4, 'act_bits': 4, 'models': ['deit_base_patch16_224', 'vit_small_patch16_224']}
}

# =============================================================================
# 2. HUGGING FACE DATALOADER (Windows-safe)
# =============================================================================
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def hf_val_transforms(examples):
    images = [base_transform(img.convert("RGB")) for img in examples["image"]]
    return {"pixel_values": images, "label": examples["label"]}


def hf_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return pixel_values, labels


def get_hf_imagenet_val_loader(batch_size, num_workers, hf_token):
    print("      [System] Loading ImageNet-1K validation set from Hugging Face...")
    dataset = load_dataset("imagenet-1k", split="validation", token=hf_token)
    dataset.set_transform(hf_val_transforms)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=hf_collate_fn, pin_memory=True
    )
    return loader


# =============================================================================
# 3. CORE QUANTIZATION & ROVIT MATH
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    """
    Simulates Symmetric Uniform Quantization.
    Formula: X_q = round(X / scale) * scale
    """
    if bits >= 32:
        return tensor

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    # Calculate step size (scale) based on absolute maximum
    max_val = tensor.abs().max().clamp(min=1e-8)
    scale = max_val / qmax

    # Quantize and Dequantize
    quantized_tensor = torch.round(tensor / scale).clamp(qmin, qmax)
    dequantized_tensor = quantized_tensor * scale
    return dequantized_tensor


# Global variable to avoid spamming the console with hook prints
hook_print_count = 0


def standard_ptq_pre_hook(module, args, a_bits):
    """
    Pre-hook to quantize input Activations (X) BEFORE the Linear layer processes them.
    """
    global hook_print_count
    if hook_print_count < 3:
        print(f"      [Debug] Standard PTQ Hook triggered on {module.__class__.__name__}!")
        hook_print_count += 1

    X = args[0]
    X_q = fake_quantize_tensor(X, a_bits)
    return (X_q,)  # Must return a tuple for pre_hook


def apply_standard_ptq(model, w_bits, a_bits):
    """Applies Standard PTQ to all Linear layers."""
    global hook_print_count
    hook_print_count = 0  # Reset counter
    print(f"      [System] Applying Standard PTQ (W{w_bits}A{a_bits})...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 1. Update Weights statically (Offline Fusion)
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)

            # 2. Register Pre-Hook for Activations
            hook = lambda mod, inp, b=a_bits: standard_ptq_pre_hook(mod, inp, b)
            module.register_forward_pre_hook(hook)

    return model


def rovit_pre_hook(module, args, a_bits, R_matrix):
    """
    Pre-hook for RoViT: Rotates X to X', then quantizes to X'_q BEFORE Linear layer.
    """
    global hook_print_count
    if hook_print_count < 3:
        print(f"      [Debug] RoViT Hook triggered on {module.__class__.__name__}!")
        hook_print_count += 1

    X = args[0]
    # Rotate Activation: X' = X @ R
    X_rotated = torch.matmul(X, R_matrix)
    # Quantize Rotated Activation
    X_q = fake_quantize_tensor(X_rotated, a_bits)

    return (X_q,)


def apply_rovit_ptq(model, w_bits, a_bits):
    """Applies RoViT (Rotation-based PTQ) to all Linear layers."""
    global hook_print_count
    hook_print_count = 0  # Reset counter
    print(f"      [System] Applying RoViT Orthogonal PTQ (W{w_bits}A{a_bits})...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            d_out, d_in = module.weight.shape

            # 1. Generate Orthogonal Matrix R using QR Decomposition
            random_matrix = torch.randn(d_in, d_in, device=device)
            Q, _ = torch.linalg.qr(random_matrix)
            R = Q  # R satisfies R @ R^T = I

            # 2. Rotate Weights: W' = W @ R
            W_rotated = torch.matmul(module.weight.data, R)

            # 3. Quantize and Update Weights statically
            module.weight.data = fake_quantize_tensor(W_rotated, w_bits)

            # 4. Register Pre-Hook to rotate & quantize Activations dynamically
            hook = lambda mod, inp, b=a_bits, r=R: rovit_pre_hook(mod, inp, b, r)
            module.register_forward_pre_hook(hook)

    return model


# =============================================================================
# 4. EVALUATION PIPELINE
# =============================================================================
def evaluate_top1_accuracy(model, data_loader, device, num_batches=None):
    model.eval()
    correct_top1 = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = outputs.topk(1, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(labels.view(1, -1).expand_as(predicted))

            correct_top1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            total_samples += labels.size(0)

    return (correct_top1 / total_samples) * 100.0


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def main():
    print(f"[*] Starting experiments on device: {DEVICE}")
    val_loader = get_hf_imagenet_val_loader(BATCH_SIZE, NUM_WORKERS, HF_TOKEN)
    results_log = []

    for exp_name, config in EXPERIMENTS.items():
        w_bits, a_bits = config['weight_bits'], config['act_bits']
        print(f"\n{'=' * 60}\nEXPERIMENT: {exp_name} (W{w_bits}A{a_bits})\n{'=' * 60}")

        for model_name in config['models']:
            print(f"\n---> Processing Model: {model_name}")

            # TODO: Giữ num_batches=5 để test. Khi lấy số liệu thật, sửa thành None.
            NUM_TEST_BATCHES = None

            # --- 1. FP32 Baseline ---
            model_fp32 = timm.create_model(model_name, pretrained=True).to(DEVICE)
            print("   [1/3] Evaluating FP32 Baseline...")
            acc_fp32 = evaluate_top1_accuracy(model_fp32, val_loader, DEVICE, num_batches=NUM_TEST_BATCHES)
            print(f"         => Top-1 Accuracy: {acc_fp32:.2f}%")

            # --- 2. Standard PTQ ---
            model_ptq = timm.create_model(model_name, pretrained=True).to(DEVICE)
            model_ptq = apply_standard_ptq(model_ptq, w_bits, a_bits)
            print("   [2/3] Evaluating Standard PTQ...")
            acc_ptq = evaluate_top1_accuracy(model_ptq, val_loader, DEVICE, num_batches=NUM_TEST_BATCHES)
            print(f"         => Top-1 Accuracy: {acc_ptq:.2f}%")

            # --- 3. RoViT PTQ ---
            model_rovit = timm.create_model(model_name, pretrained=True).to(DEVICE)
            model_rovit = apply_rovit_ptq(model_rovit, w_bits, a_bits)
            print("   [3/3] Evaluating RoViT (Proposed Method)...")
            acc_rovit = evaluate_top1_accuracy(model_rovit, val_loader, DEVICE, num_batches=NUM_TEST_BATCHES)
            print(f"         => Top-1 Accuracy: {acc_rovit:.2f}%")

            absolute_boost = acc_rovit - acc_ptq

            results_log.append({
                'Quantization_Level': exp_name,
                'Model_Architecture': model_name,
                'FP32_Baseline(%)': round(acc_fp32, 2),
                'Standard_PTQ(%)': round(acc_ptq, 2),
                'RoViT(%)': round(acc_rovit, 2),
                'Absolute_Boost(%)': round(absolute_boost, 2)
            })

    df_results = pd.DataFrame(results_log)
    output_filename = 'Experiment_Results_01.csv'
    df_results.to_csv(output_filename, index=False)

    print("\n" + "=" * 60)
    print("QUICK SUMMARY:")
    print(df_results.to_string(index=False))


if __name__ == '__main__':
    main()