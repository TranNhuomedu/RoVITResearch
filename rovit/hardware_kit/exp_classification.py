"""ImageNet-1K classification under PTQ (Tables 4 and 5).

Methods: fp32 | std (per-tensor symmetric PTQ) | advanced (per-channel
weights + asymmetric activations) | rovit (MLP-only QR rotation).
Backends: HuggingFace names contain '/', timm names do not.

If --calib is given, activation scales are frozen from that calibration
subset (static MinMax); otherwise activations are quantized dynamically.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, calibrate, is_mlp, prepare, uniform_policy
from rovit.data import load_imagenet_val, load_calibration_indices
from rovit.eval_utils import top1_hf, top1_timm, hf_calibration_runner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODELS = [
    "google/vit-base-patch16-224",
    "facebook/deit-small-patch16-224",
    "deit_base_patch16_224",
    "vit_small_patch16_224",
]
BIT_CONFIGS = {"W8A8": (8, 8), "W6A6": (6, 6), "W4A4": (4, 4)}


def load_model(name):
    if "/" in name:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(name).to(DEVICE)
        return model, AutoImageProcessor.from_pretrained(name), "hf"
    import timm
    return timm.create_model(name, pretrained=True).to(DEVICE), None, "timm"


def evaluate(model, processor, backend, dataset, batch, workers):
    if backend == "hf":
        return top1_hf(model, processor, dataset, DEVICE, batch, workers)
    return top1_timm(model, dataset, DEVICE, batch, workers)


def run_method(name, method, bits, dataset, args, calib_idx):
    w, a = bits
    model, processor, backend = load_model(name)
    if method == "fp32":
        hooks = {}
    elif method == "std":
        hooks = prepare(model, uniform_policy(w, a))
    elif method == "advanced":
        hooks = prepare(model, uniform_policy(w, a),
                        weight_per_channel=True, act_symmetric=False)
    elif method == "rovit":
        rot = build_rotations(model, is_mlp, kind=args.rotation,
                              seed=args.seed, device=DEVICE)
        hooks = prepare(model, uniform_policy(w, a), rotations=rot)
    else:
        raise ValueError(method)

    if calib_idx is not None and hooks and backend == "hf":
        calibrate(hooks, hf_calibration_runner(
            model, processor, dataset, calib_idx, DEVICE))

    acc = evaluate(model, processor, backend, dataset, args.batch, args.workers)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--bits", nargs="+", default=["W8A8", "W6A6", "W4A4"])
    p.add_argument("--methods", nargs="+",
                   default=["fp32", "std", "advanced", "rovit"])
    p.add_argument("--calib", default=None,
                   help="calibration_indices.txt for static activation scales")
    p.add_argument("--rotation", default="qr",
                   help="qr | qr_block | qr_block:<size> (block-diagonal QR)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/classification.csv")
    args = p.parse_args()

    dataset = load_imagenet_val()
    calib_idx = load_calibration_indices(args.calib) if args.calib else None
    rows = []
    for name in args.models:
        fp32_acc = None                      # FP32 is bit-independent: run once
        for tag in args.bits:
            row = {"model": name, "bits": tag}
            for method in args.methods:
                if method == "fp32":
                    if fp32_acc is None:
                        fp32_acc = run_method(name, method, (32, 32),
                                              dataset, args, calib_idx)
                    acc = fp32_acc
                else:
                    acc = run_method(name, method, BIT_CONFIGS[tag],
                                     dataset, args, calib_idx)
                row[method] = round(acc, 2)
                print(f"{name:40s} {tag} {method:9s} {acc:6.2f}%")
            rows.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
