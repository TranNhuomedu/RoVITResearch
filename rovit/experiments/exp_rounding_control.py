"""Rounding-only control for the RoViT + rounding combination (closes L2).

Configuration: per-output-channel weight rounding, symmetric per-tensor
dynamic activations, NO rotation -- exactly the combined system of
exp_robustness.py gptq minus the rotation term. Together with the existing
measurements this completes the 2x2 decomposition at W4A4 on ViT-Base:

    neither          : 0.09   (classification.csv, std)
    rotation only    : 31.27  (classification.csv, rovit)
    rounding only    : THIS SCRIPT
    both             : 46.60  (robustness_gptq.csv)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import prepare, uniform_policy
from rovit.data import load_imagenet_val
from rovit.eval_utils import top1_hf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/vit-base-patch16-224")
    p.add_argument("--bits", nargs="+", default=["W4A4"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/gptq_only.csv")
    args = p.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    dataset = load_imagenet_val()
    bit_map = {"W4A4": (4, 4), "W6A6": (6, 6), "W8A8": (8, 8)}
    rows = []
    for tag in args.bits:
        w, a = bit_map[tag]
        model = AutoModelForImageClassification.from_pretrained(args.model).to(DEVICE)
        proc = AutoImageProcessor.from_pretrained(args.model)
        prepare(model, uniform_policy(w, a), weight_per_channel=True)
        acc = top1_hf(model, proc, dataset, DEVICE, args.batch, args.workers)
        rows.append({"model": args.model, "bits": tag,
                     "rounding_only_top1": round(acc, 2)})
        print(f"{args.model} {tag} rounding-only (no rotation): {acc:.2f}%")
        del model
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
