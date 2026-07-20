"""Ablations on ViT-Base at W6A6 (Tables 11 and 12).

Part A: rotation-matrix type {identity, gaussian, householder,
        truncated hadamard, qr}, applied to all Linear layers.
Part B: layer targeting {attn, mlp, all} with QR rotation.

Each layer draws its own matrix from a single seeded generator.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp, layer_role, prepare, select_all, uniform_policy
from rovit.data import load_imagenet_val
from rovit.eval_utils import top1_hf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "google/vit-base-patch16-224"


def fresh_model():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    return (AutoModelForImageClassification.from_pretrained(MODEL).to(DEVICE),
            AutoImageProcessor.from_pretrained(MODEL))


def run(select, kind, bits, seed, dataset, batch, workers):
    model, proc = fresh_model()
    if kind is not None:
        rot = build_rotations(model, select, kind=kind, seed=seed, device=DEVICE)
        prepare(model, uniform_policy(*bits), rotations=rot)
    else:
        prepare(model, uniform_policy(*bits))
    acc = top1_hf(model, proc, dataset, DEVICE, batch, workers)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w-bits", type=int, default=6)
    p.add_argument("--a-bits", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results/rotation_ablation.csv")
    args = p.parse_args()
    bits = (args.w_bits, args.a_bits)
    dataset = load_imagenet_val()
    rows = []

    for kind in [None, "gaussian", "householder", "hadamard", "qr"]:
        acc = run(select_all, kind, bits, args.seed, dataset,
                  args.batch, args.workers)
        label = kind or "std_ptq"
        rows.append({"study": "matrix_type", "config": label,
                     "top1": round(acc, 2)})
        print(f"matrix={label:12s} {acc:.2f}%")

    targets = {"attn_only": lambda n: layer_role(n) == "attn",
               "mlp_only": is_mlp, "all": select_all}
    for label, select in targets.items():
        acc = run(select, "qr", bits, args.seed, dataset,
                  args.batch, args.workers)
        rows.append({"study": "targeting", "config": label,
                     "top1": round(acc, 2)})
        print(f"target={label:10s} {acc:.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
