"""Layer-wise relative quantization error at W4A4 on ViT-Base
(Table 14, Figure 6): standard PTQ vs RoViT-rotated activations."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp
from rovit.quant import quantize
from rovit.data import load_imagenet_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "google/vit-base-patch16-224"


def measure(mode, dataset, num_calib, a_bits, seed):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(MODEL).to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model.eval()

    rot = (build_rotations(model, is_mlp, "qr", seed=seed, device=DEVICE)
           if mode == "rovit" else {})
    records = {}
    captures = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "encoder.layer" in name:
            handles.append(mod.register_forward_hook(
                lambda m, inp, out, n=name: captures.__setitem__(n, inp[0].detach())))

    with torch.no_grad():
        for i in tqdm(range(num_calib), desc=f"collect[{mode}]", leave=False):
            img = dataset[i]["image"].convert("RGB")
            model(**proc(images=img, return_tensors="pt").to(DEVICE))
            for name, x in captures.items():
                if name in rot:
                    x = x @ rot[name]
                err = (x - quantize(x, a_bits)).pow(2).mean().item()
                rel = err / x.pow(2).mean().clamp(min=1e-12).item()
                block = next((int(t) for t in name.split(".") if t.isdigit()), -1)
                rec = records.setdefault(name, {
                    "block": block, "is_mlp": is_mlp(name), "rel": []})
                rec["rel"].append(rel)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame([{"name": n, "block": r["block"], "is_mlp": r["is_mlp"],
                          "mean_rel_err": np.mean(r["rel"]), "mode": mode}
                         for n, r in records.items()])


def plot(df_std, df_rov, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, title in [(axes[0], df_std, "Standard PTQ (W4A4)"),
                          (axes[1], df_rov, "RoViT (W4A4, MLP-rotated)")]:
        blocks = sorted(df.block.unique())
        for offset, flag, label, color in [(-0.2, True, "MLP layers", "#d62728"),
                                           (0.2, False, "Attention layers", "#1f77b4")]:
            vals = [df[(df.block == b) & (df.is_mlp == flag)].mean_rel_err.mean()
                    for b in blocks]
            ax.bar(np.arange(len(blocks)) + offset, vals, width=0.4,
                   label=label, color=color)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks)
        ax.set_xlabel("Encoder block index")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(r"Relative quantization error $\|\Delta X\|^2/\|X\|^2$ (log)")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-calib", type=int, default=64)
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dataset = load_imagenet_val()
    df_std = measure("standard", dataset, args.num_calib, args.a_bits, args.seed)
    df_rov = measure("rovit", dataset, args.num_calib, args.a_bits, args.seed)
    pd.concat([df_std, df_rov]).to_csv(
        Path(args.outdir) / "layerwise_error.csv", index=False)
    plot(df_std, df_rov, Path(args.outdir) / "z_fig_error_breakdown.pdf")

    for label, df in [("standard", df_std), ("rovit", df_rov)]:
        mlp = df[df.is_mlp].mean_rel_err.mean()
        attn = df[~df.is_mlp].mean_rel_err.mean()
        print(f"{label:8s} mlp={mlp:.4f} attn={attn:.4f} ratio={mlp/attn:.2f}x")


if __name__ == "__main__":
    main()
