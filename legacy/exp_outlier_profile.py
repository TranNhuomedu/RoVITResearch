"""M/mu outlier profiling across architectures (Table 6, Figure 3).

Rotation matrices are sized to each layer's actual input dimension
(intermediate.dense: d, output.dense: 4d)."""

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
from rovit import is_mlp
from rovit.rotation import make_matrix
from rovit.data import load_imagenet_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHS = {
    "ViT-Small": ("WinKawaks/vit-small-patch16-224", 384),
    "ViT-Base": ("google/vit-base-patch16-224", 768),
    "DeiT-Base": ("facebook/deit-base-distilled-patch16-224", 768),
}


def channel_stats(x):
    flat = x.reshape(-1, x.shape[-1])
    m = flat.abs().amax(0).max().item()
    mu = flat.abs().mean(0).mean().item()
    return m, mu, m / max(mu, 1e-12)


def profile(label, hf_name, d_model, dataset, num_calib, trials, seed):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(hf_name).to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(hf_name)
    model.eval()

    captures = {n: [] for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and is_mlp(n)}
    handles = [m.register_forward_hook(
        lambda mod, inp, out, n=n: captures[n].append(inp[0].detach().cpu()))
        for n, m in model.named_modules() if n in captures]
    with torch.no_grad():
        for i in tqdm(range(num_calib), desc=label, leave=False):
            img = dataset[i]["image"].convert("RGB")
            model(**proc(images=img, return_tensors="pt").to(DEVICE))
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    rows = []
    gen = torch.Generator()
    for name, chunks in captures.items():
        x = torch.cat(chunks).reshape(-1, chunks[0].shape[-1]).float()
        d_in, n_rows = x.shape[-1], x.shape[0]
        rms = x.norm().item() / ((n_rows * d_in) ** 0.5)
        m0, _, r0 = channel_stats(x)
        post_ratio, post_kappa = [], []
        for t in range(trials):
            gen.manual_seed(seed + t)
            xr = x @ make_matrix("qr", d_in, gen)
            _, _, r1 = channel_stats(xr)
            post_ratio.append(r1)
            post_kappa.append(xr.abs().amax(0).max().item() / max(rms, 1e-12))
        kind = "mlp_in" if "intermediate" in name else "mlp_out"
        rows.append({
            "arch": label, "d_model": d_model, "d_in": d_in, "layer": name,
            "layer_kind": kind, "M_over_mu_pre": r0,
            "M_over_mu_post": float(np.mean(post_ratio)),
            "kappa_pre": x.abs().amax(0).max().item() / max(rms, 1e-12),
            "kappa_post": float(np.mean(post_kappa)),
            "kappa_post_std": float(np.std(post_kappa)),
        })
    return pd.DataFrame(rows)


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(summary))
    axes[0].bar(x - 0.2, summary.M_over_mu_pre, 0.4,
                label="Before rotation", color="#d62728")
    axes[0].bar(x + 0.2, summary.M_over_mu_post, 0.4,
                label="After QR rotation", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary.arch)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Average M/mu ratio (post-GELU MLP layers)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].plot(summary.d_model, summary.kappa_post, "o-", markersize=10)
    for _, r in summary.iterrows():
        axes[1].annotate(r.arch, (r.d_model, r.kappa_post),
                         textcoords="offset points", xytext=(8, 4))
    axes[1].set_xlabel("Hidden dimension d_model")
    axes[1].set_ylabel("Empirical kappa_post")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-calib", type=int, default=64)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dataset = load_imagenet_val()
    df = pd.concat([profile(k, n, d, dataset, args.num_calib,
                            args.trials, args.seed)
                    for k, (n, d) in ARCHS.items()], ignore_index=True)
    df.to_csv(Path(args.outdir) / "outlier_profile.csv", index=False)
    summary = (df[df.layer_kind == "mlp_out"].groupby("arch", sort=False)
               .agg(d_model=("d_model", "first"),
                    M_over_mu_pre=("M_over_mu_pre", "mean"),
                    M_over_mu_post=("M_over_mu_post", "mean"),
                    kappa_pre=("kappa_pre", "mean"),
                    kappa_post=("kappa_post", "mean")).reset_index())
    print(summary.to_string(index=False, float_format="%.2f"))
    plot(summary, Path(args.outdir) / "z_fig_outlier_profile.pdf")


if __name__ == "__main__":
    main()
