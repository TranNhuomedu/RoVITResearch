"""Figure 4: CLS-token attention spatial maps, FP32 vs RoViT-QR vs
truncated Hadamard (ViT-Base, last encoder layer, head 0).

Each heatmap is min-max normalized independently; state this in the
LaTeX caption. Correlations are reported per image and averaged.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp, prepare, uniform_policy
from rovit.data import load_imagenet_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "google/vit-base-patch16-224"


def _patched_forward(module, hidden_states, *args, **kwargs):
    b, n, _ = hidden_states.shape
    h, d = module.num_attention_heads, module.attention_head_size
    q = module.query(hidden_states).reshape(b, n, h, d).transpose(1, 2)
    k = module.key(hidden_states).reshape(b, n, h, d).transpose(1, 2)
    v = module.value(hidden_states).reshape(b, n, h, d).transpose(1, 2)
    attn = torch.softmax(q @ k.transpose(-2, -1) / d ** 0.5, dim=-1)
    module._captured = attn.detach()
    out = (attn @ v).transpose(1, 2).reshape(b, n, h * d)
    return (out, attn)


def cls_spatial_map(model, processor, image, layer_idx=-1, head=0, side=14):
    mods = [(n, m) for n, m in model.named_modules()
            if n.endswith("attention.attention")]
    name, target = mods[layer_idx]
    original = target.forward
    target.forward = types.MethodType(_patched_forward, target)
    try:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        attn = target._captured[0, head].cpu().numpy()
    finally:
        target.forward = original
    return attn[0, 1:].reshape(side, side)


def build(variant, w_bits, a_bits, seed):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(MODEL).to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    if variant != "fp32":
        rot = build_rotations(model, is_mlp, kind=variant,
                              seed=seed, device=DEVICE)
        prepare(model, uniform_policy(w_bits, a_bits), rotations=rot)
    return model, proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indices", type=int, nargs="+",
                   default=[100, 1000, 5000, 12000])
    p.add_argument("--w-bits", type=int, default=6)
    p.add_argument("--a-bits", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dataset = load_imagenet_val()
    images = [dataset[i]["image"].convert("RGB") for i in args.indices]

    maps = {}
    for variant in ["fp32", "qr", "hadamard"]:
        model, proc = build(variant, args.w_bits, args.a_bits, args.seed)
        maps[variant] = [cls_spatial_map(model, proc, im) for im in images]
        del model; torch.cuda.empty_cache()

    n = len(images)
    fig, axes = plt.subplots(4, n, figsize=(3.2 * n, 12))
    labels = ["Original", "FP32 (Reference)",
              f"RoViT-QR W{args.w_bits}A{args.a_bits}",
              f"Truncated Hadamard W{args.w_bits}A{args.a_bits}"]
    for c in range(n):
        axes[0, c].imshow(images[c])
        for r, key in zip([1, 2, 3], ["fp32", "qr", "hadamard"]):
            axes[r, c].imshow(maps[key][c], cmap="viridis",
                              interpolation="nearest")
        for r in range(4):
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(labels[r], fontsize=11, loc="left")
    plt.tight_layout()
    out_pdf = Path(args.outdir) / "z_fig_attention_heatmap.pdf"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)

    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    rows = [{"index": idx,
             "corr_qr": corr(maps["fp32"][i], maps["qr"][i]),
             "corr_hadamard": corr(maps["fp32"][i], maps["hadamard"][i])}
            for i, idx in enumerate(args.indices)]
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.outdir) / "attention_correlation.csv", index=False)
    print(df.to_string(index=False))
    print(f"mean corr: qr={df.corr_qr.mean():.3f} "
          f"hadamard={df.corr_hadamard.mean():.3f}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
