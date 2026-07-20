"""PTQ4ViT reproduced in this pipeline (Table 4 baseline row).

Implements twin-uniform quantization for post-GELU activations, per-channel
symmetric weight quantization, and a diagonal-Hessian scale search on the
calibration set. The twin log-scale path for post-softmax attention from the
original paper is not reproduced; see the manuscript's baseline note.

Reference: Yuan et al., ECCV 2022 (github.com/hahnyuan/PTQ4ViT).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit.quant import quantize
from rovit.data import load_imagenet_val, sample_calibration_indices
from rovit.eval_utils import top1_hf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["google/vit-base-patch16-224", "facebook/deit-small-patch16-224"]


def twin_quant(x, bits, scale_pos=None, scale_neg=None):
    """Separate uniform scales for the positive and negative parts."""
    if bits >= 32:
        return x
    qmax = (2 ** (bits - 1)) - 1
    pos, neg = x.clamp(min=0), (-x).clamp(min=0)
    if scale_pos is None:
        scale_pos = pos.max().clamp(min=1e-8) / qmax
    if scale_neg is None:
        scale_neg = neg.max().clamp(min=1e-8) / qmax
    return (torch.round(pos / scale_pos).clamp(0, qmax) * scale_pos
            - torch.round(neg / scale_neg).clamp(0, qmax) * scale_neg)


def scale_search(x_calib, weight, bits, n_candidates=20):
    """Grid search over twin scales, weighted by diag(W^T W) sensitivity."""
    qmax = (2 ** (bits - 1)) - 1
    imp = (weight ** 2).sum(0)
    imp = imp / imp.sum().clamp(min=1e-8)
    flat = x_calib.reshape(-1, x_calib.shape[-1])
    pos_max = flat.clamp(min=0).max().clamp(min=1e-8).item()
    neg_max = (-flat).clamp(min=0).max().clamp(min=1e-8).item()
    best = (pos_max / qmax, max(neg_max / qmax, 1e-8), float("inf"))
    for ap in torch.linspace(0.7, 1.2, n_candidates):
        for an in torch.linspace(0.7, 1.2, n_candidates):
            sp = float(ap) * pos_max / qmax
            sn = max(float(an) * neg_max / qmax, 1e-8)
            xq = twin_quant(flat, bits,
                            torch.tensor(sp, device=flat.device),
                            torch.tensor(sn, device=flat.device))
            loss = (((flat - xq) ** 2).mean(0) * imp.to(flat.device)).sum().item()
            if loss < best[2]:
                best = (sp, sn, loss)
    return best[0], best[1]


def is_post_gelu(name):
    n = name.lower()
    return "output.dense" in n and "attention" not in n


def collect(model, proc, dataset, indices, layer_names):
    captures = {n: [] for n in layer_names}
    handles = [m.register_forward_hook(
        lambda mod, inp, out, n=n: captures[n].append(inp[0].detach().cpu()))
        for n, m in model.named_modules() if n in captures]
    model.eval()
    with torch.no_grad():
        for i in tqdm(indices, desc="calibrate", leave=False):
            img = dataset[int(i)]["image"].convert("RGB")
            model(**proc(images=img, return_tensors="pt").to(DEVICE))
    for h in handles:
        h.remove()
    return {n: torch.cat(v) for n, v in captures.items()}


def apply_ptq4vit(model, w_bits, a_bits, calib):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        mod.weight.data = quantize(mod.weight.data, w_bits, per_channel_dim=1)
        if is_post_gelu(name) and name in calib:
            sp, sn = scale_search(calib[name].to(DEVICE), mod.weight.data, a_bits)
            mod.register_forward_pre_hook(
                lambda m, inp, b=a_bits,
                p=torch.tensor(sp, device=DEVICE),
                n_=torch.tensor(sn, device=DEVICE):
                (twin_quant(inp[0], b, p, n_),))
        else:
            mod.register_forward_pre_hook(
                lambda m, inp, b=a_bits: (quantize(inp[0], b),))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--bits", nargs="+", default=["W6A6", "W4A4"])
    p.add_argument("--num-calib", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/ptq4vit_baseline.csv")
    args = p.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    dataset = load_imagenet_val()
    indices = sample_calibration_indices(args.num_calib, len(dataset), args.seed)
    bit_map = {"W6A6": (6, 6), "W4A4": (4, 4)}
    rows = []
    for name in args.models:
        proc = AutoImageProcessor.from_pretrained(name)
        base = AutoModelForImageClassification.from_pretrained(name).to(DEVICE)
        targets = [n for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and is_post_gelu(n)]
        calib = collect(base, proc, dataset, indices, targets)
        del base
        torch.cuda.empty_cache()
        for tag in args.bits:
            model = AutoModelForImageClassification.from_pretrained(name).to(DEVICE)
            apply_ptq4vit(model, *bit_map[tag], calib)
            acc = top1_hf(model, proc, dataset, DEVICE)
            rows.append({"model": name, "bits": tag, "top1": round(acc, 2)})
            print(f"{name} {tag}: {acc:.2f}%")
            del model
            torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
