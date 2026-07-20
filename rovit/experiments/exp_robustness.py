"""Robustness studies on ViT-Base (Section 4.1.3 and Table 13 context).

Subcommands:
  seeds  : Top-1 over independent rotation seeds (W6A6)
  calib  : sensitivity to calibration-set size (static MinMax scales)
  kappa  : dispersion ratio on real activations, before/after rotation
  gptq   : RoViT + per-output-channel weight rounding (W4A4)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, calibrate, is_mlp, prepare, uniform_policy
from rovit.data import load_imagenet_val, sample_calibration_indices
from rovit.eval_utils import top1_hf, hf_calibration_runner
from rovit.rotation import make_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "google/vit-base-patch16-224"


def fresh_model():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    return (AutoModelForImageClassification.from_pretrained(MODEL).to(DEVICE),
            AutoImageProcessor.from_pretrained(MODEL))


def cmd_seeds(args, dataset):
    rows = []
    for seed in args.seeds:
        model, proc = fresh_model()
        rot = build_rotations(model, is_mlp, "qr", seed=seed, device=DEVICE)
        prepare(model, uniform_policy(args.w_bits, args.a_bits), rotations=rot)
        acc = top1_hf(model, proc, dataset, DEVICE, args.batch, args.workers)
        rows.append({"seed": seed, "top1": round(acc, 2)})
        print(f"seed={seed}: {acc:.2f}%")
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(f"mean={df.top1.mean():.2f}  std={df.top1.std(ddof=1):.3f}")
    return df


def cmd_calib(args, dataset):
    rows = []
    for size in args.calib_sizes:
        model, proc = fresh_model()
        rot = build_rotations(model, is_mlp, "qr", seed=args.seed, device=DEVICE)
        hooks = prepare(model, uniform_policy(args.w_bits, args.a_bits),
                        rotations=rot)
        idx = sample_calibration_indices(size, len(dataset), seed=args.seed)
        calibrate(hooks, hf_calibration_runner(model, proc, dataset, idx, DEVICE))
        acc = top1_hf(model, proc, dataset, DEVICE, args.batch, args.workers)
        rows.append({"calib_size": size, "top1": round(acc, 2)})
        print(f"calib={size}: {acc:.2f}%")
        del model; torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def cmd_kappa(args, dataset):
    model, proc = fresh_model()
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and is_mlp(n)]
    captured = {n: [] for n in names}
    handles = [m.register_forward_hook(
        lambda mod, inp, out, n=n: captured[n].append(inp[0].detach().cpu()))
        for n, m in model.named_modules() if n in captured]
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(args.calib_sizes[0]), desc="collect", leave=False):
            img = dataset[i]["image"].convert("RGB")
            model(**proc(images=img, return_tensors="pt").to(DEVICE))
    for h in handles:
        h.remove()
    del model; torch.cuda.empty_cache()

    rows = []
    for n in names:
        x = torch.cat(captured[n]).reshape(-1, captured[n][0].shape[-1]).float()
        d = x.shape[-1]
        rms = x.norm() / (x.numel() ** 0.5)
        k_pre = x.abs().amax(0).max().item() / rms.item()
        gen = torch.Generator()
        post = []
        for t in range(args.trials):
            gen.manual_seed(args.seed + t)
            q = make_matrix("qr", d, gen)
            post.append(((x @ q).abs().amax(0).max() / rms).item())
        rows.append({"layer": n, "d_in": d, "kappa_pre": k_pre,
                     "kappa_post": float(np.mean(post)),
                     "kappa_post_std": float(np.std(post))})
    df = pd.DataFrame(rows)
    print(df[["kappa_pre", "kappa_post"]].agg(["mean", "std"]))
    return df


def cmd_gptq(args, dataset):
    model, proc = fresh_model()
    rot = build_rotations(model, is_mlp, "qr", seed=args.seed, device=DEVICE)
    # Per-output-channel weight scales on rotated MLP weights; per-tensor
    # symmetric activations, so tensors stay in standard INT format.
    prepare(model, uniform_policy(args.w_bits, args.a_bits),
            rotations=rot, weight_per_channel=True)
    acc = top1_hf(model, proc, dataset, DEVICE, args.batch, args.workers)
    print(f"RoViT + per-channel rounding W{args.w_bits}A{args.a_bits}: {acc:.2f}%")
    return pd.DataFrame([{"top1": round(acc, 2)}])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["seeds", "calib", "kappa", "gptq"])
    p.add_argument("--w-bits", type=int, default=6)
    p.add_argument("--a-bits", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--calib-sizes", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.cmd == "gptq":
        args.w_bits = args.a_bits = 4

    dataset = load_imagenet_val()
    df = {"seeds": cmd_seeds, "calib": cmd_calib,
          "kappa": cmd_kappa, "gptq": cmd_gptq}[args.cmd](args, dataset)
    out = args.out or f"results/robustness_{args.cmd}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
