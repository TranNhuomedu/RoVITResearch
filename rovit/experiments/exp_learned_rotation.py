"""Random QR vs calibration-optimized orthogonal rotation (Table 13,
Figure 5).

The optimized rotation is R = Q0 @ Cayley(A) with A initialized to zero,
so the optimizer starts exactly at the random-QR matrix Q0: random QR is a
feasible point of the parameterization, and any in-sample regression below
its objective value indicates an optimizer issue rather than an expressivity
limit."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import is_mlp, prepare, uniform_policy
from rovit.rotation import make_matrix
from rovit.data import load_imagenet_val, sample_calibration_indices
from rovit.eval_utils import top1_hf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "google/vit-base-patch16-224"


def fresh_model():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    return (AutoModelForImageClassification.from_pretrained(MODEL).to(DEVICE),
            AutoImageProcessor.from_pretrained(MODEL))


def collect_inputs(model, proc, dataset, indices):
    captures = {n: [] for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and is_mlp(n)}
    handles = [m.register_forward_hook(
        lambda mod, inp, out, n=n: captures[n].append(inp[0].detach().cpu()))
        for n, m in model.named_modules() if n in captures]
    model.eval()
    with torch.no_grad():
        for i in tqdm(indices, desc="collect", leave=False):
            img = dataset[int(i)]["image"].convert("RGB")
            model(**proc(images=img, return_tensors="pt").to(DEVICE))
    for h in handles:
        h.remove()
    return {n: torch.cat(v).reshape(-1, v[0].shape[-1]) for n, v in captures.items()}


def optimize_rotation(x_calib, q0, steps, lr, p_norm=8, max_rows=4096):
    x = x_calib.to(DEVICE).float()
    if x.shape[0] > max_rows:
        x = x[torch.randperm(x.shape[0])[:max_rows]]
    d = x.shape[-1]
    q0 = q0.to(DEVICE)
    a = torch.zeros(d, d, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    eye = torch.eye(d, device=DEVICE)
    history = []
    for _ in range(steps):
        skew = (a - a.T) / 2
        r = q0 @ torch.linalg.solve(eye + skew, eye - skew)
        loss = ((x @ r).abs().amax(0) ** p_norm).mean() ** (1 / p_norm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    with torch.no_grad():
        skew = (a - a.T) / 2
        r = q0 @ torch.linalg.solve(eye + skew, eye - skew)
        q, _ = torch.linalg.qr(r)          # numerical re-orthogonalization
    return q.detach(), history


def apply_with(model, rotations, w_bits, a_bits):
    prepare(model, uniform_policy(w_bits, a_bits), rotations=rotations)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib-size", type=int, default=128,
                   choices=[128, 1024, 2048])
    p.add_argument("--bits", nargs="+", default=["W6A6", "W4A4"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dataset = load_imagenet_val()
    indices = sample_calibration_indices(args.calib_size, len(dataset), args.seed)

    base, proc = fresh_model()
    calib = collect_inputs(base, proc, dataset, indices)
    del base
    torch.cuda.empty_cache()

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    rot_qr, rot_opt, curve = {}, {}, None
    for i, (name, x) in enumerate(calib.items()):
        q0 = make_matrix("qr", x.shape[-1], gen)
        rot_qr[name] = q0.to(DEVICE)
        r, hist = optimize_rotation(x, q0, args.steps, args.lr)
        rot_opt[name] = r
        if i == 0:
            curve = hist

    rows = []
    bit_map = {"W6A6": (6, 6), "W4A4": (4, 4)}
    for tag in args.bits:
        w, a = bit_map[tag]
        scores = {}
        for label, rot in [("std", None), ("random_qr", rot_qr),
                           ("optimized", rot_opt)]:
            model, proc = fresh_model()
            if rot is None:
                prepare(model, uniform_policy(w, a))
            else:
                apply_with(model, rot, w, a)
            scores[label] = top1_hf(model, proc, dataset, DEVICE)
            print(f"{tag} {label:10s} {scores[label]:.2f}%")
            del model; torch.cuda.empty_cache()
        rows.append({"bits": tag, "calib_size": args.calib_size,
                     **{k: round(v, 2) for k, v in scores.items()}})
    pd.DataFrame(rows).to_csv(
        Path(args.outdir) / f"learned_rotation_c{args.calib_size}.csv",
        index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(curve, linewidth=2, color="#2ca02c")
    plt.xlabel("Optimization step")
    plt.ylabel("Smoothed channel-max objective")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / "z_fig_optim_curve.pdf",
                dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
