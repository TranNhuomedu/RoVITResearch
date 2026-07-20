"""Rotation-overhead latency benchmark (Tables 17 and 19 protocol).

Protocol per the manuscript's hardware-efficiency section: batch size 1,
torch.cuda.Event timers with explicit synchronization, 100 untimed warmup
iterations, median over 1000 timed iterations, plus a 95th-percentile
stability check against the median.

This script measures the quantity reported as "overhead": the wall-clock
cost added by the online activation-rotation step (X @ Omega on MLP inputs)
relative to an otherwise identical model. Absolute INT8/INT4 kernel
latencies depend on the deployment stack (e.g. TensorRT engines) and are
outside the scope of this portable script; the rotation hooks here run in
the model's native dtype.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp

ARCHS = [
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "vit_small_patch16_224", "deit_base_patch16_224",
    "vit_base_patch16_224", "vit_large_patch16_224",
    "swin_tiny_patch4_window7_224", "beit_base_patch16_224",
]


def attach_rotation_hooks(model, seed, dtype, device):
    rotations = build_rotations(model, is_mlp, "qr", seed=seed, device=device)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in rotations:
            r = rotations[name].to(device=device, dtype=dtype)
            mod.register_forward_pre_hook(
                lambda m, inp, rot=r: (inp[0] @ rot,) + tuple(inp[1:]))
    return model


def measure(model, x, warmup, iters):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(iters):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times = np.array(times)
    med, p95 = float(np.median(times)), float(np.percentile(times, 95))
    if p95 > 1.20 * med:
        print(f"  [warn] p95 ({p95:.3f} ms) exceeds median ({med:.3f} ms) "
              "by >20%; consider a longer warmup or checking thermals.")
    return med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ARCHS)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/latency.csv")
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA device required"

    import timm
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"device: {torch.cuda.get_device_name(0)}  dtype: {args.dtype}  "
          f"batch: {args.batch}  iters: {args.iters}")

    rows = []
    for name in args.models:
        x = torch.randn(args.batch, 3, 224, 224, device=device, dtype=dtype)

        model = timm.create_model(name, pretrained=False).to(device, dtype).eval()
        base = measure(model, x, args.warmup, args.iters)
        del model
        torch.cuda.empty_cache()

        model = timm.create_model(name, pretrained=False).to(device, dtype).eval()
        attach_rotation_hooks(model, args.seed, dtype, device)
        rotated = measure(model, x, args.warmup, args.iters)
        del model
        torch.cuda.empty_cache()

        overhead = 100.0 * (rotated - base) / base
        rows.append({"model": name, "baseline_ms": round(base, 3),
                     "rovit_ms": round(rotated, 3),
                     "overhead_pct": round(overhead, 2)})
        print(f"{name:32s} base={base:8.3f} ms  +rot={rotated:8.3f} ms  "
              f"overhead={overhead:+.2f}%")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nmean overhead: {df.overhead_pct.mean():.2f}% "
          f"+/- {df.overhead_pct.std(ddof=1):.2f}%")


if __name__ == "__main__":
    main()
