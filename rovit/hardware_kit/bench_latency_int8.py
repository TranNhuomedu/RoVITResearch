"""INT8 rotation-overhead benchmark (real integer-GEMM kernels via torchao).

Measures the wall-clock cost of the online rotation step on top of an
INT8 dynamic-activation / INT8-weight model, using torchao's quantized
Linear (dispatching to cuBLAS integer GEMM). This is the measurement that
backs the latency claims in the hardware-efficiency section; the deployed
stack should be named as "PyTorch + torchao INT8 dynamic quantization".

Protocol: batch 1, torch.cuda.Event timers, 100-warmup, and ROUND-ROBIN
measurement -- baseline and RoViT variants are timed in alternating rounds
so that thermal/clock drift cancels instead of biasing one variant (the
cause of the negative overhead previously observed on ViT-Large).
For stable numbers, lock GPU clocks first, e.g.:
    nvidia-smi -pm 1
    nvidia-smi -lgc <base_clock>,<base_clock>

Requires:  pip install torchao
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


def _int8_recipe():
    """Return (quantize_, config) across torchao API generations.

    torchao >= ~0.10 uses config classes (Int8DynamicActivationInt8WeightConfig);
    older releases exposed the functional int8_dynamic_activation_int8_weight().
    """
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        return quantize_, Int8DynamicActivationInt8WeightConfig()
    except ImportError:
        from torchao.quantization import int8_dynamic_activation_int8_weight
        return quantize_, int8_dynamic_activation_int8_weight()


def build_int8_model(name, rotate, kind, seed, device, dtype):
    import timm

    model = timm.create_model(name, pretrained=False).to(device, dtype).eval()
    if rotate:
        rotations = build_rotations(model, is_mlp, kind, seed=seed, device=device)
        for lname, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and lname in rotations:
                r = rotations[lname].to(device, dtype)
                mod.weight.data = mod.weight.data @ r      # fold into weights
                mod.register_forward_pre_hook(
                    lambda m, inp, rot=r: (inp[0] @ rot,) + tuple(inp[1:]))
    quantize_, config = _int8_recipe()
    quantize_(model, config)
    return model


def timed_pass(model, x, iters):
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.no_grad():
        for _ in range(iters):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ARCHS)
    p.add_argument("--rotation", default="qr",
                   help="qr | qr_block | qr_block:<size>")
    p.add_argument("--rounds", type=int, default=4,
                   help="alternating baseline/RoViT rounds")
    p.add_argument("--iters", type=int, default=250,
                   help="timed iterations per round (total = rounds x iters)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/latency_int8.csv")
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA device required"

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"device: {torch.cuda.get_device_name(0)}  int8 via torchao  "
          f"dtype(act): {args.dtype}  batch: {args.batch}  "
          f"rounds x iters: {args.rounds} x {args.iters}  "
          f"rotation: {args.rotation}")

    rows = []
    for name in args.models:
        x = torch.randn(args.batch, 3, 224, 224, device=device, dtype=dtype)
        base = build_int8_model(name, False, args.rotation, args.seed, device, dtype)
        rov = build_int8_model(name, True, args.rotation, args.seed, device, dtype)

        with torch.no_grad():                      # joint warmup
            for _ in range(args.warmup):
                base(x); rov(x)
        torch.cuda.synchronize()

        t_base, t_rov = [], []
        for _ in range(args.rounds):               # alternate to cancel drift
            t_base += timed_pass(base, x, args.iters)
            t_rov += timed_pass(rov, x, args.iters)
        med_b, med_r = float(np.median(t_base)), float(np.median(t_rov))
        p95_b = float(np.percentile(t_base, 95))
        if p95_b > 1.20 * med_b:
            print(f"  [warn] {name}: p95/median = {p95_b/med_b:.2f}; "
                  "lock GPU clocks (nvidia-smi -lgc) and re-run.")
        overhead = 100.0 * (med_r - med_b) / med_b
        rows.append({"model": name, "int8_baseline_ms": round(med_b, 3),
                     "int8_rovit_ms": round(med_r, 3),
                     "overhead_pct": round(overhead, 2)})
        print(f"{name:32s} int8={med_b:8.3f} ms  +rot={med_r:8.3f} ms  "
              f"overhead={overhead:+.2f}%")
        del base, rov
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nmean overhead: {df.overhead_pct.mean():.2f}% "
          f"+/- {df.overhead_pct.std(ddof=1):.2f}%")


if __name__ == "__main__":
    main()
