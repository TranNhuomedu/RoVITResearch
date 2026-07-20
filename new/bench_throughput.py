import argparse
import copy
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (DEVICE, MODEL_NAME, SELECTORS,  # noqa: E402
                        load_model, quantize_model, write_result)
from rotation import build_rotations  # noqa: E402


@torch.no_grad()
def time_model(model, batch, iters=250, warmup=100):
    model.eval().to(DEVICE)
    x = torch.randn(batch, 3, 224, 224, device=DEVICE)
    for _ in range(warmup):
        model(pixel_values=x)
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        model(pixel_values=x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    fp, _ = load_model(args.model)
    fp = fp.cpu()
    models = {}
    for name, kind in (("baseline", None), ("dense_qr", "qr"),
                       ("block_128", "qr_block:128")):
        m = copy.deepcopy(fp)
        rot = (build_rotations(m, SELECTORS["mlp"], kind=kind, seed=args.seed)
               if kind else None)
        models[name] = quantize_model(m, 8, 8, rotations=rot)

    results = {}
    for batch in args.batches:
        meds = {n: [] for n in models}
        for _ in range(args.rounds):                # alternating rounds
            for name in models:
                meds[name].append(time_model(models[name], batch,
                                             iters=args.iters, warmup=25))
        for name in models:
            results[(name, batch)] = statistics.median(meds[name])
        base = results[("baseline", batch)]
        for name in ("dense_qr", "block_128"):
            ms = results[(name, batch)]
            write_result(f"{args.out}/throughput_batch_sweep.csv",
                         [args.model, name, batch, f"{ms:.3f}",
                          f"{1000.0 * batch / ms:.1f}",
                          f"{100.0 * (ms - base) / base:.2f}"],
                         ["model", "config", "batch", "ms_per_batch",
                          "img_per_s", "overhead_pct"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, marker in (("dense_qr", "o"), ("block_128", "s")):
        ys = [100.0 * (results[(name, b)] - results[("baseline", b)])
              / results[("baseline", b)] for b in args.batches]
        ax.plot(args.batches, ys, marker + "-", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("rotation overhead (%)")
    ax.axhline(0, lw=0.5, color="k")
    ax.legend(frameon=False)
    ax.set_title("Launch-bound to compute-bound transition")
    fig.tight_layout()
    fig.savefig(f"{args.out}/z_fig_throughput.pdf")
    print("figure ->", f"{args.out}/z_fig_throughput.pdf")


if __name__ == "__main__":
    main()
