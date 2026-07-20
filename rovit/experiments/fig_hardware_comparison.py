"""Figure 8: latency overhead and accuracy/overhead trade-off.

Own numbers are read from measurement CSVs (bench_latency.py and
exp_classification.py). FQ-ViT and PTQ4ViT overheads are literature-quoted
and marked as such in the manuscript (Table 17, note ‡).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

LITERATURE = {"FQ-ViT": 58.0, "PTQ4ViT": 79.0}   # ‡ latency overhead (%)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latency-csv", default="results/latency.csv")
    p.add_argument("--model", default="deit_base_patch16_224")
    p.add_argument("--acc-std", type=float, required=True,
                   help="Top-1 of Standard PTQ at INT8 (from exp_classification)")
    p.add_argument("--acc-rovit", type=float, required=True)
    p.add_argument("--acc-fp32", type=float, required=True)
    p.add_argument("--acc-fqvit", type=float, default=81.18)
    p.add_argument("--acc-ptq4vit", type=float, default=81.65)
    p.add_argument("--out", default="results/z_fig_hardware_comparison.pdf")
    args = p.parse_args()

    lat = pd.read_csv(args.latency_csv)
    row = lat[lat.model == args.model].iloc[0]
    rovit_overhead = float(row.overhead_pct)

    methods = ["Standard\nPTQ", "FQ-ViT", "PTQ4ViT", "RoViT\n(Ours)"]
    overhead = [0.0, LITERATURE["FQ-ViT"], LITERATURE["PTQ4ViT"], rovit_overhead]
    accuracy = [args.acc_std, args.acc_fqvit, args.acc_ptq4vit, args.acc_rovit]
    colors = ["#888888", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(methods, overhead, color=colors, edgecolor="black")
    for bar, v in zip(bars, overhead):
        axes[0].annotate("baseline" if v == 0 else f"{v:.1f}%",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", fontweight="bold")
    axes[0].set_ylabel("Latency overhead (% vs Standard INT8)")
    axes[0].grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.axhline(args.acc_fp32, color="gray", linestyle="--", alpha=0.7)
    for m, o, a, c in zip(methods, overhead, accuracy, colors):
        label = m.replace("\n", " ")
        ax.scatter([o], [a], s=300, c=c, edgecolors="black", zorder=5)
        ax.annotate(label, (o, a), xytext=(6 if o < 50 else -6, 0),
                    textcoords="offset points",
                    ha="left" if o < 50 else "right", fontweight="bold")
    ax.set_xlabel("Latency overhead (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out} (RoViT overhead measured: {rovit_overhead:.2f}%)")


if __name__ == "__main__":
    main()
