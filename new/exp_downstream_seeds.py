import argparse
import csv
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEGACY = os.path.join(ROOT, "legacy", "exp_downstream.py")

METRIC_KEYS = {"segmentation": ("miou", "mIoU"), "detection": ("map", "mAP")}


def read_metrics(csv_path, task):
    """-> {"qr": x, "none": y} tu CSV cua exp_downstream.py.
    Schema that (wide): task,bits,target,fp32,none,qr[,hadamard]."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:                                   # wide schema
        if row.get("task") == task and row.get("qr"):
            return {k: float(row[k]) for k in ("fp32", "none", "qr")
                    if row.get(k)}
    key_lower = METRIC_KEYS[task][0]                   # fallback: long schema
    out = {}
    for row in rows:
        metric_col = next((k for k in row
                           if k.lower().replace("_", "") == key_lower), None)
        if metric_col and row.get(metric_col):
            out[row.get("rotation", "qr")] = float(row[metric_col])
    if not out:
        raise RuntimeError(f"khong doc duoc diem {task} tu {csv_path}; "
                           "kiem tra schema exp_downstream.py")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["segmentation", "detection"],
                   required=True)
    p.add_argument("--bits", required=True)               # e.g. W6A6
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 123, 456, 789, 1024])     # đầu SEED_LIST của Q09
    p.add_argument("--target", default="mlp")
    args = p.parse_args()

    qr, none = [], []
    for seed in args.seeds:
        out_csv = os.path.join(ROOT, "results",
                               f"downstream_{args.task}_{args.bits}_s{seed}.csv")
        if os.path.exists(out_csv):
            try:
                metrics = read_metrics(out_csv, args.task)
                print(f"[SKIP] seed {seed}: dung lai {out_csv}")
                qr.append(metrics["qr"])
                none.append(metrics.get("none", float("nan")))
                continue
            except Exception:
                pass                                   # file hong -> chay lai
        cmd = [sys.executable, LEGACY,
               "--tasks", args.task, "--bits", args.bits,
               "--rotations", "none", "qr", "--target", args.target,
               "--seed", str(seed), "--out", out_csv]
        print("\n====", " ".join(cmd[1:]), "====")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"seed {seed} failed")
        metrics = read_metrics(out_csv, args.task)
        qr.append(metrics["qr"])
        none.append(metrics.get("none", float("nan")))

    mean, std = statistics.mean(qr), (statistics.stdev(qr)
                                      if len(qr) > 1 else 0.0)
    gains = [a - b for a, b in zip(qr, none)]
    gmean = statistics.mean(gains)
    gstd = statistics.stdev(gains) if len(gains) > 1 else 0.0
    name = METRIC_KEYS[args.task][1]
    print(f"\n{args.task} {args.bits}: {name} = {mean:.2f} +/- {std:.2f} "
          f"({len(qr)} seeds); gain over Std PTQ = {gmean:.2f} +/- {gstd:.2f} pp")

    agg = os.path.join(ROOT, "results", "downstream_seed_variance.csv")
    new = not os.path.exists(agg)
    with open(agg, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "bits", "n_seeds", "mean", "std",
                        "gain_mean", "gain_std", "per_seed"])
        w.writerow([args.task, args.bits, len(qr), f"{mean:.2f}",
                    f"{std:.2f}", f"{gmean:.2f}", f"{gstd:.2f}", qr])
    print("->", agg)


if __name__ == "__main__":
    main()
