import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = os.path.join(ROOT, "results", "classification.csv")
LAT = os.path.join(ROOT, "results", "latency.csv")


def main():
    for path, hint in ((CLS, "legacy/exp_classification.py"),
                       (LAT, "legacy/bench_latency.py")):
        if not os.path.exists(path):
            print(f"[SKIP] thieu {path} -- chay {hint} truoc; "
                  f"fig_hardware se tu chay o lan sau.")
            return 0
    with open(CLS) as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r.get("bits") == "W8A8"
                and "deit" in r.get("model", "")), None) \
        or next((r for r in rows if r.get("bits") == "W8A8"), None)
    if row is None or not row.get("std") or not row.get("rovit"):
        print("[SKIP] classification.csv chua co hang W8A8 std/rovit.")
        return 0
    cmd = [sys.executable, os.path.join(ROOT, "legacy",
                                        "fig_hardware_comparison.py"),
           "--latency-csv", LAT,
           "--acc-std", row["std"], "--acc-rovit", row["rovit"],
           "--acc-fp32", row.get("fp32") or row["rovit"]]
    print("->", " ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
