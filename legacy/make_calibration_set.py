"""Generate calibration_indices.txt with numpy default_rng(seed).choice.

The paper's PTQ protocol fixes 128 calibration images chosen by
numpy.random.default_rng(42) over the sorted image manifest. Whichever split
is used here must match the sentence in Section 4.1 of the manuscript.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit.data import sample_calibration_indices

SPLIT_SIZES = {"train": 1_281_167, "validation": 50_000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=SPLIT_SIZES, default="validation")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="calibration_indices.txt")
    args = p.parse_args()

    idx = sample_calibration_indices(args.n, SPLIT_SIZES[args.split], args.seed)
    Path(args.out).write_text("\n".join(map(str, idx)) + "\n")
    print(f"wrote {args.n} indices for split={args.split} "
          f"(seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
