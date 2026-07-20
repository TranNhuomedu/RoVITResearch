import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (BIT_CONFIGS, MODEL_NAME, SELECTORS,  # noqa: E402
                        evaluate_top1, load_imagenet_val, load_model,
                        make_loader, quantize_model, write_result)
from rotation import build_rotations  # noqa: E402

CONFIGS = ["none", "fc1", "fc2", "mlp", "attention", "full"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", choices=list(BIT_CONFIGS), required=True)
    p.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default="results/fc_targeting.csv")
    args = p.parse_args()
    w_bits, a_bits = BIT_CONFIGS[args.bits]

    fp, proc = load_model(args.model)
    fp = fp.cpu()
    loader = make_loader(load_imagenet_val(), proc, batch=args.batch)

    for target in args.configs:
        torch.manual_seed(args.seed)
        model = copy.deepcopy(fp).cuda() if torch.cuda.is_available() \
            else copy.deepcopy(fp)
        rot = build_rotations(model, SELECTORS[target], kind="qr",
                              seed=args.seed,
                              device=next(model.parameters()).device)
        # quantize ALL attention+mlp linears regardless of rotation target,
        # exactly as in Table `targeting_depth`
        model = quantize_model(model, w_bits, a_bits, rotations=rot)
        top1 = evaluate_top1(model, loader, args.max_batches,
                             desc=f"{target} {args.bits}")
        write_result(args.out,
                     [args.model, args.bits, target, len(rot), args.seed,
                      f"{top1:.2f}"],
                     ["model", "bits", "target", "rotated_layers", "seed",
                      "top1"])
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
