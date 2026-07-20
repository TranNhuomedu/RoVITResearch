import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (BIT_CONFIGS, DEVICE, MODEL_NAME, SELECTORS,  # noqa: E402
                        check_equivalence, evaluate_top1,
                        load_calibration_indices, load_imagenet_val,
                        load_model, make_loader, quantize_model, write_result)
from rotation import build_rotations  # noqa: E402


@torch.no_grad()
def collect_postln_ranges(model, calib_loader, max_batches=8):
    """Per-channel abs-max of every LN output feeding q/k/v (norm-before)
    and fc1 (norm-after)."""
    ranges, hooks = {}, []

    def mk(name):
        def hook(_m, _i, out):
            r = out.detach().abs().amax(dim=(0, 1))
            ranges[name] = torch.maximum(ranges.get(name, r), r)
        return hook

    from rovit_core import get_blocks
    for i, layer in enumerate(get_blocks(model)):
        hooks.append(layer.layernorm_before.register_forward_hook(mk(f"{i}.before")))
        hooks.append(layer.layernorm_after.register_forward_hook(mk(f"{i}.after")))
    model.eval().to(DEVICE)
    for b, (x, _) in enumerate(calib_loader):
        if b >= max_batches:
            break
        model(pixel_values=x.to(DEVICE))
    for h in hooks:
        h.remove()
    return ranges


@torch.no_grad()
def repq_reparameterize(model, ranges):
    """s_c = s_hat * t_c ;  fold t_c into LN affine (divide) and every
    reader's input weights (multiply).  Function-preserving in FP."""
    from rovit_core import block_parts, get_blocks
    for i, layer in enumerate(get_blocks(model)):
        bp = block_parts(model, layer)
        for ln, readers, key in (
                (bp["ln1"], bp["attn_readers"], f"{i}.before"),
                (bp["ln2"], [bp["fc1"]], f"{i}.after")):
            r = ranges[key].to(ln.weight.device).clamp_min(1e-8)
            t = r / r.median()
            ln.weight.data /= t
            ln.bias.data /= t
            for lin in readers:
                lin.weight.data *= t.unsqueeze(0)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", choices=list(BIT_CONFIGS), required=True)
    p.add_argument("--configs", nargs="+",
                   default=["standard", "rovit", "repq", "rovit+repq"])
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default="results/rovit_repq.csv")
    args = p.parse_args()
    w_bits, a_bits = BIT_CONFIGS[args.bits]

    fp, proc = load_model(args.model)
    loader = make_loader(load_imagenet_val(), proc, batch=args.batch)
    calib = make_loader(load_imagenet_val(indices=load_calibration_indices()),
                        proc, batch=16, workers=4)
    ranges = collect_postln_ranges(copy.deepcopy(fp), calib)

    for cfg in args.configs:
        torch.manual_seed(args.seed)
        model = copy.deepcopy(fp)
        if "repq" in cfg:
            before = copy.deepcopy(model)
            model = repq_reparameterize(model, ranges)
            assert check_equivalence(before, model), "RepQ reparam not exact"
            del before
        rot = (build_rotations(model, SELECTORS["mlp"], kind="qr",
                               seed=args.seed, device=DEVICE)
               if "rovit" in cfg else None)
        model = quantize_model(model, w_bits, a_bits, rotations=rot)
        top1 = evaluate_top1(model, loader, args.max_batches,
                             desc=f"{cfg} {args.bits}")
        write_result(args.out,
                     [args.model, args.bits, cfg, args.seed, f"{top1:.2f}"],
                     ["model", "bits", "config", "seed", "top1"])
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
