import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (DEVICE, MODEL_NAME, SELECTORS,  # noqa: E402
                        load_calibration_indices, load_imagenet_val,
                        load_model, make_loader, write_result)
from rotation import build_rotations  # noqa: E402

BLOCKS_TO_PLOT = [0, 3, 6, 9, 11]


def kappa(x2d):
    col_inf = x2d.abs().amax(dim=0)
    rms = x2d.norm() / (x2d.numel() ** 0.5)
    return (col_inf.max() / rms.clamp_min(1e-12)).item()


@torch.no_grad()
def capture(model, images, rotations):
    """{name: (pre_flat, post_flat, kappa_pre, kappa_post)} for MLP linears."""
    out, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and SELECTORS["mlp"](name):
            def hook(_m, inp, _name=name):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                R = rotations[_name].to(x.device, x.dtype)
                xr = x @ R
                out[_name] = (x.flatten().cpu(), xr.flatten().cpu(),
                              kappa(x), kappa(xr))
            hooks.append(mod.register_forward_pre_hook(hook))
    model.eval().to(DEVICE)(pixel_values=images.to(DEVICE))
    for h in hooks:
        h.remove()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-images", type=int, default=64)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    model, proc = load_model(args.model)
    idx = load_calibration_indices()[:args.n_images]
    loader = make_loader(load_imagenet_val(indices=idx), proc,
                         batch=args.n_images, workers=4)
    images, _ = next(iter(loader))
    rotations = build_rotations(model, SELECTORS["mlp"], kind="qr",
                                seed=args.seed, device=DEVICE)
    data = capture(model, images, rotations)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # panel 1: histogram grid on fc2 inputs of representative blocks
    fig, axes = plt.subplots(2, len(BLOCKS_TO_PLOT),
                             figsize=(3 * len(BLOCKS_TO_PLOT), 5), sharey="row")
    for col, b in enumerate(BLOCKS_TO_PLOT):
        name = next(n for n in data
                    if f"layer.{b}." in n and SELECTORS["fc2"](n))
        pre, post, _, _ = data[name]
        for row, (x, label) in enumerate(((pre, "pre-rotation"),
                                          (post, "post-rotation"))):
            ax = axes[row, col]
            ax.hist(x.numpy(), bins=200, log=True, density=True)
            ax.set_title(f"block {b}, {label}\nmax|x|={x.abs().max():.1f}",
                         fontsize=8)
            if col == 0:
                ax.set_ylabel("density (log)")
    fig.suptitle("Post-GELU (fc2-input) activations, pre vs post QR rotation",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}/z_fig_multilayer_hist.pdf", bbox_inches="tight")
    print("figure ->", f"{args.out}/z_fig_multilayer_hist.pdf")

    # panel 2: kappa across all 24 MLP linears
    names = sorted(data, key=lambda n: (int(n.split("layer.")[1].split(".")[0]),
                                        SELECTORS["fc2"](n)))
    for n in names:
        write_result(f"{args.out}/kappa_per_layer.csv",
                     [n, f"{data[n][2]:.2f}", f"{data[n][3]:.2f}"],
                     ["layer", "kappa_pre", "kappa_post"])
    fig, ax = plt.subplots(figsize=(7, 3))
    xs = range(len(names))
    ax.plot(xs, [data[n][2] for n in names], "o-", label=r"$\kappa$ pre")
    ax.plot(xs, [data[n][3] for n in names], "s-", label=r"$\kappa$ post")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n.split("encoder.layer.")[-1]
                        .replace("intermediate.dense", "fc1")
                        .replace("output.dense", "fc2") for n in names],
                       rotation=90, fontsize=6)
    ax.set_ylabel(r"$\kappa$ (log)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out}/z_fig_kappa_per_layer.pdf")
    print("figure ->", f"{args.out}/z_fig_kappa_per_layer.pdf")


if __name__ == "__main__":
    main()
