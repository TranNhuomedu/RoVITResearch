import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (BIT_CONFIGS, DEVICE, MODEL_NAME, SELECTORS,  # noqa: E402
                        load_calibration_indices, load_imagenet_val,
                        load_model, make_loader, quantize_model, write_result)
from rotation import build_rotations  # noqa: E402


def _qkv_hooks(model, store):
    """Version-agnostic capture: hook the OUTPUTS of the q/k/v linears and
    rebuild softmax(QK^T/sqrt(hd)) ourselves -- no forward patching, so it
    survives transformers refactors and QuantLinear replacement alike."""
    from rovit_core import block_parts, get_blocks
    handles, tmp = [], {}
    for i, layer in enumerate(get_blocks(model)):
        bp = block_parts(model, layer)
        nh, hd = bp["num_heads"], bp["head_size"]
        for role, lin in zip(("q", "k"), bp["attn_readers"][:2]):
            def hook(_m, _i, out, _idx=i, _role=role, _nh=nh, _hd=hd):
                B, N, _ = out.shape
                tmp[(_idx, _role)] = out.detach().float() \
                    .view(B, N, _nh, _hd).transpose(1, 2)
                if (_idx, "q") in tmp and (_idx, "k") in tmp:
                    q, k = tmp.pop((_idx, "q")), tmp.pop((_idx, "k"))
                    A = ((q @ k.transpose(-2, -1)) / _hd ** 0.5).softmax(-1)
                    store.setdefault(_idx, []).append(A.cpu())
            handles.append(lin.register_forward_hook(hook))
    return handles


@torch.no_grad()
def collect_attn(model, images):
    store = {}
    handles = _qkv_hooks(model, store)
    model.eval().to(DEVICE)(pixel_values=images.to(DEVICE))
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0) for k, v in store.items()}


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (X.t() @ Y).norm() ** 2
    den = (X.t() @ X).norm() * (Y.t() @ Y).norm()
    return (num / den.clamp_min(1e-12)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", choices=list(BIT_CONFIGS), required=True)
    p.add_argument("--n-images", type=int, default=64)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    w_bits, a_bits = BIT_CONFIGS[args.bits]
    torch.manual_seed(args.seed)

    fp, proc = load_model(args.model)
    idx = load_calibration_indices()[:args.n_images]
    loader = make_loader(load_imagenet_val(indices=idx), proc,
                         batch=args.n_images, workers=4)
    images, _ = next(iter(loader))

    print("capturing FP32 attention...")
    ref = collect_attn(copy.deepcopy(fp), images)

    variants = {}
    for name, target in (("standard_ptq", None), ("rovit", "mlp")):
        m = copy.deepcopy(fp)
        rot = (build_rotations(m, SELECTORS["mlp"], "qr", seed=args.seed,
                               device=DEVICE) if target else None)
        m = quantize_model(m, w_bits, a_bits, rotations=rot)
        print(f"capturing {name} attention...")
        variants[name] = collect_attn(m, images)
        del m
        torch.cuda.empty_cache()

    rows = []
    csv_path = f"{args.out}/attention_cka_{args.bits}.csv"
    for layer in sorted(ref):
        A_ref = ref[layer].flatten(0, 1).flatten(1)
        entry = [layer]
        for name in ("standard_ptq", "rovit"):
            A = variants[name][layer].flatten(0, 1).flatten(1)
            entry += [f"{linear_cka(A, A_ref):.4f}",
                      f"{F.cosine_similarity(A, A_ref, dim=1).mean():.4f}"]
        rows.append(entry)
        write_result(csv_path, entry,
                     ["layer", "cka_stdptq", "cos_stdptq",
                      "cka_rovit", "cos_rovit"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = [r[0] for r in rows]
    ax.plot(xs, [float(r[1]) for r in rows], "o-", label="Standard PTQ")
    ax.plot(xs, [float(r[3]) for r in rows], "s-", label="RoViT")
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Attention-map CKA to FP32")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Attention topology preservation ({args.bits})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out}/z_fig_attention_cka_{args.bits}.pdf")
    print("figure ->", f"{args.out}/z_fig_attention_cka_{args.bits}.pdf")


if __name__ == "__main__":
    main()
