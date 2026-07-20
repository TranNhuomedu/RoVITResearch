import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "core"))
from rovit_core import (BIT_CONFIGS, DEVICE, MODEL_NAME, SELECTORS,  # noqa: E402
                        apply_residual_rotation_hf, block_hadamard,
                        check_equivalence, convert_hf_vit_to_rotatable,
                        evaluate_top1, load_calibration_indices,
                        load_imagenet_val, load_model, make_loader,
                        quantize_model, write_result)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        qmax = 2 ** (bits - 1) - 1
        s = x.abs().amax().clamp_min(1e-8) / qmax
        return (x / s).round().clamp(-qmax - 1, qmax) * s

    @staticmethod
    def backward(ctx, g):
        return g, None


def _ste(x, bits):
    return _STE.apply(x, bits)


class CayleyRotation(nn.Module):
    """Q(A) = (I - A)(I + A)^{-1} Q0, A skew-symmetric -> always orthogonal."""

    def __init__(self, d, Q0):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(d, d))
        self.register_buffer("Q0", Q0)
        self.register_buffer("I", torch.eye(d))

    def Q(self):
        A = self.raw - self.raw.t()
        return torch.linalg.solve(self.I + A, self.I - A) @ self.Q0


class VirtualRotatedViT(nn.Module):
    """Differentiable stand-in for the Cayley search only: applies the
    current Q via explicit matmuls with STE fake-quant on readers/writers.
    The final Q is folded statically and evaluated on the real pipeline."""

    def __init__(self, rms_model, cayley, w_bits, a_bits):
        super().__init__()
        self.m, self.cayley = rms_model, cayley
        self.w_bits, self.a_bits = w_bits, a_bits

    def _lin(self, lin, x, side, Q):
        W, b = lin.weight, lin.bias
        if side == "reader":
            W = W @ Q
        else:
            W, b = Q.t() @ W, Q.t() @ b
        return F.linear(_ste(x, self.a_bits), _ste(W, self.w_bits), b)

    def forward(self, pixel_values):
        from rovit_core import block_parts, get_blocks
        m, Q = self.m, self.cayley.Q()
        emb = m.vit.embeddings
        x = emb.patch_embeddings(pixel_values)
        if isinstance(x, tuple):
            x = x[0]
        x = x @ Q
        cls = (emb.cls_token @ Q).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], 1) + emb.position_embeddings @ Q
        for layer in get_blocks(m):
            bp = block_parts(m, layer)
            h = bp["ln1"](x)
            B, N, _ = h.shape
            nh, hd = bp["num_heads"], bp["head_size"]

            def heads(t):
                return t.view(B, N, nh, hd).transpose(1, 2)

            rq, rk, rv = bp["attn_readers"]
            q = heads(self._lin(rq, h, "reader", Q))
            k = heads(self._lin(rk, h, "reader", Q))
            v = heads(self._lin(rv, h, "reader", Q))
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(B, N, -1)
            x = x + self._lin(bp["attn_out"], o, "writer", Q)
            h = bp["ln2"](x)
            h = F.gelu(self._lin(bp["fc1"], h, "reader", Q))
            x = x + self._lin(bp["fc2"], h, "writer", Q)
        x = m.vit.layernorm(x)
        return self._lin(m.classifier, x[:, 0], "reader", Q)


def optimize_rotation(rms_model, Q0, w_bits, a_bits, calib_loader,
                      steps=300, lr=1e-3):
    teacher = copy.deepcopy(rms_model).to(DEVICE).eval()
    cay = CayleyRotation(Q0.shape[0], Q0.to(DEVICE)).to(DEVICE)
    student = VirtualRotatedViT(rms_model.to(DEVICE), cay, w_bits, a_bits)
    opt = torch.optim.SGD(cay.parameters(), lr=lr, momentum=0.9)
    it, step = iter(calib_loader), 0
    while step < steps:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(calib_loader)
            continue
        x = x.to(DEVICE)
        with torch.no_grad():
            t = teacher(pixel_values=x).logits.log_softmax(-1)
        loss = F.kl_div(student(x).log_softmax(-1), t,
                        log_target=True, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"  [spinquant] step {step:4d}  KL {loss.item():.4f}")
        step += 1
    with torch.no_grad():
        return cay.Q().detach().cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["quarot", "spinquant"], required=True)
    p.add_argument("--bits", choices=list(BIT_CONFIGS), required=True)
    p.add_argument("--residual-rot", choices=["hadamard_block", "qr"],
                   default="hadamard_block")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default="results/llm_rotation_baselines.csv")
    args = p.parse_args()
    w_bits, a_bits = BIT_CONFIGS[args.bits]
    torch.manual_seed(args.seed)

    print(f"== {args.method} | {args.model} | {args.bits} | seed {args.seed} ==")
    model, proc = load_model(args.model)
    d = model.config.hidden_size

    reference = copy.deepcopy(model)
    model = convert_hf_vit_to_rotatable(model)
    assert check_equivalence(reference, model), "LN->RMS fold not exact"
    del reference

    gen = torch.Generator().manual_seed(args.seed)
    Q = (torch.linalg.qr(torch.randn(d, d, generator=gen))[0]
         if args.residual_rot == "qr" else block_hadamard(d))

    if args.method == "spinquant":
        calib = make_loader(load_imagenet_val(
            indices=load_calibration_indices()), proc, batch=16, workers=4)
        Q = optimize_rotation(copy.deepcopy(model), Q, w_bits, a_bits,
                              calib, steps=args.steps)

    model = apply_residual_rotation_hf(model, Q)
    print(f"  residual rotation folded (orth defect "
          f"{(Q @ Q.t() - torch.eye(d)).abs().max():.1e})")

    # online post-GELU rotation on fc2 (QuaRot R4) -- same per-layer code
    # path as RoViT, exact-orthogonal block-Hadamard construction
    rotations = {n: block_hadamard(m.weight.shape[1])
                 for n, m in model.named_modules()
                 if isinstance(m, nn.Linear) and SELECTORS["fc2"](n)}
    model = quantize_model(model, w_bits, a_bits, rotations=rotations)

    loader = make_loader(load_imagenet_val(), proc, batch=args.batch)
    t0 = time.time()
    top1 = evaluate_top1(model, loader, args.max_batches,
                         desc=f"{args.method} {args.bits}")
    write_result(args.out,
                 [args.method, args.model, args.bits, args.residual_rot,
                  args.seed, f"{top1:.2f}", f"{time.time() - t0:.0f}"],
                 ["method", "model", "bits", "residual_rot", "seed",
                  "top1", "wall_s"])


if __name__ == "__main__":
    main()
