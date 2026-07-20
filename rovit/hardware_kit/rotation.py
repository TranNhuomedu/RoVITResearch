"""Rotation-matrix construction.

All matrices are drawn on CPU from a single torch.Generator seeded once per
run, so (i) every target layer receives an independent matrix, and (ii) the
full set of rotations is reproducible from one integer seed regardless of
GPU model. The generator is advanced in named_modules() order.
"""

import torch

KINDS = ("identity", "gaussian", "householder", "hadamard", "qr")


def _walsh_hadamard(n, dtype=torch.float32):
    h = torch.tensor([[1.0]], dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def make_matrix(kind, d, generator):
    if kind == "identity":
        return torch.eye(d)
    if kind == "gaussian":
        return torch.randn(d, d, generator=generator) / (d ** 0.5)
    if kind == "householder":
        v = torch.randn(d, 1, generator=generator)
        v = v / v.norm()
        return torch.eye(d) - 2 * (v @ v.T)
    if kind == "hadamard":
        # Truncated construction: pad to the next power of two, cut to d x d.
        # Truncation breaks orthogonality when d is not a power of two.
        d_pad = 1
        while d_pad < d:
            d_pad *= 2
        return (_walsh_hadamard(d_pad) / d_pad ** 0.5)[:d, :d]
    if kind == "qr":
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))
        return q
    if kind.startswith("qr_block"):
        # Block-diagonal orthogonal rotation: "qr_block" or "qr_block:<size>".
        # Exactly orthogonal (each block is QR-orthogonal), with storage and
        # rotation FLOPs reduced by a factor of d / block_size.
        bs = int(kind.split(":")[1]) if ":" in kind else 128
        blocks = []
        for start in range(0, d, bs):
            b = min(bs, d - start)
            q, _ = torch.linalg.qr(torch.randn(b, b, generator=generator))
            blocks.append(q)
        return torch.block_diag(*blocks)
    raise ValueError(f"unknown rotation kind: {kind!r}")


def build_rotations(model, select, kind="qr", seed=42, device=None):
    """Return {layer_name: R} for every nn.Linear where select(name) is True.

    One matrix per layer, sized to that layer's input dimension.
    """
    import torch.nn as nn

    gen = torch.Generator()
    gen.manual_seed(seed)
    rotations = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and select(name):
            r = make_matrix(kind, mod.weight.shape[1], gen)
            rotations[name] = r.to(device) if device is not None else r
    return rotations
