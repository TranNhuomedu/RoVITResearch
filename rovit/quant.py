"""rovit.quant (tai tao) -- fake quantize voi ho tro per-channel theo chieu
chi dinh (per_channel_dim), khop moi loi goi trong cac script legacy:
  quantize(x, bits)                       # per-tensor symmetric
  quantize(W, bits, per_channel_dim=1)    # scale rieng cho tung lat theo dim 1
"""

import torch


def quantize(x, bits, per_channel_dim=None, symmetric=True):
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    if symmetric:
        if per_channel_dim is None:
            s = x.abs().amax().clamp_min(1e-8) / qmax
        else:
            dims = tuple(d for d in range(x.dim()) if d != per_channel_dim)
            s = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
        return (x / s).round().clamp(-qmax - 1, qmax) * s
    lo, hi = x.amin(), x.amax()
    s = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    z = (-lo / s).round()
    return ((x / s + z).round().clamp(0, 2 ** bits - 1) - z) * s
