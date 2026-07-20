"""Uniform fake-quantization primitives and activation hooks."""

import torch


def quantize(x, bits, scale=None, per_channel_dim=None, symmetric=True):
    """Quantize-dequantize `x` at `bits`. Dynamic scale unless `scale` given."""
    if bits >= 32:
        return x
    if symmetric:
        qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
        if scale is None:
            if per_channel_dim is not None:
                scale = x.abs().amax(dim=per_channel_dim, keepdim=True).clamp(min=1e-8) / qmax
            else:
                scale = x.abs().max().clamp(min=1e-8) / qmax
        return torch.round(x / scale).clamp(qmin, qmax) * scale
    # asymmetric (used only by the "advanced PTQ" baseline)
    qmin, qmax = 0, (2 ** bits) - 1
    lo = x.min() if per_channel_dim is None else x.amin(dim=per_channel_dim, keepdim=True)
    hi = x.max() if per_channel_dim is None else x.amax(dim=per_channel_dim, keepdim=True)
    s = (hi - lo).clamp(min=1e-8) / qmax
    zp = torch.round(-lo / s)
    return (torch.round(x / s + zp).clamp(qmin, qmax) - zp) * s


class InputQuantHook:
    """Forward-pre-hook: (optionally) rotate the layer input, then quantize it.

    Modes:
      dynamic : per-call max-abs scale (default; matches the online phase)
      observe : record running max-abs of the (rotated) input, pass through
      static  : reuse the scale fixed during calibration
    """

    def __init__(self, bits, rotation=None, mode="dynamic", symmetric=True):
        self.bits = bits
        self.R = rotation
        self.mode = mode
        self.symmetric = symmetric
        self.amax = 0.0

    @property
    def scale(self):
        qmax = (2 ** (self.bits - 1)) - 1
        return None if self.amax == 0.0 else self.amax / qmax

    def __call__(self, module, args):
        x = args[0]
        if self.R is not None:
            x = x @ self.R.to(x.device, x.dtype)
        if self.mode == "observe":
            self.amax = max(self.amax, x.detach().abs().max().item())
            return (x,) + tuple(args[1:])
        s = None
        if self.mode == "static" and self.amax > 0.0:
            s = torch.tensor(self.scale, device=x.device, dtype=x.dtype)
        return (quantize(x, self.bits, scale=s, symmetric=self.symmetric),) + tuple(args[1:])


def calibrate(hooks, run_forwards):
    """Fix static activation scales: observe over a calibration pass, then freeze."""
    for h in hooks.values():
        h.mode = "observe"
    run_forwards()
    for h in hooks.values():
        h.mode = "static"
