"""rovit -- RECONSTRUCTED package (ban tai tao).

Ban goc cua package nay khong co trong Code_New.rar; ban nay duoc tai tao de
khop CHINH XAC chu ky ham ma cac script legacy/exp_*.py goi, va dung dung
engine quantization da tai lap so lieu cua cac script RoVIT_Q* (symmetric
per-tensor, W static / A dynamic, mot torch.Generator seed duy nhat).

Neu ban tim lai duoc thu muc rovit/ goc, hay dung ban goc de so lieu khop
tuyet doi voi pipeline da cong bo; con so tu ban tai tao nay phai duoc doi
chieu voi cac gia tri da cong bo (vd. Std PTQ W4A4 ~0.1%, RoViT W6A6 ~79.9%)
truoc khi dua vao ban thao.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG_DIR)                      # RoViT_Campaign/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rotation import build_rotations, make_matrix      # noqa: E402  (kit root)

__all__ = ["build_rotations", "make_matrix", "is_mlp", "select_all",
           "layer_role", "uniform_policy", "prepare", "calibrate"]


# --------------------------------------------------------------------------
# Layer-role selectors (HF ViT naming, khop cac script Q)
# --------------------------------------------------------------------------

def is_mlp(name):
    return ("intermediate.dense" in name) \
        or ("output.dense" in name and "attention" not in name) \
        or name.endswith(("mlp.fc1", "mlp.fc2"))


def _is_attn(name):
    return (".attention.attention." in name
            and name.endswith(("query", "key", "value"))) \
        or ("attention.output.dense" in name) \
        or name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))


def select_all(name):
    return is_mlp(name) or _is_attn(name)


def layer_role(name):
    if "intermediate.dense" in name or name.endswith("mlp.fc1"):
        return "fc1"
    if ("output.dense" in name and "attention" not in name) \
            or name.endswith("mlp.fc2"):
        return "fc2"
    if _is_attn(name):
        return "attention"
    return "other"


# --------------------------------------------------------------------------
# Policies -- callable: layer name -> (w_bits, a_bits)
# --------------------------------------------------------------------------

def uniform_policy(w_bits, a_bits):
    def policy(_name):
        return (w_bits, a_bits)
    return policy


# --------------------------------------------------------------------------
# Quantization engine
# --------------------------------------------------------------------------

def _qdq(x, bits, per_channel=False, symmetric=True):
    """Fake quantize-dequantize. per_channel: theo chieu out cua weight."""
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    if symmetric:
        if per_channel:
            s = x.abs().amax(dim=tuple(range(1, x.dim())),
                             keepdim=True).clamp_min(1e-8) / qmax
        else:
            s = x.abs().amax().clamp_min(1e-8) / qmax
        return (x / s).round().clamp(-qmax - 1, qmax) * s
    lo, hi = x.amin(), x.amax()
    s = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    z = (-lo / s).round()
    return ((x / s + z).round().clamp(0, 2 ** bits - 1) - z) * s


class _QuantLinear(nn.Module):
    """Wrapper cho mot nn.Linear: W quantize tinh mot lan; A dong (mac dinh)
    hoac tinh sau calibrate; optional folded input rotation R."""

    def __init__(self, lin, w_bits, a_bits, R=None,
                 weight_per_channel=False, act_symmetric=True):
        super().__init__()
        W = lin.weight.data.clone()
        if R is not None:
            R = R.to(W.device, W.dtype)
            W = W @ R
        self.register_buffer("weight",
                             _qdq(W, w_bits, per_channel=weight_per_channel))
        self.register_buffer("bias",
                             None if lin.bias is None else lin.bias.data.clone())
        self.register_buffer("R", None if R is None else R)
        self.a_bits = a_bits
        self.act_symmetric = act_symmetric
        self.observing = False
        self.register_buffer("act_absmax", torch.tensor(0.0))
        self.static = False

    def forward(self, x):
        if self.R is not None:
            x = x @ self.R.to(x.dtype)
        if self.observing:
            self.act_absmax = torch.maximum(
                self.act_absmax, x.detach().abs().amax().to(self.act_absmax))
            xq = x                                # FP trong luc quan sat
        elif self.static:
            qmax = 2 ** (self.a_bits - 1) - 1
            s = self.act_absmax.clamp_min(1e-8) / qmax
            xq = (x / s).round().clamp(-qmax - 1, qmax) * s
        else:                                     # dynamic (mac dinh, khong calib)
            xq = _qdq(x, self.a_bits, symmetric=self.act_symmetric)
        return F.linear(xq, self.weight, self.bias)


def prepare(model, policy, rotations=None, select=None,
            weight_per_channel=False, act_symmetric=True):
    """Thay moi nn.Linear (attention + MLP) bang wrapper quantized theo
    `policy(name) -> (w_bits, a_bits)`. Tra ve danh sach hooks (wrapper)
    dung cho calibrate(); khong calibrate thi activation quantize dong."""
    rotations = rotations or {}
    select = select or select_all
    hooks = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and select(full):
                w_bits, a_bits = policy(full)
                q = _QuantLinear(child, w_bits, a_bits,
                                 R=rotations.get(full),
                                 weight_per_channel=weight_per_channel,
                                 act_symmetric=act_symmetric)
                setattr(mod, cname, q)
                hooks.append(q)
    return hooks


def calibrate(hooks, runner):
    """Bat che do quan sat absmax activation, chay runner() tren tap
    calibration, roi dong bang thanh static scales."""
    for h in hooks:
        h.observing = True
    runner()
    for h in hooks:
        h.observing = False
        h.static = True
    return hooks
