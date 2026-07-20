"""Layer-role detection (timm / HuggingFace) and PTQ model preparation."""

import torch.nn as nn

from .quant import quantize, InputQuantHook


def layer_role(name):
    """Classify a Linear by name: mlp | attn | bbox_head | cls_head | other.

    Covers timm ViT/DeiT (blocks.i.mlp.*, blocks.i.attn.*), HuggingFace
    ViT/DeiT/Swin (intermediate.dense / output.dense / attention.*), and
    YOLOS detection heads.
    """
    n = name.lower()
    if "bbox_predictor" in n:
        return "bbox_head"
    if "class_labels_classifier" in n:
        return "cls_head"
    if "attention" in n or ".attn." in n or n.endswith(".attn"):
        return "attn"
    if "intermediate.dense" in n or ".mlp." in n or n.endswith("mlp"):
        return "mlp"
    if "output.dense" in n:            # HF MLP second projection
        return "mlp"
    return "other"


def is_mlp(name):
    return layer_role(name) == "mlp"


def select_all(name):
    return True


def uniform_policy(w_bits, a_bits):
    return lambda name: (w_bits, a_bits)


def prepare(model, bit_policy, rotations=None, act_mode="dynamic",
            weight_per_channel=False, act_symmetric=True):
    """Quantize weights in place and register input hooks on every Linear.

    bit_policy(name) -> (w_bits, a_bits); rotations: {name: R} or None.
    Returns {name: InputQuantHook} so callers can calibrate static scales.
    """
    rotations = rotations or {}
    hooks = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w_bits, a_bits = bit_policy(name)
        w = mod.weight.data
        r = rotations.get(name)
        if r is not None:
            w = w @ r.to(w.device, w.dtype)
        mod.weight.data = quantize(
            w, w_bits, per_channel_dim=1 if weight_per_channel else None)
        hook = InputQuantHook(a_bits, rotation=r, mode=act_mode,
                              symmetric=act_symmetric)
        mod.register_forward_pre_hook(hook)
        hooks[name] = hook
    return hooks
