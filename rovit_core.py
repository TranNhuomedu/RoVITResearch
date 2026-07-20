"""rovit_core -- unified core for the NEW (Gop-Y-9) scripts.

Matches the conventions of the legacy standalone scripts exactly:
  * model     : google/vit-base-patch16-224 (HF ViTForImageClassification)
  * fc1       : *.intermediate.dense      fc2: *.output.dense (non-attention)
  * attention : *.attention.attention.{query,key,value} + *.attention.output.dense
  * quant     : symmetric per-tensor, W static / A dynamic (fake-quant),
                identical to fake_quantize_tensor() in RoVIT_Q07..Q12
  * rotation  : one torch.Generator seeded once per run, matrices drawn in
                named_modules() order (same contract as rotation.py)
  * token     : local_config.py -> env -> huggingface-cli login (never hardcoded)

Everything here is a superset distilled from the legacy scripts so the new
experiments land in a byte-identical numerical pipeline.
"""

import csv
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from rotation import build_rotations, make_matrix  # noqa: E402  (kit root copy)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/vit-base-patch16-224"
BIT_CONFIGS = {"W8A8": (8, 8), "W6A6": (6, 6), "W4A4": (4, 4)}


# --------------------------------------------------------------------------
# HF token / data (identical resolution order to the legacy Q-scripts)
# --------------------------------------------------------------------------

def get_hf_token():
    try:
        import importlib.util as ilu
        cfg_path = os.path.join(_ROOT, "local_config.py")
        if os.path.exists(cfg_path):
            spec = ilu.spec_from_file_location("local_config", cfg_path)
            cfg = ilu.module_from_spec(spec)
            spec.loader.exec_module(cfg)
            tok = getattr(cfg, "HF_TOKEN", None)
            if tok and tok.startswith("hf_") and "xxx" not in tok:
                return tok
    except Exception:
        pass
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def load_model(name=MODEL_NAME):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    tok = get_hf_token()
    model = AutoModelForImageClassification.from_pretrained(name, token=tok)
    proc = AutoImageProcessor.from_pretrained(name, token=tok)
    return model.to(DEVICE).eval(), proc


def load_imagenet_val(cache_dir="./imagenet_data", indices=None):
    from datasets import load_dataset
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                      cache_dir=cache_dir, token=get_hf_token())
    return ds.select(indices) if indices is not None else ds


def load_calibration_indices(path=None):
    path = path or os.path.join(_ROOT, "calibration_indices.txt")
    with open(path) as f:
        return [int(x) for x in f.read().split()]


class _Collate:
    """Top-level (picklable) collate -- Windows DataLoader workers dung
    spawn nen collate_fn phai pickle duoc; ham long ben trong se sap voi
    "Can't get local object 'make_loader.<locals>.collate'"."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        imgs = [it["image"].convert("RGB") for it in items]
        x = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        y = torch.tensor([it["label"] for it in items])
        return x, y


def make_loader(dataset, processor, batch=64, workers=8):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch, num_workers=workers,
                      collate_fn=_Collate(processor), pin_memory=True)


@torch.no_grad()
def evaluate_top1(model, loader, max_batches=None, desc=""):
    from tqdm import tqdm
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(pixel_values=x.to(DEVICE)).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


# --------------------------------------------------------------------------
# Layer-role selectors + structural adapter.
# transformers 4.x : *.intermediate.dense / *.output.dense /
#                    *.attention.attention.{query,key,value} / attention.output.dense
# transformers 5.x : *.mlp.fc1 / *.mlp.fc2 / *.attention.{q,k,v,o}_proj
# Both are supported so the campaign runs on the legacy pinned stack and on
# a fresh environment alike.
# --------------------------------------------------------------------------

def is_fc1(n):
    return "intermediate.dense" in n or n.endswith("mlp.fc1")


def is_fc2(n):
    return ("output.dense" in n and "attention" not in n) \
        or n.endswith("mlp.fc2")


def is_mlp(n):
    return is_fc1(n) or is_fc2(n)


def is_attn(n):
    return (".attention.attention." in n
            and n.endswith(("query", "key", "value"))) \
        or ("attention.output.dense" in n) \
        or n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))


def get_blocks(model):
    vit = model.vit
    return vit.encoder.layer if hasattr(vit, "encoder") else vit.layers


def block_parts(model, layer):
    """Version-agnostic handle on one transformer block's sub-modules."""
    cfg = model.config
    if hasattr(layer, "intermediate"):                      # transformers 4.x
        att = layer.attention.attention
        return dict(ln1=layer.layernorm_before, ln2=layer.layernorm_after,
                    attn_readers=[att.query, att.key, att.value],
                    attn_out=layer.attention.output.dense,
                    fc1=layer.intermediate.dense, fc2=layer.output.dense,
                    num_heads=att.num_attention_heads,
                    head_size=att.attention_head_size)
    att = layer.attention                                   # transformers 5.x
    return dict(ln1=layer.layernorm_before, ln2=layer.layernorm_after,
                attn_readers=[att.q_proj, att.k_proj, att.v_proj],
                attn_out=att.o_proj,
                fc1=layer.mlp.fc1, fc2=layer.mlp.fc2,
                num_heads=cfg.num_attention_heads,
                head_size=cfg.hidden_size // cfg.num_attention_heads)


SELECTORS = {
    "none": lambda n: False,
    "fc1": is_fc1,
    "fc2": is_fc2,
    "mlp": is_mlp,
    "attention": is_attn,
    "full": lambda n: is_mlp(n) or is_attn(n),
}


# --------------------------------------------------------------------------
# Quantization (numerically identical to the legacy fake_quantize_tensor)
# --------------------------------------------------------------------------

def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


class QuantLinear(nn.Module):
    """Weights quantized once (static, per-tensor symmetric); activations
    quantized dynamically per-tensor; optional folded input rotation R."""

    def __init__(self, lin: nn.Linear, w_bits, a_bits, R=None):
        super().__init__()
        W = lin.weight.data.clone()
        if R is not None:
            R = R.to(W.device, W.dtype)
            W = W @ R
        self.register_buffer("weight", fake_quantize_tensor(W, w_bits))
        self.register_buffer("bias",
                             None if lin.bias is None else lin.bias.data.clone())
        self.register_buffer("R", None if R is None else R)
        self.a_bits = a_bits

    def forward(self, x):
        if self.R is not None:
            x = x @ self.R.to(x.dtype)
        return F.linear(fake_quantize_tensor(x, self.a_bits),
                        self.weight, self.bias)


def quantize_model(model, w_bits, a_bits, rotations=None, select=None):
    """Swap every nn.Linear where select(name) (default: mlp+attention)
    for a QuantLinear; `rotations`: {layer_name: R}."""
    rotations = rotations or {}
    select = select or SELECTORS["full"]
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and select(full):
                setattr(mod, cname,
                        QuantLinear(child, w_bits, a_bits, rotations.get(full)))
    return model


# --------------------------------------------------------------------------
# Residual-stream machinery (QuaRot/SpinQuant port), HF ViT structure
# --------------------------------------------------------------------------

class RMSNormNoAffine(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _fold_ln_into_linears(ln: nn.LayerNorm, linears):
    """Fold LN affine into every reader:  W' = W diag(g), b' = b + W B."""
    g, B = ln.weight.data, ln.bias.data
    for lin in linears:
        W = lin.weight.data
        lin.weight.data = W * g.unsqueeze(0)
        bias = torch.zeros(W.shape[0], device=W.device) \
            if lin.bias is None else lin.bias.data
        lin.bias = nn.Parameter(bias + W @ B)
    ln.weight.data.fill_(1.0)
    ln.bias.data.zero_()


def _center_out(weight, bias=None):
    weight.data -= weight.data.mean(dim=0, keepdim=True)
    if bias is not None:
        bias.data -= bias.data.mean()


def convert_hf_vit_to_rotatable(model):
    """ViTForImageClassification -> functionally identical network whose
    residual stream commutes with a global orthogonal rotation.
    (i) fold LN affine into readers, (ii) center all residual writers,
    (iii) swap residual LayerNorms for affine-free RMS norm."""
    vit = model.vit
    for layer in get_blocks(model):
        bp = block_parts(model, layer)
        _fold_ln_into_linears(bp["ln1"], bp["attn_readers"])
        _fold_ln_into_linears(bp["ln2"], [bp["fc1"]])
        eps = bp["ln1"].eps
        layer.layernorm_before = RMSNormNoAffine(eps)
        layer.layernorm_after = RMSNormNoAffine(eps)
    _fold_ln_into_linears(vit.layernorm, [model.classifier])
    vit.layernorm = RMSNormNoAffine(vit.layernorm.eps)

    emb = vit.embeddings
    proj = emb.patch_embeddings.projection            # Conv2d, out = d
    proj.weight.data -= proj.weight.data.mean(dim=0, keepdim=True)
    if proj.bias is not None:
        proj.bias.data -= proj.bias.data.mean()
    emb.cls_token.data -= emb.cls_token.data.mean(-1, keepdim=True)
    emb.position_embeddings.data -= \
        emb.position_embeddings.data.mean(-1, keepdim=True)
    for layer in get_blocks(model):
        bp = block_parts(model, layer)
        _center_out(bp["attn_out"].weight, bp["attn_out"].bias)
        _center_out(bp["fc2"].weight, bp["fc2"].bias)
    return model


def apply_residual_rotation_hf(model, Q):
    """Fold a global orthogonal Q through the residual stream.
    Readers (q,k,v, fc1, classifier): W <- W Q.
    Writers (patch proj, cls, pos, attn.out, fc2): W <- Q^T W, b <- Q^T b."""
    p = next(model.parameters())
    Q = Q.to(p.device, p.dtype)
    Qt = Q.t()
    vit = model.vit
    emb = vit.embeddings
    proj = emb.patch_embeddings.projection
    proj.weight.data = (Qt @ proj.weight.data.flatten(1)).view_as(proj.weight.data)
    if proj.bias is not None:
        proj.bias.data = Qt @ proj.bias.data
    emb.cls_token.data = emb.cls_token.data @ Q
    emb.position_embeddings.data = emb.position_embeddings.data @ Q
    for layer in get_blocks(model):
        bp = block_parts(model, layer)
        for reader in bp["attn_readers"] + [bp["fc1"]]:
            reader.weight.data = reader.weight.data @ Q
        for writer in (bp["attn_out"], bp["fc2"]):
            writer.weight.data = Qt @ writer.weight.data
            writer.bias.data = Qt @ writer.bias.data
    model.classifier.weight.data = model.classifier.weight.data @ Q
    return model


def block_hadamard(d, block=None, dtype=torch.float32):
    """Exactly orthogonal block-diagonal Sylvester-Hadamard for arbitrary d
    (largest power-of-two block dividing d: 768 -> 3x256, 3072 -> 3x1024).
    Same exact-orthogonal Hadamard family as Table `rotation_zoo`."""
    if block is None:
        block = 1
        while block * 2 <= d and d % (block * 2) == 0:
            block *= 2
    h = torch.tensor([[1.0]], dtype=dtype)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / block ** 0.5
    return torch.block_diag(*[h] * (d // block))


@torch.no_grad()
def check_equivalence(model_a, model_b, atol=5e-3, n=2):
    x = torch.randn(n, 3, 224, 224, device=next(model_a.parameters()).device)
    ya = model_a(pixel_values=x).logits
    yb = model_b(pixel_values=x).logits
    err = (ya - yb).abs().max().item()
    print(f"  [equivalence] max |dy| = {err:.2e} (atol {atol})")
    return err < atol


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def write_result(csv_path, row, header):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    print("  ->", csv_path, row)
