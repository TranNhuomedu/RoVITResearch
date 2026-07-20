import os
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# =============================================================================
# 1. CONFIG
# =============================================================================
def _get_hf_token():
    """Doc HF token theo thu tu: local_config.py -> env var -> huggingface-cli login."""
    # (1) local_config.py dat cung thu muc voi script nay
    try:
        import importlib.util as _ilu
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "local_config.py")
        if os.path.exists(_cfg_path):
            _spec = _ilu.spec_from_file_location("local_config", _cfg_path)
            _cfg = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_cfg)
            tok = getattr(_cfg, "HF_TOKEN", None)
            if tok and tok.startswith("hf_") and "xxx" not in tok:
                return tok
    except Exception:
        pass
    # (2) bien moi truong
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    # (3) token da luu boi `huggingface-cli login`
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return tok
    except Exception:
        pass
    raise SystemExit(
        "\n[!] Khong tim thay HuggingFace token. Chon MOT trong ba cach:\n"
        "    (1) Tao file local_config.py CUNG THU MUC voi script, noi dung:\n"
        "            HF_TOKEN = 'hf_xxx'   # token that cua ban\n"
        "    (2) Chay mot lan:  huggingface-cli login\n"
        "    (3) Set bien moi truong HF_TOKEN\n"
        "    Luu y: tai khoan phai da bam Agree tren trang\n"
        "    https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
    )


HF_TOKEN = _get_hf_token()
DATA_CACHE_DIR = "./imagenet_data"
BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'google/vit-base-patch16-224'
SEED = 42
EVAL_SUBSET = None

BIT_CONFIGS = {'W6A6': (6, 6), 'W4A4': (4, 4)}

# (tên, nhóm, params)
CONFIGS = [
    # A. Initialization distribution
    ('QR of Gaussian (baseline/Haar)', 'init', {'dist': 'gaussian'}),
    ('QR of Uniform(-1,1)',            'init', {'dist': 'uniform'}),
    ('QR of Rademacher(±1)',           'init', {'dist': 'rademacher'}),
    # B. Normalization của input trước QR
    ('QR of column-normalized Gaussian', 'norm', {'dist': 'gaussian',
                                                  'normalize': True}),
    # C. Density
    ('QR of sparse Gaussian d=0.1',    'density', {'dist': 'gaussian',
                                                   'density': 0.1}),
    ('QR of sparse Gaussian d=0.5',    'density', {'dist': 'gaussian',
                                                   'density': 0.5}),
    # D. Storage precision của Omega
    ('Omega stored FP16',              'precision', {'dist': 'gaussian',
                                                     'omega_dtype': 'fp16'}),
    ('Omega stored BF16',              'precision', {'dist': 'gaussian',
                                                     'omega_dtype': 'bf16'}),
    # E. Block-diagonal QR
    ('Block-QR b=64',                  'block', {'dist': 'gaussian', 'b': 64}),
    ('Block-QR b=128',                 'block', {'dist': 'gaussian', 'b': 128}),
    ('Block-QR b=256',                 'block', {'dist': 'gaussian', 'b': 256}),
    ('Block-QR b=384',                 'block', {'dist': 'gaussian', 'b': 384}),
]


# =============================================================================
# 2. CORE
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def is_mlp_layer(name):
    n = name.lower()
    if 'intermediate.dense' in n:
        return True
    if 'output.dense' in n and 'attention' not in n:
        return True
    return False


def sample_base(dist, d, device):
    if dist == 'gaussian':
        return torch.randn(d, d, device=device)
    if dist == 'uniform':
        return torch.rand(d, d, device=device) * 2 - 1
    if dist == 'rademacher':
        return (torch.randint(0, 2, (d, d), device=device).float() * 2 - 1)
    raise ValueError(dist)


def make_qr_variant(d, device, params):
    """Trả về (Omega, orth_residual)."""
    b = params.get('b')
    if b:                                        # Block-diagonal QR
        assert d % b == 0, f'd={d} không chia hết cho block {b}'
        blocks = []
        for _ in range(d // b):
            A = sample_base(params['dist'], b, device)
            Q, _ = torch.linalg.qr(A)
            blocks.append(Q)
        M = torch.block_diag(*blocks)
    else:
        A = sample_base(params['dist'], d, device)
        if params.get('density'):
            mask = (torch.rand_like(A) < params['density']).float()
            A = A * mask
        if params.get('normalize'):
            A = A / (A.norm(dim=0, keepdim=True) + 1e-8)
        Q, _ = torch.linalg.qr(A)
        M = Q

    dtype = params.get('omega_dtype')
    if dtype == 'fp16':
        M = M.half().float()                     # round-trip qua FP16
    elif dtype == 'bf16':
        M = M.bfloat16().float()

    res = torch.norm(M @ M.T - torch.eye(d, device=device)).item()
    return M, res


def apply_qr_variant(model, w_bits, a_bits, params, log_rows, cfg_name):
    torch.manual_seed(SEED)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        d = module.weight.shape[1]
        if is_mlp_layer(name):
            M, res = make_qr_variant(d, DEVICE, params)
            log_rows.append({'Config': cfg_name, 'Layer': name, 'd': d,
                             'orth_residual': round(res, 8)})
            module.weight.data = fake_quantize_tensor(
                module.weight.data @ M, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, bb=a_bits, r=M:
                (fake_quantize_tensor(i[0] @ r, bb),))
        else:
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, bb=a_bits: (fake_quantize_tensor(i[0], bb),))
    return model


class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [ex['image'].convert("RGB") for ex in examples]
        labels = torch.tensor([ex['label'] for ex in examples])
        inputs = self.processor(images=images, return_tensors="pt")
        inputs['labels'] = labels
        return inputs


def evaluate(model, processor, dataset):
    model.eval()
    correct, total = 0, 0
    dl = DataLoader(dataset, batch_size=BATCH_SIZE,
                    collate_fn=CollateFn(processor),
                    num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for batch in tqdm(dl, desc="    [Eval]", leave=False):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(DEVICE)
            preds = model(**inputs).logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if EVAL_SUBSET and total >= EVAL_SUBSET:
                break
    return correct / total * 100.0


# =============================================================================
# 3. STORAGE FOOTPRINT (tính lý thuyết cho từng config, kèm vào CSV)
# =============================================================================
def storage_mb(params, d_list=(768, 3072), layers_per_d=(12, 12)):
    """Storage cho Omega của ViT-Base: 12 lớp fc1 (d=768) + 12 lớp fc2 (3072)."""
    bytes_per = 2 if params.get('omega_dtype') in ('fp16', 'bf16') else 4
    b = params.get('b')
    total = 0
    for d, n in zip(d_list, layers_per_d):
        if b:
            total += n * (d // b) * b * b * bytes_per   # chỉ lưu các khối
        else:
            total += n * d * d * bytes_per
    return total / 1024 / 1024


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    print(f"[*] Q10 QR INTERNALS on {DEVICE}\n")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation",
                           token=HF_TOKEN, cache_dir=DATA_CACHE_DIR,
                           trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    results, orth_log = [], []
    for bit_name, (wb, ab) in BIT_CONFIGS.items():
        print(f"{'#'*60}\nBIT CONFIG: {bit_name}\n{'#'*60}")
        for cfg_name, group, params in CONFIGS:
            print(f"\n--- {cfg_name} ---")
            model = AutoModelForImageClassification.from_pretrained(
                MODEL_NAME).to(DEVICE)
            model = apply_qr_variant(model, wb, ab, params, orth_log, cfg_name)
            acc = evaluate(model, processor, dataset)
            smb = storage_mb(params)
            print(f"    => Top-1: {acc:.2f}%  | Omega storage: {smb:.1f} MB")
            del model
            torch.cuda.empty_cache()
            results.append({'Bit_Config': bit_name, 'Config': cfg_name,
                            'Group': group, 'Top1_Acc': round(acc, 2),
                            'Omega_Storage_MB': round(smb, 1)})
            pd.DataFrame(results).to_csv(
                'Experiment_Results_10_QRInternals.csv', index=False)

    pd.DataFrame(orth_log).to_csv(
        'Experiment_Results_10_OrthResiduals.csv', index=False)
    print(f"\n{'='*60}\n[*] Q10 COMPLETED\n{'='*60}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
