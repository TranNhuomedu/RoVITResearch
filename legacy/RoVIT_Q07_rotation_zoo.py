import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
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
CAL_IMAGES = 64          # số ảnh calibration cho PCA/ICA (data-aware rotations)
ICA_MAX_SAMPLES = 8000   # số hàng activation tối đa đưa vào FastICA
RUN_ICA = True           # ICA trên d=3072 chậm (CPU); đặt False để bỏ qua
EVAL_SUBSET = None       # None = full 50K; đặt 5000 để smoke-test

BIT_CONFIGS = {'W6A6': (6, 6), 'W4A4': (4, 4)}

# Danh sách biến thể — mỗi entry: (tên hiển thị, loại, tham số)
VARIANTS = [
    ('Identity (Std PTQ)',        'identity',      {}),
    ('Random Gaussian (non-orth)','gaussian',      {}),
    ('QR / Haar',                 'qr',            {}),
    ('Hadamard truncated',        'had_trunc',     {}),
    ('Hadamard truncated + rand ±1','had_trunc_rand',{}),
    ('Hadamard padded (exact)',   'had_padded',    {}),          # xử lý đặc biệt
    ('Hadamard block-64',         'had_block',     {'b': 64}),
    ('Hadamard block-128',        'had_block',     {'b': 128}),
    ('Hadamard block-256',        'had_block',     {'b': 256}),
    ('Hadamard block-128 + rand ±1','had_block_rand',{'b': 128}),
    ('Householder k=1',           'householder',   {'k': 1}),
    ('Householder k=8',           'householder',   {'k': 8}),
    ('Householder k=64',          'householder',   {'k': 64}),
    ('Householder k=d',           'householder',   {'k': -1}),   # -1 nghĩa là k=d
    ('Cayley random',             'cayley',        {}),
    ('PCA rotation (data-aware)', 'pca',           {}),
    ('ICA orthogonalized (data-aware)','ica',      {}),
]


# =============================================================================
# 2. CORE QUANTIZATION
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def is_mlp_layer(name):
    """HF ViT: intermediate.dense (fc1, d=768) và output.dense (fc2, d=3072)."""
    n = name.lower()
    if 'intermediate.dense' in n:
        return True
    if 'output.dense' in n and 'attention' not in n:
        return True
    return False


# =============================================================================
# 3. MATRIX FACTORIES
# =============================================================================
def hadamard_matrix(n, device):
    if n == 1:
        return torch.tensor([[1.0]], device=device)
    h = hadamard_matrix(n // 2, device)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)


def next_pow2(d):
    p = 1
    while p < d:
        p *= 2
    return p


def orth_residual(M):
    d = M.shape[0]
    return torch.norm(M @ M.T - torch.eye(d, device=M.device)).item()


def make_matrix(kind, d, device, params, layer_stats=None):
    """
    Trả về (Omega hoặc None, orth_residual, note).
    layer_stats: dict {'cov': Tensor dxd} cho PCA, {'samples': Tensor Nxd} cho ICA.
    """
    if kind == 'identity':
        return None, 0.0, 'no rotation'

    if kind == 'gaussian':
        M = torch.randn(d, d, device=device) / (d ** 0.5)
        return M, orth_residual(M), 'non-orthogonal control'

    if kind == 'qr':
        # QR của ma trận Gaussian iid = sampling chuẩn từ phân phối Haar
        # trên nhóm trực giao O(d) [Stewart 1980]. "Haar rotation" == "Random QR".
        Q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        return Q, orth_residual(Q), 'Haar-distributed'

    if kind == 'had_trunc':
        dp = next_pow2(d)
        H = hadamard_matrix(dp, device) / (dp ** 0.5)
        M = H[:d, :d]
        return M, orth_residual(M), f'truncated {dp}->{d}'

    if kind == 'had_trunc_rand':
        dp = next_pow2(d)
        H = hadamard_matrix(dp, device) / (dp ** 0.5)
        M = H[:d, :d]
        s = (torch.randint(0, 2, (d,), device=device).float() * 2 - 1)
        M = torch.diag(s) @ M       # random ±1 KHÔNG cứu được truncation
        return M, orth_residual(M), 'randomized truncated'

    if kind == 'had_block':
        b = params['b']
        assert d % b == 0, f'd={d} không chia hết cho block {b}'
        Hb = hadamard_matrix(b, device) / (b ** 0.5)
        M = torch.block_diag(*([Hb] * (d // b)))
        return M, orth_residual(M), f'block-diag {d//b}x{b} (orthogonal exact)'

    if kind == 'had_block_rand':
        b = params['b']
        Hb = hadamard_matrix(b, device) / (b ** 0.5)
        M = torch.block_diag(*([Hb] * (d // b)))
        s = (torch.randint(0, 2, (d,), device=device).float() * 2 - 1)
        M = torch.diag(s) @ M       # D trực giao => D@BlockH vẫn trực giao
        return M, orth_residual(M), 'randomized block (QuIP-style)'

    if kind == 'householder':
        k = params['k'] if params['k'] > 0 else d
        M = torch.eye(d, device=device)
        for _ in range(k):
            v = torch.randn(d, 1, device=device)
            v = v / v.norm()
            M = M - 2 * v @ (v.T @ M)
        return M, orth_residual(M), f'{k} reflections'

    if kind == 'cayley':
        G = torch.randn(d, d, device=device) / (d ** 0.5)
        A = G - G.T
        I = torch.eye(d, device=device)
        M = torch.linalg.solve(I + A, I - A)
        return M, orth_residual(M), 'Cayley of random skew'

    if kind == 'pca':
        cov = layer_stats['cov'].to(device)
        # eigh trả eigenvector trực giao; sắp theo eigenvalue giảm dần
        evals, evecs = torch.linalg.eigh(cov)
        M = evecs.flip(-1)          # cột đầu = phương sai lớn nhất
        return M, orth_residual(M), 'eigenbasis of activation covariance'

    if kind == 'ica':
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            return None, -1, 'sklearn not available — skipped'
        X = layer_stats['samples'].cpu().numpy()
        n_comp = X.shape[1]
        ica = FastICA(n_components=n_comp, whiten='unit-variance',
                      max_iter=60, tol=5e-3, random_state=SEED)
        try:
            ica.fit(X)
        except Exception as e:
            return None, -1, f'ICA failed: {e}'
        W_unmix = torch.tensor(ica.components_.T, dtype=torch.float32,
                               device=device)
        # ICA unmixing KHÔNG trực giao trong không gian gốc — trực giao hóa
        # qua QR và ghi nhận đây là "ICA-initialized orthogonal" (data-aware).
        Q, _ = torch.linalg.qr(W_unmix)
        return Q, orth_residual(Q), 'QR-orthogonalized ICA unmixing'

    raise ValueError(kind)


# =============================================================================
# 4. DATA-AWARE STATS COLLECTION (cho PCA / ICA)
# =============================================================================
def collect_layer_stats(model, processor, dataset, target_names, n_images):
    """Pass FP32 thu covariance + samples của input các layer MLP."""
    stats = {n: {'sum': None, 'outer': None, 'count': 0, 'samples': []}
             for n in target_names}
    hooks = []

    def make_hook(name):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = stats[name]
            if s['sum'] is None:
                d = x.shape[1]
                s['sum'] = torch.zeros(d, device=x.device)
                s['outer'] = torch.zeros(d, d, device=x.device)
            s['sum'] += x.sum(0)
            s['outer'] += x.T @ x
            s['count'] += x.shape[0]
            if sum(t.shape[0] for t in s['samples']) < ICA_MAX_SAMPLES:
                keep = min(512, x.shape[0])
                idx = torch.randperm(x.shape[0], device=x.device)[:keep]
                s['samples'].append(x[idx].cpu())
        return hook

    for name, mod in model.named_modules():
        if name in target_names:
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    model.eval()
    seen = 0
    with torch.no_grad():
        for ex in dataset:
            img = ex['image'].convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            model(**inputs)
            seen += 1
            if seen >= n_images:
                break
    for h in hooks:
        h.remove()

    out = {}
    for name, s in stats.items():
        n = s['count']
        mean = s['sum'] / n
        cov = s['outer'] / n - torch.outer(mean, mean)
        out[name] = {'cov': cov.cpu(),
                     'samples': torch.cat(s['samples'], 0)[:ICA_MAX_SAMPLES]}
    return out


# =============================================================================
# 5. APPLY ROTATION + QUANT
# =============================================================================
def apply_variant(model, kind, params, w_bits, a_bits, layer_stats_map=None,
                  log_rows=None, variant_name=''):
    torch.manual_seed(SEED)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        d = module.weight.shape[1]

        if not is_mlp_layer(name):
            # non-MLP: standard PTQ, khớp targeting của headline RoViT
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
            continue

        # ---- MLP layer ----
        if kind == 'had_padded':
            # Xử lý đặc biệt: pad kênh d -> next_pow2(d), Hadamard đầy đủ
            dp = next_pow2(d)
            H = hadamard_matrix(dp, DEVICE) / (dp ** 0.5)
            W_pad = F.pad(module.weight.data, (0, dp - d))
            module.weight.data = fake_quantize_tensor(W_pad @ H, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits, h=H, pad=dp - d:
                (fake_quantize_tensor(F.pad(i[0], (0, pad)) @ h, b),))
            if log_rows is not None:
                log_rows.append({'Variant': variant_name, 'Layer': name,
                                 'd': d, 'orth_residual': 0.0,
                                 'note': f'padded {d}->{dp}, exact-output'})
            continue

        ls = layer_stats_map.get(name) if layer_stats_map else None
        M, res, note = make_matrix(kind, d, DEVICE, params, ls)
        if log_rows is not None:
            log_rows.append({'Variant': variant_name, 'Layer': name,
                             'd': d, 'orth_residual': round(res, 6),
                             'note': note})
        if M is None:                     # identity hoặc ICA fail
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
        else:
            module.weight.data = fake_quantize_tensor(
                module.weight.data @ M, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits, r=M:
                (fake_quantize_tensor(i[0] @ r, b),))
    return model


# =============================================================================
# 6. EVALUATION
# =============================================================================
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
# 7. FWHT LATENCY MICRO-BENCHMARK (Fast Walsh-Hadamard vs dense matmul)
# =============================================================================
def fwht_inplace(x):
    """Butterfly FWHT trên chiều cuối, độ phức tạp O(d log d). d = 2^k."""
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-3], d)
        h *= 2
    return x / (d ** 0.5)


def bench_fwht(d=1024, N=197, iters=300):
    """Trả về (t_dense_ms, t_fwht_ms) — cho phần hardware-efficiency."""
    x = torch.randn(N, d, device=DEVICE)
    H = hadamard_matrix(d, DEVICE) / (d ** 0.5)
    # verify tương đương
    assert torch.allclose(x @ H, fwht_inplace(x.clone()), atol=1e-4)
    torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = x @ H
    torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
    t_dense = (time.perf_counter() - t0) / iters * 1000
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fwht_inplace(x.clone())
    torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
    t_fwht = (time.perf_counter() - t0) / iters * 1000
    return t_dense, t_fwht


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    print(f"[*] Q07 ROTATION ZOO on {DEVICE}\n")

    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation",
                           token=HF_TOKEN, cache_dir=DATA_CACHE_DIR,
                           trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # --- Thu stats cho data-aware rotations (PCA / ICA) ---
    need_stats = any(v[1] in ('pca', 'ica') for v in VARIANTS)
    layer_stats_map = None
    if need_stats:
        print(f"[*] Collecting activation stats on {CAL_IMAGES} cal images "
              f"(for PCA/ICA)...")
        m = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        target_names = [n for n, mod in m.named_modules()
                        if isinstance(mod, nn.Linear) and is_mlp_layer(n)]
        layer_stats_map = collect_layer_stats(m, processor, dataset,
                                              target_names, CAL_IMAGES)
        del m
        torch.cuda.empty_cache()
        print(f"    -> stats for {len(layer_stats_map)} MLP layers collected\n")

    results, orth_log = [], []

    for bit_name, (wb, ab) in BIT_CONFIGS.items():
        print(f"{'#'*60}\nBIT CONFIG: {bit_name}\n{'#'*60}")
        for vname, kind, params in VARIANTS:
            if kind == 'ica' and not RUN_ICA:
                print(f"--- {vname}: SKIPPED (RUN_ICA=False)")
                continue
            print(f"\n--- {vname} ---")
            model = AutoModelForImageClassification.from_pretrained(
                MODEL_NAME).to(DEVICE)
            model = apply_variant(model, kind, params, wb, ab,
                                  layer_stats_map, orth_log, vname)
            acc = evaluate(model, processor, dataset)
            print(f"    => Top-1: {acc:.2f}%")
            del model
            torch.cuda.empty_cache()
            results.append({'Bit_Config': bit_name, 'Variant': vname,
                            'Kind': kind, 'Params': str(params),
                            'Top1_Acc': round(acc, 2)})
            # checkpoint từng bước để không mất kết quả nếu crash
            pd.DataFrame(results).to_csv(
                'Experiment_Results_07_RotationZoo.csv', index=False)

    # --- FWHT micro-benchmark ---
    print(f"\n{'#'*60}\nFWHT LATENCY MICRO-BENCHMARK\n{'#'*60}")
    bench_rows = []
    for d in [1024, 4096]:   # d_pad của fc1 (768->1024) và fc2 (3072->4096)
        td, tf = bench_fwht(d=d)
        speedup = td / tf if tf > 0 else float('inf')
        print(f"  d={d}: dense matmul {td:.4f} ms | FWHT {tf:.4f} ms "
              f"| speedup {speedup:.1f}x")
        bench_rows.append({'d': d, 'dense_ms': round(td, 4),
                           'fwht_ms': round(tf, 4),
                           'speedup': round(speedup, 2)})
    pd.DataFrame(bench_rows).to_csv(
        'Experiment_Results_07_FWHT_bench.csv', index=False)

    # --- Export ---
    pd.DataFrame(orth_log).to_csv(
        'Experiment_Results_07_OrthResiduals.csv', index=False)
    df = pd.DataFrame(results)
    print(f"\n{'='*60}\n[*] Q07 COMPLETED\n{'='*60}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
