import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from scipy import stats as sps

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


def load_imagenet_validation():
    """Tai CHI validation split (~7GB, 50K anh) qua cac shard parquet,
    tranh viec datasets tai + generate ca train split (~160GB).
    Fallback ve cach cu neu khong liet ke duoc file parquet."""
    from huggingface_hub import list_repo_files
    repo = "ILSVRC/imagenet-1k"
    cache_dir = globals().get("DATA_CACHE_DIR")
    for revision in (None, "refs/convert/parquet"):
        try:
            files = list_repo_files(repo, repo_type="dataset",
                                    revision=revision, token=HF_TOKEN)
            val_files = sorted(
                f for f in files
                if f.endswith(".parquet")
                and "val" in f.lower()
                and "train" not in f.lower()
                and "test" not in f.lower())
            if not val_files:
                continue
            base = f"hf://datasets/{repo}"
            if revision:
                base += f"@{revision}"
            print(f"[System] Tai {len(val_files)} shard parquet cua "
                  f"validation split (bo qua train)...")
            ds = load_dataset(
                "parquet",
                data_files=[f"{base}/{f}" for f in val_files],
                split="train",       # ten split cua builder parquet
                cache_dir=cache_dir,
                token=HF_TOKEN,
                storage_options={"token": HF_TOKEN})
            return ds
        except Exception as e:
            print(f"[System] Parquet route "
                  f"(revision={revision}) that bai: {e}")
    print("[System] Fallback: load_dataset day du (se cham hon nhieu)...")
    return load_dataset(repo, split="validation", token=HF_TOKEN,
                        cache_dir=cache_dir, trust_remote_code=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'google/vit-base-patch16-224'
SEED = 42
N_IMAGES = 64            # ảnh calibration để thu activation
MAX_VALUES_PER_LAYER = 400_000   # subsample giá trị mỗi layer cho metric
N_BINS = 512


# =============================================================================
# 2. ROTATION FACTORIES (đồng bộ với Q07)
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


def get_rotation(kind, d, device):
    torch.manual_seed(SEED)
    if kind == 'QR':
        Q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        return Q
    if kind == 'Hadamard_block128':
        b = 128
        Hb = hadamard_matrix(b, device) / (b ** 0.5)
        return torch.block_diag(*([Hb] * (d // b)))
    if kind == 'Hadamard_truncated':
        dp = next_pow2(d)
        H = hadamard_matrix(dp, device) / (dp ** 0.5)
        return H[:d, :d]
    raise ValueError(kind)


# =============================================================================
# 3. DISTRIBUTION METRICS
# =============================================================================
def dist_metrics(values):
    """
    values: 1-D numpy array. So sánh với Gaussian fit N(mu, sigma).
    Trả về dict: kurtosis, KL, JS, W1 (trên dữ liệu chuẩn hóa).
    """
    v = values.astype(np.float64)
    mu, sigma = v.mean(), v.std() + 1e-12
    z = (v - mu) / sigma

    kurt = sps.kurtosis(z, fisher=True)          # excess kurtosis

    # Histogram trong [-8, 8] sigma (clip phần đuôi cực đại vào bin biên)
    lo, hi = -8.0, 8.0
    zc = np.clip(z, lo, hi)
    counts, edges = np.histogram(zc, bins=N_BINS, range=(lo, hi))
    p = counts / counts.sum()
    # Gaussian mass trên cùng lưới bin
    cdf = sps.norm.cdf(edges)
    q = np.diff(cdf)
    q = q / q.sum()
    eps = 1e-12
    kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
               + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))

    # Wasserstein-1: empirical z vs quantile chuẩn N(0,1)
    n = min(len(z), 100_000)
    zs = np.sort(np.random.default_rng(SEED).choice(z, n, replace=False))
    gq = sps.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    w1 = float(np.mean(np.abs(zs - gq)))

    return {'kurtosis': kurt, 'KL': kl, 'JS': js, 'W1': w1,
            'M_over_mu': float(np.abs(v).max() / (np.abs(v).mean() + 1e-12))}


# =============================================================================
# 4. ACTIVATION CAPTURE
# =============================================================================
def is_postgelu_layer(name):
    """fc2 input = post-GELU, nơi outlier tập trung (d=3072 trên ViT-Base)."""
    n = name.lower()
    return 'output.dense' in n and 'attention' not in n


def block_index(name):
    import re
    m = re.search(r'layer\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def capture_activations(model, processor, dataset, target_names):
    """Trả về dict name -> Tensor (subsampled values, CPU) của input pre-hook."""
    store = {n: [] for n in target_names}
    hooks = []

    def make_hook(name):
        def hook(mod, inp):
            x = inp[0].detach().float().flatten()
            cur = sum(t.numel() for t in store[name])
            if cur < MAX_VALUES_PER_LAYER:
                keep = min(20_000, x.numel())
                idx = torch.randperm(x.numel(), device=x.device)[:keep]
                store[name].append(x[idx].cpu())
        return hook

    for n, mod in model.named_modules():
        if n in target_names:
            hooks.append(mod.register_forward_pre_hook(make_hook(n)))

    model.eval()
    seen = 0
    with torch.no_grad():
        for ex in dataset:
            img = ex['image'].convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            model(**inputs)
            seen += 1
            if seen >= N_IMAGES:
                break
    for h in hooks:
        h.remove()
    return {n: torch.cat(v)[:MAX_VALUES_PER_LAYER].numpy()
            for n, v in store.items()}


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print(f"[*] Q08 DISTRIBUTION METRICS on {DEVICE}\n")

    dataset = load_imagenet_validation()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)

    target_names = [n for n, m in model.named_modules()
                    if isinstance(m, nn.Linear) and is_postgelu_layer(n)]
    print(f"[*] {len(target_names)} post-GELU layers targeted "
          f"(fc2 inputs, d=3072)\n")

    # --- Pha 1: thu activation GỐC (pre-rotation) ---
    print("[*] Capturing PRE-rotation activations...")
    raw = capture_activations(model, processor, dataset, target_names)

    # --- Pha 2: tính metric cho pre + từng rotation (áp offline lên samples) ---
    # Lưu ý: rotation là phép nhân ma trận trên chiều kênh; để đo phân phối giá
    # trị sau rotation cần giữ cấu trúc (N, d). Thu lại theo (N, d):
    print("[*] Capturing PRE-rotation activations with channel structure...")
    store2 = {n: [] for n in target_names}
    hooks = []

    def make_hook2(name):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            cur = sum(t.shape[0] for t in store2[name])
            if cur < 6000:
                keep = min(512, x.shape[0])
                idx = torch.randperm(x.shape[0], device=x.device)[:keep]
                store2[name].append(x[idx].cpu())
        return hook

    for n, mod in model.named_modules():
        if n in target_names:
            hooks.append(mod.register_forward_pre_hook(make_hook2(n)))
    model.eval()
    seen = 0
    with torch.no_grad():
        for ex in dataset:
            img = ex['image'].convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            model(**inputs)
            seen += 1
            if seen >= N_IMAGES:
                break
    for h in hooks:
        h.remove()
    structured = {n: torch.cat(v, 0) for n, v in store2.items()}

    rows = []
    methods = ['None (pre-rotation)', 'QR', 'Hadamard_block128',
               'Hadamard_truncated']
    for name in target_names:
        X = structured[name]            # (N, d) CPU
        d = X.shape[1]
        blk = block_index(name)
        for method in methods:
            if method == 'None (pre-rotation)':
                vals = X.flatten().numpy()
            else:
                R = get_rotation(method, d, DEVICE)
                Xr = (X.to(DEVICE) @ R).cpu()
                vals = Xr.flatten().numpy()
            m = dist_metrics(vals)
            rows.append({'Layer': name, 'Block': blk, 'd': d,
                         'Method': method, **{k: round(v, 4)
                                              for k, v in m.items()}})
            print(f"  block {blk:2d} | {method:24s} | "
                  f"kurt={m['kurtosis']:9.2f} KL={m['KL']:.4f} "
                  f"JS={m['JS']:.4f} W1={m['W1']:.4f} "
                  f"M/mu={m['M_over_mu']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv('Experiment_Results_08_DistMetrics.csv', index=False)

    # --- Bảng tổng hợp trung bình (đưa thẳng vào paper) ---
    summary = df.groupby('Method')[['kurtosis', 'KL', 'JS', 'W1',
                                    'M_over_mu']].mean().round(3)
    summary.to_csv('Experiment_Results_08_DistMetrics_Summary.csv')
    print(f"\n=== SUMMARY (mean over {len(target_names)} layers) ===")
    print(summary.to_string())

    # --- Figure: KL và kurtosis per block, pre vs QR ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['KL', 'kurtosis']):
        for method, color in [('None (pre-rotation)', '#c0392b'),
                              ('QR', '#27ae60'),
                              ('Hadamard_block128', '#2980b9')]:
            sub = df[df['Method'] == method].sort_values('Block')
            ax.plot(sub['Block'], sub[metric], 'o-', label=method,
                    color=color, markersize=4)
        ax.set_xlabel('Transformer block')
        ax.set_ylabel(metric)
        ax.set_yscale('log' if metric == 'kurtosis' else 'linear')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title('KL divergence to fitted Gaussian')
    axes[1].set_title('Excess kurtosis (log scale)')
    plt.tight_layout()
    plt.savefig('z_fig_dist_metrics.pdf', bbox_inches='tight')
    print("\n[*] Figure saved: z_fig_dist_metrics.pdf")
    print("[*] Q08 COMPLETED")


if __name__ == '__main__':
    main()
