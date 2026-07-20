import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
EVAL_SUBSET = None       # None = full 50K

N_SEEDS = 20
SEED_LIST = [42, 123, 456, 789, 1024, 7, 21, 99, 314, 271,
             555, 888, 1111, 2024, 3141, 5926, 100, 200, 300, 400][:N_SEEDS]

# Targeting configs — mỗi config là hàm (is_mlp, is_attn, block_idx) -> rotate?
TARGETING_CONFIGS = {
    'attention_only':  lambda mlp, attn, b: attn,
    'mlp_only':        lambda mlp, attn, b: mlp,
    'all':             lambda mlp, attn, b: True,
    'mlp_first6':      lambda mlp, attn, b: mlp and 0 <= b <= 5,
    'mlp_last6':       lambda mlp, attn, b: mlp and 6 <= b <= 11,
    'all_first6':      lambda mlp, attn, b: 0 <= b <= 5,
    'all_last6':       lambda mlp, attn, b: 6 <= b <= 11,
    'mlp_blocks_0_3':  lambda mlp, attn, b: mlp and 0 <= b <= 3,
    'mlp_blocks_4_7':  lambda mlp, attn, b: mlp and 4 <= b <= 7,
    'mlp_blocks_8_11': lambda mlp, attn, b: mlp and 8 <= b <= 11,
}
# W6A6 chạy tất cả; W4A4 chạy tập trọng tâm (tiết kiệm GPU)
TARGETING_W4A4_SUBSET = ['mlp_only', 'all', 'mlp_first6', 'mlp_last6']


# =============================================================================
# 2. CORE
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def classify_layer(name):
    """Trả về (is_mlp, is_attn, block_idx)."""
    n = name.lower()
    is_mlp = ('intermediate.dense' in n) or \
             ('output.dense' in n and 'attention' not in n)
    is_attn = ('attention' in n)
    m = re.search(r'layer\.(\d+)\.', n)
    b = int(m.group(1)) if m else -1
    return is_mlp, is_attn, b


def apply_targeted_rovit(model, w_bits, a_bits, target_fn, seed=42):
    torch.manual_seed(seed)
    n_rotated = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_mlp, is_attn, b = classify_layer(name)
        d = module.weight.shape[1]
        if target_fn(is_mlp, is_attn, b):
            Q, _ = torch.linalg.qr(torch.randn(d, d, device=DEVICE))
            module.weight.data = fake_quantize_tensor(
                module.weight.data @ Q, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, bb=a_bits, r=Q:
                (fake_quantize_tensor(i[0] @ r, bb),))
            n_rotated += 1
        else:
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, bb=a_bits: (fake_quantize_tensor(i[0], bb),))
    return model, n_rotated


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
# 3. MAIN
# =============================================================================
def main():
    print(f"[*] Q09 TARGETING + SEEDS on {DEVICE}\n")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation",
                           token=HF_TOKEN, cache_dir=DATA_CACHE_DIR,
                           trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # =====================================================================
    # PART A: TARGETING ABLATION
    # =====================================================================
    print(f"{'#'*60}\nPART A: MODULE/DEPTH TARGETING\n{'#'*60}")
    rows = []
    for bit_name, (wb, ab) in [('W6A6', (6, 6)), ('W4A4', (4, 4))]:
        configs = (TARGETING_CONFIGS.keys() if bit_name == 'W6A6'
                   else TARGETING_W4A4_SUBSET)
        for cfg in configs:
            print(f"\n--- {bit_name} | targeting = {cfg} ---")
            model = AutoModelForImageClassification.from_pretrained(
                MODEL_NAME).to(DEVICE)
            model, n_rot = apply_targeted_rovit(
                model, wb, ab, TARGETING_CONFIGS[cfg])
            acc = evaluate(model, processor, dataset)
            print(f"    => Top-1: {acc:.2f}%  ({n_rot} layers rotated)")
            del model
            torch.cuda.empty_cache()
            rows.append({'Bit_Config': bit_name, 'Targeting': cfg,
                         'Layers_Rotated': n_rot, 'Top1_Acc': round(acc, 2)})
            pd.DataFrame(rows).to_csv(
                'Experiment_Results_09_Targeting.csv', index=False)

    # =====================================================================
    # PART B: 20-SEED ROBUSTNESS + BOXPLOT
    # =====================================================================
    print(f"\n{'#'*60}\nPART B: {N_SEEDS}-SEED ROBUSTNESS\n{'#'*60}")
    seed_rows = []
    for bit_name, (wb, ab) in [('W6A6', (6, 6)), ('W4A4', (4, 4))]:
        accs = []
        for seed in SEED_LIST:
            print(f"\n--- {bit_name} | seed = {seed} ---")
            model = AutoModelForImageClassification.from_pretrained(
                MODEL_NAME).to(DEVICE)
            model, _ = apply_targeted_rovit(
                model, wb, ab, TARGETING_CONFIGS['mlp_only'], seed=seed)
            acc = evaluate(model, processor, dataset)
            accs.append(acc)
            print(f"    => Top-1: {acc:.2f}%")
            del model
            torch.cuda.empty_cache()
            seed_rows.append({'Bit_Config': bit_name, 'Seed': seed,
                              'Top1_Acc': round(acc, 2)})
            pd.DataFrame(seed_rows).to_csv(
                'Experiment_Results_09_Seeds.csv', index=False)
        a = np.array(accs)
        print(f"\n==> {bit_name}: mean {a.mean():.3f} | std {a.std(ddof=1):.3f} "
              f"| min {a.min():.2f} | max {a.max():.2f} | n={len(a)}")

    # --- Boxplot ---
    df = pd.DataFrame(seed_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, bit in zip(axes, ['W6A6', 'W4A4']):
        vals = df[df['Bit_Config'] == bit]['Top1_Acc'].values
        bp = ax.boxplot([vals], widths=0.5, patch_artist=True,
                        showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('#a8d5ba')
        ax.scatter(np.random.normal(1, 0.04, len(vals)), vals,
                   s=18, alpha=0.6, color='#2c3e50', zorder=3)
        ax.set_title(f'RoViT ViT-Base {bit}\n'
                     f'mean {vals.mean():.2f} ± {vals.std(ddof=1):.3f} '
                     f'(n={len(vals)})')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_xticks([])
        ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('z_fig_seed_boxplot.pdf', bbox_inches='tight')
    print("\n[*] Boxplot saved: z_fig_seed_boxplot.pdf")
    print("[*] Q09 COMPLETED")


if __name__ == '__main__':
    main()
