import os
import torch
import torch.nn as nn
import pandas as pd
import timm
import torchvision.transforms as T
from datasets import load_dataset
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
os.environ["HF_TOKEN"] = HF_TOKEN  # cho timm/hub tai model co xac thuc


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
DATA_CACHE_DIR = "./imagenet_data"
BATCH_SIZE = 48
NUM_WORKERS = 8   # Windows: neu van loi pickle/spawn, dat = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EVAL_SUBSET = None

BIT_CONFIGS = {'W6A6': (6, 6), 'W4A4': (4, 4)}

# (tên, timm model, img_size, patch)
SWEEP = [
    ('ViT-B/16 @160', 'vit_base_patch16_224', 160, 16),
    ('ViT-B/16 @224', 'vit_base_patch16_224', 224, 16),
    ('ViT-B/16 @288', 'vit_base_patch16_224', 288, 16),
    ('ViT-B/16 @384', 'vit_base_patch16_224', 384, 16),
    ('ViT-B/32 @224', 'vit_base_patch32_224', 224, 32),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# 2. QUANT CORE (timm naming: blocks.N.mlp.fc1 / fc2)
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def is_mlp_layer_timm(name):
    return 'mlp' in name.lower()


def apply_ptq(model, w_bits, a_bits, rotate, seed=SEED):
    torch.manual_seed(seed)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        d = module.weight.shape[1]
        if rotate and is_mlp_layer_timm(name):
            Q, _ = torch.linalg.qr(torch.randn(d, d, device=DEVICE))
            module.weight.data = fake_quantize_tensor(
                module.weight.data @ Q, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits, r=Q:
                (fake_quantize_tensor(i[0] @ r, b),))
        else:
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
    return model


# =============================================================================
# 3. DATA + EVAL
# =============================================================================
def make_transform(res):
    return T.Compose([
        T.Resize(int(res / 0.875), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(res),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TimmCollate:
    """Class cap module de picklable tren Windows (DataLoader spawn workers)."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, examples):
        imgs = torch.stack([self.transform(ex['image'].convert("RGB"))
                            for ex in examples])
        labels = torch.tensor([ex['label'] for ex in examples])
        return imgs, labels


def evaluate(model, transform, dataset):
    model.eval()
    correct, total = 0, 0

    dl = DataLoader(dataset, batch_size=BATCH_SIZE,
                    collate_fn=TimmCollate(transform),
                    num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for imgs, labels in tqdm(dl, desc="    [Eval]", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if EVAL_SUBSET and total >= EVAL_SUBSET:
                break
    return correct / total * 100.0


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    print(f"[*] Q12 SENSITIVITY on {DEVICE}\n")
    dataset = load_imagenet_validation()

    results = []
    for disp, model_id, res, patch in SWEEP:
        n_tokens = (res // patch) ** 2 + 1
        print(f"\n{'#'*60}\n{disp}  (seq_len = {n_tokens} tokens)\n{'#'*60}")
        transform = make_transform(res)

        def fresh():
            # timm nội suy pos-embed khi img_size khác 224
            return timm.create_model(model_id, pretrained=True,
                                     img_size=res).to(DEVICE)

        print("--- FP32 ---")
        model = fresh()
        acc_fp32 = evaluate(model, transform, dataset)
        print(f"    => {acc_fp32:.2f}%")
        del model
        torch.cuda.empty_cache()

        for bit_name, (wb, ab) in BIT_CONFIGS.items():
            accs = {}
            for method, rotate in [('Std_PTQ', False), ('RoViT', True)]:
                print(f"--- {bit_name} | {method} ---")
                model = fresh()
                model = apply_ptq(model, wb, ab, rotate)
                accs[method] = evaluate(model, transform, dataset)
                print(f"    => {accs[method]:.2f}%")
                del model
                torch.cuda.empty_cache()
            results.append({
                'Config': disp, 'Resolution': res, 'Patch': patch,
                'Seq_Len': n_tokens, 'Bit_Config': bit_name,
                'FP32': round(acc_fp32, 2),
                'Std_PTQ': round(accs['Std_PTQ'], 2),
                'RoViT': round(accs['RoViT'], 2),
                'Boost': round(accs['RoViT'] - accs['Std_PTQ'], 2),
            })
            pd.DataFrame(results).to_csv(
                'Experiment_Results_12_Sensitivity.csv', index=False)

    df = pd.DataFrame(results)
    print(f"\n{'='*60}\n[*] Q12 COMPLETED\n{'='*60}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
