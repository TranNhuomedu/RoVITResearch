
import os
import torch
import torch.nn as nn
import pandas as pd
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
BATCH_SIZE = 32
NUM_WORKERS = 8   # Windows: neu van loi pickle/spawn, dat = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EVAL_SUBSET = None       # None = full 50K; đặt 10000 cho lần chạy nhanh

BIT_CONFIGS = {'W6A6': (6, 6), 'W4A4': (4, 4)}

BACKBONES = [
    # (tên hiển thị, loại, model_id, recipe)
    ('DINOv2-Base (linear head)', 'hf_cls',
     'facebook/dinov2-base-imagenet1k-1-layer', 'self-distillation (DINOv2)'),
    ('BEiT-Base',                 'hf_cls',
     'microsoft/beit-base-patch16-224', 'masked image modeling (BEiT)'),
    ('EVA02-Base',                'timm_cls',
     'eva02_base_patch14_224.mim_in22k_ft_in1k', 'MIM + EVA-CLIP teacher'),
    ('CLIP ViT-B/16 (zero-shot)', 'clip_zs',
     'openai/clip-vit-base-patch16', 'contrastive image-text (CLIP)'),
    ('SigLIP-Base (zero-shot)',   'siglip_zs',
     'google/siglip-base-patch16-224', 'sigmoid contrastive (SigLIP)'),
]


# =============================================================================
# 2. QUANTIZATION CORE
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def is_mlp_layer(name):
    """Bao phủ HF ViT/DINOv2/BEiT, timm EVA02, CLIP/SigLIP vision tower."""
    n = name.lower()
    attn = ('attention' in n) or ('attn' in n)
    if attn:
        return False
    if 'mlp.fc1' in n or 'mlp.fc2' in n:      # timm, CLIP, SigLIP
        return True
    if 'intermediate.dense' in n:             # HF ViT/BEiT
        return True
    if 'output.dense' in n:                   # HF ViT/BEiT fc2
        return True
    if n.endswith('mlp.w12') or n.endswith('mlp.w3'):  # EVA02 SwiGLU naming
        return True
    return False


def in_vision_tower(name, model_type):
    """CLIP/SigLIP: chỉ rotate vision tower, giữ nguyên text tower."""
    if model_type in ('clip_zs', 'siglip_zs'):
        return 'vision_model' in name
    return True


def apply_ptq(model, w_bits, a_bits, rotate, model_type, seed=SEED):
    torch.manual_seed(seed)
    n_rot = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # zero-shot: KHÔNG quantize text tower (đo riêng vision encoder)
        if model_type in ('clip_zs', 'siglip_zs') and \
                not in_vision_tower(name, model_type):
            continue
        d = module.weight.shape[1]
        if rotate and is_mlp_layer(name):
            Q, _ = torch.linalg.qr(torch.randn(d, d, device=DEVICE))
            module.weight.data = fake_quantize_tensor(
                module.weight.data @ Q, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits, r=Q:
                (fake_quantize_tensor(i[0] @ r, b),))
            n_rot += 1
        else:
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(
                lambda m, i, b=a_bits: (fake_quantize_tensor(i[0], b),))
    return model, n_rot


# =============================================================================
# 3. MODEL LOADERS + EVAL PATHS
# =============================================================================
def load_model(model_type, model_id):
    if model_type == 'hf_cls':
        from transformers import AutoImageProcessor, \
            AutoModelForImageClassification
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(
            model_id).to(DEVICE)
        return model, proc
    if model_type == 'timm_cls':
        import timm
        from timm.data import resolve_data_config, create_transform
        model = timm.create_model(model_id, pretrained=True).to(DEVICE)
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg)
        return model, transform
    if model_type == 'clip_zs':
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        proc = CLIPProcessor.from_pretrained(model_id)
        return model, proc
    if model_type == 'siglip_zs':
        from transformers import SiglipModel, SiglipProcessor
        model = SiglipModel.from_pretrained(model_id).to(DEVICE)
        proc = SiglipProcessor.from_pretrained(model_id)
        return model, proc
    raise ValueError(model_type)


def build_text_features(model, proc, class_names, model_type):
    """Zero-shot: encode 'a photo of a {name}.' cho 1000 lớp, chuẩn hóa L2."""
    prompts = [f"a photo of a {n.replace('_', ' ')}." for n in class_names]
    feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            batch = prompts[i:i + 256]
            inputs = proc(text=batch, return_tensors="pt",
                          padding=True, truncation=True).to(DEVICE)
            if model_type == 'clip_zs':
                f = model.get_text_features(**inputs)
            else:
                f = model.get_text_features(**inputs)
            feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats, 0)          # (1000, dim)


class HFClsCollate:
    """Picklable collate cho HF classifier (Windows spawn workers)."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, examples):
        images = [ex['image'].convert("RGB") for ex in examples]
        labels = torch.tensor([ex['label'] for ex in examples])
        inputs = self.proc(images=images, return_tensors="pt")
        inputs['labels'] = labels
        return inputs


class TimmClsCollate:
    """Picklable collate cho timm classifier."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, examples):
        imgs = torch.stack([self.transform(ex['image'].convert("RGB"))
                            for ex in examples])
        labels = torch.tensor([ex['label'] for ex in examples])
        return imgs, labels


class PixelCollate:
    """Picklable collate cho zero-shot (tra pixel_values, labels)."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, examples):
        images = [ex['image'].convert("RGB") for ex in examples]
        labels = torch.tensor([ex['label'] for ex in examples])
        inputs = self.proc(images=images, return_tensors="pt")
        return inputs['pixel_values'], labels


def eval_hf_classifier(model, proc, dataset):
    model.eval()
    correct, total = 0, 0

    dl = DataLoader(dataset, batch_size=BATCH_SIZE,
                    collate_fn=HFClsCollate(proc),
                    num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for batch in tqdm(dl, desc="    [Eval]", leave=False):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k != 'labels'}
            labels = batch['labels'].to(DEVICE)
            preds = model(**inputs).logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if EVAL_SUBSET and total >= EVAL_SUBSET:
                break
    return correct / total * 100.0


def eval_timm_classifier(model, transform, dataset):
    model.eval()
    correct, total = 0, 0

    dl = DataLoader(dataset, batch_size=BATCH_SIZE,
                    collate_fn=TimmClsCollate(transform),
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


def eval_zeroshot(model, proc, dataset, text_feats, model_type):
    model.eval()
    correct, total = 0, 0

    dl = DataLoader(dataset, batch_size=BATCH_SIZE,
                    collate_fn=PixelCollate(proc),
                    num_workers=NUM_WORKERS, pin_memory=True)
    with torch.no_grad():
        for pixel_values, labels in tqdm(dl, desc="    [Eval]", leave=False):
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)
            img_f = model.get_image_features(pixel_values=pixel_values)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            logits = img_f @ text_feats.T
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if EVAL_SUBSET and total >= EVAL_SUBSET:
                break
    return correct / total * 100.0


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    print(f"[*] Q11 BACKBONE GENERALIZATION on {DEVICE}\n")
    dataset = load_imagenet_validation()
    # Ten 1000 lop cho zero-shot prompt: uu tien ClassLabel metadata trong
    # parquet; fallback qua timm ImageNetInfo neu metadata khong di kem.
    class_names = getattr(dataset.features.get('label'), 'names', None)
    if not class_names:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        class_names = [info.index_to_description(i) for i in range(1000)]
        print("[System] ClassLabel metadata khong co trong parquet -> "
              "dung ten lop tu timm.ImageNetInfo")

    results = []
    for disp_name, mtype, model_id, recipe in BACKBONES:
        print(f"\n{'#'*60}\nBACKBONE: {disp_name}\n{'#'*60}")
        try:
            # ---------- FP32 baseline ----------
            model, proc = load_model(mtype, model_id)
            text_feats = None
            if mtype in ('clip_zs', 'siglip_zs'):
                print("    [System] Building zero-shot text features...")
                text_feats = build_text_features(model, proc, class_names, mtype)

            def run_eval(m):
                if mtype == 'hf_cls':
                    return eval_hf_classifier(m, proc, dataset)
                if mtype == 'timm_cls':
                    return eval_timm_classifier(m, proc, dataset)
                return eval_zeroshot(m, proc, dataset, text_feats, mtype)

            print("--- FP32 Baseline ---")
            acc_fp32 = run_eval(model)
            print(f"    => {acc_fp32:.2f}%")
            del model
            torch.cuda.empty_cache()

            # ---------- Quantized: Std PTQ vs RoViT ----------
            for bit_name, (wb, ab) in BIT_CONFIGS.items():
                accs = {}
                for method, rotate in [('Std_PTQ', False), ('RoViT', True)]:
                    print(f"--- {bit_name} | {method} ---")
                    model, _proc2 = load_model(mtype, model_id)
                    model, n_rot = apply_ptq(model, wb, ab, rotate, mtype)
                    accs[method] = run_eval(model)
                    print(f"    => {accs[method]:.2f}%"
                          + (f"  ({n_rot} layers rotated)" if rotate else ""))
                    del model
                    torch.cuda.empty_cache()
                results.append({
                    'Backbone': disp_name, 'Recipe': recipe,
                    'Model_ID': model_id, 'Bit_Config': bit_name,
                    'FP32': round(acc_fp32, 2),
                    'Std_PTQ': round(accs['Std_PTQ'], 2),
                    'RoViT': round(accs['RoViT'], 2),
                    'Boost': round(accs['RoViT'] - accs['Std_PTQ'], 2),
                })
                pd.DataFrame(results).to_csv(
                    'Experiment_Results_11_Backbones.csv', index=False)
        except Exception as e:
            print(f"    [!] SKIPPED {disp_name}: {e}")
            results.append({'Backbone': disp_name, 'Recipe': recipe,
                            'Model_ID': model_id, 'Bit_Config': 'ERROR',
                            'FP32': None, 'Std_PTQ': None, 'RoViT': None,
                            'Boost': None})

    df = pd.DataFrame(results)
    df.to_csv('Experiment_Results_11_Backbones.csv', index=False)
    print(f"\n{'='*60}\n[*] Q11 COMPLETED\n{'='*60}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
