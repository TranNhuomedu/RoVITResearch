"""rovit.data (tai tao) -- token, dataset, calibration indices, data dirs."""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_token():
    cfg_path = os.path.join(_ROOT, "local_config.py")
    if os.path.exists(cfg_path):
        try:
            spec = importlib.util.spec_from_file_location("local_config", cfg_path)
            cfg = importlib.util.module_from_spec(spec)
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


def load_imagenet_val(cache_dir=None):
    from datasets import load_dataset
    return load_dataset("ILSVRC/imagenet-1k", split="validation",
                        cache_dir=cache_dir or os.path.join(".", "imagenet_data"),
                        token=resolve_token())


def load_calibration_indices(path=None):
    path = path or os.path.join(_ROOT, "calibration_indices.txt")
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def sample_calibration_indices(size, total, seed=42):
    import numpy as np
    return sorted(np.random.default_rng(seed)
                  .choice(total, size=size, replace=False).tolist())


def find_data_dir(name):
    """Tim thu muc du lieu cuc bo (vd. coco_data_official). Tim theo thu tu:
    ./name, <campaign>/name, ../name, $ROVIT_DATA/name. Neu khong thay, tra
    ve duong dan mac dinh kem canh bao (de cac task khong can no van chay)."""
    from pathlib import Path
    candidates = [Path(name), Path(_ROOT) / name, Path(_ROOT).parent / name]
    env = os.environ.get("ROVIT_DATA")
    if env:
        candidates.append(Path(env) / name)
    for c in candidates:
        if c.exists():
            return c
    sys.stderr.write(
        f"[rovit.data] Chua tim thay thu muc du lieu '{name}'. Cac task can "
        f"no se loi; dat vao {Path(_ROOT) / name} hoac set ROVIT_DATA.\n")
    return Path(_ROOT) / name


def load_ade20k_val(cache_dir=None):
    from datasets import load_dataset
    return load_dataset("scene_parse_150", split="validation",
                        cache_dir=cache_dir or os.path.join(".", "ade20k_data"),
                        token=resolve_token(), trust_remote_code=True)
