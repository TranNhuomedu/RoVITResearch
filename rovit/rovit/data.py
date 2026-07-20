"""Dataset access. HF token comes from the environment or `huggingface-cli
login`; it is never hard-coded in this repository."""

import os

import numpy as np
import torch


def _first_existing_dir(*candidates, default):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return default


def find_data_dir(name):
    """Prefer ./name, fall back to ../name (caches from earlier runs)."""
    return _first_existing_dir(name, os.path.join("..", name), default=name)


def resolve_token():
    """Priority: local_config.py (gitignored) > HF_TOKEN env > hf login."""
    try:
        from local_config import HF_TOKEN
        if HF_TOKEN:
            return HF_TOKEN
    except ImportError:
        pass
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def load_imagenet_val(cache_dir=None):
    """Load the ImageNet-1K validation split.

    Cache location: `cache_dir` arg > IMAGENET_CACHE env var >
    ./imagenet_data relative to the current working directory.
    Requires either a cached copy or valid HF credentials
    (`huggingface-cli login`) for the gated ILSVRC/imagenet-1k repo.
    """
    from datasets import load_dataset
    cache_dir = (cache_dir or os.environ.get("IMAGENET_CACHE")
                 or find_data_dir("imagenet_data"))
    return load_dataset(
        "ILSVRC/imagenet-1k", split="validation", token=resolve_token(),
        streaming=False, cache_dir=cache_dir, trust_remote_code=True)


def load_ade20k_val():
    from datasets import load_dataset
    return load_dataset(
        "scene_parse_150", split="validation", token=resolve_token(),
        streaming=False, trust_remote_code=True)


def load_calibration_indices(path):
    return [int(line) for line in open(path) if line.strip()]


def sample_calibration_indices(n, split_size, seed=42):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(split_size, size=n, replace=False).tolist())


class HFCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        imgs = [e["image"].convert("RGB") for e in examples]
        out = self.processor(images=imgs, return_tensors="pt")
        out["labels"] = torch.tensor([e["label"] for e in examples])
        return out
