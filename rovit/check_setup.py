"""Pre-flight check: verify the environment before starting the campaign.

Run from the repository root:  python check_setup.py
All items should print [OK]; fix any [FAIL] before running run_all.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ok = True


def report(label, passed, detail=""):
    global ok
    print(f"[{'OK' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    ok = ok and passed


# 1) required packages
for pkg in ["torch", "torchvision", "timm", "transformers", "datasets",
            "pycocotools", "torchmetrics", "matplotlib", "pandas", "PIL", "tqdm"]:
    try:
        importlib.import_module(pkg)
        report(f"package {pkg}", True)
    except Exception as e:
        report(f"package {pkg}", False, str(e))

# 2) datasets version (<3.0 required for script-based loaders)
try:
    import datasets
    major = int(datasets.__version__.split(".")[0])
    report("datasets < 3.0", major < 3, f"found {datasets.__version__}")
except Exception as e:
    report("datasets version", False, str(e))

# 3) CUDA
try:
    import torch
    report("CUDA available", torch.cuda.is_available(),
           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    report("CUDA", False, str(e))

# 4) HF token resolution
try:
    from rovit.data import resolve_token, find_data_dir
    tok = resolve_token()
    report("HF token", bool(tok), (tok[:8] + "...") if tok else "none found")
except Exception as e:
    report("HF token", False, str(e))
    find_data_dir = None

# 5) dataset caches
if find_data_dir:
    for name in ["imagenet_data", "coco_data_official"]:
        path = find_data_dir(name)
        exists = os.path.isdir(path)
        report(f"cache {name}", exists,
               os.path.abspath(path) if exists else
               "not found (will download on first use)")

# 6) library sanity: rotation invariance on a toy layer
try:
    import torch
    import torch.nn as nn
    from rovit import build_rotations, prepare, uniform_policy
    m = nn.Sequential()
    m.mlp = nn.Linear(8, 8)
    x = torch.randn(2, 8)
    y0 = m.mlp(x)
    rot = build_rotations(m, lambda n: True, "qr", seed=1)
    prepare(m, uniform_policy(32, 32), rotations=rot)
    y1 = m.mlp(x)
    report("rotation invariance", torch.allclose(y0, y1, atol=1e-4))
except Exception as e:
    report("rotation invariance", False, str(e))

print("\n" + ("READY: run run_all.bat (or run_all.ps1)." if ok
              else "NOT READY: fix the [FAIL] items above first."))
sys.exit(0 if ok else 1)
