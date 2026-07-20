import importlib
import os
import sys

ok = True


def report(label, passed, detail=""):
    global ok
    print(f"[{'OK' if passed else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    ok = ok and passed


# 0) phien ban Python (datasets<3 + dill 0.3.8 khong chay tren 3.13/3.14)
v = sys.version_info
report("Python 3.10-3.12", (3, 10) <= (v.major, v.minor) <= (3, 12),
       f"found {v.major}.{v.minor}.{v.micro}"
       + ("" if v.minor <= 12 else
          " -- cai Python 3.12, tao venv: py -3.12 -m venv .venv"))

# 1) packages
for pkg in ["torch", "torchvision", "transformers", "datasets", "timm",
            "numpy", "pandas", "matplotlib", "tqdm", "PIL",
            "pycocotools", "torchmetrics", "scipy", "sklearn", "torchao"]:
    try:
        importlib.import_module(pkg)
        report(f"package {pkg}", True)
    except Exception as e:
        report(f"package {pkg}", False, str(e).splitlines()[0])

# 2) transformers < 5 (legacy Q-scripts nhan dien layer theo ten module 4.x)
try:
    import transformers
    major = int(transformers.__version__.split(".")[0])
    report("transformers < 5 (cho script legacy)", major < 5,
           f"found {transformers.__version__}"
           + ("" if major < 5 else
              " -- script new/ van chay, nhung RoVIT_Q* se xoay 0 layer!"))
except Exception as e:
    report("transformers version", False, str(e))

# 3) datasets < 3
try:
    import datasets
    report("datasets < 3.0", int(datasets.__version__.split(".")[0]) < 3,
           f"found {datasets.__version__}")
except Exception:
    pass

# 4) CUDA + GPU
try:
    import torch
    has = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if has else "khong thay GPU"
    report("CUDA available", has, f"{name}; torch {torch.__version__}")
    if has:
        x = (torch.randn(64, 64, device="cuda") @
             torch.randn(64, 64, device="cuda")).sum().item()
        report("GPU matmul chay duoc (kernel hop kien truc)", x == x,
               "neu FAIL: cai torch tu index cu128 (xem setup_env.ps1)")
except Exception as e:
    report("CUDA", False, str(e).splitlines()[0])

# 5) HF token
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.insert(0, "core")
    from rovit_core import get_hf_token
    tok = get_hf_token()
    report("HF token", bool(tok),
           (tok[:8] + "...") if tok else
           "copy local_config.example.py -> local_config.py va dien token; "
           "tai khoan phai Agree tren huggingface.co/datasets/ILSVRC/imagenet-1k")
except Exception as e:
    report("HF token", False, str(e).splitlines()[0])

# 6) package rovit (tuy chon -- thieu thi cac buoc legacy exp_* bi SKIP)
have_rovit = os.path.exists(os.path.join("rovit", "__init__.py"))
print(f"[{'OK' if have_rovit else 'WARN'}] package rovit/ "
      + ("co mat" if have_rovit else
         "chua co -- cac buoc legacy exp_* se bi SKIP (Q-scripts va new/ van chay)"))

print("\n" + ("SAN SANG: chay run_campaign.ps1" if ok
              else "CHUA SAN SANG: sua cac muc [FAIL] o tren."))
sys.exit(0 if ok else 1)
