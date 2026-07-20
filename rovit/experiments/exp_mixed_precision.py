"""Mixed-precision ablation on YOLOS-Tiny / COCO at W4A4 (Table 9).

Selectively allocates INT8 to candidate bottleneck sub-modules while the
rest of the network stays at INT4 with MLP-only QR rotation.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp, layer_role, prepare
from exp_downstream import prepare_coco, eval_detection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "hustvl/yolos-tiny"

CONFIGS = {
    # name: rule(role) -> True if the layer is promoted to INT8
    "all_int4": lambda role: False,
    "all_int8": lambda role: True,
    "int8_bbox_head": lambda role: role == "bbox_head",
    "int8_cls_head": lambda role: role == "cls_head",
    "int8_both_heads": lambda role: role in ("bbox_head", "cls_head"),
    "int8_backbone": lambda role: role not in ("bbox_head", "cls_head"),
    "int8_attn_only": lambda role: role == "attn",
    "int8_mlp_only": lambda role: role == "mlp",
}


def build_model(config, seed):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(MODEL).to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    if config == "fp32":
        return model, proc
    promote = CONFIGS[config]
    policy = lambda name: (8, 8) if promote(layer_role(name)) else (4, 4)
    rot = build_rotations(model, is_mlp, "qr", seed=seed, device=DEVICE)
    prepare(model, policy, rotations=rot)
    return model, proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+",
                   default=["fp32"] + list(CONFIGS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/mixed_precision.csv")
    args = p.parse_args()

    img_dir, ann_file = prepare_coco()
    rows = []
    for config in args.configs:
        model, proc = build_model(config, args.seed)
        mAP = eval_detection(model, proc, img_dir, ann_file)
        rows.append({"config": config, "mAP": round(mAP, 2)})
        print(f"{config:18s} mAP={mAP:.2f}%")
        del model
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
