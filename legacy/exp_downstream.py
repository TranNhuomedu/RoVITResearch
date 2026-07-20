"""Downstream evaluation: COCO detection (YOLOS-Tiny) and ADE20K
segmentation (UperNet+Swin-Tiny).

--target all  + --rotation qr        -> Table 8 configuration
--target mlp  + --rotation qr/hadamard/none (matched targeting) -> Table 10
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovit import build_rotations, is_mlp, prepare, select_all, uniform_policy
from rovit.data import load_ade20k_val, find_data_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_DIR = find_data_dir("coco_data_official")
DET_MODEL = "hustvl/yolos-tiny"
SEG_MODEL = "openmmlab/upernet-swin-tiny"
BIT_CONFIGS = {"W8A8": (8, 8), "W6A6": (6, 6), "W4A4": (4, 4)}


def prepare_coco(root=COCO_DIR):
    os.makedirs(root, exist_ok=True)
    ann = os.path.join(root, "annotations/instances_val2017.json")
    if not os.path.exists(ann):
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        z = os.path.join(root, "ann.zip")
        urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(root)
    if not os.path.isdir(os.path.join(root, "val2017")):
        url = "http://images.cocodataset.org/zips/val2017.zip"
        z = os.path.join(root, "val.zip")
        urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(root)
    return os.path.join(root, "val2017"), ann


def eval_detection(model, processor, img_dir, ann_file):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    results = []
    model.eval()
    for img_id in tqdm(coco_gt.getImgIds(), desc="coco", leave=False):
        info = coco_gt.loadImgs(img_id)[0]
        image = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        sizes = torch.tensor([[info["height"], info["width"]]]).to(DEVICE)
        preds = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.001)[0]
        for box, score, label in zip(preds["boxes"].cpu().numpy(),
                                     preds["scores"].cpu().numpy(),
                                     preds["labels"].cpu().numpy()):
            x1, y1, x2, y2 = box
            results.append({"image_id": img_id, "category_id": int(label),
                            "bbox": [float(x1), float(y1),
                                     float(x2 - x1), float(y2 - y1)],
                            "score": float(score)})
    if not results:
        return 0.0
    ev = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0] * 100.0


def eval_segmentation(model, processor, batch_size=16):
    from torchmetrics.classification import MulticlassJaccardIndex

    dataset = list(load_ade20k_val())
    metric = MulticlassJaccardIndex(num_classes=151, ignore_index=0).to(DEVICE)
    model.eval()
    for i in tqdm(range(0, len(dataset), batch_size), desc="ade20k", leave=False):
        batch = dataset[i:i + batch_size]
        images = [b["image"].convert("RGB") for b in batch]
        masks = [torch.tensor(np.array(b["annotation"]), dtype=torch.long,
                              device=DEVICE) for b in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        for j, mask in enumerate(masks):
            up = nn.functional.interpolate(logits[j:j + 1], size=mask.shape,
                                           mode="bilinear", align_corners=False)
            metric.update(up.argmax(dim=1).squeeze(0) + 1, mask)
    return metric.compute().item() * 100.0


def build(task):
    if task == "detection":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        return (AutoModelForObjectDetection.from_pretrained(DET_MODEL).to(DEVICE),
                AutoImageProcessor.from_pretrained(DET_MODEL))
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    return (AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(DEVICE),
            AutoImageProcessor.from_pretrained(SEG_MODEL))


def quantize_model(model, bits, rotation, target, seed):
    w, a = bits
    if rotation == "none":
        prepare(model, uniform_policy(w, a))
        return model
    select = select_all if target == "all" else is_mlp
    rot = build_rotations(model, select, kind=rotation, seed=seed, device=DEVICE)
    prepare(model, uniform_policy(w, a), rotations=rot)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["detection", "segmentation"])
    p.add_argument("--bits", nargs="+", default=["W8A8", "W6A6", "W4A4"])
    p.add_argument("--rotations", nargs="+", default=["none", "qr"],
                   choices=["none", "qr", "hadamard"])
    p.add_argument("--target", choices=["all", "mlp"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/downstream.csv")
    args = p.parse_args()

    img_dir, ann_file = prepare_coco() if "detection" in args.tasks else (None, None)
    rows = []
    for task in args.tasks:
        run = ((lambda m, pr: eval_detection(m, pr, img_dir, ann_file))
               if task == "detection" else eval_segmentation)
        model, proc = build(task)
        fp32 = run(model, proc)
        del model; torch.cuda.empty_cache()
        print(f"{task} FP32: {fp32:.2f}")
        for tag in args.bits:
            row = {"task": task, "bits": tag, "target": args.target, "fp32": round(fp32, 2)}
            for rotation in args.rotations:
                model, proc = build(task)
                score = run(quantize_model(model, BIT_CONFIGS[tag], rotation,
                                           args.target, args.seed), proc)
                row[rotation] = round(score, 2)
                print(f"{task} {tag} rotation={rotation:8s} "
                      f"target={args.target}: {score:.2f}")
                del model; torch.cuda.empty_cache()
            rows.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
