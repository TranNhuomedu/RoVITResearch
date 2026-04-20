import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# =============================================================================
# 1. HARDWARE & EXPERIMENT CONFIGURATION
# =============================================================================
HF_TOKEN = ""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE_SEG = 16  # Batch size cho ADE20K (RTX 5080)
COCO_DATA_DIR = "coco_data_official"  # Thư mục lưu dữ liệu COCO gốc

EXPERIMENTS = {
    'W8A8': {'weight_bits': 8, 'act_bits': 8},
    'W6A6': {'weight_bits': 6, 'act_bits': 6}
}

MODELS = {
    'Object_Detection': 'hustvl/yolos-tiny',
    'Semantic_Seg': 'openmmlab/upernet-swin-tiny'
}


# =============================================================================
# 2. CORE QUANTIZATION & ROVIT ALGORITHM
# =============================================================================
def fake_quantize_tensor(tensor, bits):
    if bits >= 32: return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    scale = tensor.abs().max().clamp(min=1e-8) / qmax
    return torch.round(tensor / scale).clamp(qmin, qmax) * scale


def apply_standard_ptq(model, w_bits, a_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = fake_quantize_tensor(module.weight.data, w_bits)
            module.register_forward_pre_hook(lambda mod, inp, b=a_bits: (fake_quantize_tensor(inp[0], b),))
    return model


def apply_rovit_ptq(model, w_bits, a_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            Q, _ = torch.linalg.qr(torch.randn(module.weight.shape[1], module.weight.shape[1], device=device))
            module.weight.data = fake_quantize_tensor(torch.matmul(module.weight.data, Q), w_bits)
            module.register_forward_pre_hook(
                lambda mod, inp, b=a_bits, r=Q: (fake_quantize_tensor(torch.matmul(inp[0], r), b),))
    return model


# =============================================================================
# 3. TASK 1: OBJECT DETECTION (COCO) PIPELINE
# =============================================================================
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)


def prepare_official_coco_dataset(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    ann_zip = os.path.join(root_dir, "annotations_trainval2017.zip")
    val_zip = os.path.join(root_dir, "val2017.zip")

    if not os.path.exists(os.path.join(root_dir, "annotations/instances_val2017.json")):
        print("      [System] Downloading COCO Annotations (241MB)...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
            urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                       filename=ann_zip, reporthook=t.update_to)
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref: zip_ref.extractall(root_dir)

    if not os.path.exists(os.path.join(root_dir, "val2017")):
        print("      [System] Downloading COCO Val Images (778MB)...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
            urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", filename=val_zip,
                                       reporthook=t.update_to)
        with zipfile.ZipFile(val_zip, 'r') as zip_ref: zip_ref.extractall(root_dir)


def evaluate_object_detection_map(model, processor, img_dir, ann_file, device):
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    results = []

    model.eval()
    for img_id in tqdm(img_ids, desc="      [Inferring COCO]", leave=False):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[img_info['height'], img_info['width']]]).to(device)
        preds = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.001)[0]

        boxes, scores, labels = preds['boxes'].cpu().numpy(), preds['scores'].cpu().numpy(), preds[
            'labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            results.append({
                "image_id": img_id, "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)], "score": float(score)
            })

    if not results: return 0.0

    print("      [Calculating pycocotools mAP...]")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0] * 100.0


# =============================================================================
# 4. TASK 2: SEMANTIC SEGMENTATION (ADE20K) PIPELINE
# =============================================================================
def evaluate_semantic_segmentation_miou(model, processor, device):
    dataset = load_dataset("scene_parse_150", split="validation", token=HF_TOKEN, streaming=False,
                           trust_remote_code=True)
    metric = MulticlassJaccardIndex(num_classes=151, ignore_index=0).to(device)
    model.eval()

    # Batched inference cho Segmentation
    dataset_list = list(dataset)
    for i in tqdm(range(0, len(dataset_list), BATCH_SIZE_SEG), desc="      [Inferring ADE20K]", leave=False):
        batch = dataset_list[i:i + BATCH_SIZE_SEG]
        images = [item['image'].convert("RGB") for item in batch]

        mask_tensors = []
        for item in batch:
            mask_np = np.array(item['annotation'])
            mask_tensors.append(torch.tensor(mask_np, dtype=torch.long).to(device))

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        for j, mask in enumerate(mask_tensors):
            logits_single = outputs.logits[j:j + 1]
            logits_resized = nn.functional.interpolate(logits_single, size=mask.shape, mode="bilinear",
                                                       align_corners=False)
            preds = logits_resized.argmax(dim=1).squeeze(0) + 1  # Căn chỉnh index ADE20K
            metric.update(preds, mask)

    print("      [Calculating mIoU...]")
    return metric.compute().item() * 100.0


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def main():
    print(f"[*] STARTING UNIFIED RUN: GROUP 2 (Downstream Tasks) on {DEVICE}\n")
    results_log = []

    prepare_official_coco_dataset(COCO_DATA_DIR)
    img_dir = os.path.join(COCO_DATA_DIR, "val2017")
    ann_file = os.path.join(COCO_DATA_DIR, "annotations/instances_val2017.json")

    # -------------------------------------------------------------------------
    # PART A: OBJECT DETECTION (COCO)
    # -------------------------------------------------------------------------
    print(f"{'#' * 60}\nTASK 1: OBJECT DETECTION (COCO)\n{'#' * 60}")
    model_name_det = MODELS['Object_Detection']
    processor_det = AutoImageProcessor.from_pretrained(model_name_det)

    print(f"---> Evaluating Baseline FP32 Model: {model_name_det}")
    model_det = AutoModelForObjectDetection.from_pretrained(model_name_det).to(DEVICE)
    map_fp32 = evaluate_object_detection_map(model_det, processor_det, img_dir, ann_file, DEVICE)
    print(f"     => Baseline mAP: {map_fp32:.2f}%\n")
    del model_det;
    torch.cuda.empty_cache()

    for exp_name, config in EXPERIMENTS.items():
        w_bits, a_bits = config['weight_bits'], config['act_bits']
        print(f"--- Evaluating {exp_name} (W{w_bits}A{a_bits}) ---")

        model_det = AutoModelForObjectDetection.from_pretrained(model_name_det).to(DEVICE)
        map_ptq = evaluate_object_detection_map(apply_standard_ptq(model_det, w_bits, a_bits), processor_det, img_dir,
                                                ann_file, DEVICE)
        print(f"     => Standard PTQ mAP: {map_ptq:.2f}%")
        del model_det;
        torch.cuda.empty_cache()

        model_det = AutoModelForObjectDetection.from_pretrained(model_name_det).to(DEVICE)
        map_rovit = evaluate_object_detection_map(apply_rovit_ptq(model_det, w_bits, a_bits), processor_det, img_dir,
                                                  ann_file, DEVICE)
        print(f"     => RoViT mAP: {map_rovit:.2f}%\n")
        del model_det;
        torch.cuda.empty_cache()

        results_log.append({
            'Task': 'mAP (COCO)', 'Quantization': exp_name, 'Model': model_name_det,
            'FP32': round(map_fp32, 2), 'Standard_PTQ': round(map_ptq, 2),
            'RoViT': round(map_rovit, 2), 'Boost': round(map_rovit - map_ptq, 2)
        })

    # -------------------------------------------------------------------------
    # PART B: SEMANTIC SEGMENTATION (ADE20K)
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 60}\nTASK 2: SEMANTIC SEGMENTATION (ADE20K)\n{'#' * 60}")
    model_name_seg = MODELS['Semantic_Seg']
    processor_seg = AutoImageProcessor.from_pretrained(model_name_seg)

    print(f"---> Evaluating Baseline FP32 Model: {model_name_seg}")
    model_seg = AutoModelForSemanticSegmentation.from_pretrained(model_name_seg).to(DEVICE)
    miou_fp32 = evaluate_semantic_segmentation_miou(model_seg, processor_seg, DEVICE)
    print(f"     => Baseline mIoU: {miou_fp32:.2f}%\n")
    del model_seg;
    torch.cuda.empty_cache()

    for exp_name, config in EXPERIMENTS.items():
        w_bits, a_bits = config['weight_bits'], config['act_bits']
        print(f"--- Evaluating {exp_name} (W{w_bits}A{a_bits}) ---")

        model_seg = AutoModelForSemanticSegmentation.from_pretrained(model_name_seg).to(DEVICE)
        miou_ptq = evaluate_semantic_segmentation_miou(apply_standard_ptq(model_seg, w_bits, a_bits), processor_seg,
                                                       DEVICE)
        print(f"     => Standard PTQ mIoU: {miou_ptq:.2f}%")
        del model_seg;
        torch.cuda.empty_cache()

        model_seg = AutoModelForSemanticSegmentation.from_pretrained(model_name_seg).to(DEVICE)
        miou_rovit = evaluate_semantic_segmentation_miou(apply_rovit_ptq(model_seg, w_bits, a_bits), processor_seg,
                                                         DEVICE)
        print(f"     => RoViT mIoU: {miou_rovit:.2f}%\n")
        del model_seg;
        torch.cuda.empty_cache()

        results_log.append({
            'Task': 'mIoU (ADE20K)', 'Quantization': exp_name, 'Model': model_name_seg,
            'FP32': round(miou_fp32, 2), 'Standard_PTQ': round(miou_ptq, 2),
            'RoViT': round(miou_rovit, 2), 'Boost': round(miou_rovit - miou_ptq, 2)
        })

    # =============================================================================
    # 6. EXPORT FINAL RESULTS
    # =============================================================================
    df_results = pd.DataFrame(results_log)
    output_filename = 'Experiment_Results_02.csv'
    df_results.to_csv(output_filename, index=False)
    print("\n" + "=" * 60 + f"\n[*] GROUP 2 EXPERIMENT COMPLETED!\n[*] Results saved to: {output_filename}\n")
    print(df_results.to_string(index=False))


if __name__ == '__main__':
    main()