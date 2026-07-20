"""Figure 2: dataset overview (real samples + computed statistics).

See module body for data-source notes.
RoVIT_Q15_DatasetOverview.py — Figure 2: z_fig_dataset_overview.pdf
====================================================================
Replaces the placeholder sample boxes in Figure 2 with REAL images from
the exact same data sources used by the experiment scripts:

  * ImageNet-1K : HuggingFace  ILSVRC/imagenet-1k  (validation split)
                  -- same loader as Q03/Q04/Q07-Q10, cache ./imagenet_data
  * ADE20K      : HuggingFace  scene_parse_150     (validation split)
                  -- same loader as Q02/Q06; masks in item['annotation']
  * MS COCO 2017: local disk   coco_data_official/val2017/*.jpg
                  + coco_data_official/annotations/instances_val2017.json
                  -- same layout that Q02/Q06 auto-download

Left column : 3 real samples per dataset
              (ADE20K samples overlaid with per-pixel masks;
               COCO samples overlaid with ground-truth bounding boxes,
               so the figure matches what the caption promises).
Right column: class-frequency statistics COMPUTED FROM THE DATA
              (not hard-coded), cached to ./figures_q15/*.npz.

Output:  ./figures_q15/z_fig_dataset_overview.pdf  (+ .png preview)
Copy the PDF next to 00-main.tex before recompiling.

Runtime: ~2-4 min on first run (ADE20K pixel counting over 2,000 masks);
seconds afterwards thanks to caching. No GPU required.

SECURITY NOTE: do NOT hard-code the HF token here. Either run
`huggingface-cli login` once (recommended), or set it per-session:
    export HF_TOKEN=hf_xxx           (Linux/macOS)
    set HF_TOKEN=hf_xxx              (Windows cmd.exe)
    $env:HF_TOKEN = "hf_xxx"         (Windows PowerShell)
If the datasets are already cached (you ran Q01-Q10 on this machine),
NO token is needed at all -- the script runs cache-only.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image

# =============================================================================
# 1. CONFIGURATION  (paths identical to Q01-Q10)
# =============================================================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from rovit.data import resolve_token, find_data_dir

HF_TOKEN = resolve_token()                        # local_config > env > login
HF_TOKEN_SOURCE = "shared resolver (rovit.data)"
DATA_CACHE_DIR = find_data_dir("imagenet_data")   # ./ or ../ auto-detected
COCO_DATA_DIR = find_data_dir("coco_data_official")
COCO_IMG_DIR = os.path.join(COCO_DATA_DIR, "val2017")
COCO_ANN_FILE = os.path.join(COCO_DATA_DIR, "annotations/instances_val2017.json")
OUTPUT_DIR = "./figures_q15"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42

# Deterministic sample choices (recorded here for reproducibility).
# ImageNet indices follow the Q07 convention of fixed validation indices.
IMAGENET_SAMPLE_IDX = [100, 5000, 12000]
ADE20K_SAMPLE_IDX = [3, 27, 101]
COCO_SAMPLE_IDX = [12, 40, 77]        # positions in sorted getImgIds()

ROW_COLORS = {"imagenet": "#4472c4", "ade20k": "#ed7d31", "coco": "#2ca02c"}

# Standard ADE20K / SceneParse150 class names (indices 1..150 in the masks).
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television", "airplane", "dirt track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
    "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower",
    "radiator", "glass", "clock", "flag",
]


# =============================================================================
# 2. DATA LOADING  (same sources as the experiment scripts)
# =============================================================================
def load_imagenet():
    from datasets import load_dataset
    print("[System] Loading ImageNet-1K validation set "
          "(local cache first, then Hugging Face)...")
    # NOTE: if you see red warnings like
    #   "couldn't be found on the Hugging Face Hub"
    #   "Using the latest cached version of the dataset"
    # these are NOT errors: without a token the gated repo is invisible,
    # so `datasets` silently falls back to your local cache in
    # ./imagenet_data — which is exactly what we want here.
    try:
        ds = load_dataset(
            "ILSVRC/imagenet-1k", split="validation", token=HF_TOKEN,
            streaming=False, cache_dir=DATA_CACHE_DIR, trust_remote_code=True,
        )
    except Exception as e:
        raise SystemExit(
            "\n[!] Could not load ImageNet-1K from cache or the Hub.\n"
            "    Fix EITHER of the following, then re-run:\n"
            "    (a) One-time login (recommended):  huggingface-cli login\n"
            "    (b) Set the token for this shell session:\n"
            "        PowerShell:  $env:HF_TOKEN = \"hf_xxx\"\n"
            "        cmd.exe   :  set HF_TOKEN=hf_xxx\n"
            "        Linux/mac :  export HF_TOKEN=hf_xxx\n"
            "    (`export` does not work on Windows -- use the lines above.)\n"
            f"    Original error: {type(e).__name__}: {e}"
        )
    print(f"[System] ImageNet ready: {len(ds)} images.")
    return ds


def load_ade20k():
    from datasets import load_dataset
    print("[System] Loading ADE20K (scene_parse_150) validation set...")
    ds = load_dataset(
        "scene_parse_150", split="validation",
        token=HF_TOKEN, streaming=False, trust_remote_code=True,
    )
    print(f"[System] ADE20K ready: {len(ds)} images.")
    return ds


def load_coco():
    from pycocotools.coco import COCO
    if not (os.path.isdir(COCO_IMG_DIR) and os.path.isfile(COCO_ANN_FILE)):
        raise FileNotFoundError(
            f"COCO not found at '{COCO_DATA_DIR}/'. Run RoVIT_Q02.py or "
            f"RoVIT_Q06.py once (their prepare_coco step downloads val2017 "
            f"and the annotations), then re-run this script."
        )
    print("[System] Loading COCO val2017 annotations...")
    return COCO(COCO_ANN_FILE)


# =============================================================================
# 3. STATISTICS (computed from the data, cached)
# =============================================================================
def imagenet_class_histogram(dataset):
    cache = os.path.join(OUTPUT_DIR, "imagenet_hist.npz")
    if os.path.exists(cache):
        return np.load(cache)["counts"]
    print("[Stats] Counting ImageNet validation labels (50,000)...")
    labels = np.asarray(dataset["label"])
    counts = np.bincount(labels, minlength=1000)
    np.savez(cache, counts=counts)
    return counts


def ade20k_pixel_share(dataset):
    cache = os.path.join(OUTPUT_DIR, "ade20k_pixel_share.npz")
    if os.path.exists(cache):
        return np.load(cache)["share"]
    print("[Stats] Counting ADE20K pixels over 2,000 masks (one-off, ~2 min)...")
    counts = np.zeros(151, dtype=np.int64)          # 0 = ignore, 1..150 classes
    for i in range(len(dataset)):
        mask = np.asarray(dataset[i]["annotation"], dtype=np.int64)
        counts += np.bincount(mask.ravel(), minlength=151)
        if (i + 1) % 500 == 0:
            print(f"        {i + 1}/{len(dataset)} masks")
    share = counts[1:] / counts[1:].sum() * 100.0   # % per class, ignore label 0
    np.savez(cache, share=share)
    return share


def coco_size_partition(coco):
    cache = os.path.join(OUTPUT_DIR, "coco_size_partition.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return float(d["small"]), float(d["medium"]), float(d["large"])
    print("[Stats] Computing COCO val2017 instance-size partition...")
    areas = np.array([a["area"] for a in coco.loadAnns(coco.getAnnIds())
                      if not a.get("iscrowd", 0)])
    small = float((areas < 32 ** 2).mean() * 100)
    medium = float(((areas >= 32 ** 2) & (areas < 96 ** 2)).mean() * 100)
    large = float((areas >= 96 ** 2).mean() * 100)
    np.savez(cache, small=small, medium=medium, large=large)
    return small, medium, large


# =============================================================================
# 4. SAMPLE PREPARATION
# =============================================================================
def get_imagenet_samples(dataset):
    print(f"[Samples] ImageNet indices: {IMAGENET_SAMPLE_IDX}")
    return [dataset[i]["image"].convert("RGB") for i in IMAGENET_SAMPLE_IDX]


def get_ade20k_samples(dataset):
    """Return (image, colorized mask) pairs for alpha-blended overlay."""
    print(f"[Samples] ADE20K indices: {ADE20K_SAMPLE_IDX}")
    out = []
    cmap = plt.get_cmap("nipy_spectral")
    for i in ADE20K_SAMPLE_IDX:
        img = dataset[i]["image"].convert("RGB")
        mask = np.asarray(dataset[i]["annotation"], dtype=np.int64)
        rgba = cmap(mask / 150.0)
        rgba[..., 3] = np.where(mask > 0, 0.45, 0.0)     # transparent on ignore
        out.append((img, rgba))
    return out


def get_coco_samples(coco):
    """Return (image, [bbox...]) pairs; bbox = [x, y, w, h]."""
    img_ids = sorted(coco.getImgIds())
    chosen = [img_ids[i] for i in COCO_SAMPLE_IDX]
    print(f"[Samples] COCO image ids: {chosen}")
    out = []
    for img_id in chosen:
        info = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(COCO_IMG_DIR, info["file_name"])).convert("RGB")
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        out.append((img, [a["bbox"] for a in anns], info["file_name"]))
    return out


# =============================================================================
# 5. RENDERING
# =============================================================================
def render(imagenet_imgs, ade_pairs, coco_triples,
           in_counts, ade_share, coco_sizes, out_pdf, out_png):
    fig = plt.figure(figsize=(13.5, 10))
    outer = gridspec.GridSpec(3, 2, width_ratios=[1.05, 1.6],
                              hspace=0.42, wspace=0.28)

    def sample_row(row, draw_fn, label, color):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row, 0], wspace=0.06)
        for c in range(3):
            ax = fig.add_subplot(inner[0, c])
            draw_fn(ax, c)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(1.6)
            if c == 0:
                ax.set_ylabel(label, color=color, fontsize=11,
                              fontweight="bold", labelpad=8)

    # --- Row 1: ImageNet ---
    sample_row(0, lambda ax, c: ax.imshow(imagenet_imgs[c]),
               "ImageNet-1K", ROW_COLORS["imagenet"])
    ax = fig.add_subplot(outer[0, 1])
    ax.bar(np.arange(1000), in_counts, width=1.0,
           color=ROW_COLORS["imagenet"], alpha=0.85)
    ax.axhline(50, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("class index (1–1000)", fontsize=9)
    ax.set_ylabel("val images / class", fontsize=9)
    ax.set_title("ImageNet-1K: exactly balanced validation split "
                 "(50 images per class)", fontsize=10)
    ax.set_ylim(0, 70)

    # --- Row 2: ADE20K ---
    def draw_ade(ax, c):
        img, rgba = ade_pairs[c]
        ax.imshow(img); ax.imshow(rgba)
    sample_row(1, draw_ade, "ADE20K", ROW_COLORS["ade20k"])
    ax = fig.add_subplot(outer[1, 1])
    order = np.argsort(ade_share)[::-1][:30]
    ax.bar(np.arange(30), ade_share[order],
           color=ROW_COLORS["ade20k"], alpha=0.9)
    ax.set_xticks(np.arange(30))
    ax.set_xticklabels([ADE20K_CLASSES[i] for i in order],
                       rotation=90, fontsize=6.5)
    ax.set_ylabel("pixel share (%)", fontsize=9)
    ax.set_title("ADE20K: long-tailed pixel distribution "
                 "(top-30 of 150 classes)", fontsize=10)

    # --- Row 3: COCO ---
    def draw_coco(ax, c):
        img, boxes, _ = coco_triples[c]
        ax.imshow(img)
        for (x, y, w, h) in boxes:
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor="#00e5ff", linewidth=1.0))
    sample_row(2, draw_coco, "MS COCO 2017", ROW_COLORS["coco"])
    ax = fig.add_subplot(outer[2, 1])
    s, m, l = coco_sizes
    vals = [l, m, s]
    labels = ["large ($>96^2$)", "medium ($32^2$–$96^2$)", "small ($<32^2$)"]
    shades = ["#a8ddb5", "#7bccc4", "#2ca02c"]
    ax.barh(np.arange(3), vals, color=shades)
    ax.set_yticks([])                       # no left-side labels: they were
    for yv, v, lab in zip(np.arange(3), vals, labels):   # spilling onto the
        ax.text(1.0, yv, lab, va="center", ha="left",    # sample images
                fontsize=8.5, color="black")
        ax.text(v + 0.8, yv, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("share of annotated instances (%)", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title("MS COCO 2017: instance size partition "
                 "(small objects dominate)", fontsize=10)

    # No editorial suptitle: interpretation belongs to the LaTeX caption.
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved: {out_pdf}\n[*] Saved: {out_png}")


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    if HF_TOKEN is None:
        print("[i] No HF token found (env var or huggingface-cli login).")
        print("    Proceeding in CACHE-ONLY mode -- fine if you already ran")
        print("    Q01-Q10 on this machine. Red 'couldn't be found on the")
        print("    Hugging Face Hub' warnings are expected and harmless.")
        print("    To silence them: run `huggingface-cli login` once, or set")
        print('    PowerShell: $env:HF_TOKEN="hf_xxx"   cmd: set HF_TOKEN=hf_xxx')
    else:
        print(f"[i] Using HF token from {HF_TOKEN_SOURCE}.")

    imagenet = load_imagenet()
    ade20k = load_ade20k()
    coco = load_coco()

    in_counts = imagenet_class_histogram(imagenet)
    ade_share = ade20k_pixel_share(ade20k)
    coco_sizes = coco_size_partition(coco)
    print(f"[Stats] COCO size partition (val2017, computed): "
          f"small={coco_sizes[0]:.1f}%  medium={coco_sizes[1]:.1f}%  "
          f"large={coco_sizes[2]:.1f}%")

    imagenet_imgs = get_imagenet_samples(imagenet)
    ade_pairs = get_ade20k_samples(ade20k)
    coco_triples = get_coco_samples(coco)
    print("[Samples] COCO files used:",
          [t[2] for t in coco_triples])

    render(imagenet_imgs, ade_pairs, coco_triples,
           in_counts, ade_share, coco_sizes,
           os.path.join(OUTPUT_DIR, "z_fig_dataset_overview.pdf"),
           os.path.join(OUTPUT_DIR, "z_fig_dataset_overview.png"))

    print("\n[*] Done. Copy z_fig_dataset_overview.pdf next to 00-main.tex "
          "and recompile.")


if __name__ == "__main__":
    main()
