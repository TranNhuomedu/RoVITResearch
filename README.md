# RoViT — Rotation-based Vision Transformer Quantization

Reproducibility kit for "Mitigating Quantization Errors in Vision
Transformers via Orthogonal Feature Rotation".

## Layout

    rovit/                  core library (quantization, rotations, layer roles)
    experiments/            one script per experiment / figure

## Setup

    pip install -r requirements.txt
    huggingface-cli login          # required once for ILSVRC/imagenet-1k

## Script-to-paper mapping

| Script                        | Paper artifact                         |
|-------------------------------|----------------------------------------|
| make_calibration_set.py       | calibration_indices.txt (Sec. 4.1)     |
| exp_classification.py         | Tables 4, 5                            |
| exp_downstream.py             | Table 8 (--target all), Table 10 (--target mlp, qr vs hadamard) |
| exp_rotation_ablation.py      | Tables 11, 12                          |
| exp_robustness.py             | Sec. 4.1.3 (seeds, calib), Table 6 kappa, RoViT+GPTQ |
| exp_ptq4vit_baseline.py       | PTQ4ViT (reproduced) rows, Table 4     |
| exp_mixed_precision.py        | Table 9                                |
| exp_learned_rotation.py       | Table 13, Figure 5                     |
| exp_error_breakdown.py        | Table 14, Figure 6                     |
| exp_outlier_profile.py        | Table 6, Figure 3                      |
| exp_attention_maps.py         | Figure 4                               |
| bench_latency.py              | Tables 17, 19 (rotation overhead)      |
| fig_dataset_overview.py       | Figure 2                               |
| fig_hardware_comparison.py    | Figure 8                               |

## Reproducibility notes

* All rotation matrices are drawn from one `torch.Generator` seeded once per
  run (default 42), on CPU, so every target layer receives an independent
  matrix and results are identical across GPUs.
* Activation quantization is dynamic per-tensor by default; pass a
  calibration index file (`--calib`) for static MinMax scales. The mode used
  must match the protocol sentence in Sec. 4.1 of the manuscript.
* COCO val2017 is downloaded automatically to `coco_data_official/` on first
  use; ImageNet-1K and ADE20K load through HuggingFace `datasets`.
