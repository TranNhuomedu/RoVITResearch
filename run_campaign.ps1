# run_campaign.ps1 -- toan bo chien dich thuc nghiem, chay lai tu dau theo phase.
# Chay tu thu muc goc RoViT_Campaign:
#   powershell -ExecutionPolicy Bypass -File run_campaign.ps1            # tat ca
#   powershell -ExecutionPolicy Bypass -File run_campaign.ps1 -Phase 2   # 1 phase
# Moi buoc log vao logs\, dung lai khi exit code != 0.
#
# YEU CAU: thu muc `rovit\` (package cua kit day du) phai nam BEN TRONG
# RoViT_Campaign\ (canh file nay). Neu thieu, cac buoc phu thuoc se tu dong
# bi BO QUA (SKIP) va duoc liet ke o cuoi; cac script standalone (RoVIT_Q*,
# new\*) van chay binh thuong.

param([int]$Phase = 0, [switch]$Redo)   # 0 = chay het; -Redo = chay lai ca buoc da xong

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path logs, results | Out-Null

# Tu kich hoat venv neu co (khoi quen activate o cua so moi)
if ($env:VIRTUAL_ENV -eq $null -and (Test-Path ".venv\Scripts\Activate.ps1")) {
    . .venv\Scripts\Activate.ps1
    Write-Host "[venv] Da kich hoat .venv" -ForegroundColor DarkGray
}

# ---- Guard: package rovit ------------------------------------------------
$HasRovit = Test-Path "rovit\__init__.py"
if (-not $HasRovit) {
    Write-Host "[!] KHONG tim thay rovit\__init__.py trong RoViT_Campaign\." -ForegroundColor Yellow
    Write-Host "    Copy thu muc rovit\ tu repo day du cua ban vao DAY (ben trong" -ForegroundColor Yellow
    Write-Host "    RoViT_Campaign\, canh run_campaign.ps1). Cac buoc legacy\exp_*" -ForegroundColor Yellow
    Write-Host "    se bi SKIP cho den khi co no; script RoVIT_Q* va new\* van chay." -ForegroundColor Yellow
}
$Skipped = @()

# Script legacy (RoVIT_Q*) tim local_config.py CANH CHINH NO -> tu dong copy
if (Test-Path "local_config.py") {
    Copy-Item local_config.py legacy\local_config.py -Force
}

function Step($name, $cmd) {
    $done = "logs\$name.done"
    if ((Test-Path $done) -and (-not $Redo)) {
        Write-Host "`n==== $name ==== [SKIP: da xong truoc do -- xoa $done hoac them -Redo de chay lai]" -ForegroundColor DarkGray
        return
    }
    Write-Host "`n==== $name ====" -ForegroundColor Cyan
    $log = "logs\$name.log"
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] $name (exit $LASTEXITCODE). Xem $log" -ForegroundColor Red
        Get-Content $log -Tail 25
        exit 1
    }
    New-Item -ItemType File -Force -Path $done | Out-Null
}

# Buoc can package rovit: tu dong skip neu thieu
function StepR($name, $cmd) {
    if (-not $HasRovit) {
        Write-Host "`n==== $name ==== [SKIP: thieu rovit\]" -ForegroundColor Yellow
        $script:Skipped += $name
        return
    }
    Step $name $cmd
}

function RunPhase($n) { return ($Phase -eq 0) -or ($Phase -eq $n) }

# ---------------- PHASE 0: kiem tra moi truong ----------------------------
Step "p0_check_env" "python check_env.py"

# ---------------- PHASE 1: setup + core classification -------------------
if (RunPhase 1) {
    if (Test-Path "calibration_indices.txt") {
        Write-Host "`n==== p1_calibration ==== [SKIP: calibration_indices.txt da co san]" -ForegroundColor DarkGray
    } else {
        StepR "p1_calibration"     "python legacy\make_calibration_set.py"
    }
    StepR "p1_classification"      "python legacy\exp_classification.py --out results\classification.csv"
    StepR "p1_outlier_profile"     "python legacy\exp_outlier_profile.py"
    Step  "p1_multilayer_hist"     "python new\fig_multilayer_hist.py"
}

# ---------------- PHASE 2: baselines (Gop Y 9 muc 3.1 - QUAN TRONG NHAT) --
if (RunPhase 2) {
    StepR "p2_ptq4vit"             "python legacy\exp_ptq4vit_baseline.py"
    Step  "p2_quarot_w6"           "python new\exp_llm_rotation_baselines.py --method quarot --bits W6A6"
    Step  "p2_quarot_w4"           "python new\exp_llm_rotation_baselines.py --method quarot --bits W4A4"
    Step  "p2_spinquant_w4"        "python new\exp_llm_rotation_baselines.py --method spinquant --bits W4A4 --steps 300"
    Step  "p2_spinquant_w6"        "python new\exp_llm_rotation_baselines.py --method spinquant --bits W6A6 --steps 300"
}

# ---------------- PHASE 3: ablations --------------------------------------
if (RunPhase 3) {
    Step  "p3_rotation_zoo"        "python legacy\RoVIT_Q07_rotation_zoo.py"
    Step  "p3_dist_metrics"        "python legacy\RoVIT_Q08_distribution_metrics.py"
    Step  "p3_targeting_seeds"     "python legacy\RoVIT_Q09_targeting_and_seeds.py"
    Step  "p3_fc_targeting_w6"     "python new\exp_fc_targeting.py --bits W6A6 --configs fc1 fc2"
    Step  "p3_fc_targeting_w4"     "python new\exp_fc_targeting.py --bits W4A4 --configs fc1 fc2"
    Step  "p3_qr_internals"        "python legacy\RoVIT_Q10_qr_internals.py"
    StepR "p3_learned_rotation"    "python legacy\exp_learned_rotation.py"
    Step  "p3_rovit_repq_w4"       "python new\exp_rovit_repq.py --bits W4A4"
    Step  "p3_rovit_repq_w6"       "python new\exp_rovit_repq.py --bits W6A6"
    StepR "p3_robustness_gptq"     "python legacy\exp_robustness.py gptq"
    StepR "p3_robustness_kappa"    "python legacy\exp_robustness.py kappa"
    StepR "p3_robustness_calib"    "python legacy\exp_robustness.py calib"
    StepR "p3_error_breakdown"     "python legacy\exp_error_breakdown.py"
    StepR "p3_rounding_control"    "python legacy\exp_rounding_control.py"
    StepR "p3_mixed_precision"     "python legacy\exp_mixed_precision.py"
}

# ---------------- PHASE 4: generalization ---------------------------------
if (RunPhase 4) {
    Step  "p4_backbones"           "python legacy\RoVIT_Q11_backbones.py"
    Step  "p4_sensitivity"         "python legacy\RoVIT_Q12_sensitivity.py"
}

# ---------------- PHASE 5: downstream + attention topology ----------------
if (RunPhase 5) {
    StepR "p5_downstream_all"      "python legacy\exp_downstream.py --target all --out results\downstream.csv"
    StepR "p5_downstream_mlp"      "python legacy\exp_downstream.py --target mlp --rotations none qr hadamard --out results\downstream_matched.csv"
    StepR "p5_seg_seeds_w6"        "python new\exp_downstream_seeds.py --task segmentation --bits W6A6 --seeds 42 123 456 789 1024"
    StepR "p5_det_seeds_w6"        "python new\exp_downstream_seeds.py --task detection --bits W6A6 --seeds 42 123 456 789 1024"
    StepR "p5_seg_seeds_w4"        "python new\exp_downstream_seeds.py --task segmentation --bits W4A4 --seeds 42 123 456 789 1024"
    Step  "p5_attention_cka_w6"    "python new\exp_attention_cka.py --bits W6A6"
    Step  "p5_attention_cka_w4"    "python new\exp_attention_cka.py --bits W4A4"
    StepR "p5_attention_maps"      "python legacy\exp_attention_maps.py"
}

# ---------------- PHASE 6: hardware + figures -----------------------------
if (RunPhase 6) {
    StepR "p6_latency_int8"        "python legacy\bench_latency_int8.py"
    Step  "p6_throughput"          "python new\bench_throughput.py --batches 1 8 32 128"
    StepR "p6_latency"             "python legacy\bench_latency.py"
    Step  "p6_fig_hardware"        "python new\run_fig_hardware.py"
    StepR "p6_fig_dataset"         "python legacy\fig_dataset_overview.py"
}

# ---------------- Dien so lieu vao ban thao -------------------------------
Remove-Item -ErrorAction SilentlyContinue "logs\fill_macros.done"
Step "fill_macros" "python paper\fill_macros.py"
if ($Skipped.Count -gt 0) {
    Write-Host "`nCac buoc da SKIP vi thieu rovit\ :" -ForegroundColor Yellow
    $Skipped | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host "Copy rovit\ vao RoViT_Campaign\ roi chay lai phase tuong ung." -ForegroundColor Yellow
}
Write-Host "`nHOAN TAT. So lieu da duoc dien vao paper\99-results-macros.tex" -ForegroundColor Green
