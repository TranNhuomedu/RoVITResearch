# run_all.ps1 -- execute the full experimental campaign in dependency order.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File run_all.ps1
# Each step streams to the console, logs to logs\, and stops on real failures
# (non-zero exit code), not on stderr chatter like tqdm progress bars.

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path logs, results | Out-Null

function Step($name, $cmd) {
    Write-Host "`n==== $name ====" -ForegroundColor Cyan
    $log = "logs\$name.log"
    # cmd /c merges stderr into stdout before PowerShell sees it, so native
    # stderr output is plain text here (no NativeCommandError on PS 5.1).
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[FAILED] $name (exit $LASTEXITCODE). Last lines of $log :" -ForegroundColor Red
        Get-Content $log -Tail 25
        exit 1
    }
}

# ---- Pre-flight environment check ----
Step "pre_check" "python check_setup.py"

# ---- Phase 0: calibration protocol (CHOOSE THE SPLIT -- see README) ----
Step "00_calibration" "python experiments\make_calibration_set.py --split validation --n 128 --seed 42"

# ---- Phase 1: fast pipeline validation (~1-2 h total) ----
Step "01_error_breakdown"  "python experiments\exp_error_breakdown.py"
Step "02_outlier_profile"  "python experiments\exp_outlier_profile.py"
Step "03_attention_maps"   "python experiments\exp_attention_maps.py"
Step "04_bench_latency"    "python experiments\bench_latency.py"
Step "05_fig_datasets"     "python experiments\fig_dataset_overview.py"

# ---- Phase 2: headline accuracy tables (longest) ----
Step "06_classification"   "python experiments\exp_classification.py"
Step "07_ptq4vit"          "python experiments\exp_ptq4vit_baseline.py"
Step "08_ablation"         "python experiments\exp_rotation_ablation.py"

# ---- Phase 3: downstream tasks ----
Step "09_downstream_all"   "python experiments\exp_downstream.py --target all --rotations none qr"
Step "10_downstream_mlp"   "python experiments\exp_downstream.py --target mlp --rotations none qr hadamard --out results\downstream_mlp.csv"
Step "11_mixed_precision"  "python experiments\exp_mixed_precision.py"

# ---- Phase 4: robustness and learned rotation ----
Step "12_seeds"            "python experiments\exp_robustness.py seeds"
Step "13_calib_size"       "python experiments\exp_robustness.py calib"
Step "14_kappa"            "python experiments\exp_robustness.py kappa"
Step "15_gptq"             "python experiments\exp_robustness.py gptq"
Step "16_learned_128"      "python experiments\exp_learned_rotation.py --calib-size 128"
Step "17_learned_1024"     "python experiments\exp_learned_rotation.py --calib-size 1024"
Step "18_learned_2048"     "python experiments\exp_learned_rotation.py --calib-size 2048"

Write-Host "`nAll steps complete. CSVs and figures are in results\." -ForegroundColor Green
