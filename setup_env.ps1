# setup_env.ps1 -- cai moi truong mot lan. Chay tu thu muc RoViT_Campaign:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
Write-Host "== Cai PyTorch (CUDA 12.8, can cho RTX 5080) ==" -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Write-Host "[FAILED] cai torch" -ForegroundColor Red; exit 1 }
Write-Host "== Cai cac thu vien con lai ==" -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[FAILED] cai requirements" -ForegroundColor Red; exit 1 }
Write-Host "== Kiem tra moi truong ==" -ForegroundColor Cyan
python check_env.py
