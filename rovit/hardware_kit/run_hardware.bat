@echo off
setlocal
rem ============================================================
rem  RoViT hardware kit - chay 3 thi nghiem hardware con thieu
rem  Cach dung: chep ca thu muc hardware_kit vao trong thu muc
rem  RoViT (noi co san experiments\ va results\), roi double-click
rem  file nay. Khuyen nghi: chuot phai -> Run as administrator
rem  de khoa duoc xung nhip GPU (do latency chinh xac hon).
rem ============================================================
cd /d %~dp0

rem ---- 1. Tim thu muc repo (chinh no hoac thu muc cha) ----
set REPO=
if exist experiments\exp_downstream.py set REPO=%CD%
if not defined REPO if exist ..\experiments\exp_downstream.py set REPO=%CD%\..
if not defined REPO (
    echo [LOI] Khong tim thay repo. Hay dat thu muc hardware_kit
    echo        vao BEN TRONG thu muc RoViT roi chay lai.
    pause & exit /b 1
)

rem ---- 2. Cai cac file code va (neu di kem trong kit nay) ----
if exist bench_latency_int8.py copy /Y bench_latency_int8.py "%REPO%\experiments\" >nul
if exist exp_classification.py copy /Y exp_classification.py "%REPO%\experiments\" >nul
if exist rotation.py copy /Y rotation.py "%REPO%\rovit\" >nul
cd /d "%REPO%"

rem ---- 3. Kich hoat venv ----
if exist "..\.venv\Scripts\activate.bat" call "..\.venv\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -c "import torch" 2>nul || (
    echo [LOI] Khong tim thay moi truong Python co torch.
    echo        Kiem tra thu muc .venv nam o dau.
    pause & exit /b 1
)
pip show torchao >nul 2>&1 || pip install torchao

rem ---- 4. Kiem tra cac ban va da vao dung cho ----
findstr /C:"qr_block" rovit\rotation.py >nul || (
    echo [LOI] rovit\rotation.py chua co qr_block. & pause & exit /b 1)
findstr /C:"--rotation" experiments\exp_classification.py >nul || (
    echo [LOI] exp_classification.py chua co co --rotation. & pause & exit /b 1)
findstr /C:"Int8DynamicActivationInt8WeightConfig" experiments\bench_latency_int8.py >nul || (
    echo [LOI] bench_latency_int8.py chua phai ban va torchao moi. & pause & exit /b 1)
if not exist results mkdir results

rem ---- 5. Thu khoa xung nhip GPU (can quyen Admin) ----
set ROUNDS=4
nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -lgc 2000,2000 >nul 2>&1 && (
    set LOCKED=1
    echo [i] Da khoa clock GPU o 2000 MHz de do on dinh.
) || (
    echo [!] Khong khoa duoc clock GPU ^(thieu quyen Admin^).
    echo     Se tang so vong do tu 4 len 8 de bu tru nhieu nhiet.
    set ROUNDS=8
)

echo.
echo ==== BUOC 1/3: accuracy block-QR ^(khoang 1-2 gio^) ====
python experiments\exp_classification.py --models google/vit-base-patch16-224 --bits W4A4 W6A6 --methods fp32 std rovit --rotation qr_block:128 --out results\block_qr.csv
if errorlevel 1 (echo [LOI] Buoc 1 that bai - chup man hinh phia tren. & goto :cleanup)
echo --- Ket qua block_qr.csv: ---
type results\block_qr.csv

echo.
echo ==== BUOC 2/3: latency INT8 - block QR ^(khoang 1 gio^) ====
python experiments\bench_latency_int8.py --rotation qr_block:128 --rounds %ROUNDS% --out results\latency_int8_block.csv
if errorlevel 1 (echo [LOI] Buoc 2 that bai - chup man hinh phia tren. & goto :cleanup)

echo.
echo ==== BUOC 3/3: latency INT8 - dense QR ^(khoang 1 gio^) ====
python experiments\bench_latency_int8.py --rotation qr --rounds %ROUNDS% --out results\latency_int8_dense.csv
if errorlevel 1 (echo [LOI] Buoc 3 that bai - chup man hinh phia tren. & goto :cleanup)

echo.
echo ============================================================
echo  HOAN THANH. Gui 3 file sau trong thu muc results\ :
echo    block_qr.csv
echo    latency_int8_block.csv
echo    latency_int8_dense.csv
echo ============================================================

:cleanup
if defined LOCKED nvidia-smi -rgc >nul 2>&1 && echo [i] Da tra clock GPU ve mac dinh.
pause
