@echo off
rem Launch the experiment runner: handles drive change, venv, and policy.
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\activate.bat" (
    call "%~dp0..\.venv\Scripts\activate.bat"
) else if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
pause
