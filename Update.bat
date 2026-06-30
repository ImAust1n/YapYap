@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Update

set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "UPDATE_LOG=%LOG_DIR%\update.log"
set "VENV_PYTHON=backend\.venv\Scripts\python.exe"

echo ===================================================
echo  AI Speech Dictation Updater
echo ===================================================

echo [*] Pulling latest code from GitHub...
git pull >> "%UPDATE_LOG%" 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] git pull failed. Check your internet connection.
    echo [*] Continuing with local dependency update...
)
echo [*] Code updated.

echo [*] Updating Python dependencies...
if exist "%VENV_PYTHON%" (
    if exist "backend\requirements.txt" (
        "%VENV_PYTHON%" -m pip install -r backend\requirements.txt >> "%UPDATE_LOG%" 2>&1
        echo [*] Python dependencies updated.
    )
) else (
    echo [*] No virtual environment found - skipping Python update.
    echo     Run Start.bat first to create the environment.
)

echo [*] Updating Node.js dependencies...
if exist "electron-app\package.json" (
    if exist "electron-app\node_modules\electron" (
        rmdir /S /Q "electron-app\node_modules\electron" >nul 2>&1
    )
    cmd /c "cd electron-app && npm install >> ..\logs\update_frontend.log 2>&1"
    echo [*] Node dependencies updated.
)

echo ===================================================
echo  Update complete! You can now run Start.bat
echo ===================================================
pause
exit /b 0
