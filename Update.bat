@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Update

set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "UPDATE_LOG=%LOG_DIR%\update.log"

echo ===================================================
echo AI Speech Dictation Updater
echo ===================================================

echo [*] Pulling latest code from GitHub...
git pull >> "%UPDATE_LOG%" 2>&1
echo [*] Update pulled successfully.

echo [*] Updating Python dependencies...
if exist "backend\.venv\Scripts\activate.bat" (
    call backend\.venv\Scripts\activate.bat
    if exist "backend\requirements.txt" (
        pip install -r backend\requirements.txt >> "%UPDATE_LOG%" 2>&1
    )
)

echo [*] Updating Node dependencies...
if exist "electron-app" (
    cd electron-app
    call npm install >> "..\logs\update_frontend.log" 2>&1
    cd ..
)

echo ===================================================
echo Update complete! You can now run Start.bat
echo ===================================================
pause
exit /b 0
