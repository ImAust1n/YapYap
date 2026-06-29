@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Update

:: ==============================================================================
:: Configuration
:: ==============================================================================
set "FOLDER_NAME=Speech2Text2"
set "LOG_DIR=logs"
:: ==============================================================================

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "UPDATE_LOG=%LOG_DIR%\update.log"

echo ===================================================
echo AI Speech Dictation Updater
echo ===================================================

if not exist "%FOLDER_NAME%\.git" (
    echo [ERROR] Project folder not found. Please run Start.bat first to download it.
    pause
    exit /b 1
)

echo [*] Pulling latest code from GitHub...
cd "%FOLDER_NAME%"
git pull >> "..\CURRENT_UPDATE_LOG.tmp" 2>&1
cd ..
type "CURRENT_UPDATE_LOG.tmp" >> "%UPDATE_LOG%"
del "CURRENT_UPDATE_LOG.tmp"
echo [*] Update pulled successfully.

echo [*] Updating dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    if exist "%FOLDER_NAME%\requirements.txt" (
        pip install -r "%FOLDER_NAME%\requirements.txt" >> "%UPDATE_LOG%" 2>&1
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to update dependencies. See %UPDATE_LOG% for details.
        ) else (
            echo [*] Dependencies updated successfully.
        )
    )
) else (
    echo [WARNING] Virtual environment not found. Please run Start.bat to set it up.
)

echo ===================================================
echo Update complete! You can now run Start.bat
echo ===================================================
pause
exit /b 0
