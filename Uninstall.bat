@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Uninstall

echo ===================================================
echo WARNING: UNINSTALLATION
echo This will completely remove the application data,
echo including all models, logs, cache, and virtual environments.
echo ===================================================

set /p CONFIRM="Are you sure you want to proceed? (Type YES to confirm): "
if /I "!CONFIRM!" neq "YES" (
    echo Uninstall cancelled.
    pause
    exit /b
)

echo [*] Stopping backend...
if exist "Stop.bat" call Stop.bat >nul 2>&1

echo [*] Deleting environment and data...
if exist "backend\.venv" rmdir /S /Q "backend\.venv"
if exist "electron-app\node_modules" rmdir /S /Q "electron-app\node_modules"
if exist "models" rmdir /S /Q "models"
if exist "logs" rmdir /S /Q "logs"
if exist "cache" rmdir /S /Q "cache"

echo [*] Data removed successfully. 
echo To completely remove the application, you can now safely delete this entire folder.
pause
exit /b 0
