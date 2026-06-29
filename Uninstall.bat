@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Uninstall

echo ===================================================
echo WARNING: UNINSTALLATION
echo This will completely remove the application,
echo including all models, logs, cache, and code.
echo ===================================================

set /p CONFIRM="Are you sure you want to proceed? (Type YES to confirm): "
if /I "!CONFIRM!" neq "YES" (
    echo Uninstall cancelled.
    pause
    exit /b
)

set "FOLDER_NAME=Speech2Text2"

echo [*] Stopping backend...
if exist "Stop.bat" call Stop.bat >nul 2>&1

echo [*] Deleting environment and data...
if exist "%FOLDER_NAME%" rmdir /S /Q "%FOLDER_NAME%"
if exist ".venv" rmdir /S /Q ".venv"
if exist "models" rmdir /S /Q "models"
if exist "logs" rmdir /S /Q "logs"
if exist "cache" rmdir /S /Q "cache"

echo [*] Data removed successfully.
echo [*] Self-destructing batch scripts...

:: Create a temporary script to delete the batch files and then delete itself
echo @echo off > temp_del.bat
echo timeout /t 2 /nobreak ^>nul >> temp_del.bat
echo del Start.bat >> temp_del.bat
echo del Stop.bat >> temp_del.bat
echo del Update.bat >> temp_del.bat
echo del Uninstall.bat >> temp_del.bat
echo del temp_del.bat >> temp_del.bat

:: Launch the temporary deletion script in the background and exit immediately
start /b "" cmd /c temp_del.bat
exit
