@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Stop

:: ==============================================================================
:: Configuration
:: ==============================================================================
set "FOLDER_NAME=Speech2Text2"
:: ==============================================================================

echo ===================================================
echo Stopping AI Speech Dictation Backend
echo ===================================================

echo [*] Terminating backend processes...

:: Terminate any python or pythonw process that has our folder name in its command line
wmic process where "name='python.exe' and commandline like '%%%FOLDER_NAME%%%'" call terminate >nul 2>&1
wmic process where "name='pythonw.exe' and commandline like '%%%FOLDER_NAME%%%'" call terminate >nul 2>&1

echo [*] Backend stopped successfully!
timeout /t 2 >nul
exit /b 0
