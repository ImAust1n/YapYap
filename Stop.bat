@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation - Stop

echo ===================================================
echo  Stopping AI Speech Dictation
echo ===================================================

echo [*] Stopping Python backend...
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo [*] Stopping Electron app and terminal...
taskkill /f /im electron.exe >nul 2>&1
taskkill /fi "windowtitle eq SpeechForge-App" /f >nul 2>&1

echo [*] All processes stopped.
timeout /t 2 >nul
exit /b 0
