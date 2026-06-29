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

echo [*] Stopping Electron app...
taskkill /f /im electron.exe >nul 2>&1

echo [*] Closing Node.js and terminal...
taskkill /f /im node.exe >nul 2>&1

:: Kill the SpeechForge-App cmd window by its title
taskkill /fi "windowtitle eq SpeechForge-App" /f >nul 2>&1

:: Also close any cmd windows with npm in the title (fallback)
taskkill /fi "windowtitle eq npm" /f >nul 2>&1

echo [*] All processes stopped.
timeout /t 1 >nul

:: Close this Stop.bat window too
exit
