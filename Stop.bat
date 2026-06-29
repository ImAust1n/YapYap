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

echo [*] Closing Node.js processes...
taskkill /f /im node.exe >nul 2>&1

echo [*] Closing Electron terminal window...
if exist "logs\frontend.pid" (
    set /p FRONTEND_PID=<logs\frontend.pid
    if defined FRONTEND_PID (
        taskkill /f /pid !FRONTEND_PID! >nul 2>&1
    )
    del "logs\frontend.pid" >nul 2>&1
)

echo [*] All processes stopped.
timeout /t 1 >nul
exit
