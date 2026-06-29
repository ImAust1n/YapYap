@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation Launcher

set "BACKEND_DIR=backend"
set "VENV_PYTHON=backend\.venv\Scripts\python.exe"
set "VENV_PIP=backend\.venv\Scripts\python.exe -m pip"
set "BACKEND_SCRIPT=backend\app.py"
set "LOG_DIR=logs"
set "STARTUP_LOG=%LOG_DIR%\startup.log"
set "INSTALL_LOG=%LOG_DIR%\install.log"
set "UPDATE_LOG=%LOG_DIR%\update.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "models" mkdir "models"

echo ===================================================
echo  AI Speech Dictation Launcher
echo ===================================================
echo [%date% %time%] Launcher started >> "%STARTUP_LOG%"

echo [*] Checking prerequisites...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [*] Checking for updates...
git pull >> "%UPDATE_LOG%" 2>&1

echo [*] Setting up Python virtual environment...
if not exist "backend\.venv\Scripts\python.exe" (
    echo [*] Creating new virtual environment...
    python -m venv backend\.venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [*] Installing Python dependencies...
"%VENV_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt" >> "%INSTALL_LOG%" 2>&1
echo [*] Python dependencies ready.

echo [*] Checking if backend is already running...
tasklist /fi "imagename eq pythonw.exe" | find /i "pythonw.exe" >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Backend is already running.
) else (
    echo [*] Starting backend server in background...
    start "" "%VENV_PYTHON%" "%BACKEND_SCRIPT%"
    echo [*] Waiting for backend to initialize...
    timeout /t 5 /nobreak >nul
)

echo [*] Installing Node.js dependencies...
if exist "electron-app\package.json" (
    cmd /c "cd electron-app && npm install >> ..\logs\install_frontend.log 2>&1"
)

echo [*] Launching application...
for /f %%i in ('powershell -NoProfile -Command "(Start-Process cmd -ArgumentList '/k cd /d \"%~dp0electron-app\" ^&^& npm start' -PassThru).Id"') do set FRONTEND_PID=%%i
echo !FRONTEND_PID! > "%LOG_DIR%\frontend.pid"

echo [*] Application started successfully!
echo [%date% %time%] Startup complete >> "%STARTUP_LOG%"
timeout /t 3 >nul
exit /b 0
