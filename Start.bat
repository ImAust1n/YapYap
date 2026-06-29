@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Speech Dictation Launcher

:: ==============================================================================
:: Configuration - Replace these values as needed
:: ==============================================================================
set "REPO_URL=https://github.com/ImAust1n/YapYap.git"
set "FOLDER_NAME=Speech2Text2"
:: Command to launch backend. Using 'app.py' as a placeholder, update as needed.
set "BACKEND_SCRIPT=%FOLDER_NAME%\app.py"
:: Local URL to open when backend is ready
set "FRONTEND_URL=http://localhost:5000"
:: ==============================================================================

set "LOG_DIR=logs"
set "STARTUP_LOG=%LOG_DIR%\startup.log"
set "INSTALL_LOG=%LOG_DIR%\install.log"
set "UPDATE_LOG=%LOG_DIR%\update.log"

:: ==============================================================================
:: 13. Folder Structure Setup
:: ==============================================================================
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "cache" mkdir "cache"
if not exist "models" mkdir "models"

echo ===================================================
echo AI Speech Dictation Launcher
echo ===================================================
echo [%date% %time%] Launcher started >> "%STARTUP_LOG%"

:: ==============================================================================
:: 1. Check prerequisites (Python and Git)
:: ==============================================================================
echo [*] Checking prerequisites...

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your system PATH.
    echo Please install Git from https://git-scm.com/downloads
    echo [%date% %time%] ERROR: Git missing >> "%STARTUP_LOG%"
    pause
    exit /b 1
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo [%date% %time%] ERROR: Python missing >> "%STARTUP_LOG%"
    pause
    exit /b 1
)

:: ==============================================================================
:: 2. Download / Update project
:: ==============================================================================
if not exist "%FOLDER_NAME%\.git" (
    echo [*] Downloading project from GitHub...
    git clone "%REPO_URL%" "%FOLDER_NAME%" >> "%INSTALL_LOG%" 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to clone repository. Check your internet connection.
        echo [%date% %time%] ERROR: Git clone failed >> "%STARTUP_LOG%"
        pause
        exit /b 1
    )
) else (
    echo [*] Checking for updates...
    cd "%FOLDER_NAME%"
    git pull >> "..\CURRENT_UPDATE_LOG.tmp" 2>&1
    cd ..
    type "CURRENT_UPDATE_LOG.tmp" >> "%UPDATE_LOG%"
    del "CURRENT_UPDATE_LOG.tmp"
)

:: ==============================================================================
:: 3. Create virtual environment
:: ==============================================================================
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating Python virtual environment...
    python -m venv .venv >> "%INSTALL_LOG%" 2>&1
)

:: ==============================================================================
:: 4. Activate virtual environment
:: ==============================================================================
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: ==============================================================================
:: 5. Install dependencies
:: ==============================================================================
if exist "%FOLDER_NAME%\requirements.txt" (
    echo [*] Installing/Verifying dependencies...
    :: pip caches built wheels, so subsequent runs are fast automatically
    pip install -r "%FOLDER_NAME%\requirements.txt" >> "%INSTALL_LOG%" 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install dependencies. Check logs/install.log for details.
        echo [%date% %time%] ERROR: pip install failed >> "%STARTUP_LOG%"
        pause
        exit /b 1
    )
) else (
    echo [WARNING] requirements.txt not found. Skipping dependency installation.
    echo [%date% %time%] WARN: requirements.txt missing >> "%STARTUP_LOG%"
)

:: ==============================================================================
:: 6. Download AI models
:: ==============================================================================
echo [*] Checking AI models...
:: Add your specific model download logic here if needed. 
:: Alternatively, your Python backend can handle the downloads automatically.

:: ==============================================================================
:: 7. Launch backend
:: ==============================================================================
echo [*] Checking if backend is already running...

:: Check if a pythonw process is running associated with our folder
wmic process where "name='pythonw.exe' and commandline like '%%%FOLDER_NAME%%%'" get processid | findstr [0-9] >nul
if %errorlevel% equ 0 (
    echo [*] Backend is already running in the background.
) else (
    echo [*] Starting backend server...
    :: Start backend using pythonw (no visible command window)
    start "" pythonw "%BACKEND_SCRIPT%"
    
    :: Wait a few seconds for backend to initialize
    echo [*] Waiting for backend to initialize...
    timeout /t 5 /nobreak >nul
)

:: ==============================================================================
:: 8. Launch frontend
:: ==============================================================================
echo [*] Launching frontend...
start "" "%FRONTEND_URL%"

echo [*] Application started successfully!
echo [%date% %time%] Startup complete >> "%STARTUP_LOG%"
timeout /t 3 >nul
exit /b 0
