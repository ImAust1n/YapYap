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

:: ─── BOOTSTRAP ──────────────────────────────────────────────────────────────
:: If project files are missing (running Start.bat standalone), clone the repo
:: and relaunch from the installed location automatically.
if not exist "%~dp0backend\app.py" (
    echo ===================================================
    echo  SpeechForge First-Time Setup
    echo ===================================================
    echo [*] Project files not found.
    echo [*] Downloading SpeechForge from GitHub...
    echo.

    set "INSTALL_DIR=%USERPROFILE%\SpeechForge"

    git --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Git is not installed.
        echo [*] Please install Git from: https://git-scm.com/downloads
        echo [*] Then run this file again.
        pause
        exit /b 1
    )

    :: ── Auto-cleanup broken states from old/failed attempts ──────────────────
    if exist "!INSTALL_DIR!" (
        :: Old wrong structure: had Speech2Text2 subfolder inside SpeechForge
        if exist "!INSTALL_DIR!\Speech2Text2" (
            echo [*] Removing old incorrect installation structure...
            rmdir /S /Q "!INSTALL_DIR!" >nul 2>&1
        )
        :: Invalid git repo: SpeechForge exists but has no .git folder
        if not exist "!INSTALL_DIR!\.git" (
            echo [*] Removing incomplete installation (no git data)...
            rmdir /S /Q "!INSTALL_DIR!" >nul 2>&1
        )
    )
    :: ─────────────────────────────────────────────────────────────────────────

    :: Check if a VALID install already exists (check for actual file, not just folder)
    if exist "!INSTALL_DIR!\backend\app.py" (
        echo [*] Found existing install - updating...
        cd /d "!INSTALL_DIR!"
        git pull >> "%TEMP%\sf_update.log" 2>&1
        cd /d "%~dp0"
    ) else (
        :: Clean up anything left over before fresh clone
        if exist "!INSTALL_DIR!" rmdir /S /Q "!INSTALL_DIR!" >nul 2>&1

        echo [*] Cloning to !INSTALL_DIR! ...
        git clone --depth=1 https://github.com/ImAust1n/YapYap.git "!INSTALL_DIR!"

        :: Verify the clone actually produced the expected files
        if not exist "!INSTALL_DIR!\backend\app.py" (
            echo [ERROR] Download failed or was incomplete.
            echo [*] Check your internet connection and try again.
            echo [*] Or download manually from: https://github.com/ImAust1n/YapYap
            pause
            exit /b 1
        )
        echo [*] Download complete!
    )

    echo [*] Relaunching from installed location...
    start "" "!INSTALL_DIR!\Start.bat"
    exit /b 0
)
:: ─── END BOOTSTRAP ──────────────────────────────────────────────────────────

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
:: Only run pip install on first setup or if requirements changed
if not exist "%INSTALL_LOG%" (
    "%VENV_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt" >> "%INSTALL_LOG%" 2>&1
    echo [*] Python dependencies installed.
) else (
    echo [*] Python dependencies already installed. Skipping.
)

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

echo [*] Checking Node.js dependencies...
if exist "electron-app\package.json" (
    :: Only run npm install if node_modules doesn't exist yet
    if not exist "electron-app\node_modules" (
        echo [*] Installing Node.js dependencies for first time...
        cmd /c "cd electron-app && npm install >> ..\logs\install_frontend.log 2>&1"
    ) else (
        echo [*] Node.js dependencies already installed. Skipping.
    )
)

:: Only download Electron binary if it's actually missing
if not exist "electron-app\node_modules\electron\dist\electron.exe" (
    echo [*] Electron binary missing - downloading...
    if exist "electron-app\node_modules\electron" (
        rmdir /S /Q "electron-app\node_modules\electron" >nul 2>&1
    )
    cmd /c "cd electron-app && npm install electron --save-dev >> ..\logs\install_frontend.log 2>&1"
)

if not exist "electron-app\node_modules\electron\dist\electron.exe" (
    echo [ERROR] Electron could not be downloaded.
    echo [*] Please check your internet connection and try again.
    echo [*] You can also try running manually: cd electron-app ^&^& npm install
    pause
    exit /b 1
)

echo [*] Launching application...
:: Write a tiny helper script to avoid path escaping issues
set "SF_LAUNCHER=%TEMP%\sf_start_frontend.bat"
echo @echo off > "%SF_LAUNCHER%"
echo cd /d "%~dp0electron-app" >> "%SF_LAUNCHER%"
echo npm start >> "%SF_LAUNCHER%"

:: Launch it via PowerShell and capture PID
for /f %%i in ('powershell -NoProfile -Command "(Start-Process cmd -ArgumentList '/k \"%SF_LAUNCHER%\"' -PassThru).Id"') do set FRONTEND_PID=%%i
echo !FRONTEND_PID! > "%LOG_DIR%\frontend.pid"

echo [*] Application started successfully!
echo [%date% %time%] Startup complete >> "%STARTUP_LOG%"
timeout /t 3 >nul
exit /b 0
