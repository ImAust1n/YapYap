# SpeechForge Dictation

An AI-powered speech dictation desktop application that runs completely locally. It features an automated setup, offline model management, and a seamless native Windows experience.

## Download & Installation (For Users)

1. Navigate to the **[Releases](../../releases)** page on GitHub.
2. Download the latest `SpeechDictationSetup.exe` installer.
3. Run the installer and follow the standard setup prompts.
4. Launch **SpeechForge Dictation** from your Desktop or Start Menu.
   - **First Launch:** The launcher will display a progress window while it automatically sets up the Python environment and downloads the required offline AI models (approx. 1GB). This will take a few minutes depending on your internet connection.
   - **Subsequent Launches:** The application will instantly and silently spin up the backend processes and open the frontend interface.

## Development & Building from Source

This project includes a complete launcher, installer, and uninstaller distribution system.

### Architecture Overview
- `launcher/`: Contains the Python code for the single-click executable. Handles auto-venv, dependency installation, model downloading, and background process orchestration.
- `installer/`: Contains the Inno Setup script (`SpeechDictationSetup.iss`) to compile the final `.exe` Windows installer and post-uninstall cleanup scripts.
- `backend/`: The FastAPI local server and `faster-whisper` transcription engine.
- `electron-app/` & `frontend/`: The user interfaces for the application.

### Step 1: Build the Electron App (Frontend)
To ensure the launcher starts the native Electron window instead of the default web browser:
1. Navigate to the `electron-app/` directory.
2. Install dependencies: `npm install`
3. Build the unpacked directory: `npx electron-builder --win --dir`
4. Ensure `SpeechForge Desktop.exe` is generated inside `electron-app/dist/win-unpacked/`.

### Step 2: Build the Launcher Executable
1. Open a command prompt or PowerShell at the root of the project.
2. Run `build_exe.bat`.
3. This will install PyInstaller (if not present) and compile `launcher/main.py` into a single background executable located at `build/dist/SpeechDictation.exe`.

### Step 3: Build the Professional Installer
1. Download and install [Inno Setup 6](https://jrsoftware.org/isdl.php).
2. Open `installer/SpeechDictationSetup.iss` in the Inno Setup Compiler.
3. Click **Compile** (or press `Ctrl+F9`).
4. This will bundle the launcher executable, backend, frontend, electron app, and uninstallation scripts into a single, professional setup wizard.
5. The final installer will be saved to `build/installer/SpeechDictationSetup.exe`.

## Uninstalling
The installer automatically registers a native Windows Uninstaller. When you uninstall from "Add or Remove Programs", it will:
1. Run `uninstall_cleanup.bat` to gracefully shut down the background Python process.
2. Forcefully delete the `.venv` and `models` folders to free up disk space.
3. Remove all program files, shortcuts, and registry keys.
