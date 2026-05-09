import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def launch_frontend(base_dir: Path):
    """
    Launches the frontend.
    Tries to launch the compiled Electron app first.
    If not found, falls back to opening the default web browser.
    """
    # Look for compiled electron app
    # Usually in electron-app/dist/win-unpacked/speechforge-desktop.exe
    # Or just bundled in the root folder as SpeechForge.exe
    
    electron_exe = base_dir / "electron-app" / "dist" / "win-unpacked" / "SpeechForge Desktop.exe"
    if not electron_exe.exists():
        # Fallback for dev environment or portable
        electron_exe = base_dir / "SpeechForge Desktop.exe"
        
    if electron_exe.exists():
        print("Launching Electron app...")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        subprocess.Popen(
            [str(electron_exe)],
            cwd=str(electron_exe.parent),
            creationflags=creationflags
        )
    else:
        print("Electron app not found, opening browser...")
        webbrowser.open("http://127.0.0.1:5000")
