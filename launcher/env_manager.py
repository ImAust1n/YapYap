import os
import sys
import subprocess
import venv
from pathlib import Path

def setup_environment(base_dir: Path) -> Path:
    """
    Ensures the virtual environment exists and dependencies are installed.
    Returns the path to the python executable.
    """
    venv_dir = base_dir / ".venv"
    python_exe = venv_dir / "Scripts" / "python.exe" if os.name == 'nt' else venv_dir / "bin" / "python"
    
    if not venv_dir.exists() or not python_exe.exists():
        print("Creating virtual environment...")
        import shutil
        system_python = shutil.which("python") or shutil.which("python3")
        if not system_python:
            raise RuntimeError("Python is not installed or not in PATH. Please install Python 3.10+.")
        
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        subprocess.check_call([system_python, "-m", "venv", str(venv_dir)], creationflags=creationflags)
        print("Virtual environment created.")
    
    # Check dependencies
    requirements_files = [
        base_dir / "requirements.txt",
        base_dir / "backend" / "requirements.txt"
    ]
    
    # We use a simple marker file to know if reqs were installed successfully
    marker_file = venv_dir / ".reqs_installed"
    
    if not marker_file.exists():
        print("Installing dependencies...")
        # Upgrade pip
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        for req_file in requirements_files:
            if req_file.exists():
                print(f"Installing {req_file.name}...")
                subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(req_file)],
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        # Touch marker
        with open(marker_file, 'w') as f:
            f.write("done")
        print("Dependencies installed.")
        
    return python_exe
