import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk

def predownload_model(python_exe: Path, base_dir: Path, model_size: str = "small"):
    """
    Runs a tiny script using the venv python to download the faster-whisper model
    into the models/ directory before the backend starts, showing a progress UI.
    """
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # We create a temporary script to run in the venv
    script_content = f"""
import sys
from faster_whisper import download_model

model_size = "{model_size}"
download_root = r"{str(models_dir)}"

try:
    print(f"Checking/Downloading model {{model_size}}...", flush=True)
    download_model(model_size, output_dir=download_root)
    print("DONE", flush=True)
except Exception as e:
    print(f"ERROR: {{e}}", flush=True)
    sys.exit(1)
"""
    temp_script = base_dir / "launcher" / "temp_download.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)

    import subprocess
    
    # We show a simple tkinter window with an indeterminate progress bar
    root = tk.Tk()
    root.title("Speech Dictation Setup")
    root.geometry("400x150")
    root.eval('tk::PlaceWindow . center')
    root.resizable(False, False)
    
    label = ttk.Label(root, text=f"Checking AI Models ({model_size})...\nThis may take a few minutes on first launch.", justify="center")
    label.pack(pady=20)
    
    progress = ttk.Progressbar(root, mode='indeterminate', length=300)
    progress.pack(pady=10)
    progress.start(15)
    
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen(
        [str(python_exe), str(temp_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        text=True
    )
    
    def check_process():
        if process.poll() is None:
            root.after(100, check_process)
        else:
            root.destroy()
            if temp_script.exists():
                temp_script.unlink()
            
    root.after(100, check_process)
    root.mainloop()

    if process.returncode != 0:
        print("Model download failed!")
        return False
        
    return True
