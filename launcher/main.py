import os
import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# Add current directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from env_manager import setup_environment
from model_manager import predownload_model
from process_manager import ProcessManager
from frontend_launcher import launch_frontend

def show_splash_and_setup(base_dir: Path):
    """Shows a small setup UI while venv is being created."""
    root = tk.Tk()
    root.title("Speech Dictation Setup")
    root.geometry("400x150")
    root.eval('tk::PlaceWindow . center')
    root.resizable(False, False)
    
    label = ttk.Label(root, text="Setting up environment...\nThis happens once and may take a few minutes.", justify="center")
    label.pack(pady=20)
    
    progress = ttk.Progressbar(root, mode='indeterminate', length=300)
    progress.pack(pady=10)
    progress.start(15)
    
    result_exe = [None]
    
    def do_setup():
        try:
            exe = setup_environment(base_dir)
            result_exe[0] = exe
        except Exception as e:
            print(f"Setup failed: {e}")
        finally:
            root.destroy()
            
    root.after(100, do_setup)
    root.mainloop()
    return result_exe[0]

def cleanup_and_exit(base_dir: Path):
    """Forcefully kills backend processes running from this directory."""
    import psutil
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['exe'] and str(base_dir) in proc.info['exe']:
                print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    sys.exit(0)

def main():
    if getattr(sys, 'frozen', False):
        # Running as compiled PyInstaller executable
        launcher_dir = Path(sys.executable).parent
    else:
        # Running as python script
        launcher_dir = Path(__file__).resolve().parent.parent
        
    base_dir = launcher_dir
    
    if "--cleanup" in sys.argv:
        cleanup_and_exit(base_dir)

    # 1. Setup Environment
    venv_dir = base_dir / ".venv"
    marker_file = venv_dir / ".reqs_installed"
    
    if not venv_dir.exists() or not marker_file.exists():
        python_exe = show_splash_and_setup(base_dir)
    else:
        python_exe = venv_dir / "Scripts" / "python.exe" if os.name == 'nt' else venv_dir / "bin" / "python"
        
    if not python_exe or not python_exe.exists():
        import tkinter.messagebox
        tk.Tk().withdraw()
        tkinter.messagebox.showerror("Error", "Failed to setup Python environment.")
        sys.exit(1)

    # 2. Check/Download Models
    if not predownload_model(python_exe, base_dir, model_size="small"):
        import tkinter.messagebox
        tk.Tk().withdraw()
        tkinter.messagebox.showerror("Error", "Failed to download required AI models.")
        sys.exit(1)

    # 3. Start Backend
    manager = ProcessManager(python_exe, base_dir)
    if not manager.start_backend():
        import tkinter.messagebox
        tk.Tk().withdraw()
        tkinter.messagebox.showerror("Error", "Failed to start backend process.")
        sys.exit(1)

    # 4. Wait for Health Check
    if manager.wait_for_health_check(timeout=45):
        # 5. Launch Frontend
        launch_frontend(base_dir)
    else:
        import tkinter.messagebox
        tk.Tk().withdraw()
        tkinter.messagebox.showerror("Error", "Backend failed to become ready in time.")
        manager.stop_backend()
        sys.exit(1)
        
    # Do NOT exit, wait for backend process to finish (which should be forever unless user kills it)
    # Actually, we can exit the launcher and leave the background process running, 
    # but then we lose the ability to stop it when the launcher closes.
    # We will just exit, the process was started with CREATE_NO_WINDOW and detached.
    sys.exit(0)

if __name__ == "__main__":
    main()
