import os
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

class ProcessManager:
    def __init__(self, python_exe: Path, base_dir: Path):
        self.python_exe = python_exe
        self.base_dir = base_dir
        self.backend_process = None

    def start_backend(self) -> bool:
        """Starts the backend in the background."""
        backend_script = self.base_dir / "backend" / "app.py"
        if not backend_script.exists():
            print(f"Backend script not found at {backend_script}")
            return False

        print("Starting backend...")
        # CREATE_NO_WINDOW = 0x08000000 for Windows
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        
        # Start the process and detach
        self.backend_process = subprocess.Popen(
            [str(self.python_exe), str(backend_script)],
            cwd=str(self.base_dir),
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True

    def wait_for_health_check(self, url: str = "http://127.0.0.1:5000/api/status", timeout: int = 30) -> bool:
        """Polls the health check URL until it succeeds or times out."""
        print("Waiting for backend to become ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = urllib.request.urlopen(url)
                if response.getcode() == 200:
                    print("Backend is ready.")
                    return True
            except urllib.error.URLError:
                pass
            
            # Check if process crashed
            if self.backend_process and self.backend_process.poll() is not None:
                print(f"Backend process crashed with exit code {self.backend_process.returncode}")
                return False
                
            time.sleep(1)
            
        print("Timeout waiting for backend.")
        return False
        
    def stop_backend(self):
        """Stops the backend process if running."""
        if self.backend_process and self.backend_process.poll() is None:
            print("Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
