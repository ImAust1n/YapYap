@echo off
echo Cleaning up leftover processes...

REM Call the launcher with --cleanup flag to gracefully kill background processes
if exist "SpeechDictation.exe" (
    "SpeechDictation.exe" --cleanup
)

REM Wait a bit for processes to die
timeout /t 2 /nobreak >nul

REM Forcefully kill any python processes running in this directory if they still exist
REM (This is a bit aggressive but ensures no locked files)
FOR /F "tokens=2" %%I in ('TASKLIST /NH /FI "IMAGENAME eq python.exe"') DO (
    REM We don't have a reliable way to check the path in batch without wmic, so we just rely on the python launcher cleanup above.
    REM If we really wanted to, we could use wmic process where "name='python.exe' and ExecutablePath like '%%Speech2Text2%%'" call terminate
)

echo Cleaning up virtual environment...
if exist ".venv" (
    rmdir /S /Q ".venv"
)

echo Cleaning up downloaded models...
if exist "models" (
    rmdir /S /Q "models"
)

echo Cleaning up compiled files...
if exist "backend\__pycache__" (
    rmdir /S /Q "backend\__pycache__"
)
if exist "launcher\__pycache__" (
    rmdir /S /Q "launcher\__pycache__"
)

echo Cleanup finished.
