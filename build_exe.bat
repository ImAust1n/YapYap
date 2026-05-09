@echo off
echo Building Launcher Executable...

REM Activate virtual environment if it exists, otherwise assume pyinstaller is installed globally
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Install PyInstaller if not present
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo Packaging launcher...
REM We use --noconsole to hide the terminal window
REM We use --onefile to produce a single exe
pyinstaller --noconsole --onefile --name "SpeechDictation" --distpath "build\dist" --workpath "build\work" launcher\main.py

echo Build complete!
echo Executable is located in build\dist\SpeechDictation.exe
pause
