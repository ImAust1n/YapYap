[Setup]
AppName=SpeechForge Dictation
AppVersion=1.0.0
AppPublisher=SpeechForge
DefaultDirName={autopf}\SpeechForge
DefaultGroupName=SpeechForge
UninstallDisplayIcon={app}\SpeechDictation.exe
Compression=lzma2
SolidCompression=yes
OutputDir=..\build\installer
OutputBaseFilename=SpeechDictationSetup
PrivilegesRequired=lowest

[Files]
; The main launcher executable
Source: "..\build\dist\SpeechDictation.exe"; DestDir: "{app}"; Flags: ignoreversion
; Launcher python scripts (if not bundled, but pyinstaller bundles them. We include them just in case or if we use python directly)
Source: "..\launcher\*"; DestDir: "{app}\launcher"; Flags: ignoreversion recursesubdirs createallsubdirs
; The backend API
Source: "..\backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs createallsubdirs
; The frontend UI
Source: "..\frontend\*"; DestDir: "{app}\frontend"; Flags: ignoreversion recursesubdirs createallsubdirs
; The Electron app
Source: "..\electron-app\*"; DestDir: "{app}\electron-app"; Flags: ignoreversion recursesubdirs createallsubdirs
; Root Requirements
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
; Uninstall cleanup script
Source: "uninstall_cleanup.bat"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\SpeechForge Dictation"; Filename: "{app}\SpeechDictation.exe"
Name: "{autodesktop}\SpeechForge Dictation"; Filename: "{app}\SpeechDictation.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
; Run the app after installation
Filename: "{app}\SpeechDictation.exe"; Description: "Launch SpeechForge Dictation"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Run cleanup script before deleting files
Filename: "{app}\installer\uninstall_cleanup.bat"; Flags: waituntilterminated runhidden
