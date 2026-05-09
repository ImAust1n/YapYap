const { app, BrowserWindow, globalShortcut, ipcMain, Tray, Menu, screen, clipboard, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let tray;

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  
  mainWindow = new BrowserWindow({
    width: 320,
    height: 180,
    x: width - 340,
    y: height - 200,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile('index.html');
  // Initially show it or hide it, let's keep it hidden until shortcut
  mainWindow.hide();
}

app.whenReady().then(() => {
  createWindow();

  // Create Tray
  const icon = nativeImage.createEmpty();
  tray = new Tray(icon);
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show UI', click: () => mainWindow.show() },
    { label: 'Quit', click: () => { app.isQuiting = true; app.quit(); } }
  ]);
  tray.setToolTip('SpeechForge dictation');
  tray.setContextMenu(contextMenu);

  // Register Global Shortcut
  let lastTrigger = 0;
  const registered = globalShortcut.register('CommandOrControl+Shift+D', () => {
    const now = Date.now();
    if (now - lastTrigger < 1000) return; // 1 second debounce against auto-repeat
    lastTrigger = now;
    
    if (!mainWindow.isVisible()) {
      // showInactive() reveals the overlay WITHOUT stealing focus from the
      // current text field the user was typing in.
      mainWindow.showInactive();
    }
    // Send trigger to renderer
    mainWindow.webContents.send('shortcut-trigger');
  });

  if (!registered) {
    console.error('Shortcut registration failed');
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Incremental paste: renderer sends one chunk at a time as speech is recognized.
// Because the overlay was shown with showInactive(), the original text field
// retains focus and receives the paste directly.
function doPaste(text) {
  if (!text) return;
  clipboard.writeText(text);
  setTimeout(() => {
    const script = `
      $wshell = New-Object -ComObject wscript.shell;
      $wshell.SendKeys('^v')
    `;
    const ps = spawn('powershell.exe', ['-NoProfile', '-Command', script]);
    ps.on('error', (err) => console.error('Paste failed:', err));
  }, 100); // small delay for clipboard write to propagate
}

ipcMain.on('paste-text', (event, text) => {
  doPaste(text);
});

ipcMain.on('transcription-complete', (event, _text) => {
  // All pasting is now done incrementally via 'paste-text' during recording.
  // Just hide the overlay after a brief pause so the user sees the final text.
  setTimeout(() => mainWindow.hide(), 1200);
});


// App lifecycle
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
