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
      mainWindow.show();
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

ipcMain.on('transcription-complete', (event, text) => {
  if (text) {
    clipboard.writeText(text);
    
    // Auto paste via powershell
    const script = `
      $wshell = New-Object -ComObject wscript.shell;
      $wshell.SendKeys('^v')
    `;
    const ps = spawn('powershell.exe', ['-NoProfile', '-Command', script]);
    ps.on('error', (err) => {
      console.error('Failed to auto-paste:', err);
    });
  }
  
  // Hide window after a brief delay
  setTimeout(() => {
    mainWindow.hide();
  }, 4000);
});

// App lifecycle
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
