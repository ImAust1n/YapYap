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
    icon: path.join(__dirname, '..', 'resources', 'app_icon.png'),
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
  const iconPath = path.join(__dirname, '..', 'resources', 'app_icon.png');
  tray = new Tray(iconPath);
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
    if (now - lastTrigger < 500) return; // debounce
    lastTrigger = now;

    // Force window to appear on top and be visible no matter what state it's in.
    // Windows has focus-stealing prevention, so we temporarily boost alwaysOnTop
    // level to 'screen-saver' to guarantee it surfaces.
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.setAlwaysOnTop(true, 'screen-saver');
    mainWindow.showInactive();
    mainWindow.moveTop();

    // Drop back to normal always-on-top after a short delay so other windows
    // can still go over it if needed.
    setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.setAlwaysOnTop(true, 'normal');
      }
    }, 300);

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
