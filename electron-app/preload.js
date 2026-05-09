const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onShortcutTrigger: (callback) => ipcRenderer.on('shortcut-trigger', callback),
  transcriptionComplete: (text) => ipcRenderer.send('transcription-complete', text),
  pasteText: (text) => ipcRenderer.send('paste-text', text)
});
