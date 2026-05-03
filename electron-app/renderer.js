let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const rawTextDiv = document.getElementById('raw-text');
const processedTextDiv = document.getElementById('processed-text');
const footer = document.getElementById('footer');

const latStt = document.getElementById('lat-stt');
const latClean = document.getElementById('lat-clean');
const latTotal = document.getElementById('lat-total');

async function startBackendRecording() {
  try {
    // Clear previous history so we only get this session's text
    await fetch('http://localhost:5000/api/reset', { method: 'POST' });

    const res = await fetch('http://localhost:5000/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' })
    });
    if (!res.ok) throw new Error('Start failed');
  } catch (err) {
    console.error('Error starting backend:', err);
    updateStatus('error', 'Backend not reachable');
  }
}

function updateStatus(state, text) {
  statusIndicator.className = 'status-indicator ' + state;
  if (text) statusText.innerText = text;
  
  if (state === 'idle') {
    rawTextDiv.innerText = '';
    processedTextDiv.innerText = '';
    footer.style.display = 'none';
  }
}

async function stopBackendRecording() {
  updateStatus('processing', '⚙ Processing...');
  
  try {
    const res = await fetch('http://localhost:5000/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' })
    });
    if (!res.ok) throw new Error('Stop failed');

    // Wait a tiny bit just to ensure backend flushed to history
    await new Promise(r => setTimeout(r, 500));

    // Poll for result
    let attempts = 0;
    const pollInterval = setInterval(async () => {
      attempts++;
      try {
        const transRes = await fetch('http://localhost:5000/api/transcripts');
        if (transRes.ok) {
          const data = await transRes.json();
          // Wait until we have at least one transcript item, or timeout
          if (data.count > 0 || attempts > 10) {
            clearInterval(pollInterval);
            
            const fullText = data.paragraph || '';
            rawTextDiv.innerText = '';
            processedTextDiv.innerText = fullText || 'No speech detected.';
            
            updateStatus('done', 'Done');
            
            if (fullText) {
              window.electronAPI.transcriptionComplete(fullText);
            } else {
              window.electronAPI.transcriptionComplete("");
            }
          }
        }
      } catch (err) {
        console.error(err);
      }
      
      if (attempts > 10) { // 5 seconds max wait
        clearInterval(pollInterval);
        updateStatus('error', 'Timeout waiting for backend');
        window.electronAPI.transcriptionComplete("");
      }
    }, 500);

  } catch (error) {
    console.error('Backend error:', error);
    updateStatus('error', 'Backend not reachable');
    rawTextDiv.innerText = '';
    processedTextDiv.innerText = error.message;
    window.electronAPI.transcriptionComplete("");
  }
}

// Listen to shortcut from main process
window.electronAPI.onShortcutTrigger(() => {
  if (!isRecording) {
    isRecording = true;
    startBackendRecording().then(() => {
      updateStatus('recording', '🎤 Listening... (Press again to stop)');
    });
  } else {
    isRecording = false;
    stopBackendRecording();
  }
});


