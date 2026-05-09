let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let livePollerInterval = null;  // real-time transcript updates while recording
let lastPastedCount = 0;         // how many transcript items have been pasted so far

const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const rawTextDiv = document.getElementById('raw-text');
const processedTextDiv = document.getElementById('processed-text');
const footer = document.getElementById('footer');

const latStt = document.getElementById('lat-stt');
const latClean = document.getElementById('lat-clean');
const latTotal = document.getElementById('lat-total');

// Poll the backend every second while recording and update the popup live.
function startLivePolling() {
  if (livePollerInterval) return;
  lastPastedCount = 0;

  livePollerInterval = setInterval(async () => {
    try {
      const res = await fetch('http://localhost:5000/api/transcripts');
      if (res.ok) {
        const data = await res.json();
        const items = data.items || [];
        const text = (data.paragraph || '').trim();

        // Update popup with the full rolling paragraph
        if (text) processedTextDiv.innerText = text;

        // Paste only the chunks that have arrived since the last paste
        if (items.length > lastPastedCount) {
          const newChunks = items.slice(lastPastedCount);
          const newText = newChunks.map(i => (i.text || '').trim()).filter(Boolean).join(' ');
          if (newText) {
            // Prepend a space so it slots into existing text without smashing
            const toPaste = lastPastedCount > 0 ? ' ' + newText : newText;
            lastPastedCount = items.length;
            window.electronAPI.pasteText(toPaste);
          }
        }
      }
    } catch (_) {
      // Silently ignore network hiccups during recording
    }
  }, 1000);
}

function stopLivePolling() {
  if (livePollerInterval) {
    clearInterval(livePollerInterval);
    livePollerInterval = null;
  }
}

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
  stopLivePolling();  // stop live updates before fetching final result
  updateStatus('processing', '⚙ Processing...');
  
  try {
    const res = await fetch('http://localhost:5000/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' })
    });
    if (!res.ok) throw new Error('Stop failed');

    // Wait for the backend to finish flushing the tail audio before polling.
    // The flush can take several seconds if the leftover buffer is large.
    await new Promise(r => setTimeout(r, 2000));

    // Poll for result — up to 24 attempts × 500 ms = 12 seconds max
    let attempts = 0;
    const pollInterval = setInterval(async () => {
      attempts++;
      try {
        const transRes = await fetch('http://localhost:5000/api/transcripts');
        if (transRes.ok) {
          const data = await transRes.json();
          if (data.count > 0 || attempts > 24) {
            clearInterval(pollInterval);

            const fullText = data.paragraph || '';
            const items = data.items || [];
            processedTextDiv.innerText = fullText || 'No speech detected.';
            updateStatus('done', 'Done');

            // Paste any tail items that arrived after the live poller stopped
            if (items.length > lastPastedCount) {
              const newChunks = items.slice(lastPastedCount);
              const newText = newChunks.map(i => (i.text || '').trim()).filter(Boolean).join(' ');
              if (newText) {
                const toPaste = lastPastedCount > 0 ? ' ' + newText : newText;
                lastPastedCount = items.length;
                window.electronAPI.pasteText(toPaste);
              }
            }

            window.electronAPI.transcriptionComplete(fullText);
          }
        }
      } catch (err) {
        console.error(err);
      }
      
      if (attempts > 24) {
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
    // --- New session: wipe all stale state immediately ---
    rawTextDiv.innerText = '';
    processedTextDiv.innerText = '';
    footer.style.display = 'none';
    if (latStt)   latStt.innerText   = '';
    if (latClean) latClean.innerText = '';
    if (latTotal) latTotal.innerText = '';
    // -----------------------------------------------------
    isRecording = true;
    startBackendRecording().then(() => {
      updateStatus('recording', '🎤 Listening... (Press again to stop)');
      startLivePolling();  // ← start showing text as it arrives
    });
  } else {
    isRecording = false;
    stopBackendRecording();  // stopLivePolling() is called inside this
  }
});
