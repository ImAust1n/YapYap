const BACKEND_URL = "http://localhost:5000/api/extension/latest";
const STATUS_URL = "http://localhost:5000/api/status";
const CONTROL_URL = "http://localhost:5000/api/control";
const POLL_INTERVAL_MS = 2000;

async function ensureBackendRunning() {
  try {
    const statusResp = await fetch(STATUS_URL, { cache: "no-store" });
    if (statusResp.ok) {
      const status = await statusResp.json();
      if (status.capture_active) {
        return true;
      }
    }

    const resp = await fetch(CONTROL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "start" }),
    });
    return resp.ok;
  } catch (err) {
    console.warn("Failed to ensure backend running", err);
    return false;
  }
}

let polling = false;
let pollTimeout = null;
let lastTranscriptId = null;
const activePorts = new Set();

function broadcast(message) {
  activePorts.forEach((port) => {
    try {
      port.postMessage(message);
    } catch (err) {
      console.warn("Failed to post message to port", err);
    }
  });
}

async function pollBackend() {
  if (!polling) {
    return;
  }

  try {
    const resp = await fetch(BACKEND_URL, { cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`Backend responded with ${resp.status}`);
    }

    const payload = await resp.json();
    if (payload && payload.id && payload.id !== lastTranscriptId) {
      lastTranscriptId = payload.id;
      broadcast({ transcript: payload.text || "", timestamp: payload.timestamp });
    }
  } catch (err) {
    console.error("YapYap poll failed", err);
    broadcast({ error: "Unable to reach YapYap backend." });
  } finally {
    pollTimeout = setTimeout(pollBackend, POLL_INTERVAL_MS);
  }
}

async function startPolling() {
  if (polling) {
    return;
  }

  const ready = await ensureBackendRunning();
  if (!ready) {
    broadcast({ error: "Unable to connect to YapYap backend" });
    return;
  }

  polling = true;
  lastTranscriptId = null;
  broadcast({ status: "listening" });
  pollBackend();
}

function stopPolling() {
  polling = false;
  if (pollTimeout) {
    clearTimeout(pollTimeout);
    pollTimeout = null;
  }
  broadcast({ status: "stopped" });
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "yapyap") {
    return;
  }

  activePorts.add(port);

  port.onMessage.addListener((msg) => {
    if (!msg || !msg.action) {
      return;
    }

    if (msg.action === "START") {
      startPolling();
    } else if (msg.action === "STOP") {
      stopPolling();
    }
  });

  port.onDisconnect.addListener(() => {
    activePorts.delete(port);
    if (activePorts.size === 0) {
      stopPolling();
    }
  });
});
