let port = chrome.runtime.connect({ name: "yapyap" });

const micButton = document.getElementById("micButton");
const statusBadge = document.getElementById("status");
const output = document.getElementById("output");
const copyButton = document.getElementById("copyButton");
const searchButton = document.getElementById("searchButton");

let listening = false;

function setListeningState(isListening) {
  listening = isListening;
  micButton.classList.toggle("listening", isListening);
  statusBadge.textContent = isListening ? "Listening" : "Idle";
}

micButton.addEventListener("click", () => {
  if (!listening) {
    port.postMessage({ action: "START" });
  } else {
    port.postMessage({ action: "STOP" });
  }
});

copyButton?.addEventListener("click", async () => {
  const text = output.value.trim();
  if (!text) {
    statusBadge.textContent = "Nothing to copy";
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    statusBadge.textContent = "Copied";
    setTimeout(() => {
      statusBadge.textContent = listening ? "Listening" : "Idle";
    }, 1500);
  } catch (error) {
    statusBadge.textContent = "Copy failed";
    console.error("Clipboard copy failed", error);
  }
});

searchButton?.addEventListener("click", () => {
  const text = output.value.trim();
  if (!text) {
    statusBadge.textContent = "Nothing to search";
    return;
  }

  const encoded = encodeURIComponent(text);
  chrome.tabs.create({ url: `https://www.google.com/search?q=${encoded}` });
});

port.onMessage.addListener((msg) => {
  if (!msg) return;

  if (msg.status) {
    if (msg.status === "listening") {
      setListeningState(true);
    } else if (msg.status === "stopped") {
      setListeningState(false);
    } else {
      statusBadge.textContent = msg.status;
    }
  }

  if (msg.error) {
    statusBadge.textContent = `Error`;
  }

  if (msg.transcript) {
    output.value += msg.transcript + "\n";
    output.scrollTop = output.scrollHeight;
  }
});
