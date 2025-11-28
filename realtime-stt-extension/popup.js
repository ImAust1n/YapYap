let port = chrome.runtime.connect({ name: "yapyap" });

const micButton = document.getElementById("micButton");
const statusBadge = document.getElementById("status");
const output = document.getElementById("output");

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
