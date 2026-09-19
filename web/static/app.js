// State & Configuration
let isBroadcasting = false;
let isMuted = false;
let audioMonitorEnabled = false;

let videoStream = null;
let audioContext = null;
let audioProcessor = null;
let audioSource = null;

let videoWs = null;
let audioWs = null;

let activeBgPath = "";
let currentSettings = {
  bg_path: "",
  threshold: 0.5,
  feather: 7,
  face_swap: false,
};

// Elements
const videoEl = document.getElementById("webcamFeed");
const outputCanvas = document.getElementById("outputCanvas");
const outputCtx = outputCanvas.getContext("2d");
const pipCanvas = document.getElementById("pipCanvas");
const pipCtx = pipCanvas.getContext("2d");

const btnToggleBroadcast = document.getElementById("btnToggleBroadcast");
const btnMuteAudio = document.getElementById("btnMuteAudio");
const toggleAudioMonitor = document.getElementById("toggleAudioMonitor");
const streamStatusBadge = document.getElementById("streamStatusBadge");
const streamStatusText = document.getElementById("streamStatusText");

const pitchSlider = document.getElementById("pitchSlider");
const pitchDisplay = document.getElementById("pitchDisplay");
const pitchSliderVal = document.getElementById("pitchSliderVal");

const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdVal = document.getElementById("thresholdVal");
const featherSlider = document.getElementById("featherSlider");
const featherVal = document.getElementById("featherVal");
const toggleFaceSwap = document.getElementById("toggleFaceSwap");

const hudFps = document.getElementById("hudFps");
const hudLatency = document.getElementById("hudLatency");
const bgPresetsGrid = document.getElementById("bgPresetsGrid");
const bgUploadInput = document.getElementById("bgUploadInput");

// Hidden canvas for resizing/capturing raw webcam frames
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");
captureCanvas.width = 960;
captureCanvas.height = 540;

// Performance metrics
let frameCount = 0;
let lastFpsTime = performance.now();
let isAwaitingFrame = false;

// 1. Initialize Preset Backgrounds
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();
    bgPresetsGrid.innerHTML = "";

    // Add a "None / Original" option
    const noneItem = document.createElement("div");
    noneItem.className = "bg-item" + (activeBgPath === "" ? " active" : "");
    noneItem.innerHTML = `<div style="width:100%;height:100%;background:#1e2330;display:flex;align-items:center;justify-content:center;font-size:0.75rem;color:#aaa;">No Backdrop</div><span>None</span>`;
    noneItem.onclick = () => selectBackground("", noneItem);
    bgPresetsGrid.appendChild(noneItem);

    data.presets.forEach((preset, idx) => {
      const item = document.createElement("div");
      item.className = "bg-item";
      item.innerHTML = `<img src="${preset.url}" alt="${preset.name}"><span>${preset.name}</span>`;
      item.onclick = () => selectBackground(preset.path, item);
      bgPresetsGrid.appendChild(item);

      // Default to the first built-in studio preset
      if (idx === 0 && !activeBgPath) {
        selectBackground(preset.path, item);
      }
    });
  } catch (err) {
    console.error("Failed loading presets:", err);
  }
}

function selectBackground(path, element) {
  activeBgPath = path;
  currentSettings.bg_path = path;
  document.querySelectorAll(".bg-item").forEach(el => el.classList.remove("active"));
  if (element) element.classList.add("active");
}

// 2. Custom Background Upload
bgUploadInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload-background", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    if (result.status === "success") {
      await loadPresets();
      currentSettings.bg_path = result.path;
      activeBgPath = result.path;
    }
  } catch (err) {
    console.error("Upload failed:", err);
  }
});

// 3. UI Control Event Listeners
pitchSlider.addEventListener("input", (e) => {
  const val = parseFloat(e.target.value);
  setPitch(val);
});

function setPitch(val) {
  pitchSlider.value = val;
  pitchSliderVal.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + " st";
  pitchDisplay.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + " Semitones";
  if (audioWs && audioWs.readyState === WebSocket.OPEN) {
    audioWs.send(JSON.stringify({ semitones: val }));
  }
}
window.setPitch = setPitch;

thresholdSlider.addEventListener("input", (e) => {
  const val = parseFloat(e.target.value);
  thresholdVal.textContent = val.toFixed(2);
  currentSettings.threshold = val;
});

featherSlider.addEventListener("input", (e) => {
  const val = parseInt(e.target.value);
  featherVal.textContent = val + " px";
  currentSettings.feather = val;
});

toggleFaceSwap.addEventListener("change", (e) => {
  currentSettings.face_swap = e.target.checked;
});

btnMuteAudio.addEventListener("click", () => {
  isMuted = !isMuted;
  btnMuteAudio.textContent = isMuted ? "Mic: MUTED" : "Mic: ON";
  btnMuteAudio.style.borderColor = isMuted ? "#ef4444" : "var(--border-color)";
  btnMuteAudio.style.color = isMuted ? "#ef4444" : "var(--text-main)";
});

toggleAudioMonitor.addEventListener("change", (e) => {
  audioMonitorEnabled = e.target.checked;
});

// 4. WebSocket Connections
function initWebSockets() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  
  // Video WS
  videoWs = new WebSocket(`${proto}//${location.host}/ws/video`);
  videoWs.onopen = () => console.log("[WS] Video connected.");
  videoWs.onmessage = (event) => {
    isAwaitingFrame = false;
    const data = JSON.parse(event.data);
    
    // Draw incoming processed frame onto the main stage
    const img = new Image();
    img.onload = () => {
      outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
      updateFps();
      hudLatency.textContent = `Latency: ${data.processing_ms} ms`;
    };
    img.src = data.image;
  };
  videoWs.onclose = () => console.log("[WS] Video disconnected.");

  // Audio WS
  audioWs = new WebSocket(`${proto}//${location.host}/ws/audio`);
  audioWs.binaryType = "arraybuffer";
  audioWs.onopen = () => {
    console.log("[WS] Audio connected.");
    setPitch(parseFloat(pitchSlider.value));
  };
  audioWs.onmessage = (event) => {
    if (audioMonitorEnabled && audioContext && event.data instanceof ArrayBuffer) {
      playTransformedAudio(event.data);
    }
  };
}

// 5. Video Broadcast Loop
function sendFrameLoop() {
  if (!isBroadcasting) return;

  if (videoEl.readyState >= 2 && videoWs && videoWs.readyState === WebSocket.OPEN && !isAwaitingFrame) {
    // Capture to invisible canvas
    captureCtx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Also update PIP preview
    pipCtx.drawImage(videoEl, 0, 0, pipCanvas.width, pipCanvas.height);

    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.8);
    const packet = {
      image: dataUrl,
      settings: currentSettings,
    };

    isAwaitingFrame = true;
    videoWs.send(JSON.stringify(packet));
  }

  requestAnimationFrame(sendFrameLoop);
}

function updateFps() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    hudFps.textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastFpsTime = now;
  }
}

// 6. Audio Capture & Monitor Playback
async function initAudioPipeline(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
  audioSource = audioContext.createMediaStreamSource(stream);

  // ScriptProcessor with 256-sample buffer for low-latency transmission
  audioProcessor = audioContext.createScriptProcessor(256, 1, 1);

  audioProcessor.onaudioprocess = (e) => {
    if (!isBroadcasting || isMuted || !audioWs || audioWs.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);
    // Send binary Float32Array to server
    audioWs.send(inputData.buffer);
  };

  audioSource.connect(audioProcessor);
  audioProcessor.connect(audioContext.destination); // Required for onaudioprocess to tick
}

let nextPlayTime = 0;
function playTransformedAudio(arrayBuffer) {
  const floatData = new Float32Array(arrayBuffer);
  const audioBuffer = audioContext.createBuffer(1, floatData.length, 44100);
  audioBuffer.copyToChannel(floatData, 0);

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  const currentTime = audioContext.currentTime;
  if (nextPlayTime < currentTime) nextPlayTime = currentTime;
  source.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;
}

const standbyOverlay = document.getElementById("standbyOverlay");
if (standbyOverlay) {
  standbyOverlay.addEventListener("click", () => {
    if (!isBroadcasting) {
      btnToggleBroadcast.click();
    }
  });
}

// 7. Start / Stop Broadcast Toggle
btnToggleBroadcast.addEventListener("click", async () => {
  if (!isBroadcasting) {
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });

      videoEl.srcObject = videoStream;
      await videoEl.play();

      initWebSockets();
      await initAudioPipeline(videoStream);

      isBroadcasting = true;
      if (standbyOverlay) standbyOverlay.classList.add("hidden");
      btnToggleBroadcast.textContent = "Stop Broadcast";
      btnToggleBroadcast.className = "btn-danger";
      streamStatusText.textContent = "ON AIR";
      streamStatusBadge.style.color = "var(--accent-cyan)";
      streamStatusBadge.style.borderColor = "var(--accent-cyan)";

      sendFrameLoop();
    } catch (err) {
      alert("Could not access camera/mic: " + err.message);
    }
  } else {
    isBroadcasting = false;
    if (standbyOverlay) standbyOverlay.classList.remove("hidden");
    if (videoStream) videoStream.getTracks().forEach(t => t.stop());
    if (videoWs) videoWs.close();
    if (audioWs) audioWs.close();
    if (audioContext) audioContext.close();

    btnToggleBroadcast.textContent = "Start Broadcast";
    btnToggleBroadcast.className = "btn-primary";
    streamStatusText.textContent = "OFFLINE";
    streamStatusBadge.style.color = "var(--text-muted)";
    streamStatusBadge.style.borderColor = "var(--border-color)";
    hudFps.textContent = "FPS: 0";
    hudLatency.textContent = "Latency: 0 ms";
  }
});

// Load presets on boot
loadPresets();
