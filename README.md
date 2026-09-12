# Live Avatar Pipeline

A real-time AI streaming pipeline combining neural facial landmark/image transformation, MediaPipe selfie background segmentation, and low-latency mathematical audio pitch-shifting, routed directly to a virtual webcam (`pyvirtualcam`) and a virtual audio cable.

---

## Features

- **Vision Engine (`src/vision/swapper.py`)**:
  - Accelerated ONNX Runtime inference using `CUDAExecutionProvider` with fallback to `CPUExecutionProvider`.
  - Automatic frame preprocessing (BGR to RGB, float32 normalization, NCHW tensor layout) and postprocessing back to OpenCV BGR frames.

- **Background Compositor (`src/vision/segmenter.py`)**:
  - MediaPipe Selfie Segmentation for real-time human silhouette extraction.
  - Gaussian blur edge feathering on alpha masks to prevent jagged edges or green fringes.
  - Compositing onto static background images loaded from local disk.

- **Audio Engine (`src/audio/rvc_engine.py`)**:
  - Low-latency microphone capture in 128–256 sample chunks.
  - Mathematical pitch-shifting transformation array using time-domain resampling and linear phase interpolation in NumPy.
  - Non-blocking background worker thread routing output to a Virtual Audio Cable sink.

- **Integrated Pipeline (`main.py`)**:
  - Non-blocking multithreaded orchestrator.
  - Pushes video directly to OBS Virtual Camera / Unity Capture via `pyvirtualcam`.
  - Responsive `Ctrl+C` interrupt handler for clean termination on Windows.

---

## Directory Structure

```
avatar_pipeline/
├── src/
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── swapper.py       # ONNX Runtime GPU vision engine
│   │   └── segmenter.py     # MediaPipe selfie segmentation & background overlay
│   └── audio/
│       ├── __init__.py
│       └── rvc_engine.py    # Low-latency PyAudio pitch streamer
├── models/                  # Local model weights and background assets (gitignored)
│   ├── face/
│   └── background/
├── main.py                  # Main orchestration script
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

---

## Prerequisites

1. **Python 3.10+**
2. **Virtual Camera Driver:**
   - Install [OBS Studio](https://obsproject.com/) to register the Windows DirectShow virtual camera driver.
3. **Virtual Audio Cable:**
   - Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) to route audio to Discord, Zoom, or Google Meet.
4. **GPU Acceleration (Optional):**
   - NVIDIA GPU with CUDA 11.x/12.x and cuDNN for `onnxruntime-gpu`.

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd avatar_pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the integrated pipeline:

```bash
python main.py
```

Press `Ctrl+C` in the terminal to gracefully stop both video and audio streams.
