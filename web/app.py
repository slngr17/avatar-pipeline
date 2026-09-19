import os
import sys
import io
import time
import base64
import json
import logging
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebStudio")

app = None  # Will be assigned below

# ── Lazy globals (initialized in startup event, NOT at import time) ──
face_swapper: Optional[Any] = None
audio_engine: Optional[Any] = None

# Directory paths
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PRESETS_DIR = os.path.join(STATIC_DIR, "presets")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "models", "background")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "face")

# Create dirs eagerly (cheap, no I/O beyond mkdir)
os.makedirs(PRESETS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Import cv2 safely ──
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available — video processing disabled")

# ── FastAPI app ──
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Avatar Pipeline Web Studio")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def create_default_presets():
    """Generates attractive default background gradient presets if missing."""
    if not CV2_AVAILABLE:
        logger.warning("Skipping preset generation (OpenCV unavailable)")
        return

    presets = {
        "studio.jpg": ((20, 20, 30), (70, 50, 40)),
        "cyberpunk.jpg": ((40, 10, 50), (120, 20, 180)),
        "office.jpg": ((35, 45, 55), (100, 120, 130)),
        "synthwave.jpg": ((10, 10, 40), (200, 60, 120)),
    }
    for filename, (c1, c2) in presets.items():
        path = os.path.join(PRESETS_DIR, filename)
        if not os.path.exists(path):
            try:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                for y in range(720):
                    alpha = y / 720.0
                    color = [
                        int(c1[i] * (1.0 - alpha) + c2[i] * alpha) for i in range(3)
                    ]
                    img[y, :] = color
                cv2.imwrite(path, img)
                logger.info(f"Created preset: {filename}")
            except Exception as e:
                logger.warning(f"Failed to create preset {filename}: {e}")


@app.on_event("startup")
async def startup_event():
    """Deferred initialization — runs AFTER Uvicorn binds the port."""
    global face_swapper, audio_engine
    logger.info("Running startup initialization...")

    # 1. Generate preset backgrounds
    try:
        create_default_presets()
    except Exception as e:
        logger.warning(f"Preset generation failed: {e}")

    # 2. Load face swapper model (optional)
    ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            from src.vision.swapper import FaceSwapper
            logger.info(f"Loading FaceSwapper model: {ONNX_MODEL_PATH}")
            face_swapper = FaceSwapper(model_path=ONNX_MODEL_PATH, use_gpu=True)
        except Exception as e:
            logger.warning(f"FaceSwapper load failed (non-fatal): {e}")

    # 3. Audio engine (pure NumPy, no hardware needed)
    try:
        from src.audio.rvc_engine import RVCStreamer
        audio_engine = RVCStreamer(sample_rate=44100, chunk_size=256, semitones=0.0)
        logger.info("Audio engine initialized")
    except Exception as e:
        logger.warning(f"Audio engine init failed (non-fatal): {e}")

    logger.info("Startup complete — ready to serve requests")


# ── Health & Index routes ──

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "app": "AvatarPipelineWebStudio"})


@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Avatar Pipeline Web Studio: index.html not found</h2>")


# ── Preset / Background API ──

@app.get("/api/presets")
async def list_presets():
    """Returns all preset and uploaded background files."""
    presets = []
    if os.path.exists(PRESETS_DIR):
        for f in os.listdir(PRESETS_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                presets.append({
                    "name": os.path.splitext(f)[0].capitalize(),
                    "filename": f,
                    "url": f"/static/presets/{f}",
                    "type": "preset",
                    "path": os.path.join(PRESETS_DIR, f),
                })
    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                presets.append({
                    "name": os.path.splitext(f)[0].capitalize(),
                    "filename": f,
                    "url": f"/api/background/{f}",
                    "type": "custom",
                    "path": os.path.join(UPLOADS_DIR, f),
                })
    return {"presets": presets}


@app.get("/api/background/{filename}")
async def get_custom_background(filename: str):
    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse(status_code=404, content="Background not found")


@app.post("/api/upload-background")
async def upload_background(file: UploadFile = File(...)):
    """Uploads a custom background image."""
    safe_name = os.path.basename(file.filename or "uploaded_bg.jpg")
    dest_path = os.path.join(UPLOADS_DIR, safe_name)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)
    return {
        "status": "success",
        "filename": safe_name,
        "path": dest_path,
        "url": f"/api/background/{safe_name}",
    }


# ── Video WebSocket ──

@app.websocket("/ws/video")
async def websocket_video_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket for live video frame processing:
    1. Client sends base64 image + controls configuration.
    2. Server performs background matting & optional face swap.
    3. Server replies with processed base64 frame + latency stats.
    """
    await websocket.accept()
    logger.info("Video client connected.")

    # Lazy import segmenter only when a client connects
    overlay_fn = None
    try:
        from src.vision.segmenter import overlay_on_background
        overlay_fn = overlay_on_background
    except Exception as e:
        logger.warning(f"Segmenter not available: {e}")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            t_start = time.perf_counter()
            msg = json.loads(raw_msg)

            data_url = msg.get("image", "")
            if not data_url or "," not in data_url:
                continue

            if not CV2_AVAILABLE:
                continue

            base64_data = data_url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            settings = msg.get("settings", {})
            bg_path = settings.get("bg_path", "")
            threshold = float(settings.get("threshold", 0.5))
            feather = int(settings.get("feather", 7))
            enable_face_swap = bool(settings.get("face_swap", False))

            # Step 1: Face Swapping
            if enable_face_swap and face_swapper is not None:
                try:
                    frame = face_swapper.process_frame(frame)
                except Exception:
                    pass

            # Step 2: Background Compositor
            if bg_path and os.path.exists(bg_path) and overlay_fn is not None:
                try:
                    frame = overlay_fn(
                        frame,
                        bg_image_path=bg_path,
                        threshold=threshold,
                        feather_kernel=feather,
                    )
                except Exception:
                    pass

            # Encode back to JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode(".jpg", frame, encode_params)
            encoded_jpg = base64.b64encode(buffer).decode("utf-8")

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            await websocket.send_text(
                json.dumps({
                    "image": f"data:image/jpeg;base64,{encoded_jpg}",
                    "processing_ms": round(t_elapsed_ms, 1),
                })
            )
    except WebSocketDisconnect:
        logger.info("Video client disconnected.")
    except Exception as e:
        logger.error(f"Video WebSocket error: {e}")


# ── Audio WebSocket ──

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket for live audio pitch shifting.
    Receives Float32 PCM arrays, applies pitch shift, returns transformed PCM.
    """
    await websocket.accept()
    logger.info("Audio client connected.")

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and audio_engine is not None:
                audio_bytes = msg["bytes"]
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                transformed = audio_engine.pitch_shift(audio_chunk)
                await websocket.send_bytes(transformed.astype(np.float32).tobytes())

            elif "text" in msg and audio_engine is not None:
                payload = json.loads(msg["text"])
                if "semitones" in payload:
                    new_semitones = float(payload["semitones"])
                    audio_engine.semitones = new_semitones
                    audio_engine.pitch_factor = 2.0 ** (new_semitones / 12.0)
                    await websocket.send_text(json.dumps({
                        "status": "updated",
                        "semitones": new_semitones,
                    }))
    except WebSocketDisconnect:
        logger.info("Audio client disconnected.")
    except Exception as e:
        logger.error(f"Audio WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
