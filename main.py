import os
import sys
import time
import threading
import cv2
import pyvirtualcam
import numpy as np

from src.vision.swapper import FaceSwapper
from src.vision.segmenter import overlay_on_background
from src.audio.rvc_engine import RVCStreamer


class AvatarPipeline:
    def __init__(
        self,
        onnx_model_path: str = "models/face/inswapper_128.onnx",
        bg_image_path: str = "models/background/studio.jpg",
        pitch_shift_semitones: float = -2.5,
        cam_index: int = 0,
        fps: int = 30,
    ):
        self.stop_event = threading.Event()
        self.cam_index = cam_index
        self.fps = fps
        self.bg_image_path = bg_image_path

        # 1. Vision Engine (Face Swapper)
        if os.path.exists(onnx_model_path):
            print(f"[Pipeline] Loading ONNX model from {onnx_model_path}...")
            self.swapper = FaceSwapper(model_path=onnx_model_path, use_gpu=True)
        else:
            print(f"[Pipeline] Notice: '{onnx_model_path}' not found. Running in passthrough mode.")
            self.swapper = None

        # 2. Audio Engine (RVC Streamer)
        self.audio = RVCStreamer(
            sample_rate=44100,
            chunk_size=256,
            semitones=pitch_shift_semitones,
            output_device_name="CABLE Input",
        )

    def run_video_pipeline(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Pipeline] Error: Unable to open camera {self.cam_index}")
            self.stop_event.set()
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] Camera opened: {width}x{height} @ {self.fps} FPS")

        try:
            with pyvirtualcam.Camera(width=width, height=height, fps=self.fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
                print(f"[Pipeline] Virtual Camera active: {cam.device}")
                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    # Step 1: Facial Landmark / ONNX Swap
                    if self.swapper is not None:
                        frame = self.swapper.process_frame(frame)

                    # Step 2: Background Compositor
                    if os.path.exists(self.bg_image_path):
                        frame = overlay_on_background(frame, self.bg_image_path)

                    # Stream BGR directly to virtual camera (no RGB conversion overhead)
                    cam.send(frame)
                    cam.sleep_until_next_frame()
        except Exception as e:
            print(f"[Pipeline] Video pipeline exception: {e}")
        finally:
            cap.release()
            print("[Pipeline] Video capture released.")

    def run_audio_pipeline(self):
        print("[Pipeline] Starting Audio Engine...")
        self.audio.start()
        # Keep alive without spinning CPU
        while not self.stop_event.is_set():
            self.stop_event.wait(timeout=0.5)
        self.audio.stop()
        print("[Pipeline] Audio Engine stopped.")

    def start(self):
        video_thread = threading.Thread(target=self.run_video_pipeline, daemon=True)
        audio_thread = threading.Thread(target=self.run_audio_pipeline, daemon=True)

        video_thread.start()
        audio_thread.start()

        print("[Pipeline] AvatarPipeline running. Press Ctrl+C to terminate.")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[Pipeline] User interrupt received. Shutting down...")
        finally:
            self.stop_event.set()
            video_thread.join(timeout=2.0)
            audio_thread.join(timeout=2.0)
            print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    app = AvatarPipeline()
    app.start()
