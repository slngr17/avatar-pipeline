import os
import cv2
import numpy as np
from typing import Optional, Any

# Robust MediaPipe import to handle different MediaPipe packaging & versions
mp_selfie = None
try:
    import mediapipe.python.solutions.selfie_segmentation as mp_selfie
except (ImportError, AttributeError):
    try:
        from mediapipe import solutions
        mp_selfie = getattr(solutions, "selfie_segmentation", None)
    except (ImportError, AttributeError):
        try:
            import mediapipe as mp
            mp_selfie = getattr(getattr(mp, "solutions", None), "selfie_segmentation", None)
        except Exception:
            mp_selfie = None

# Global cache to prevent re-reading the background image and re-instantiating MediaPipe on every frame
_GLOBAL_SEGMENTER: Optional[Any] = None
_CACHED_BG_PATH: Optional[str] = None
_CACHED_BG_IMG: Optional[np.ndarray] = None
_WARNED_MISSING_MEDIAPIPE = False


def get_segmenter(model_selection: int = 1) -> Optional[Any]:
    """
    Retrieves or initializes a singleton MediaPipe SelfieSegmentation instance.
    :param model_selection: 0 for general model (256x256), 1 for landscape model (best for webcams).
    """
    global _GLOBAL_SEGMENTER, _WARNED_MISSING_MEDIAPIPE
    if _GLOBAL_SEGMENTER is None:
        if mp_selfie is not None:
            _GLOBAL_SEGMENTER = mp_selfie.SelfieSegmentation(model_selection=model_selection)
        else:
            if not _WARNED_MISSING_MEDIAPIPE:
                print(
                    "[BackgroundSegmenter] Warning: MediaPipe selfie_segmentation solution "
                    "is not available in this environment. Running in pass-through mode."
                )
                _WARNED_MISSING_MEDIAPIPE = True
    return _GLOBAL_SEGMENTER


def overlay_on_background(
    frame: np.ndarray,
    bg_image_path: str,
    threshold: float = 0.5,
    feather_kernel: int = 7,
) -> np.ndarray:
    """
    Extracts the human subject silhouette from a live OpenCV video frame using
    MediaPipe Selfie Segmentation and overlays it onto a static background image.

    :param frame: Input live BGR frame from OpenCV (H, W, 3).
    :param bg_image_path: Path to the static background image on the local filesystem.
    :param threshold: Confidence threshold [0.0 - 1.0] for the segmentation mask.
    :param feather_kernel: Gaussian blur kernel size for alpha mask edge smoothing.
    :return: Composited OpenCV BGR frame (H, W, 3).
    """
    global _CACHED_BG_PATH, _CACHED_BG_IMG

    if frame is None or frame.size == 0:
        return frame

    segmenter = get_segmenter(model_selection=1)
    if segmenter is None:
        return frame

    h, w, _ = frame.shape

    # 1. Load and cache the static background image from the local directory
    if _CACHED_BG_PATH != bg_image_path or _CACHED_BG_IMG is None:
        if not os.path.exists(bg_image_path):
            raise FileNotFoundError(f"Background image not found at '{bg_image_path}'")
        raw_bg = cv2.imread(bg_image_path)
        if raw_bg is None:
            raise ValueError(f"Could not decode image at '{bg_image_path}'")
        _CACHED_BG_IMG = raw_bg
        _CACHED_BG_PATH = bg_image_path

    # Ensure background matches incoming frame dimensions
    if _CACHED_BG_IMG.shape[:2] != (h, w):
        bg_resized = cv2.resize(_CACHED_BG_IMG, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg_resized = _CACHED_BG_IMG

    # 2. Convert BGR to RGB for MediaPipe inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False  # Performance optimization for MediaPipe

    results = segmenter.process(rgb_frame)

    if results is None or results.segmentation_mask is None:
        return frame

    # 3. Extract the alpha mask
    raw_mask = results.segmentation_mask

    # Threshold the probability mask into a binary or soft mask
    binary_mask = (raw_mask > threshold).astype(np.float32)

    # Apply Gaussian feathering to soften edges and eliminate harsh fringes
    if feather_kernel > 1:
        if feather_kernel % 2 == 0:
            feather_kernel += 1  # Kernel size must be odd
        alpha = cv2.GaussianBlur(binary_mask, (feather_kernel, feather_kernel), 0)
    else:
        alpha = binary_mask

    # Expand alpha from (H, W) to (H, W, 1) for 3-channel broadcasting
    alpha_3d = np.expand_dims(alpha, axis=-1)

    # 4. Composite: Foreground * Alpha + Background * (1 - Alpha)
    foreground = frame.astype(np.float32)
    background = bg_resized.astype(np.float32)

    composited = (foreground * alpha_3d) + (background * (1.0 - alpha_3d))
    return np.clip(composited, 0, 255).astype(np.uint8)


class BackgroundSegmenter:
    """
    Object-oriented wrapper for persistent real-time streaming compositing.
    """

    def __init__(
        self,
        bg_image_path: str,
        threshold: float = 0.5,
        feather_kernel: int = 7,
        model_selection: int = 1,
    ):
        self.bg_image_path = bg_image_path
        self.threshold = threshold
        self.feather_kernel = feather_kernel
        self.model_selection = model_selection
        self.segmenter = get_segmenter(model_selection=model_selection)
        self.bg_image: Optional[np.ndarray] = None
        self._load_background(bg_image_path)

    def _load_background(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Background image not found at '{path}'")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image at '{path}'")
        self.bg_image = img

    def process(self, frame: np.ndarray) -> np.ndarray:
        return overlay_on_background(
            frame=frame,
            bg_image_path=self.bg_image_path,
            threshold=self.threshold,
            feather_kernel=self.feather_kernel,
        )


if __name__ == "__main__":
    test_bg_path = "models/background/studio.jpg"
    os.makedirs(os.path.dirname(test_bg_path), exist_ok=True)
    if not os.path.exists(test_bg_path):
        gradient = np.zeros((720, 1280, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.linspace(30, 120, 1280, dtype=np.uint8)
        gradient[:, :, 1] = 60
        gradient[:, :, 2] = np.linspace(150, 40, 720, dtype=np.uint8)[:, None]
        cv2.imwrite(test_bg_path, gradient)
        print(f"Generated sample background image at '{test_bg_path}'.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found. Exiting test.")
    else:
        print("Running segmenter test. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = overlay_on_background(frame, test_bg_path)
            cv2.imshow("Composited Frame", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
