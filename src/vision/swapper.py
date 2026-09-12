import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, List


class FaceSwapper:
    """
    Inference engine for loading ONNX facial landmark modification/image processing
    models using ONNX Runtime GPU with fallback to CPU.
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        device_id: int = 0,
        input_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initializes the ONNX runtime session with GPU/CUDA execution provider logic.

        :param model_path: Path to the .onnx model file.
        :param use_gpu: Whether to prioritize GPU (CUDAExecutionProvider).
        :param device_id: CUDA device ID (default 0).
        :param input_size: (width, height) expected by the ONNX model.
        """
        self.model_path = model_path
        self.input_size = input_size
        self.device_id = device_id

        self.session = self._create_inference_session(use_gpu)

        # Inspect model input/output signatures
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def _create_inference_session(self, use_gpu: bool) -> ort.InferenceSession:
        """
        Configures execution providers prioritizing CUDA with graceful fallback to CPU.
        """
        providers: List[object] = []
        available_providers = ort.get_available_providers()

        if use_gpu and "CUDAExecutionProvider" in available_providers:
            cuda_provider_options = {
                "device_id": self.device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB limit
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            providers.append(("CUDAExecutionProvider", cuda_provider_options))

        # Always append CPU as fallback
        providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers,
        )

        active_providers = session.get_providers()
        print(f"[FaceSwapper] Active ONNX Execution Providers: {active_providers}")
        return session

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Prepares an OpenCV BGR frame for the ONNX model:
        1. Resizes to model input dimensions.
        2. Converts BGR to RGB.
        3. Normalizes pixel values to [0.0, 1.0].
        4. Transposes from HWC to NCHW batch format.
        """
        orig_h, orig_w = frame.shape[:2]

        # Resize to network target dimensions
        resized = cv2.resize(
            frame, self.input_size, interpolation=cv2.INTER_LINEAR
        )

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalization (float32, [0, 1])
        tensor = rgb.astype(np.float32) / 255.0

        # HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
        tensor = np.transpose(tensor, (2, 0, 1))

        # Add batch dimension -> NCHW: (1, 3, H, W)
        tensor = np.expand_dims(tensor, axis=0)

        return np.ascontiguousarray(tensor, dtype=np.float32), (orig_w, orig_h)

    def postprocess(
        self, output_tensor: np.ndarray, original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Converts the raw model output tensor back into a displayable OpenCV BGR image:
        1. Squeezes batch dimension.
        2. Transposes NCHW -> HWC.
        3. Denormalizes to uint8 [0, 255].
        4. Converts RGB -> BGR.
        5. Resizes back to the original webcam frame resolution.
        """
        # Squeeze batch dimension if present: (1, 3, H, W) -> (3, H, W)
        if output_tensor.ndim == 4:
            tensor = output_tensor[0]
        else:
            tensor = output_tensor

        # CHW -> HWC
        img = np.transpose(tensor, (1, 2, 0))

        # Denormalize & clip [0.0, 1.0] -> [0, 255]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # RGB -> BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Scale back to webcam frame size
        orig_w, orig_h = original_size
        return cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        End-to-end inference pass: Preprocess -> ONNX Run -> Postprocess.

        :param frame: Live BGR frame from cv2.VideoCapture
        :return: Processed BGR frame
        """
        if frame is None or frame.size == 0:
            return frame

        input_tensor, orig_dims = self.preprocess(frame)

        # Run ONNX inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        return self.postprocess(outputs[0], orig_dims)


if __name__ == "__main__":
    import os

    # Quick test harness
    model_file = "models/face/inswapper_128.onnx"
    if os.path.exists(model_file):
        swapper = FaceSwapper(model_path=model_file, use_gpu=True)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = swapper.process_frame(dummy_frame)
        print(f"Processed frame shape: {processed.shape}")
    else:
        print(f"To run standalone test, place an ONNX model at '{model_file}'.")
