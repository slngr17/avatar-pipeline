import time
import threading
import numpy as np
import pyaudio
from typing import Optional, List, Dict, Any


class RVCStreamer:
    """
    Low-latency audio streaming engine capturing live microphone input,
    applying a mathematical pitch-shift transformation array in NumPy,
    and streaming to a Virtual Audio Cable sink via a non-blocking background thread.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 256,
        semitones: float = -3.0,
        input_device_index: Optional[int] = None,
        output_device_name: str = "CABLE Input",
    ):
        """
        :param sample_rate: Audio sampling frequency (Hz).
        :param chunk_size: Buffer size in frames (128 or 256 for minimal latency).
        :param semitones: Pitch shifting shift amount in semitones (+ for higher, - for lower).
        :param input_device_index: Specific mic index, or None for system default.
        :param output_device_name: Substring to match the Virtual Cable sink device.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.semitones = semitones
        self.input_device_index = input_device_index
        self.output_device_name = output_device_name

        self.p = pyaudio.PyAudio()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Precompute pitch shift factor: factor = 2^(semitones / 12)
        self.pitch_factor = 2.0 ** (self.semitones / 12.0)

        # Locate Virtual Audio Cable output sink
        self.output_device_index = self._find_output_device(self.output_device_name)

    def _find_output_device(self, name_substr: str) -> Optional[int]:
        """
        Scans all available audio output devices to locate the Virtual Audio Cable sink.
        """
        device_count = self.p.get_device_count()
        matched_index = None

        for i in range(device_count):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info.get("maxOutputChannels", 0) > 0:
                name = dev_info.get("name", "")
                if name_substr.lower() in name.lower():
                    matched_index = i
                    print(f"[RVCStreamer] Found matching output sink '{name}' at index {i}")
                    break

        if matched_index is None:
            default_out = self.p.get_default_output_device_info()
            matched_index = default_out.get("index")
            print(
                f"[RVCStreamer] Warning: Sink '{name_substr}' not found. "
                f"Falling back to default output: '{default_out.get('name')}' (index {matched_index})"
            )

        return matched_index

    def pitch_shift(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Mathematical pitch-shifting transformation using time-domain resampling
        and linear interpolation over the NumPy sample array.

        Pitch Shift Factor: s = 2^(n / 12)
        - Higher pitch (s > 1.0): Compresses wavelength, shifts spectral frequencies up.
        - Lower pitch (s < 1.0): Expands wavelength, shifts spectral frequencies down.
        """
        if self.pitch_factor == 1.0 or len(audio_chunk) == 0:
            return audio_chunk

        orig_len = len(audio_chunk)
        # Generate resampled time indices based on the pitch factor
        time_orig = np.arange(orig_len)
        time_resampled = np.linspace(0, orig_len - 1, int(orig_len / self.pitch_factor))

        # Mathematical interpolation for fractional phase indices
        stretched = np.interp(time_resampled, time_orig, audio_chunk)

        # Conform back to chunk_size to maintain continuous audio streaming
        time_final = np.linspace(0, len(stretched) - 1, orig_len)
        shifted = np.interp(time_final, np.arange(len(stretched)), stretched)

        return shifted.astype(np.float32)

    def _stream_loop(self):
        """
        Continuous non-blocking audio capture, mathematical transformation,
        and sink transmission loop.
        """
        try:
            stream_in = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
            )

            stream_out = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size,
            )

            print(
                f"[RVCStreamer] Audio stream started (Chunk: {self.chunk_size} samples, "
                f"Pitch shift: {self.semitones:+.1f} semitones)"
            )

            while not self.stop_event.is_set():
                # Read raw low-latency audio chunk from mic
                raw_data = stream_in.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(raw_data, dtype=np.float32)

                # Apply pitch shifting array transformation
                transformed_np = self.pitch_shift(audio_np)

                # Output to Virtual Audio Cable sink
                stream_out.write(transformed_np.tobytes())

        except Exception as e:
            print(f"[RVCStreamer] Error in audio loop: {e}")
        finally:
            if "stream_in" in locals() and stream_in.is_active():
                stream_in.stop_stream()
                stream_in.close()
            if "stream_out" in locals() and stream_out.is_active():
                stream_out.stop_stream()
                stream_out.close()
            print("[RVCStreamer] Audio streams closed.")

    def start(self):
        """Starts the audio pipeline inside a non-blocking background thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            print("[RVCStreamer] Stream is already running.")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Signals the background thread to terminate and cleans up resources."""
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        self.p.terminate()
        print("[RVCStreamer] PyAudio terminated.")


if __name__ == "__main__":
    streamer = RVCStreamer(chunk_size=256, semitones=3.0)
    streamer.start()
    print("Streaming live pitch-shifted audio... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping audio engine...")
        streamer.stop()
