"""Music Looper Desktop App - Single file, no server."""

import io
import os
import base64
import wave
from pathlib import Path
from enum import Enum

import numpy as np
import librosa
import webview
from webview.dom import DOMEventHandler
from pymusiclooper.core import MusicLooper

from analyzer import SSMLoopAnalyzer

# Global window reference for API access
window = None


class AnalyzerType(str, Enum):
    """Available analyzer types."""
    PYMUSICLOOPER = "pymusiclooper"  # Original: chroma + beat detection
    SSM = "ssm"                       # New: Self-Similarity Matrix


class Api:
    """Python API exposed to JavaScript via PyWebView bridge."""

    def __init__(self):
        self._looper: MusicLooper | None = None
        self._ssm_analyzer: SSMLoopAnalyzer | None = None
        self._current_file: str | None = None
        self._audio: np.ndarray | None = None
        self._sr: int | None = None

    def select_file(self) -> dict | None:
        """Open native file dialog and load the selected audio file."""
        file_types = ("Audio Files (*.mp3;*.wav;*.flac;*.ogg;*.m4a)",)
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types,
        )

        if not result or len(result) == 0:
            return None

        file_path = Path(result[0])
        return {
            "filename": file_path.name,
            "path": str(file_path),
        }

    def analyze(self, file_path: str, method: str = "pymusiclooper") -> dict:
        """Analyze audio file for loop points.

        Args:
            file_path: Path to audio file
            method: "pymusiclooper" (original) or "ssm" (Self-Similarity Matrix)
        """
        self._current_file = file_path

        try:
            if method == AnalyzerType.SSM:
                return self._analyze_ssm(file_path)
            else:
                return self._analyze_pymusiclooper(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_pymusiclooper(self, file_path: str) -> dict:
        """Original PyMusicLooper analysis."""
        self._looper = MusicLooper(file_path)
        loop_pairs = self._looper.find_loop_pairs()

        loops = []
        for i, pair in enumerate(loop_pairs):
            start_time = self._looper.samples_to_ftime(pair.loop_start)
            end_time = self._looper.samples_to_ftime(pair.loop_end)
            duration_samples = pair.loop_end - pair.loop_start
            duration_time = self._looper.samples_to_ftime(duration_samples)

            loops.append({
                "index": i,
                "start_sample": pair.loop_start,
                "end_sample": pair.loop_end,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_time,
                "score": pair.score,
                "method": "pymusiclooper",
            })

        audio = self._looper.mlaudio.playback_audio
        self._sr = self._looper.mlaudio.rate

        # Ensure consistent shape (samples, channels)
        if len(audio.shape) == 1:
            self._audio = audio.reshape(-1, 1)
        else:
            self._audio = audio

        duration = len(self._audio) / self._sr

        return {
            "success": True,
            "duration": duration,
            "sample_rate": self._sr,
            "loops": loops,
            "method": "pymusiclooper",
        }

    def _analyze_ssm(self, file_path: str) -> dict:
        """SSM-based analysis (Self-Similarity Matrix)."""
        self._ssm_analyzer = SSMLoopAnalyzer(file_path)
        candidates = self._ssm_analyzer.find_loop_points()

        # Also load for playback
        self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
        if len(self._audio.shape) == 1:
            self._audio = self._audio.reshape(-1, 1)
        else:
            self._audio = self._audio.T  # (samples, channels)

        loops = []
        for i, cand in enumerate(candidates):
            start_time = self._samples_to_ftime(cand.loop_start, self._ssm_analyzer.sr)
            end_time = self._samples_to_ftime(cand.loop_end, self._ssm_analyzer.sr)
            duration_samples = cand.loop_end - cand.loop_start
            duration_time = self._samples_to_ftime(duration_samples, self._ssm_analyzer.sr)

            loops.append({
                "index": i,
                "start_sample": cand.loop_start,
                "end_sample": cand.loop_end,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_time,
                "score": cand.score,
                "ssm_score": cand.ssm_score,
                "correlation_score": cand.correlation_score,
                "method": "ssm",
            })

        duration = len(self._audio) / self._sr

        return {
            "success": True,
            "duration": duration,
            "sample_rate": self._sr,
            "loops": loops,
            "method": "ssm",
        }

    def _samples_to_ftime(self, samples: int, sr: int) -> str:
        """Convert samples to MM:SS.mmm format."""
        seconds = samples / sr
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"

    def get_available_methods(self) -> list[dict]:
        """Return available analysis methods."""
        return [
            {
                "id": "pymusiclooper",
                "name": "PyMusicLooper",
                "description": "Chroma + Beat Detection + Cosine Similarity (Original)",
            },
            {
                "id": "ssm",
                "name": "SSM (Self-Similarity Matrix)",
                "description": "Advanced: finds repeating structures via similarity matrix",
            },
        ]

    def get_audio_base64(self) -> str | None:
        """Get full audio as base64 WAV for playback."""
        if self._audio is None or self._sr is None:
            return None

        audio = self._audio
        sample_rate = self._sr

        # Handle both (samples,) and (samples, channels) shapes
        if len(audio.shape) == 1:
            n_channels = 1
            audio_data = audio
        else:
            n_channels = audio.shape[1]
            audio_data = audio

        # Normalize if needed
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        audio_int16 = (audio_data * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_waveform(self, points: int = 1000) -> list[float] | None:
        """Get downsampled waveform data for visualization."""
        if self._audio is None:
            return None

        audio = self._audio

        # Convert to mono for visualization
        if len(audio.shape) > 1:
            mono = audio[:, 0]
        else:
            mono = audio

        step = max(1, len(mono) // points)
        downsampled = mono[::step][:points]

        return downsampled.tolist()

    def export_loop(self, loop_start: int, loop_end: int) -> bool:
        """Export loop segment - opens save dialog."""
        if self._audio is None or self._sr is None:
            return False

        try:
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename="loop_export.wav",
                file_types=("WAV Files (*.wav)",),
            )

            if not result:
                return False

            output_path = result if isinstance(result, str) else result[0]

            audio = self._audio
            sample_rate = self._sr

            # Handle both shapes
            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            loop_audio = audio[loop_start:loop_end]

            # Normalize if needed
            max_val = np.max(np.abs(loop_audio))
            if max_val > 1.0:
                loop_audio = loop_audio / max_val

            audio_int16 = (loop_audio * 32767).astype(np.int16)

            with wave.open(output_path, "wb") as wav:
                wav.setnchannels(n_channels)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_int16.tobytes())

            return True
        except Exception:
            return False


def _is_truthy_env(var_name: str) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_html_path() -> str:
    """Get path to the HTML file."""
    if _is_truthy_env("MUSIC_LOOPER_DEV"):
        return os.getenv("MUSIC_LOOPER_DEV_URL", "http://localhost:3000")

    base = Path(__file__).parent
    static_dir = base / "static"
    if static_dir.exists():
        return str(static_dir / "index.html")
    # Development: use frontend dev server
    return os.getenv("MUSIC_LOOPER_DEV_URL", "http://localhost:3000")


def on_drag(e):
    """Handle drag events."""
    pass


def on_drop(e):
    """Handle file drop event."""
    files = e.get("dataTransfer", {}).get("files", [])
    if not files:
        return

    file_info = files[0]
    full_path = file_info.get("pywebviewFullPath")

    if full_path:
        ext = Path(full_path).suffix.lower()
        if ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
            # Notify frontend about the dropped file
            filename = Path(full_path).name
            window.evaluate_js(f'window.onFileDropped("{full_path}", "{filename}")')


def bind_drag_drop(win):
    """Bind drag and drop events after window is loaded."""
    global window
    window = win

    # Bind drag and drop events to the document
    window.dom.document.events.dragenter += DOMEventHandler(on_drag, True, True)
    window.dom.document.events.dragover += DOMEventHandler(on_drag, True, True)
    window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)


def main():
    """Main entry point."""
    global window

    api = Api()
    window = webview.create_window(
        title="Music Looper",
        url=get_html_path(),
        js_api=api,
        width=1000,
        height=750,
        min_size=(800, 600),
    )
    webview.start(bind_drag_drop, window)


if __name__ == "__main__":
    main()
