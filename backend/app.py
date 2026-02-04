"""Music Looper Desktop App - Single file, no server."""

import io
import os
import base64
import wave
from pathlib import Path

import numpy as np
import webview
from webview.dom import DOMEventHandler
from pymusiclooper.core import MusicLooper

# Global window reference for API access
window = None


class Api:
    """Python API exposed to JavaScript via PyWebView bridge."""

    def __init__(self):
        self._looper: MusicLooper | None = None

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

    def analyze(self, file_path: str) -> dict:
        """Analyze audio file for loop points."""
        try:
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
                })

            audio = self._looper.mlaudio.playback_audio
            duration = len(audio) / self._looper.mlaudio.rate

            return {
                "success": True,
                "duration": duration,
                "sample_rate": self._looper.mlaudio.rate,
                "loops": loops,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_audio_base64(self) -> str | None:
        """Get full audio as base64 WAV for playback."""
        if not self._looper:
            return None

        audio = self._looper.mlaudio.playback_audio
        sample_rate = self._looper.mlaudio.rate
        n_channels = self._looper.mlaudio.n_channels

        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_waveform(self, points: int = 1000) -> list[float] | None:
        """Get downsampled waveform data for visualization."""
        if not self._looper:
            return None

        audio = self._looper.mlaudio.playback_audio

        if len(audio.shape) > 1:
            mono = audio[:, 0]
        else:
            mono = audio

        step = max(1, len(mono) // points)
        downsampled = mono[::step][:points]

        return downsampled.tolist()

    def export_loop(self, loop_start: int, loop_end: int) -> bool:
        """Export loop segment - opens save dialog."""
        if not self._looper:
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

            audio = self._looper.mlaudio.playback_audio
            sample_rate = self._looper.mlaudio.rate
            n_channels = self._looper.mlaudio.n_channels

            loop_audio = audio[loop_start:loop_end]
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
