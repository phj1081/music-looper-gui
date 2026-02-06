"""Music Looper Desktop App - PyWebView frontend (legacy mode).

This is a thin wrapper around core.py that provides the PyWebView GUI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _maybe_exec_embedded_python_cli() -> None:
    """Emulate `python -c` / `python -m` in frozen apps.

    On macOS, when this app is packaged as a GUI executable (PyInstaller
    `--windowed`), the frozen bootloader always runs our entry-point script.
    Some libraries (std `multiprocessing`, joblib, etc.) spawn helper processes
    by invoking `sys.executable` with `-c ...` or `-m ...`. If `sys.executable`
    points to our GUI executable, those helper processes would start a second
    GUI window.

    We detect that invocation pattern early (before importing heavy/UI modules)
    and execute the requested snippet/module instead, then exit.
    """
    if not getattr(sys, "frozen", False):
        return

    argv = sys.argv
    if os.getenv("MUSIC_LOOPER_DEBUG_BOOT") == "1":
        try:
            with open("/tmp/music-looper-boot.log", "a", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} argv={argv!r}\n")
        except Exception:
            pass

    # `python -c <code> [args...]`
    try:
        c_index = argv.index("-c")
    except ValueError:
        c_index = -1

    if c_index != -1 and c_index + 1 < len(argv):
        code = argv[c_index + 1]
        if isinstance(code, str):
            sys.argv = ["-c", *argv[c_index + 2 :]]
            globals_dict: dict[str, object] = {"__name__": "__main__"}
            exec(code, globals_dict, globals_dict)
            raise SystemExit(0)

    # `python -m <module> [args...]`
    try:
        m_index = argv.index("-m")
    except ValueError:
        m_index = -1

    if m_index != -1 and m_index + 1 < len(argv):
        module = argv[m_index + 1]
        if isinstance(module, str):
            sys.argv = [module, *argv[m_index + 2 :]]
            import runpy

            runpy.run_module(module, run_name="__main__", alter_sys=True)
            raise SystemExit(0)


_maybe_exec_embedded_python_cli()

import webview
from webview.dom import DOMEventHandler

from core import MusicLooperCore

# Global window reference for API access
window = None


def _is_truthy_env(var_name: str) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Api:
    """PyWebView API wrapper around MusicLooperCore.

    Maintains the same interface as the original for frontend compatibility.
    """

    def __init__(self):
        self._core = MusicLooperCore()

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
        return self._core.analyze(file_path)

    def analyze_async(self, file_path: str) -> dict:
        """Start asynchronous analysis in a background thread."""
        return self._core.analyze_async(file_path)

    def get_progress(self) -> dict:
        """Get the current analysis progress."""
        return self._core.get_progress()

    def get_analysis_result(self) -> dict | None:
        """Get the analysis result if completed."""
        return self._core.get_analysis_result()

    def get_audio_base64(self) -> str | None:
        """Get full audio as base64 WAV for playback."""
        return self._core.get_audio_base64()

    def get_waveform(self, points: int = 1000) -> list[float] | None:
        """Get downsampled waveform data for visualization."""
        return self._core.get_waveform(points)

    def export_loop(self, loop_start: int, loop_end: int) -> bool:
        """Export loop segment - opens save dialog."""
        if self._core._audio is None or self._core._sr is None:
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
            return self._core.export_loop(loop_start, loop_end, output_path)
        except Exception:
            return False

    def export_with_loop_tags(self, loop_start: int, loop_end: int, format: str) -> bool:
        """Export file with loop metadata tags (OGG or WAV with smpl chunk)."""
        if self._core._audio is None or self._core._sr is None:
            return False

        try:
            ext = "ogg" if format == "ogg" else "wav"
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=f"loop_tagged.{ext}",
                file_types=(f"{ext.upper()} Files (*.{ext})",),
            )

            if not result:
                return False

            output_path = result if isinstance(result, str) else result[0]
            return self._core.export_with_loop_tags(loop_start, loop_end, format, output_path)
        except Exception:
            return False

    def export_split_sections(self, loop_start: int, loop_end: int) -> bool:
        """Export intro, loop, and outro as separate WAV files."""
        if self._core._audio is None or self._core._sr is None or self._core._current_file is None:
            return False

        try:
            result = window.create_file_dialog(
                webview.FOLDER_DIALOG,
            )

            if not result:
                return False

            output_dir = result[0] if isinstance(result, (list, tuple)) else result
            return self._core.export_split_sections(loop_start, loop_end, output_dir)
        except Exception:
            return False

    def export_extended(self, loop_start: int, loop_end: int, loop_count: int) -> bool:
        """Export extended version with loop repeated N times."""
        if self._core._audio is None or self._core._sr is None:
            return False

        try:
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename="extended_loop.wav",
                file_types=("WAV Files (*.wav)",),
            )

            if not result:
                return False

            output_path = result if isinstance(result, str) else result[0]
            return self._core.export_extended(loop_start, loop_end, loop_count, output_path)
        except Exception:
            return False

    def export_loop_info(self, loop_start: int, loop_end: int, format: str) -> bool:
        """Export loop point information as JSON or TXT."""
        if self._core._sr is None:
            return False

        try:
            ext = "json" if format == "json" else "txt"
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=f"loop_info.{ext}",
                file_types=(f"{ext.upper()} Files (*.{ext})",),
            )

            if not result:
                return False

            output_path = result if isinstance(result, str) else result[0]
            return self._core.export_loop_info(loop_start, loop_end, format, output_path)
        except Exception:
            return False


def _get_base_path() -> Path:
    """Get base path for resources (PyInstaller compatible)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).parent


def get_html_path() -> str:
    """Get path to the HTML file."""
    if _is_truthy_env("MUSIC_LOOPER_DEV"):
        return os.getenv("MUSIC_LOOPER_DEV_URL", "http://localhost:3000")

    base = _get_base_path()
    static_dir = base / "static"
    if static_dir.exists():
        return str(static_dir / "index.html")
    return os.getenv("MUSIC_LOOPER_DEV_URL", "http://localhost:3000")


def _wait_for_frontend_dev_server(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return

    try:
        wait_seconds = float(os.getenv("MUSIC_LOOPER_DEV_WAIT_SECONDS", "60"))
    except ValueError:
        wait_seconds = 60.0

    deadline = time.monotonic() + wait_seconds
    last_log = 0.0

    while True:
        try:
            req = Request(url, headers={"User-Agent": "music-looper-dev-wait"})
            with urlopen(req, timeout=5):
                return
        except HTTPError:
            return
        except (URLError, TimeoutError) as e:
            if time.monotonic() >= deadline:
                print(f"[music-looper] Frontend dev server not reachable at {url}: {e}", flush=True)
                return

            now = time.monotonic()
            if now - last_log >= 1.0:
                print(f"[music-looper] Waiting for frontend dev server at {url}...", flush=True)
                last_log = now

            time.sleep(0.25)


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
            filename = Path(full_path).name
            payload = json.dumps(
                {"path": full_path, "filename": filename},
                ensure_ascii=False,
            )
            script = f"""
            (() => {{
                const payload = {payload};
                if (typeof window.onFileDropped === "function") {{
                    window.onFileDropped(payload.path, payload.filename);
                    return;
                }}
                if (!Array.isArray(window.__musicLooperPendingDrops)) {{
                    window.__musicLooperPendingDrops = [];
                }}
                window.__musicLooperPendingDrops.push(payload);
                window.dispatchEvent(new CustomEvent("music-looper-file-drop", {{ detail: payload }}));
            }})();
            """
            try:
                window.evaluate_js(script)
            except Exception as e:
                print(f"[music-looper] Failed to deliver dropped file event: {e}", flush=True)


def bind_drag_drop(win):
    """Bind drag and drop events after window is loaded."""
    global window
    window = win

    window.dom.document.events.dragenter += DOMEventHandler(on_drag, True, True)
    window.dom.document.events.dragover += DOMEventHandler(on_drag, True, True)
    window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)


def main():
    """Main entry point."""
    global window

    api = Api()
    html_path = get_html_path()
    if _is_truthy_env("MUSIC_LOOPER_DEV"):
        _wait_for_frontend_dev_server(html_path)
    window = webview.create_window(
        title="Music Looper",
        url=html_path,
        js_api=api,
        width=1000,
        height=750,
        min_size=(800, 600),
    )
    webview.start(bind_drag_drop, window)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
