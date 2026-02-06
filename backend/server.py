"""JSON-RPC stdin/stdout server for Tauri sidecar.

Protocol:
  Request  (stdin):  {"id": 1, "method": "analyze", "params": {"file_path": "/path/to/audio.mp3"}}
  Response (stdout): {"id": 1, "result": {"success": true, "loops": [...]}}
  Event    (stdout): {"event": "progress", "data": {"current": 5, "total": 10, "stage": "extracting"}}
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import runpy


def _maybe_exec_embedded_python_cli() -> None:
    """Emulate `python -c` / `python -m` when running as a frozen executable.

    allin1 internally calls demucs via:
      `sys.executable -m demucs.separate ...`

    In a PyInstaller sidecar, `sys.executable` points to this sidecar binary.
    Without this handler, child processes re-enter this JSON-RPC server and wait
    on stdin forever, causing structure analysis to hang.
    """
    if not getattr(sys, "frozen", False):
        return

    argv = sys.argv

    def _redirect_child_stdout_to_stderr() -> None:
        """Keep sidecar JSON-RPC channel clean (stdout reserved for JSON only)."""
        try:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        except Exception:
            pass

    # Handle `python -c "<code>" ...`
    try:
        c_index = argv.index("-c")
    except ValueError:
        c_index = -1

    if c_index != -1 and c_index + 1 < len(argv):
        code = argv[c_index + 1]
        if isinstance(code, str):
            _redirect_child_stdout_to_stderr()
            sys.argv = ["-c", *argv[c_index + 2 :]]
            globals_dict: dict[str, object] = {"__name__": "__main__"}
            exec(code, globals_dict, globals_dict)
            raise SystemExit(0)

    # Handle `python -m <module> ...`
    try:
        m_index = argv.index("-m")
    except ValueError:
        m_index = -1

    if m_index != -1 and m_index + 1 < len(argv):
        module = argv[m_index + 1]
        if isinstance(module, str):
            _redirect_child_stdout_to_stderr()
            sys.argv = [module, *argv[m_index + 2 :]]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
            raise SystemExit(0)


_maybe_exec_embedded_python_cli()

from core import MusicLooperCore


def _write_json(obj: dict) -> None:
    """Write a JSON object as a single line to stdout and flush."""
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _send_response(request_id: int | str, result) -> None:
    """Send a JSON-RPC response."""
    _write_json({"id": request_id, "result": result})


def _send_error(request_id: int | str | None, error: str) -> None:
    """Send a JSON-RPC error response."""
    msg = {"error": error}
    if request_id is not None:
        msg["id"] = request_id
    _write_json(msg)


def _send_event(event_name: str, data: dict) -> None:
    """Send a push event (no id)."""
    _write_json({"event": event_name, "data": data})


class Server:
    """JSON-RPC server wrapping MusicLooperCore."""

    def __init__(self):
        self.core = MusicLooperCore()
        self.core.set_progress_callback(self._on_progress)

    def _on_progress(self, current: int, total: int, stage: str) -> None:
        """Push progress events to Tauri via stdout."""
        _send_event("progress", {"current": current, "total": total, "stage": stage})

    def handle_request(self, request: dict) -> None:
        """Dispatch a single JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            result = self._dispatch(method, params)
            _send_response(request_id, result)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _send_error(request_id, str(e))

    def _dispatch(self, method: str, params: dict):
        """Route method name to core function."""
        if method == "analyze":
            # Use async analysis with progress push
            file_path = params["file_path"]
            self.core.analyze_async(file_path)
            # The analysis runs in a background thread; progress events are pushed.
            # When complete, the core sets stage="completed".
            # We need to wait for completion and then send the result.
            self.core._analysis_thread.join()

            result = self.core.get_analysis_result()
            if result is None:
                # Check for error
                progress = self.core.get_progress()
                if progress.get("stage") == "error":
                    raise RuntimeError(progress.get("error", "Analysis failed"))
                raise RuntimeError("Analysis completed but no result available")

            # Also send a completion event
            _send_event("complete", result)
            return result

        elif method == "get_waveform":
            points = params.get("points", 1000)
            result = self.core.get_waveform(points)
            if result is None:
                raise RuntimeError("No audio loaded")
            return result

        elif method == "get_audio_file":
            path = self.core.write_audio_to_temp()
            if path is None:
                raise RuntimeError("No audio loaded")
            return path

        elif method == "export_loop":
            return self.core.export_loop(
                int(params["loop_start"]),
                int(params["loop_end"]),
                params["output_path"],
            )

        elif method == "export_with_loop_tags":
            return self.core.export_with_loop_tags(
                int(params["loop_start"]),
                int(params["loop_end"]),
                params["format"],
                params["output_path"],
            )

        elif method == "export_split_sections":
            return self.core.export_split_sections(
                int(params["loop_start"]),
                int(params["loop_end"]),
                params["output_dir"],
            )

        elif method == "export_extended":
            return self.core.export_extended(
                int(params["loop_start"]),
                int(params["loop_end"]),
                int(params["loop_count"]),
                params["output_path"],
            )

        elif method == "export_loop_info":
            return self.core.export_loop_info(
                int(params["loop_start"]),
                int(params["loop_end"]),
                params["format"],
                params["output_path"],
            )

        else:
            raise ValueError(f"Unknown method: {method}")


def main():
    """Main entry point: read JSON-RPC requests from stdin, write responses to stdout."""
    server = Server()

    # Signal ready
    _send_event("ready", {})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _send_error(None, f"Invalid JSON: {e}")
            continue

        server.handle_request(request)


if __name__ == "__main__":
    main()
