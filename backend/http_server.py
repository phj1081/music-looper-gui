"""HTTP server for Tauri sidecar (replaces JSON-RPC stdin/stdout).

Startup:
  1. Binds to a free port on 127.0.0.1
  2. Prints {"port": N} to stdout (Rust reads this once)
  3. Serves FastAPI endpoints matching the old JSON-RPC methods

Progress is streamed via SSE on GET /progress.
"""

from __future__ import annotations

import asyncio
import json
import os
import runpy
import signal
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# ── PyInstaller child-process handler (must run before any imports) ──


def _maybe_exec_embedded_python_cli() -> None:
    """Emulate `python -c` / `python -m` when running as a frozen executable.

    allin1 internally calls demucs via:
      `sys.executable -m demucs.separate ...`

    In a PyInstaller sidecar, `sys.executable` points to this sidecar binary.
    Without this handler, child processes re-enter this server and wait
    forever, causing structure analysis to hang.
    """
    if not getattr(sys, "frozen", False):
        return

    argv = sys.argv

    def _redirect_child_stdout_to_stderr() -> None:
        """Keep sidecar stdout clean (reserved for the port JSON line)."""
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

# ── Imports (after PyInstaller guard) ────────────────────────────────

import socket

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from core import MusicLooperCore

# ── Request models ───────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    file_path: str


class ExportLoopRequest(BaseModel):
    loop_start: int
    loop_end: int
    output_path: str


class ExportLoopTagsRequest(BaseModel):
    loop_start: int
    loop_end: int
    format: str
    output_path: str


class ExportSplitRequest(BaseModel):
    loop_start: int
    loop_end: int
    output_dir: str


class ExportExtendedRequest(BaseModel):
    loop_start: int
    loop_end: int
    loop_count: int
    output_path: str


class ExportInfoRequest(BaseModel):
    loop_start: int
    loop_end: int
    format: str
    output_path: str


# ── Globals ──────────────────────────────────────────────────────────

core = MusicLooperCore()
# asyncio.Queue for SSE progress broadcasting
_progress_queue: asyncio.Queue[dict] | None = None
_event_loop: asyncio.AbstractEventLoop | None = None


def _on_progress(current: int, total: int, stage: str) -> None:
    """Progress callback from core (may be called from a background thread)."""
    data = {"current": current, "total": total, "stage": stage}
    if _progress_queue is not None and _event_loop is not None:
        _event_loop.call_soon_threadsafe(_progress_queue.put_nowait, data)


core.set_progress_callback(_on_progress)

# ── Model preload state ──────────────────────────────────────────────

_preload_status: str = "idle"  # "idle" | "loading" | "ready" | "error"
_preload_error: str | None = None


def _preload_ast_model() -> None:
    """Background worker: pre-load the AST model at startup."""
    global _preload_status, _preload_error
    _preload_status = "loading"
    try:
        core.preload_model()
        _preload_status = "ready"
    except Exception as e:
        _preload_status = "error"
        _preload_error = str(e)


# ── App lifecycle ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _progress_queue, _event_loop
    _event_loop = asyncio.get_running_loop()
    _progress_queue = asyncio.Queue()

    # Start background model preloading
    threading.Thread(target=_preload_ast_model, daemon=True).start()

    yield
    _progress_queue = None
    _event_loop = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ────────────────────────────────────────────────────────


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze audio file for loop points (blocking)."""
    try:
        # Run the synchronous analysis in a thread so the event loop stays free
        # and progress callbacks can be forwarded via SSE concurrently.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _analyze_sync, req.file_path)
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


def _analyze_sync(file_path: str) -> dict:
    """Run analysis in background thread (called via run_in_executor)."""
    core.analyze_async(file_path)
    core._analysis_thread.join()

    result = core.get_analysis_result()
    if result is None:
        progress = core.get_progress()
        if progress.get("stage") == "error":
            raise RuntimeError(progress.get("error", "Analysis failed"))
        raise RuntimeError("Analysis completed but no result available")

    return result


@app.get("/waveform")
async def get_waveform(points: int = Query(default=1000)):
    """Get downsampled waveform data."""
    result = core.get_waveform(points)
    if result is None:
        return JSONResponse(content={"error": "No audio loaded"}, status_code=400)
    return JSONResponse(content=result)


@app.get("/audio")
async def get_audio():
    """Serve the loaded audio as a WAV file."""
    path = core.write_audio_to_temp()
    if path is None:
        return JSONResponse(content={"error": "No audio loaded"}, status_code=400)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="audio.wav",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/export/loop")
async def export_loop(req: ExportLoopRequest):
    result = core.export_loop(req.loop_start, req.loop_end, req.output_path)
    return JSONResponse(content=result)


@app.post("/export/loop-tags")
async def export_loop_tags(req: ExportLoopTagsRequest):
    result = core.export_with_loop_tags(req.loop_start, req.loop_end, req.format, req.output_path)
    return JSONResponse(content=result)


@app.post("/export/split")
async def export_split(req: ExportSplitRequest):
    result = core.export_split_sections(req.loop_start, req.loop_end, req.output_dir)
    return JSONResponse(content=result)


@app.post("/export/extended")
async def export_extended(req: ExportExtendedRequest):
    result = core.export_extended(req.loop_start, req.loop_end, req.loop_count, req.output_path)
    return JSONResponse(content=result)


@app.post("/export/info")
async def export_info(req: ExportInfoRequest):
    result = core.export_loop_info(req.loop_start, req.loop_end, req.format, req.output_path)
    return JSONResponse(content=result)


@app.get("/progress")
async def progress_stream(request: Request):
    """SSE endpoint for analysis progress events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        if _progress_queue is None:
            return
        while True:
            if await request.is_disconnected():
                break
            try:
                data = await asyncio.wait_for(_progress_queue.get(), timeout=30.0)
                yield f"data: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive comment to prevent connection timeout
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/preload-status")
async def preload_status():
    """Return the current AST model preload status."""
    return JSONResponse(content={"status": _preload_status, "error": _preload_error})


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown endpoint (defensive fallback)."""
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse(content={"status": "shutting_down"})


# ── Main ─────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = _find_free_port()

    # Signal port to Rust (must be the first and only stdout line)
    sys.stdout.write(json.dumps({"port": port}) + "\n")
    sys.stdout.flush()

    # Redirect further stdout to stderr so library prints don't corrupt the channel
    sys.stdout = sys.stderr

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
