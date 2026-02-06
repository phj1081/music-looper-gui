"""MusicLooperCore - Headless business logic for audio loop detection.

Extracted from app.py's Api class. No PyWebView or GUI dependencies.
"""

from __future__ import annotations

import collections
import collections.abc
import io
import json
import os
import struct
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
from mutagen.oggvorbis import OggVorbis

from deep_analyzer import DeepLoopAnalyzer, BeatAlignedLoopAnalyzer
from allin1_enhancer import Allin1Enhancer

# Compatibility shim: some optional deps (e.g., madmom) still import legacy
# classes from `collections`, which were removed in Python 3.11.
for _name in ("MutableSequence", "MutableMapping", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# Compatibility shim: older audio libs (e.g., madmom) still use deprecated NumPy
# aliases removed in NumPy 2.x.
for _name, _alias in (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("complex", complex),
    ("object", object),
):
    if _name not in np.__dict__:
        setattr(np, _name, _alias)


def _is_truthy_env(var_name: str) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class MusicLooperCore:
    """Headless audio loop detection and export engine."""

    def __init__(self):
        self._deep_analyzer: DeepLoopAnalyzer | None = None
        self._current_file: str | None = None
        self._audio: np.ndarray | None = None
        self._sr: int | None = None
        # Async analysis state
        self._analysis_progress = {"current": 0, "total": 0, "stage": "idle"}
        self._analysis_result: dict | None = None
        self._analysis_thread: threading.Thread | None = None
        # Enhancement flags (auto-detected)
        self._use_beat_alignment: bool = False
        self._use_allin1: bool = False
        self._allin1_enhancer: Allin1Enhancer | None = None
        # Cached mono + downsampled audio for faster coarse seam refinement
        self._mono_audio: np.ndarray | None = None
        self._mono_ds_audio: np.ndarray | None = None
        self._mono_ds_sr: int | None = None
        # Progress callback (can be overridden by server for push events)
        self._progress_callback_fn: Callable[[int, int, str], None] | None = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None] | None):
        """Set an external progress callback for push-based progress reporting."""
        self._progress_callback_fn = callback

    @staticmethod
    def _is_madmom_available() -> bool:
        """Check if madmom is available for beat alignment."""
        try:
            import madmom
            return True
        except Exception:
            return False

    def analyze(self, file_path: str) -> dict:
        """Analyze audio file for loop points (synchronous).

        Auto-detects and uses available enhancements:
        - madmom: beat alignment (if installed)
        - allin1: structure analysis (if installed)
        """
        self._current_file = file_path
        self._use_beat_alignment = self._is_madmom_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_BEAT_ALIGNMENT"
        )
        self._use_allin1 = Allin1Enhancer.is_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_STRUCTURE"
        )
        self._deep_analyzer = None
        self._allin1_enhancer = None

        try:
            self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
            if len(self._audio.shape) == 1:
                self._audio = self._audio.reshape(-1, 1)
            else:
                self._audio = self._audio.T
            self._mono_audio = self._audio[:, 0] if self._audio is not None else None
            self._mono_ds_audio = None
            self._mono_ds_sr = None

            return self._analyze_ast(file_path, self._use_beat_alignment, self._use_allin1)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_async(self, file_path: str) -> dict:
        """Start asynchronous analysis in a background thread."""
        self._current_file = file_path
        self._use_beat_alignment = self._is_madmom_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_BEAT_ALIGNMENT"
        )
        self._use_allin1 = Allin1Enhancer.is_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_STRUCTURE"
        )
        self._deep_analyzer = None
        self._allin1_enhancer = None
        self._analysis_progress = {"current": 0, "total": 0, "stage": "starting"}
        self._analysis_result = None

        self._analysis_thread = threading.Thread(
            target=self._analyze_with_progress,
            args=(file_path, self._use_beat_alignment, self._use_allin1),
            daemon=True,
        )
        self._analysis_thread.start()

        return {"started": True}

    def get_progress(self) -> dict:
        """Get the current analysis progress."""
        return self._analysis_progress

    def get_analysis_result(self) -> dict | None:
        """Get the analysis result if completed."""
        if self._analysis_progress.get("stage") == "completed":
            return self._analysis_result
        return None

    def get_waveform(self, points: int = 1000) -> list[float] | None:
        """Get downsampled waveform data for visualization."""
        if self._audio is None:
            return None

        audio = self._audio
        if len(audio.shape) > 1:
            mono = audio[:, 0]
        else:
            mono = audio

        step = max(1, len(mono) // points)
        downsampled = mono[::step][:points]
        return downsampled.tolist()

    def write_audio_to_temp(self) -> str | None:
        """Write full audio as WAV to a temp file and return the path.

        This replaces the base64 approach for Tauri (convertFileSrc).
        """
        if self._audio is None or self._sr is None:
            return None

        audio = self._audio
        sample_rate = self._sr

        if len(audio.shape) == 1:
            n_channels = 1
            audio_data = audio
        else:
            n_channels = audio.shape[1]
            audio_data = audio

        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        audio_int16 = (audio_data * 32767).astype(np.int16)

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with wave.open(temp_path, "wb") as wav:
                wav.setnchannels(n_channels)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_int16.tobytes())
        except Exception:
            os.close(fd)
            raise
        else:
            os.close(fd)

        return temp_path

    def get_audio_base64(self) -> str | None:
        """Get full audio as base64 WAV for playback (legacy PyWebView mode)."""
        import base64

        if self._audio is None or self._sr is None:
            return None

        audio = self._audio
        sample_rate = self._sr

        if len(audio.shape) == 1:
            n_channels = 1
            audio_data = audio
        else:
            n_channels = audio.shape[1]
            audio_data = audio

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

    def export_loop(self, loop_start: int, loop_end: int, output_path: str) -> bool:
        """Export loop segment as WAV."""
        if self._audio is None or self._sr is None:
            return False

        try:
            audio = self._audio
            sample_rate = self._sr

            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            loop_audio = audio[loop_start:loop_end]

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

    def export_with_loop_tags(self, loop_start: int, loop_end: int, format: str, output_path: str) -> bool:
        """Export file with loop metadata tags (OGG or WAV with smpl chunk)."""
        if self._audio is None or self._sr is None:
            return False

        try:
            if format == "ogg":
                self._write_ogg_with_loop_tags(self._audio, self._sr, loop_start, loop_end, output_path)
            else:
                self._write_wav_with_smpl(self._audio, self._sr, loop_start, loop_end, output_path)
            return True
        except Exception:
            return False

    def export_split_sections(self, loop_start: int, loop_end: int, output_dir: str) -> bool:
        """Export intro, loop, and outro as separate WAV files."""
        if self._audio is None or self._sr is None or self._current_file is None:
            return False

        try:
            output_dir_path = Path(output_dir)
            base_name = Path(self._current_file).stem

            audio = self._audio
            sr = self._sr

            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            intro = audio[:loop_start]
            loop = audio[loop_start:loop_end]
            outro = audio[loop_end:]

            sections = [
                (intro, f"{base_name}_intro.wav"),
                (loop, f"{base_name}_loop.wav"),
                (outro, f"{base_name}_outro.wav"),
            ]

            for section_audio, filename in sections:
                if len(section_audio) == 0:
                    continue

                out_path = output_dir_path / filename

                max_val = np.max(np.abs(section_audio))
                if max_val > 0:
                    if max_val > 1.0:
                        section_audio = section_audio / max_val

                audio_int16 = (section_audio * 32767).astype(np.int16)

                with wave.open(str(out_path), "wb") as wav:
                    wav.setnchannels(n_channels)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    wav.writeframes(audio_int16.tobytes())

            return True
        except Exception:
            return False

    def export_extended(self, loop_start: int, loop_end: int, loop_count: int, output_path: str) -> bool:
        """Export extended version with loop repeated N times."""
        if self._audio is None or self._sr is None:
            return False

        try:
            audio = self._audio
            sr = self._sr

            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            intro = audio[:loop_start]
            loop = audio[loop_start:loop_end]
            outro = audio[loop_end:]

            crossfade_ms = 80.0
            fade_samples = int((crossfade_ms / 1000.0) * sr)

            repeated = self._repeat_with_crossfade(loop, loop_count, fade_samples)
            extended = np.concatenate([intro, repeated, outro], axis=0)

            max_val = np.max(np.abs(extended))
            if max_val > 1.0:
                extended = extended / max_val

            audio_int16 = (extended * 32767).astype(np.int16)

            with wave.open(output_path, "wb") as wav:
                wav.setnchannels(n_channels)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(audio_int16.tobytes())

            return True
        except Exception:
            return False

    def export_loop_info(self, loop_start: int, loop_end: int, format: str, output_path: str) -> bool:
        """Export loop point information as JSON or TXT."""
        if self._sr is None:
            return False

        try:
            sr = self._sr
            start_seconds = loop_start / sr
            end_seconds = loop_end / sr
            duration_seconds = end_seconds - start_seconds

            start_time = self._samples_to_ftime(loop_start, sr)
            end_time = self._samples_to_ftime(loop_end, sr)
            duration_time = self._samples_to_ftime(loop_end - loop_start, sr)

            if format == "json":
                info = {
                    "sample_rate": sr,
                    "loop_start": {
                        "sample": loop_start,
                        "seconds": round(start_seconds, 6),
                        "time": start_time,
                    },
                    "loop_end": {
                        "sample": loop_end,
                        "seconds": round(end_seconds, 6),
                        "time": end_time,
                    },
                    "loop_duration": {
                        "samples": loop_end - loop_start,
                        "seconds": round(duration_seconds, 6),
                        "time": duration_time,
                    },
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
            else:
                lines = [
                    "=== Loop Point Information ===",
                    f"Sample Rate: {sr} Hz",
                    "",
                    "Loop Start:",
                    f"  Sample: {loop_start}",
                    f"  Seconds: {start_seconds:.6f}",
                    f"  Time: {start_time}",
                    "",
                    "Loop End:",
                    f"  Sample: {loop_end}",
                    f"  Seconds: {end_seconds:.6f}",
                    f"  Time: {end_time}",
                    "",
                    "Loop Duration:",
                    f"  Samples: {loop_end - loop_start}",
                    f"  Seconds: {duration_seconds:.6f}",
                    f"  Time: {duration_time}",
                ]
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            return True
        except Exception:
            return False

    # ── Internal methods ──────────────────────────────────────────────

    def _progress_callback(self, current: int, total: int, stage: str):
        """Callback for analysis progress updates."""
        self._analysis_progress = {
            "current": current,
            "total": total,
            "stage": stage,
        }
        if self._progress_callback_fn:
            self._progress_callback_fn(current, total, stage)

    def _analyze_with_progress(self, file_path: str, use_beat_alignment: bool = False, use_allin1: bool = False):
        """Run analysis with progress tracking in background thread."""
        try:
            self._analysis_progress = {"current": 0, "total": 0, "stage": "loading_model"}
            self._analysis_result = None
            if self._progress_callback_fn:
                self._progress_callback_fn(0, 0, "loading_model")

            self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
            if len(self._audio.shape) == 1:
                self._audio = self._audio.reshape(-1, 1)
            else:
                self._audio = self._audio.T
            self._mono_audio = self._audio[:, 0] if self._audio is not None else None
            self._mono_ds_audio = None
            self._mono_ds_sr = None

            self._analysis_result = self._analyze_ast_with_progress(file_path, use_beat_alignment, use_allin1)

            self._analysis_progress = {"current": 1, "total": 1, "stage": "completed"}
            if self._progress_callback_fn:
                self._progress_callback_fn(1, 1, "completed")
        except Exception as e:
            self._analysis_result = {"success": False, "error": str(e)}
            self._analysis_progress = {"current": 0, "total": 0, "stage": "error", "error": str(e)}
            if self._progress_callback_fn:
                self._progress_callback_fn(0, 0, "error")

    def _analyze_ast(self, file_path: str, use_beat_alignment: bool = False, use_allin1: bool = False) -> dict:
        """Deep learning-based analysis (Audio Spectrogram Transformer)."""
        if use_beat_alignment:
            self._deep_analyzer = BeatAlignedLoopAnalyzer(file_path, use_beat_alignment=True)
        else:
            self._deep_analyzer = DeepLoopAnalyzer(file_path)

        candidates = []
        for use_recurrence in (True, False):
            try:
                candidates.extend(
                    self._deep_analyzer.find_loop_points(
                        use_librosa_recurrence=use_recurrence,
                        min_loop_fraction=0.0,
                        n_candidates=30,
                    )
                )
            except Exception:
                continue

        scale_factor = self._sr / self._deep_analyzer.sr

        for cand in candidates:
            cand.loop_start = int(cand.loop_start * scale_factor)
            cand.loop_end = int(cand.loop_end * scale_factor)

        candidates.sort(key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)
        deduped = []
        tol = int(self._sr * 0.35)
        for cand in candidates:
            if any(
                abs(int(cand.loop_start) - int(prev.loop_start)) <= tol
                and abs(int(cand.loop_end) - int(prev.loop_end)) <= tol
                for prev in deduped
            ):
                continue
            deduped.append(cand)
        candidates = deduped

        if use_allin1:
            candidates = self._apply_allin1_enhancement(file_path, candidates)

        try:
            mono_audio = self._mono_audio
            if mono_audio is not None:
                coarse_audio, coarse_sr = self._get_downsampled_mono(target_sr=2000)

                coarse_search_ms = 1200.0 if use_beat_alignment else 2000.0
                coarse_window_ms = 180.0 if use_beat_alignment else 220.0
                fine_search_ms = 22.0 if use_beat_alignment else 35.0
                seam_zero_ms = 6.0 if use_beat_alignment else 8.0

                for cand in candidates:
                    refined_start, refined_end, seam_corr = self._refine_seam_two_stage_original_sr(
                        mono_audio, self._sr, cand.loop_start, cand.loop_end,
                        coarse_audio=coarse_audio, coarse_sr=coarse_sr,
                        coarse_search_window_ms=coarse_search_ms,
                        coarse_window_ms=coarse_window_ms,
                        fine_search_window_ms=fine_search_ms,
                        fine_window_ms=50.0, zero_cross_ms=seam_zero_ms,
                    )
                    cand.loop_start = refined_start
                    cand.loop_end = refined_end

                    base_score = cand.similarity_score * 0.7 + max(0.0, seam_corr) * 0.3
                    cand.score = min(1.0, base_score + float(getattr(cand, "structure_boost", 0.0) or 0.0))

                candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            pass

        beat_effective = self._compute_beat_effective(candidates, use_beat_alignment)
        structure_effective = self._compute_structure_effective(candidates, use_allin1)
        return self._build_result(
            candidates,
            use_allin1,
            beat_effective=beat_effective,
            structure_effective=structure_effective,
        )

    def _analyze_ast_with_progress(self, file_path: str, use_beat_alignment: bool = False, use_allin1: bool = False) -> dict:
        """AST analysis with progress tracking."""
        if use_beat_alignment:
            self._deep_analyzer = BeatAlignedLoopAnalyzer(file_path, use_beat_alignment=True)
        else:
            self._deep_analyzer = DeepLoopAnalyzer(file_path)

        candidates = []
        for use_recurrence in (True, False):
            try:
                candidates.extend(
                    self._deep_analyzer.find_loop_points(
                        use_librosa_recurrence=use_recurrence,
                        min_loop_fraction=0.0,
                        n_candidates=30,
                        progress_callback=self._progress_callback,
                    )
                )
            except Exception:
                continue

        scale_factor = self._sr / self._deep_analyzer.sr

        for cand in candidates:
            cand.loop_start = int(cand.loop_start * scale_factor)
            cand.loop_end = int(cand.loop_end * scale_factor)

        candidates.sort(key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)
        deduped = []
        tol = int(self._sr * 0.35)
        for cand in candidates:
            if any(
                abs(int(cand.loop_start) - int(prev.loop_start)) <= tol
                and abs(int(cand.loop_end) - int(prev.loop_end)) <= tol
                for prev in deduped
            ):
                continue
            deduped.append(cand)
        candidates = deduped

        if use_allin1:
            candidates = self._apply_allin1_enhancement(
                file_path,
                candidates,
                progress_callback=self._progress_callback,
            )

        try:
            mono_audio = self._mono_audio
            if mono_audio is not None:
                coarse_audio, coarse_sr = self._get_downsampled_mono(target_sr=2000)

                coarse_search_ms = 1200.0 if use_beat_alignment else 2000.0
                coarse_window_ms = 180.0 if use_beat_alignment else 220.0
                fine_search_ms = 22.0 if use_beat_alignment else 35.0
                seam_zero_ms = 6.0 if use_beat_alignment else 8.0

                self._progress_callback(0, len(candidates) or 1, "refining_seam")
                for idx, cand in enumerate(candidates):
                    refined_start, refined_end, seam_corr = self._refine_seam_two_stage_original_sr(
                        mono_audio, self._sr, cand.loop_start, cand.loop_end,
                        coarse_audio=coarse_audio, coarse_sr=coarse_sr,
                        coarse_search_window_ms=coarse_search_ms,
                        coarse_window_ms=coarse_window_ms,
                        fine_search_window_ms=fine_search_ms,
                        fine_window_ms=50.0, zero_cross_ms=seam_zero_ms,
                    )
                    cand.loop_start = refined_start
                    cand.loop_end = refined_end

                    base_score = cand.similarity_score * 0.7 + max(0.0, seam_corr) * 0.3
                    cand.score = min(1.0, base_score + float(getattr(cand, "structure_boost", 0.0) or 0.0))

                    self._progress_callback(idx + 1, len(candidates) or 1, "refining_seam")

                candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            pass

        beat_effective = self._compute_beat_effective(candidates, use_beat_alignment)
        structure_effective = self._compute_structure_effective(candidates, use_allin1)
        return self._build_result(
            candidates,
            use_allin1,
            beat_effective=beat_effective,
            structure_effective=structure_effective,
        )

    def _compute_beat_effective(self, candidates, use_beat_alignment: bool) -> bool:
        """Determine whether beat alignment was actually applied."""
        if not use_beat_alignment:
            return False

        if any("beat" in str(getattr(cand, "algorithm", "")).lower() for cand in candidates):
            return True

        beat_analyzer = getattr(getattr(self._deep_analyzer, "_beat_analyzer", None), "_downbeats", None)
        if beat_analyzer is None:
            return False
        try:
            return len(beat_analyzer) > 0
        except Exception:
            return True

    def _compute_structure_effective(self, candidates, use_allin1: bool) -> bool:
        """Determine whether structure analysis produced usable structure info."""
        if not use_allin1:
            return False

        candidate_marked = any(
            bool(getattr(cand, "start_segment", None))
            or bool(getattr(cand, "end_segment", None))
            or float(getattr(cand, "structure_boost", 0.0) or 0.0) > 0.0
            for cand in candidates
        )
        if candidate_marked:
            return True

        if self._allin1_enhancer is None:
            return False

        try:
            structure_info = self._allin1_enhancer.get_structure_info()
        except Exception:
            return False

        if structure_info is None:
            return False

        return (
            bool(getattr(structure_info, "downbeats", None))
            or bool(getattr(structure_info, "segments", None))
            or float(getattr(structure_info, "bpm", 0.0) or 0.0) > 0.0
        )

    def _build_result(
        self,
        candidates,
        use_allin1: bool,
        *,
        beat_effective: bool = False,
        structure_effective: bool = False,
    ) -> dict:
        """Build the analysis result dict from candidates."""
        loops = []
        for i, cand in enumerate(candidates):
            loop_data = {
                "index": i,
                "start_sample": int(cand.loop_start),
                "end_sample": int(cand.loop_end),
                "start_time": self._samples_to_ftime(cand.loop_start, self._sr),
                "end_time": self._samples_to_ftime(cand.loop_end, self._sr),
                "duration": self._samples_to_ftime(cand.loop_end - cand.loop_start, self._sr),
                "score": float(cand.score),
                "similarity_score": float(cand.similarity_score),
            }
            if use_allin1:
                loop_data["start_segment"] = cand.start_segment
                loop_data["end_segment"] = cand.end_segment
                loop_data["is_downbeat_aligned"] = cand.is_downbeat_aligned
                loop_data["structure_boost"] = float(cand.structure_boost)
            loops.append(loop_data)

        duration = len(self._audio) / self._sr

        return {
            "success": True,
            "duration": duration,
            "sample_rate": self._sr,
            "enhancements": {
                "beat_alignment": {"enabled": bool(self._use_beat_alignment), "effective": bool(beat_effective)},
                "structure": {"enabled": bool(use_allin1), "effective": bool(structure_effective)},
                "seam_refinement": "two_stage",
            },
            "loops": loops,
        }

    def _apply_allin1_enhancement(self, file_path: str, candidates, progress_callback=None):
        """Apply allin1 structure analysis enhancement to candidates."""
        if not Allin1Enhancer.is_available():
            return candidates

        try:
            self._allin1_enhancer = Allin1Enhancer(file_path)
            self._allin1_enhancer.analyze_structure(progress_callback=progress_callback)
            return self._allin1_enhancer.enhance_candidates(
                candidates,
                sr=self._sr,
                snap_to_downbeats=not self._use_beat_alignment,
                downbeat_filter=False,
                same_segment_boost=0.15,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"allin1 enhancement failed: {e}")
            return candidates

    def _samples_to_ftime(self, samples: int, sr: int) -> str:
        """Convert samples to MM:SS.mmm format."""
        seconds = samples / sr
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"

    def _find_zero_crossing(self, mono_audio: np.ndarray, sample_idx: int, window: int) -> int:
        """Find nearest zero crossing to sample index in mono audio."""
        if mono_audio.size == 0:
            return sample_idx

        sample_idx = int(np.clip(sample_idx, 0, mono_audio.size - 1))
        window = max(1, int(window))

        start = max(0, sample_idx - window)
        end = min(mono_audio.size, sample_idx + window)

        segment = mono_audio[start:end]

        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        if zero_crossings.size == 0:
            return sample_idx

        center = sample_idx - start
        nearest_idx = int(zero_crossings[np.argmin(np.abs(zero_crossings - center))])
        return start + nearest_idx

    def _get_downsampled_mono(self, target_sr: int = 2000) -> tuple[np.ndarray, int] | tuple[None, None]:
        """Get (cached) downsampled mono audio for coarse seam searches."""
        if self._mono_audio is None or self._sr is None:
            return None, None

        if self._mono_ds_audio is not None and self._mono_ds_sr == int(target_sr):
            return self._mono_ds_audio, self._mono_ds_sr

        try:
            ds = librosa.resample(
                self._mono_audio.astype(np.float32, copy=False),
                orig_sr=int(self._sr),
                target_sr=int(target_sr),
                res_type="kaiser_fast",
            )
            self._mono_ds_audio = ds
            self._mono_ds_sr = int(target_sr)
            return ds, int(target_sr)
        except Exception:
            return None, None

    def _best_match_center_xcorr(
        self,
        mono_audio: np.ndarray,
        sr: int,
        ref_center: int,
        search_center: int,
        *,
        search_window_ms: float,
        window_ms: float,
    ) -> tuple[int, float]:
        """Find the best-matching window center near `search_center` for a window around `ref_center`."""
        if mono_audio.size == 0 or sr <= 0:
            return int(search_center), 0.0

        ref_center = int(np.clip(ref_center, 0, mono_audio.size - 1))
        search_center = int(np.clip(search_center, 0, mono_audio.size - 1))

        window_size = max(32, int((window_ms / 1000.0) * sr))
        half = window_size // 2

        ref_start = max(0, ref_center - half)
        ref_end = min(mono_audio.size, ref_center + half)
        reference = mono_audio[ref_start:ref_end]
        if reference.size < max(16, window_size // 4):
            return int(search_center), 0.0

        half_ref = int(reference.size // 2)
        ref0 = reference.astype(np.float64, copy=False) - float(np.mean(reference))
        ref_norm = float(np.linalg.norm(ref0))
        if not np.isfinite(ref_norm) or ref_norm < 1e-8:
            return int(search_center), 0.0

        search_window = max(1, int((search_window_ms / 1000.0) * sr))
        search_start = max(0, search_center - search_window)
        search_end = min(mono_audio.size, search_center + search_window + ref0.size)

        y = mono_audio[search_start:search_end].astype(np.float64, copy=False)
        if y.size < ref0.size:
            return int(search_center), 0.0

        numerators = np.correlate(y, ref0, mode="valid").astype(np.float64, copy=False)

        y2 = y * y
        csum = np.cumsum(np.concatenate(([0.0], y)))
        csum2 = np.cumsum(np.concatenate(([0.0], y2)))

        L = ref0.size
        window_sums = csum[L:] - csum[:-L]
        window_sums2 = csum2[L:] - csum2[:-L]
        var_sums = window_sums2 - (window_sums * window_sums) / float(L)
        denom_y = np.sqrt(np.maximum(var_sums, 1e-12))

        denom = ref_norm * denom_y
        corr = np.where(denom > 0, numerators / denom, -np.inf)

        best_idx = int(np.nanargmax(corr))
        best_corr = float(corr[best_idx])
        best_corr = float(np.clip(best_corr, -1.0, 1.0))

        best_start = search_start + best_idx
        best_center = int(np.clip(best_start + half_ref, 0, mono_audio.size - 1))
        return best_center, best_corr

    def _refine_seam_original_sr(
        self,
        mono_audio: np.ndarray,
        sr: int,
        loop_start: int,
        loop_end: int,
        *,
        search_window_ms: float = 30.0,
        window_ms: float = 50.0,
        zero_cross_ms: float = 8.0,
    ) -> tuple[int, int, float]:
        """Refine loop seam at original sample rate using normalized cross-correlation."""
        if mono_audio.size == 0 or sr <= 0:
            return int(loop_start), int(loop_end), 0.0

        loop_start = int(np.clip(loop_start, 0, mono_audio.size - 1))
        loop_end = int(np.clip(loop_end, 0, mono_audio.size - 1))

        best_center, best_corr = self._best_match_center_xcorr(
            mono_audio, sr, loop_start, loop_end,
            search_window_ms=search_window_ms, window_ms=window_ms,
        )

        zc_window = max(1, int((zero_cross_ms / 1000.0) * sr))
        refined_start = self._find_zero_crossing(mono_audio, loop_start, zc_window)
        refined_end = self._find_zero_crossing(mono_audio, best_center, zc_window)

        if refined_end <= refined_start:
            refined_start = loop_start
            refined_end = best_center

        return int(refined_start), int(refined_end), float(best_corr)

    def _refine_seam_two_stage_original_sr(
        self,
        mono_audio: np.ndarray,
        sr: int,
        loop_start: int,
        loop_end: int,
        *,
        coarse_audio: np.ndarray | None = None,
        coarse_sr: int | None = None,
        coarse_search_window_ms: float = 2000.0,
        coarse_window_ms: float = 200.0,
        fine_search_window_ms: float = 35.0,
        fine_window_ms: float = 50.0,
        zero_cross_ms: float = 8.0,
    ) -> tuple[int, int, float]:
        """Two-stage seam refinement: coarse seed search (downsampled) -> fine sample-accurate refinement."""
        end_seed = int(loop_end)

        if (
            coarse_audio is not None
            and coarse_sr is not None
            and isinstance(coarse_sr, int)
            and coarse_sr > 0
            and sr > 0
            and coarse_audio.size > 0
        ):
            try:
                scale = float(coarse_sr) / float(sr)
                start_c = int(np.clip(int(loop_start * scale), 0, coarse_audio.size - 1))
                end_c = int(np.clip(int(loop_end * scale), 0, coarse_audio.size - 1))
                best_center_c, _ = self._best_match_center_xcorr(
                    coarse_audio, coarse_sr, start_c, end_c,
                    search_window_ms=coarse_search_window_ms,
                    window_ms=coarse_window_ms,
                )
                end_seed = int(np.clip(int(best_center_c / scale), 0, mono_audio.size - 1))
            except Exception:
                end_seed = int(loop_end)

        return self._refine_seam_original_sr(
            mono_audio, sr, loop_start, end_seed,
            search_window_ms=fine_search_window_ms,
            window_ms=fine_window_ms,
            zero_cross_ms=zero_cross_ms,
        )

    def _write_ogg_with_loop_tags(self, audio: np.ndarray, sr: int, loop_start: int, loop_end: int, output_path: str):
        """Write OGG file with LOOPSTART/LOOPLENGTH tags (RPG Maker compatible)."""
        audio_to_write = np.asarray(audio, dtype=np.float32)

        if (
            audio_to_write.ndim == 2
            and audio_to_write.shape[0] <= 8
            and audio_to_write.shape[1] > 8
            and audio_to_write.shape[0] < audio_to_write.shape[1]
        ):
            audio_to_write = audio_to_write.T

        scale = 1.0
        if audio_to_write.size:
            max_val = float(np.max(np.abs(audio_to_write)))
            if max_val > 1.0:
                scale = 1.0 / max_val

        channels = 1 if audio_to_write.ndim == 1 else int(audio_to_write.shape[1])

        chunk_frames = 65_536
        with sf.SoundFile(
            output_path, mode="w", samplerate=sr, channels=channels,
            format="OGG", subtype="VORBIS",
        ) as f:
            total_frames = int(audio_to_write.shape[0])
            for start in range(0, total_frames, chunk_frames):
                chunk = audio_to_write[start : start + chunk_frames]
                if scale != 1.0:
                    chunk = chunk * scale
                elif not chunk.flags["C_CONTIGUOUS"]:
                    chunk = np.ascontiguousarray(chunk)
                f.write(chunk)

        ogg = OggVorbis(output_path)
        ogg["LOOPSTART"] = str(loop_start)
        ogg["LOOPLENGTH"] = str(loop_end - loop_start)
        ogg.save()

    def _write_wav_with_smpl(self, audio: np.ndarray, sr: int, loop_start: int, loop_end: int, output_path: str):
        """Write WAV file with smpl chunk containing loop points."""
        if len(audio.shape) == 1:
            n_channels = 1
            audio_data = audio
        else:
            n_channels = audio.shape[1]
            audio_data = audio

        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(audio_int16.tobytes())

        self._append_smpl_chunk(output_path, sr, loop_start, loop_end)

    def _append_smpl_chunk(self, wav_path: str, sr: int, loop_start: int, loop_end: int):
        """Append smpl chunk to existing WAV file."""
        with open(wav_path, "rb") as f:
            data = bytearray(f.read())

        sample_period = int(1000000000 / sr)

        smpl_data = struct.pack(
            "<IIIIIIIIII",
            0, 0, sample_period, 60, 0, 0, 0, 1, 0, 0,
        )

        loop_data = struct.pack(
            "<IIIIII",
            0, 0, loop_start, loop_end - 1, 0, 0,
        )

        smpl_chunk = b"smpl" + struct.pack("<I", len(smpl_data) + len(loop_data) - 4) + smpl_data[:-4] + loop_data

        riff_size = struct.unpack("<I", data[4:8])[0]
        new_riff_size = riff_size + len(smpl_chunk)
        data[4:8] = struct.pack("<I", new_riff_size)

        data.extend(smpl_chunk)

        with open(wav_path, "wb") as f:
            f.write(data)

    def _repeat_with_crossfade(self, segment: np.ndarray, count: int, fade_samples: int) -> np.ndarray:
        """Repeat a segment N times, crossfading each boundary (equal-power)."""
        if count <= 1:
            return segment

        if segment.size == 0:
            return segment

        if segment.ndim == 1:
            seg = segment[:, None]
            squeeze = True
        else:
            seg = segment
            squeeze = False

        fade_samples = int(max(0, fade_samples))
        if fade_samples <= 0:
            out = np.concatenate([seg] * count, axis=0)
            return out[:, 0] if squeeze else out

        max_fade = int(seg.shape[0] // 2)
        if max_fade <= 0:
            out = np.concatenate([seg] * count, axis=0)
            return out[:, 0] if squeeze else out

        fade = int(min(fade_samples, max_fade))

        t = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32)
        fade_out = np.cos(0.5 * np.pi * t)[:, None]
        fade_in = np.sin(0.5 * np.pi * t)[:, None]

        head = seg[:fade]
        tail = seg[-fade:]
        overlap = tail * fade_out + head * fade_in

        pieces = [seg[:-fade]]
        mid = seg[fade:-fade]
        last = seg[fade:]
        for i in range(count - 1):
            pieces.append(overlap)
            pieces.append(mid if i < count - 2 else last)

        out = np.concatenate(pieces, axis=0)
        return out[:, 0] if squeeze else out
