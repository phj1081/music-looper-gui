"""Music Looper Desktop App - Single file, no server."""

from __future__ import annotations

import io
import json
import os
import base64
import struct
import time
import wave
import threading
import collections
import collections.abc
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import librosa
import soundfile as sf
import webview
from mutagen.oggvorbis import OggVorbis
from webview.dom import DOMEventHandler

from deep_analyzer import DeepLoopAnalyzer, BeatAlignedLoopAnalyzer
from allin1_enhancer import Allin1Enhancer

# Global window reference for API access
window = None


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
    # Avoid triggering NumPy's deprecated alias warnings via `hasattr`.
    if _name not in np.__dict__:
        setattr(np, _name, _alias)


class Api:
    """Python API exposed to JavaScript via PyWebView bridge."""

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

    @staticmethod
    def _is_madmom_available() -> bool:
        """Check if madmom is available for beat alignment."""
        try:
            import madmom
            return True
        except Exception:
            return False

    def analyze(self, file_path: str) -> dict:
        """Analyze audio file for loop points.

        Auto-detects and uses available enhancements:
        - madmom: beat alignment (if installed)
        - allin1: structure analysis (if installed)

        Args:
            file_path: Path to audio file
        """
        self._current_file = file_path
        # Auto-detect available enhancements (can be disabled via env flags)
        self._use_beat_alignment = self._is_madmom_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_BEAT_ALIGNMENT"
        )
        self._use_allin1 = Allin1Enhancer.is_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_STRUCTURE"
        )

        try:
            # Load audio for playback at original sample rate
            self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
            if len(self._audio.shape) == 1:
                self._audio = self._audio.reshape(-1, 1)
            else:
                self._audio = self._audio.T  # (samples, channels)
            # Cache mono audio for analysis helpers
            self._mono_audio = self._audio[:, 0] if self._audio is not None else None
            self._mono_ds_audio = None
            self._mono_ds_sr = None

            return self._analyze_ast(file_path, self._use_beat_alignment, self._use_allin1)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_ast(self, file_path: str, use_beat_alignment: bool = False, use_allin1: bool = False) -> dict:
        """Deep learning-based analysis (Audio Spectrogram Transformer)."""
        if use_beat_alignment:
            self._deep_analyzer = BeatAlignedLoopAnalyzer(file_path, use_beat_alignment=True)
        else:
            self._deep_analyzer = DeepLoopAnalyzer(file_path)

        # Max-quality strategy:
        # - Run BOTH recurrence_matrix(+path_enhance) and raw similarity scans.
        # - Merge + dedupe candidates before post-processing.
        # - Avoid tying loop length to total duration (min_loop_fraction=0).
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

        # Scale sample indices from 16kHz (model) to original sample rate
        scale_factor = self._sr / self._deep_analyzer.sr

        # Scale candidates to original sample rate before allin1 enhancement
        for cand in candidates:
            cand.loop_start = int(cand.loop_start * scale_factor)
            cand.loop_end = int(cand.loop_end * scale_factor)

        # Deduplicate near-identical candidates (keep best-scoring first)
        candidates.sort(key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)
        deduped = []
        tol = int(self._sr * 0.35)  # 350ms tolerance at original SR
        for cand in candidates:
            if any(
                abs(int(cand.loop_start) - int(prev.loop_start)) <= tol
                and abs(int(cand.loop_end) - int(prev.loop_end)) <= tol
                for prev in deduped
            ):
                continue
            deduped.append(cand)
        candidates = deduped

        # Apply allin1 enhancement if enabled
        if use_allin1:
            candidates = self._apply_allin1_enhancement(file_path, candidates)

        # Final seam refinement at original sample rate (post-snapping)
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
                        mono_audio,
                        self._sr,
                        cand.loop_start,
                        cand.loop_end,
                        coarse_audio=coarse_audio,
                        coarse_sr=coarse_sr,
                        coarse_search_window_ms=coarse_search_ms,
                        coarse_window_ms=coarse_window_ms,
                        fine_search_window_ms=fine_search_ms,
                        fine_window_ms=50.0,
                        zero_cross_ms=seam_zero_ms,
                    )
                    cand.loop_start = refined_start
                    cand.loop_end = refined_end

                    base_score = cand.similarity_score * 0.7 + max(0.0, seam_corr) * 0.3
                    cand.score = min(1.0, base_score + float(getattr(cand, "structure_boost", 0.0) or 0.0))

                # Re-sort after seam re-scoring
                candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            # Seam refinement is best-effort; fall back to original candidates.
            pass

        loops = []
        beat_effective = any("beat" in str(getattr(cand, "algorithm", "")).lower() for cand in candidates)
        structure_effective = any(
            bool(getattr(cand, "start_segment", None))
            or bool(getattr(cand, "end_segment", None))
            or float(getattr(cand, "structure_boost", 0.0) or 0.0) > 0.0
            for cand in candidates
        )
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

            # Add allin1 fields if available
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
                "beat_alignment": {"enabled": bool(use_beat_alignment), "effective": bool(beat_effective)},
                "structure": {"enabled": bool(use_allin1), "effective": bool(structure_effective)},
                "seam_refinement": "two_stage",
            },
            "loops": loops,
        }

    def _apply_allin1_enhancement(self, file_path: str, candidates):
        """Apply allin1 structure analysis enhancement to candidates.

        Args:
            file_path: Path to audio file
            candidates: List of loop candidates (already scaled to original sample rate)

        Returns:
            Enhanced list of candidates
        """
        if not Allin1Enhancer.is_available():
            return candidates

        try:
            self._allin1_enhancer = Allin1Enhancer(file_path)
            self._allin1_enhancer.analyze_structure()
            return self._allin1_enhancer.enhance_candidates(
                candidates,
                sr=self._sr,
                snap_to_downbeats=not self._use_beat_alignment,
                downbeat_filter=False,  # Don't filter, just mark and boost
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

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        if zero_crossings.size == 0:
            return sample_idx

        # Find nearest to center
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
            # Use a fast resampler for coarse alignment (seam seed search only).
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
        """Refine loop seam at original sample rate using normalized cross-correlation.

        This runs after any beat/structure snapping to nudge the exact sample
        positions for a cleaner transition (pop/click reduction).
        """
        if mono_audio.size == 0 or sr <= 0:
            return int(loop_start), int(loop_end), 0.0

        loop_start = int(np.clip(loop_start, 0, mono_audio.size - 1))
        loop_end = int(np.clip(loop_end, 0, mono_audio.size - 1))

        best_center, best_corr = self._best_match_center_xcorr(
            mono_audio,
            sr,
            loop_start,
            loop_end,
            search_window_ms=search_window_ms,
            window_ms=window_ms,
        )

        # Snap to nearest zero-crossing in a small window
        zc_window = max(1, int((zero_cross_ms / 1000.0) * sr))
        refined_start = self._find_zero_crossing(mono_audio, loop_start, zc_window)
        refined_end = self._find_zero_crossing(mono_audio, best_center, zc_window)

        # Keep ordering sane
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
                    coarse_audio,
                    coarse_sr,
                    start_c,
                    end_c,
                    search_window_ms=coarse_search_window_ms,
                    window_ms=coarse_window_ms,
                )
                end_seed = int(np.clip(int(best_center_c / scale), 0, mono_audio.size - 1))
            except Exception:
                end_seed = int(loop_end)

        return self._refine_seam_original_sr(
            mono_audio,
            sr,
            loop_start,
            end_seed,
            search_window_ms=fine_search_window_ms,
            window_ms=fine_window_ms,
            zero_cross_ms=zero_cross_ms,
        )

    def _progress_callback(self, current: int, total: int, stage: str):
        """Callback for analysis progress updates."""
        self._analysis_progress = {
            "current": current,
            "total": total,
            "stage": stage,
        }

    def _analyze_with_progress(self, file_path: str, use_beat_alignment: bool = False, use_allin1: bool = False):
        """Run analysis with progress tracking in background thread."""
        try:
            self._analysis_progress = {"current": 0, "total": 0, "stage": "loading_model"}
            self._analysis_result = None

            # Load audio for playback at original sample rate
            self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
            if len(self._audio.shape) == 1:
                self._audio = self._audio.reshape(-1, 1)
            else:
                self._audio = self._audio.T  # (samples, channels)
            # Cache mono audio for analysis helpers
            self._mono_audio = self._audio[:, 0] if self._audio is not None else None
            self._mono_ds_audio = None
            self._mono_ds_sr = None

            self._analysis_result = self._analyze_ast_with_progress(file_path, use_beat_alignment, use_allin1)

            self._analysis_progress = {"current": 1, "total": 1, "stage": "completed"}
        except Exception as e:
            self._analysis_result = {"success": False, "error": str(e)}
            self._analysis_progress = {"current": 0, "total": 0, "stage": "error", "error": str(e)}

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

        # Scale candidates to original sample rate before allin1 enhancement
        for cand in candidates:
            cand.loop_start = int(cand.loop_start * scale_factor)
            cand.loop_end = int(cand.loop_end * scale_factor)

        # Deduplicate near-identical candidates (keep best-scoring first)
        candidates.sort(key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)
        deduped = []
        tol = int(self._sr * 0.35)  # 350ms tolerance at original SR
        for cand in candidates:
            if any(
                abs(int(cand.loop_start) - int(prev.loop_start)) <= tol
                and abs(int(cand.loop_end) - int(prev.loop_end)) <= tol
                for prev in deduped
            ):
                continue
            deduped.append(cand)
        candidates = deduped

        # Apply allin1 enhancement if enabled
        if use_allin1:
            self._progress_callback(0, 1, "analyzing_structure")
            candidates = self._apply_allin1_enhancement(file_path, candidates)

        # Final seam refinement at original sample rate (post-snapping)
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
                        mono_audio,
                        self._sr,
                        cand.loop_start,
                        cand.loop_end,
                        coarse_audio=coarse_audio,
                        coarse_sr=coarse_sr,
                        coarse_search_window_ms=coarse_search_ms,
                        coarse_window_ms=coarse_window_ms,
                        fine_search_window_ms=fine_search_ms,
                        fine_window_ms=50.0,
                        zero_cross_ms=seam_zero_ms,
                    )
                    cand.loop_start = refined_start
                    cand.loop_end = refined_end

                    base_score = cand.similarity_score * 0.7 + max(0.0, seam_corr) * 0.3
                    cand.score = min(1.0, base_score + float(getattr(cand, "structure_boost", 0.0) or 0.0))

                    self._progress_callback(idx + 1, len(candidates) or 1, "refining_seam")

                candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            pass

        loops = []
        beat_effective = any("beat" in str(getattr(cand, "algorithm", "")).lower() for cand in candidates)
        structure_effective = any(
            bool(getattr(cand, "start_segment", None))
            or bool(getattr(cand, "end_segment", None))
            or float(getattr(cand, "structure_boost", 0.0) or 0.0) > 0.0
            for cand in candidates
        )
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

            # Add allin1 fields if available
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
                "beat_alignment": {"enabled": bool(use_beat_alignment), "effective": bool(beat_effective)},
                "structure": {"enabled": bool(use_allin1), "effective": bool(structure_effective)},
                "seam_refinement": "two_stage",
            },
            "loops": loops,
        }

    def analyze_async(self, file_path: str) -> dict:
        """Start asynchronous analysis in a background thread.

        Auto-detects and uses available enhancements:
        - madmom: beat alignment (if installed)
        - allin1: structure analysis (if installed)

        Args:
            file_path: Path to audio file
        """
        self._current_file = file_path
        # Auto-detect available enhancements (can be disabled via env flags)
        self._use_beat_alignment = self._is_madmom_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_BEAT_ALIGNMENT"
        )
        self._use_allin1 = Allin1Enhancer.is_available() and not _is_truthy_env(
            "MUSIC_LOOPER_DISABLE_STRUCTURE"
        )
        self._analysis_progress = {"current": 0, "total": 0, "stage": "starting"}
        self._analysis_result = None

        # Start analysis in background thread
        self._analysis_thread = threading.Thread(
            target=self._analyze_with_progress,
            args=(file_path, self._use_beat_alignment, self._use_allin1),
            daemon=True
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

    def export_with_loop_tags(self, loop_start: int, loop_end: int, format: str) -> bool:
        """Export file with loop metadata tags (OGG or WAV with smpl chunk)."""
        if self._audio is None or self._sr is None:
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

            if format == "ogg":
                self._write_ogg_with_loop_tags(self._audio, self._sr, loop_start, loop_end, output_path)
            else:
                self._write_wav_with_smpl(self._audio, self._sr, loop_start, loop_end, output_path)

            return True
        except Exception:
            return False

    def _write_ogg_with_loop_tags(self, audio: np.ndarray, sr: int, loop_start: int, loop_end: int, output_path: str):
        """Write OGG file with LOOPSTART/LOOPLENGTH tags (RPG Maker compatible)."""
        # Ensure audio is in correct shape for soundfile (samples, channels)
        if len(audio.shape) == 1:
            audio_to_write = audio
        else:
            audio_to_write = audio

        # Write OGG file first
        sf.write(output_path, audio_to_write, sr, format='OGG', subtype='VORBIS')

        # Add loop tags using mutagen
        ogg = OggVorbis(output_path)
        ogg['LOOPSTART'] = str(loop_start)
        ogg['LOOPLENGTH'] = str(loop_end - loop_start)
        ogg.save()

    def _write_wav_with_smpl(self, audio: np.ndarray, sr: int, loop_start: int, loop_end: int, output_path: str):
        """Write WAV file with smpl chunk containing loop points."""
        # Handle channel configuration
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

        # Write basic WAV first
        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(audio_int16.tobytes())

        # Now append smpl chunk
        self._append_smpl_chunk(output_path, sr, loop_start, loop_end)

    def _append_smpl_chunk(self, wav_path: str, sr: int, loop_start: int, loop_end: int):
        """Append smpl chunk to existing WAV file."""
        # Read the entire WAV file
        with open(wav_path, 'rb') as f:
            data = bytearray(f.read())

        # Build smpl chunk
        # smpl chunk structure:
        # - Manufacturer (4 bytes): 0
        # - Product (4 bytes): 0
        # - Sample Period (4 bytes): 1000000000 / sample_rate (nanoseconds per sample)
        # - MIDI Unity Note (4 bytes): 60 (middle C)
        # - MIDI Pitch Fraction (4 bytes): 0
        # - SMPTE Format (4 bytes): 0
        # - SMPTE Offset (4 bytes): 0
        # - Num Sample Loops (4 bytes): 1
        # - Sampler Data (4 bytes): 0
        # Then for each loop:
        # - Cue Point ID (4 bytes): 0
        # - Type (4 bytes): 0 (forward loop)
        # - Start (4 bytes): loop_start
        # - End (4 bytes): loop_end
        # - Fraction (4 bytes): 0
        # - Play Count (4 bytes): 0 (infinite)

        sample_period = int(1000000000 / sr)

        smpl_data = struct.pack('<IIIIIIIIII',
            0,                  # Manufacturer
            0,                  # Product
            sample_period,      # Sample Period
            60,                 # MIDI Unity Note (middle C)
            0,                  # MIDI Pitch Fraction
            0,                  # SMPTE Format
            0,                  # SMPTE Offset
            1,                  # Num Sample Loops
            0,                  # Sampler Data size
            0,                  # Cue Point ID
        )

        # Add loop data
        loop_data = struct.pack('<IIIIII',
            0,                  # Cue Point ID
            0,                  # Type (0 = forward loop)
            loop_start,         # Start
            loop_end - 1,       # End (inclusive, so subtract 1)
            0,                  # Fraction
            0,                  # Play Count (0 = infinite)
        )

        smpl_chunk = b'smpl' + struct.pack('<I', len(smpl_data) + len(loop_data) - 4) + smpl_data[:-4] + loop_data

        # Find the position to insert (before the end of RIFF)
        # Update RIFF size
        riff_size = struct.unpack('<I', data[4:8])[0]
        new_riff_size = riff_size + len(smpl_chunk)
        data[4:8] = struct.pack('<I', new_riff_size)

        # Append smpl chunk at the end
        data.extend(smpl_chunk)

        # Write back
        with open(wav_path, 'wb') as f:
            f.write(data)

    def export_split_sections(self, loop_start: int, loop_end: int) -> bool:
        """Export intro, loop, and outro as separate WAV files."""
        if self._audio is None or self._sr is None or self._current_file is None:
            return False

        try:
            # Ask for output directory
            result = window.create_file_dialog(
                webview.FOLDER_DIALOG,
            )

            if not result:
                return False

            output_dir = Path(result[0]) if isinstance(result, (list, tuple)) else Path(result)
            base_name = Path(self._current_file).stem

            audio = self._audio
            sr = self._sr

            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            # Split sections
            intro = audio[:loop_start]
            loop = audio[loop_start:loop_end]
            outro = audio[loop_end:]

            # Export each section
            sections = [
                (intro, f"{base_name}_intro.wav"),
                (loop, f"{base_name}_loop.wav"),
                (outro, f"{base_name}_outro.wav"),
            ]

            for section_audio, filename in sections:
                if len(section_audio) == 0:
                    continue

                output_path = output_dir / filename

                # Normalize
                max_val = np.max(np.abs(section_audio))
                if max_val > 0:
                    if max_val > 1.0:
                        section_audio = section_audio / max_val

                audio_int16 = (section_audio * 32767).astype(np.int16)

                with wave.open(str(output_path), "wb") as wav:
                    wav.setnchannels(n_channels)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    wav.writeframes(audio_int16.tobytes())

            return True
        except Exception:
            return False

    def export_extended(self, loop_start: int, loop_end: int, loop_count: int) -> bool:
        """Export extended version with loop repeated N times."""
        if self._audio is None or self._sr is None:
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

            audio = self._audio
            sr = self._sr

            if len(audio.shape) == 1:
                n_channels = 1
            else:
                n_channels = audio.shape[1]

            # Split and concatenate
            intro = audio[:loop_start]
            loop = audio[loop_start:loop_end]
            outro = audio[loop_end:]

            # Build extended version: intro + (loop × N) + outro
            # For "hard" seams (common with AI-gen tracks), a short crossfade between
            # each loop repetition improves perceived seamlessness a lot.
            crossfade_ms = 80.0
            fade_samples = int((crossfade_ms / 1000.0) * sr)

            repeated = self._repeat_with_crossfade(loop, loop_count, fade_samples)
            extended = np.concatenate([intro, repeated, outro], axis=0)

            # Normalize
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

    def _repeat_with_crossfade(self, segment: np.ndarray, count: int, fade_samples: int) -> np.ndarray:
        """Repeat a segment N times, crossfading each boundary (equal-power).

        The crossfade is only used between repetitions; the first/last samples
        of the overall output are unchanged.
        """
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

        # Prevent pathological overlap on very short loops
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
        # Middle repetitions contribute seg[fade:-fade] (head + tail are crossfaded)
        mid = seg[fade:-fade]
        last = seg[fade:]
        for i in range(count - 1):
            pieces.append(overlap)
            pieces.append(mid if i < count - 2 else last)

        out = np.concatenate(pieces, axis=0)
        return out[:, 0] if squeeze else out

    def export_loop_info(self, loop_start: int, loop_end: int, format: str) -> bool:
        """Export loop point information as JSON or TXT."""
        if self._sr is None:
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
            sr = self._sr

            # Calculate time values
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
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
            else:
                # TXT format
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
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

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
            # Server is reachable even if it responds with an error page.
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
    main()
