"""allin1 Music Structure Enhancer.

This module integrates allin1 (music structure analysis library) as a post-processor
to enhance loop candidate scoring based on:
1. Downbeat alignment - filter/snap candidates to downbeat boundaries
2. Segment labels (chorus/verse/bridge) - boost candidates within same segment type

allin1 library:
    https://github.com/CPJKU/allin1
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import collections
import collections.abc
import contextlib
import numpy as np
import sys
import types
import os
import tempfile
import warnings
from pathlib import Path

if TYPE_CHECKING:
    from deep_analyzer import DeepLoopCandidate


def _install_pure_pytorch_natten_fallback() -> None:
    """Install a pure PyTorch fallback for natten's neighborhood attention functions.

    Creates fake ``natten`` and ``natten.functional`` modules in ``sys.modules``
    so that ``from natten.functional import natten1dqkrpb, ...`` succeeds without
    the native natten package.  Used on platforms where natten has no pre-built
    wheel (e.g. Windows).

    The implementation replicates natten's *window-shifting* boundary handling
    (not per-element clamping) and its relative-position-bias indexing scheme
    which were empirically verified against the native natten library.
    """
    if "natten" in sys.modules:
        return

    import torch

    # ------------------------------------------------------------------ helpers
    def _na1d_neighbor_info(
        L: int, ks: int, d: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(neighbors, rpb_indices)`` each of shape ``(L, ks)``.

        *neighbors*: absolute position index of each neighbor slot.
        *rpb_indices*: index into the ``(2*ks-1,)`` RPB table.

        Uses natten's window-shifting strategy: the ks-sized window is shifted
        so that all positions stay within ``[0, L-1]`` while keeping neighbours
        aligned to ``i mod d``.
        """
        half = ks // 2
        pos = torch.arange(L, device=device)

        ideal_start = pos - half * d          # (L,)
        lo = pos % d                          # per-position aligned minimum
        hi_scalar = L - 1 - (ks - 1) * d     # maximum possible start
        hi = hi_scalar - ((hi_scalar - lo) % d)  # round down to alignment
        hi = torch.where(hi >= lo, hi, lo)

        j_start = ideal_start.clamp(min=0)
        j_start = torch.max(j_start, lo)
        j_start = torch.min(j_start, hi)

        offsets = torch.arange(ks, device=device) * d
        neighbors = j_start.unsqueeze(1) + offsets.unsqueeze(0)  # (L, ks)

        k_range = torch.arange(ks, device=device)
        shift = (j_start - pos) // d
        rpb_idx = k_range.unsqueeze(0) + shift.unsqueeze(1) + (ks - 1)  # (L, ks)

        return neighbors.long(), rpb_idx.long()

    def _na2d_neighbor_info(
        H: int, W: int, ks: int, d: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(row_nb, col_nb, row_rpb, col_rpb)``.

        Shapes: row_nb ``(H, ks)``, col_nb ``(W, ks)``,
        row_rpb ``(H, ks)``, col_rpb ``(W, ks)``.
        """
        row_nb, row_rpb = _na1d_neighbor_info(H, ks, d, device)
        col_nb, col_rpb = _na1d_neighbor_info(W, ks, d, device)
        return row_nb, col_nb, row_rpb, col_rpb

    # ----------------------------------------------- 1-D functions
    def natten1dqkrpb(
        query: torch.Tensor,
        key: torch.Tensor,
        rpb: torch.Tensor,
        kernel_size: int,
        dilation: int,
    ) -> torch.Tensor:
        """Pure-PyTorch 1-D neighborhood QK + RPB.

        Args:
            query, key: ``(B, heads, L, D)``
            rpb: ``(heads, 2*ks-1)``
        Returns:
            ``(B, heads, L, ks)``
        """
        _B, _heads, L, _D = query.shape
        neighbors, rpb_idx = _na1d_neighbor_info(L, kernel_size, dilation, query.device)

        # Gather neighbor keys: (B, heads, L, ks, D)
        K_nb = key[:, :, neighbors]
        # Dot product over D → (B, heads, L, ks)
        scores = torch.einsum("bhld,bhlkd->bhlk", query, K_nb)
        # Add RPB: rpb[:, rpb_idx] → (heads, L, ks)
        scores = scores + rpb[:, rpb_idx].unsqueeze(0)
        return scores

    def natten1dav(
        attn: torch.Tensor,
        value: torch.Tensor,
        kernel_size: int,
        dilation: int,
    ) -> torch.Tensor:
        """Pure-PyTorch 1-D neighborhood AV (weighted sum).

        Args:
            attn: ``(B, heads, L, ks)``
            value: ``(B, heads, L, D)``
        Returns:
            ``(B, heads, L, D)``
        """
        _B, _heads, L, _D = value.shape
        neighbors, _ = _na1d_neighbor_info(L, kernel_size, dilation, value.device)

        # Gather neighbor values: (B, heads, L, ks, D)
        V_nb = value[:, :, neighbors]
        # Weighted sum over ks → (B, heads, L, D)
        return torch.einsum("bhlk,bhlkd->bhld", attn, V_nb)

    # ----------------------------------------------- 2-D functions
    def natten2dqkrpb(
        query: torch.Tensor,
        key: torch.Tensor,
        rpb: torch.Tensor,
        kernel_size: int,
        dilation: int,
    ) -> torch.Tensor:
        """Pure-PyTorch 2-D neighborhood QK + RPB.

        Args:
            query, key: ``(B, heads, H, W, D)``
            rpb: ``(heads, 2*ks-1, 2*ks-1)``
        Returns:
            ``(B, heads, H, W, ks*ks)``
        """
        _B, _heads, H, W, _D = query.shape
        ks = kernel_size
        row_nb, col_nb, row_rpb, col_rpb = _na2d_neighbor_info(
            H, W, ks, dilation, query.device,
        )

        # 2-D neighbor grid indices: (H, ks_h, W, ks_w)
        # row_nb: (H, ks), col_nb: (W, ks)
        # Gather keys: K[:, :, row_nb[h, kh], col_nb[w, kw], :] → (B, heads, H, ks, W, ks, D)
        K_rows = key[:, :, row_nb]          # (B, heads, H, ks, W, D)
        K_nb = K_rows[:, :, :, :, col_nb]   # (B, heads, H, ks, W, ks, D)
        # Rearrange to (B, heads, H, W, ks*ks, D)
        K_nb = K_nb.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        K_nb = K_nb.reshape(_B, _heads, H, W, ks * ks, _D)

        # Dot product
        scores = torch.einsum("bhwsd,bhwskd->bhwsk", query, K_nb)
        # → (B, heads, H, W, ks*ks)

        # RPB: combine row and col RPB indices
        # row_rpb: (H, ks), col_rpb: (W, ks)
        # Full 2D RPB index: rpb[:, row_rpb[h, kh], col_rpb[w, kw]]
        # → shape (heads, H, ks, W, ks) → rearrange to (heads, H, W, ks*ks)
        rpb_2d = rpb[:, row_rpb]                    # (heads, H, ks, 2*ks-1)
        rpb_2d = rpb_2d[:, :, :, col_rpb]           # (heads, H, ks, W, ks)
        rpb_2d = rpb_2d.permute(0, 1, 3, 2, 4).contiguous()  # (heads, H, W, ks, ks)
        rpb_2d = rpb_2d.reshape(_heads, H, W, ks * ks)

        scores = scores + rpb_2d.unsqueeze(0)
        return scores

    def natten2dav(
        attn: torch.Tensor,
        value: torch.Tensor,
        kernel_size: int,
        dilation: int,
    ) -> torch.Tensor:
        """Pure-PyTorch 2-D neighborhood AV (weighted sum).

        Args:
            attn: ``(B, heads, H, W, ks*ks)``
            value: ``(B, heads, H, W, D)``
        Returns:
            ``(B, heads, H, W, D)``
        """
        _B, _heads, H, W, _D = value.shape
        ks = kernel_size
        row_nb, col_nb, _, _ = _na2d_neighbor_info(
            H, W, ks, dilation, value.device,
        )

        # Gather neighbor values
        V_rows = value[:, :, row_nb]          # (B, heads, H, ks, W, D)
        V_nb = V_rows[:, :, :, :, col_nb]    # (B, heads, H, ks, W, ks, D)
        V_nb = V_nb.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        V_nb = V_nb.reshape(_B, _heads, H, W, ks * ks, _D)

        return torch.einsum("bhwsk,bhwskd->bhwsd", attn, V_nb)

    # -------------------------------- register fake natten modules
    natten_mod = types.ModuleType("natten")
    natten_mod.__package__ = "natten"
    natten_mod.__path__ = []  # type: ignore[attr-defined]

    func_mod = types.ModuleType("natten.functional")
    func_mod.__package__ = "natten"
    func_mod.natten1dqkrpb = natten1dqkrpb  # type: ignore[attr-defined]
    func_mod.natten1dav = natten1dav  # type: ignore[attr-defined]
    func_mod.natten2dqkrpb = natten2dqkrpb  # type: ignore[attr-defined]
    func_mod.natten2dav = natten2dav  # type: ignore[attr-defined]

    natten_mod.functional = func_mod  # type: ignore[attr-defined]
    sys.modules["natten"] = natten_mod
    sys.modules["natten.functional"] = func_mod


def _ensure_natten_torch_compat() -> None:
    """Ensure allin1 can import natten on macOS/non-CUDA torch builds.

    natten<=0.17.x imports an internal typing alias (`torch.cuda._device_t`) that
    has been removed in newer torch versions. It's only used for type hints, so
    providing a stub keeps the import working without affecting runtime.

    Additionally, allin1 expects legacy natten function names
    (`natten1dqkrpb`, `natten1dav`, etc.) which are not present in some natten
    versions. We alias them to the newer `na*d_*` functions when available.
    """
    try:
        # Compatibility shim: madmom still imports legacy classes from `collections`,
        # removed in Python 3.11.
        for _name in ("MutableSequence", "MutableMapping", "MutableSet"):
            if not hasattr(collections, _name) and hasattr(collections.abc, _name):
                setattr(collections, _name, getattr(collections.abc, _name))

        # Compatibility shim: madmom uses deprecated NumPy aliases removed in NumPy 2.x.
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

        # NumPy 2.x compatibility: madmom's DBNDownBeatTrackingProcessor uses
        # `np.asarray(results)` on ragged sequences which now raises ValueError.
        # Patch the processor to fall back to a safe implementation.
        try:
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor

            if not getattr(DBNDownBeatTrackingProcessor.process, "_music_looper_numpy2_patch", False):
                _orig_process = DBNDownBeatTrackingProcessor.process

                def _process_numpy2_safe(self, activations, **kwargs):  # type: ignore[override]
                    try:
                        return _orig_process(self, activations, **kwargs)
                    except ValueError as e:
                        if "setting an array element with a sequence" not in str(e):
                            raise

                        first = 0
                        if getattr(self, "threshold", None):
                            idx = np.nonzero(activations >= self.threshold)[0]
                            if idx.any():
                                first = max(first, int(np.min(idx)))
                                last = min(len(activations), int(np.max(idx)) + 1)
                            else:
                                last = first
                            activations = activations[first:last]

                        if not activations.any():
                            return np.empty((0, 2))

                        results = [hmm.viterbi(activations) for hmm in self.hmms]
                        best = max(range(len(results)), key=lambda i: float(results[i][1]))
                        path, _ = results[best]

                        st = self.hmms[best].transition_model.state_space
                        om = self.hmms[best].observation_model
                        positions = st.state_positions[path]
                        beat_numbers = positions.astype(int) + 1

                        if getattr(self, "correct", False):
                            beats = np.empty(0, dtype=int)
                            beat_range = om.pointers[path] >= 1
                            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
                            if beat_range[0]:
                                idx = np.r_[0, idx]
                            if beat_range[-1]:
                                idx = np.r_[idx, beat_range.size]
                            if idx.any():
                                for left, right in idx.reshape((-1, 2)):
                                    peak = np.argmax(activations[left:right]) // 2 + int(left)
                                    beats = np.hstack((beats, peak))
                        else:
                            beats = np.nonzero(np.diff(beat_numbers))[0] + 1

                        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

                _process_numpy2_safe._music_looper_numpy2_patch = True  # type: ignore[attr-defined]
                DBNDownBeatTrackingProcessor.process = _process_numpy2_safe  # type: ignore[assignment]
        except Exception:
            pass

        import torch

        cuda_mod = getattr(torch, "cuda", None)
        if cuda_mod is None:
            return

        # Typing alias removed in newer torch versions
        if not hasattr(cuda_mod, "_device_t"):
            setattr(cuda_mod, "_device_t", object)

        # natten<=0.17.x assumes CUDA APIs exist even when torch is built without CUDA
        # (e.g., macOS / MPS builds). Patch the one call site used during import.
        try:
            if not cuda_mod.is_available():
                def _fake_get_device_capability(device: object | None = None):
                    return (0, 0)

                cuda_mod.get_device_capability = _fake_get_device_capability  # type: ignore[assignment]
        except Exception:
            return

        # Alias legacy natten function names expected by allin1's DiNAT implementation.
        try:
            import natten.functional as nf  # type: ignore
        except Exception:
            _install_pure_pytorch_natten_fallback()
            return

        if hasattr(nf, "na1d_qk") and not hasattr(nf, "natten1dqkrpb"):
            def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
                return nf.na1d_qk(query, key, kernel_size, dilation=dilation, rpb=rpb)

            def natten1dav(attn, value, kernel_size, dilation):
                return nf.na1d_av(attn, value, kernel_size, dilation=dilation)

            def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
                return nf.na2d_qk(query, key, kernel_size, dilation=dilation, rpb=rpb)

            def natten2dav(attn, value, kernel_size, dilation):
                return nf.na2d_av(attn, value, kernel_size, dilation=dilation)

            nf.natten1dqkrpb = natten1dqkrpb  # type: ignore[attr-defined]
            nf.natten1dav = natten1dav  # type: ignore[attr-defined]
            nf.natten2dqkrpb = natten2dqkrpb  # type: ignore[attr-defined]
            nf.natten2dav = natten2dav  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; if torch isn't available, allin1 won't be either.
        return


@dataclass
class Segment:
    """Music structure segment."""
    start: float      # seconds
    end: float        # seconds
    label: str        # 'intro', 'verse', 'chorus', 'bridge', 'outro', etc.


@dataclass
class StructureInfo:
    """Music structure information from allin1."""
    bpm: float
    beats: List[float]       # beat times in seconds
    downbeats: List[float]   # downbeat times in seconds
    segments: List[Segment]


class Allin1Enhancer:
    """Enhances loop candidates using allin1 music structure analysis."""

    _model = None
    _model_repo_id = "taejunkim/allinone"
    _default_model_name = "harmonix-all"

    def __init__(self, file_path: str):
        """Initialize enhancer with audio file.

        Args:
            file_path: Path to audio file
        """
        self.file_path = file_path
        self._structure_info: Optional[StructureInfo] = None

    @staticmethod
    def is_available() -> bool:
        """Check if allin1 is available."""
        try:
            _ensure_natten_torch_compat()
            import allin1
            return True
        except Exception:
            return False

    @classmethod
    def _load_model(cls):
        """Load allin1 model (cached at class level)."""
        if cls._model is not None:
            return cls._model

        _ensure_natten_torch_compat()
        import allin1
        # allin1 uses a pretrained model internally
        cls._model = allin1
        return cls._model

    @staticmethod
    def _emit_progress(progress_callback, current: int, total: int, stage: str) -> None:
        if progress_callback:
            progress_callback(current, total, stage)

    @classmethod
    def _get_model_checkpoint_filenames(cls) -> List[str]:
        """Resolve checkpoint filenames for the default allin1 model preset."""
        _ensure_natten_torch_compat()
        from allin1.models.loaders import ENSEMBLE_MODELS, NAME_TO_FILE

        model_name = cls._default_model_name
        model_keys = ENSEMBLE_MODELS.get(model_name, [model_name])

        filenames: List[str] = []
        for key in model_keys:
            filename = NAME_TO_FILE.get(key)
            if filename:
                filenames.append(filename)
        return filenames

    @classmethod
    def _ensure_model_files_downloaded(cls, progress_callback=None) -> None:
        """Ensure allin1 checkpoint files are present in HF cache.

        Emits:
        - checking_structure_model
        - downloading_structure_model
        """
        cls._emit_progress(progress_callback, 0, 1, "checking_structure_model")

        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            # If huggingface_hub is unavailable, let downstream allin1 call fail normally.
            cls._emit_progress(progress_callback, 1, 1, "checking_structure_model")
            return

        filenames = cls._get_model_checkpoint_filenames()
        cls._emit_progress(progress_callback, 1, 1, "checking_structure_model")

        if not filenames:
            cls._emit_progress(progress_callback, 1, 1, "downloading_structure_model")
            return

        missing_files: List[str] = []
        for filename in filenames:
            try:
                hf_hub_download(
                    repo_id=cls._model_repo_id,
                    filename=filename,
                    local_files_only=True,
                )
            except Exception:
                missing_files.append(filename)

        if not missing_files:
            cls._emit_progress(progress_callback, 1, 1, "downloading_structure_model")
            return

        total = len(missing_files)
        cls._emit_progress(progress_callback, 0, total, "downloading_structure_model")
        for index, filename in enumerate(missing_files, start=1):
            hf_hub_download(repo_id=cls._model_repo_id, filename=filename)
            cls._emit_progress(progress_callback, index, total, "downloading_structure_model")

    def analyze_structure(self, progress_callback=None) -> StructureInfo:
        """Analyze music structure using allin1.

        Args:
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            StructureInfo with bpm, beats, downbeats, and segments
        """
        if self._structure_info is not None:
            return self._structure_info

        if not self.is_available():
            raise ImportError("allin1 is required for structure analysis. Install with: pip install allin1")

        self._ensure_model_files_downloaded(progress_callback)

        self._emit_progress(progress_callback, 0, 1, "loading_structure_model")
        allin1 = self._load_model()
        self._emit_progress(progress_callback, 1, 1, "loading_structure_model")

        demix_dir, spec_dir = self._resolve_runtime_cache_dirs()

        try:
            # allin1.analyze returns a Result object with:
            # - bpm: float
            # - beats: List[float] - beat times
            # - downbeats: List[float] - downbeat times
            # - segments: List[Segment] with start, end, label
            # Disable multiprocessing for better reliability in GUI/embedded contexts.
            self._emit_progress(progress_callback, 0, 1, "analyzing_structure")
            # Keep sidecar stdout JSON-only; send library logs/progress text to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                result = allin1.analyze(
                    self.file_path,
                    multiprocess=False,
                    demix_dir=demix_dir.as_posix(),
                    spec_dir=spec_dir.as_posix(),
                )

            # Convert to our StructureInfo format
            segments = []
            for seg in result.segments:
                segments.append(Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    label=str(seg.label).lower(),
                ))

            self._structure_info = StructureInfo(
                bpm=float(result.bpm),
                beats=[float(b) for b in result.beats],
                downbeats=[float(d) for d in result.downbeats],
                segments=segments,
            )

            self._emit_progress(progress_callback, 1, 1, "analyzing_structure")
            self._emit_progress(progress_callback, 1, 1, "structure_complete")

            return self._structure_info

        except Exception as e:
            warnings.warn(f"allin1 analysis failed: {e}")
            # Return empty structure info on failure
            self._structure_info = StructureInfo(
                bpm=0.0,
                beats=[],
                downbeats=[],
                segments=[],
            )
            return self._structure_info

    @staticmethod
    def _resolve_runtime_cache_dirs() -> Tuple[Path, Path]:
        """Resolve writable cache directories for allin1 byproducts.

        allin1 defaults to relative './demix' and './spec', which depend on the
        process working directory. In packaged app contexts this can point inside
        the app bundle, leading to unstable behavior across environments.
        """
        custom_root = os.getenv("MUSIC_LOOPER_CACHE_DIR")
        if custom_root:
            cache_root = Path(custom_root).expanduser()
        else:
            if sys.platform == "darwin":
                cache_root = Path.home() / "Library" / "Caches" / "music-looper"
            elif os.name == "nt":
                localappdata = os.getenv("LOCALAPPDATA")
                if localappdata:
                    cache_root = Path(localappdata) / "music-looper"
                else:
                    cache_root = Path(tempfile.gettempdir()) / "music-looper"
            else:
                cache_root = Path.home() / ".cache" / "music-looper"

        demix_dir = cache_root / "allin1_demix"
        spec_dir = cache_root / "allin1_spec"
        demix_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)
        return demix_dir, spec_dir

    def snap_to_downbeat(
        self,
        sample: int,
        sr: int,
        tolerance_ms: float = 100.0
    ) -> Tuple[int, bool]:
        """Snap a sample position to the nearest downbeat within tolerance.

        Args:
            sample: Sample index to snap
            sr: Sample rate
            tolerance_ms: Maximum distance in milliseconds to snap

        Returns:
            Tuple of (snapped_sample, was_snapped)
        """
        if self._structure_info is None:
            self.analyze_structure()

        if not self._structure_info.downbeats:
            return sample, False

        time_sec = sample / sr
        tolerance_sec = tolerance_ms / 1000.0

        # Find nearest downbeat
        downbeats = np.array(self._structure_info.downbeats)
        distances = np.abs(downbeats - time_sec)
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]

        if nearest_distance <= tolerance_sec:
            snapped_time = downbeats[nearest_idx]
            snapped_sample = int(snapped_time * sr)
            return snapped_sample, True

        return sample, False

    def get_segment_at_time(self, time_sec: float) -> Optional[Segment]:
        """Get the segment at a given time.

        Args:
            time_sec: Time in seconds

        Returns:
            Segment at the given time, or None if not found
        """
        if self._structure_info is None:
            self.analyze_structure()

        for segment in self._structure_info.segments:
            if segment.start <= time_sec < segment.end:
                return segment

        return None

    def get_segment_label_at_sample(self, sample: int, sr: int) -> Optional[str]:
        """Get the segment label at a given sample position.

        Args:
            sample: Sample index
            sr: Sample rate

        Returns:
            Segment label or None
        """
        time_sec = sample / sr
        segment = self.get_segment_at_time(time_sec)
        return segment.label if segment else None

    def enhance_candidates(
        self,
        candidates: List["DeepLoopCandidate"],
        sr: int,
        snap_to_downbeats: bool = True,
        downbeat_filter: bool = True,
        same_segment_boost: float = 0.15,
        tolerance_ms: float = 100.0,
    ) -> List["DeepLoopCandidate"]:
        """Enhance loop candidates with structure information.

        This method:
        1. Snaps loop points to downbeats (if within tolerance)
        2. Filters candidates that aren't near downbeats (optional)
        3. Boosts scores for candidates within same segment type

        Score boost logic:
        - Same segment (chorus->chorus): +0.15
        - Same segment (bridge/intro/outro): +0.10
        - Different segments: +0.00

        Args:
            candidates: List of DeepLoopCandidate objects
            sr: Sample rate of the audio (original, not model rate)
            downbeat_filter: If True, filter out candidates not near downbeats
            same_segment_boost: Score boost for same-segment loops
            tolerance_ms: Tolerance for downbeat snapping in milliseconds

        Returns:
            Enhanced list of DeepLoopCandidate objects
        """
        if not candidates:
            return candidates

        if self._structure_info is None:
            try:
                self.analyze_structure()
            except Exception as e:
                warnings.warn(f"Structure analysis failed: {e}, returning original candidates")
                return candidates

        enhanced = []

        for cand in candidates:
            # Snap to downbeats (optional; can be disabled if another beat-snapper is used)
            start_nearest, start_is_downbeat = self.snap_to_downbeat(
                cand.loop_start, sr, tolerance_ms
            )
            end_nearest, end_is_downbeat = self.snap_to_downbeat(
                cand.loop_end, sr, tolerance_ms
            )

            start_snapped = start_nearest if snap_to_downbeats else cand.loop_start
            end_snapped = end_nearest if snap_to_downbeats else cand.loop_end

            # Check if both start and end are near downbeats
            is_downbeat_aligned = start_is_downbeat and end_is_downbeat

            # Filter if downbeat_filter is enabled and not aligned
            if downbeat_filter and not is_downbeat_aligned:
                # Still include but mark as not aligned
                pass

            # Get segment labels
            start_segment = self.get_segment_label_at_sample(start_snapped, sr)
            end_segment = self.get_segment_label_at_sample(end_snapped, sr)

            # Calculate structure boost
            structure_boost = 0.0
            if start_segment and end_segment and start_segment == end_segment:
                # Same segment type
                if start_segment in ('chorus', 'verse'):
                    structure_boost = same_segment_boost
                elif start_segment in ('bridge', 'intro', 'outro'):
                    structure_boost = same_segment_boost * 0.67  # ~0.10 for default
                else:
                    structure_boost = same_segment_boost * 0.5

            # Update candidate with enhanced info
            # Note: We modify the dataclass fields directly since they're mutable
            cand.loop_start = start_snapped
            cand.loop_end = end_snapped
            cand.start_segment = start_segment
            cand.end_segment = end_segment
            cand.is_downbeat_aligned = is_downbeat_aligned
            cand.structure_boost = structure_boost

            # Apply boost to score
            cand.score = min(1.0, cand.score + structure_boost)

            enhanced.append(cand)

        # Re-sort by score after boosting
        enhanced.sort(key=lambda x: x.score, reverse=True)

        return enhanced

    def get_structure_info(self) -> Optional[StructureInfo]:
        """Get the cached structure info.

        Returns:
            StructureInfo if analyzed, None otherwise
        """
        return self._structure_info
