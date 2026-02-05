"""Deep Learning Loop Detection using Audio Spectrogram Transformer (AST).

This module provides an advanced loop detection algorithm that uses:
1. Audio Spectrogram Transformer (AST) - extracts semantic audio embeddings
2. Similarity Matrix - finds repeating patterns via cosine similarity
3. Zero-crossing refinement - precise sample-level alignment

AST Model:
    MIT/ast-finetuned-audioset-10-10-0.4593
    https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

    Paper: "AST: Audio Spectrogram Transformer"
    Authors: Yuan Gong, Yu-An Chung, James Glass (MIT)
    https://arxiv.org/abs/2104.01778

    License: BSD-3-Clause
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings


@dataclass
class DeepLoopCandidate:
    """Loop point candidate from deep learning analysis."""
    loop_start: int  # in samples
    loop_end: int    # in samples
    score: float
    similarity_score: float
    algorithm: str = "ast"  # Which algorithm found this candidate
    # allin1 structure analysis fields
    start_segment: Optional[str] = None  # 'intro', 'verse', 'chorus', 'bridge', 'outro'
    end_segment: Optional[str] = None
    is_downbeat_aligned: bool = False
    structure_boost: float = 0.0

    def __post_init__(self):
        # Ensure all values are Python native types for JSON serialization
        self.loop_start = int(self.loop_start)
        self.loop_end = int(self.loop_end)
        self.score = float(self.score)
        self.similarity_score = float(self.similarity_score)
        self.structure_boost = float(self.structure_boost)

    @property
    def duration_samples(self) -> int:
        return self.loop_end - self.loop_start


class DeepLoopAnalyzer:
    """Deep Learning based loop finder using Audio Spectrogram Transformer."""

    # Class-level model cache to avoid reloading
    _model = None
    _processor = None
    _device = None
    _use_half = False

    def __init__(
        self,
        file_path: str,
        sr: int = 16000,  # AST expects 16kHz
        chunk_duration: float = 2.0,  # Increased from 1.0 for better performance
        batch_size: int = 16,  # Number of chunks to process at once
    ):
        """Initialize analyzer with audio file.

        Args:
            file_path: Path to audio file
            sr: Sample rate (16kHz for AST model)
            chunk_duration: Duration of each chunk in seconds
            batch_size: Number of chunks to process in a single forward pass
        """
        self.file_path = file_path
        self.target_sr = sr
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size

        # Load audio
        self.audio, self.sr = librosa.load(file_path, sr=sr, mono=True)
        self.duration = len(self.audio) / self.sr

        # Lazy loading
        self._embeddings = None
        self._similarity_matrix = None

    @classmethod
    def _load_model(cls):
        """Load AST model (cached at class level).

        Uses float16 precision on CUDA/MPS for better performance.
        """
        if cls._model is not None:
            return cls._model, cls._processor, cls._device, cls._use_half

        import torch
        from transformers import AutoFeatureExtractor, ASTModel

        # Determine device and precision
        if torch.cuda.is_available():
            cls._device = torch.device("cuda")
            cls._use_half = True
        elif torch.backends.mps.is_available():
            cls._device = torch.device("mps")
            cls._use_half = True
        else:
            cls._device = torch.device("cpu")
            cls._use_half = False

        # Load model and processor
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls._processor = AutoFeatureExtractor.from_pretrained(model_name)
            cls._model = ASTModel.from_pretrained(model_name)

            # Use half precision on GPU/MPS for better performance
            if cls._use_half:
                cls._model = cls._model.half()

            cls._model.to(cls._device)
            cls._model.eval()

        return cls._model, cls._processor, cls._device, cls._use_half

    def extract_embeddings(self, progress_callback=None) -> np.ndarray:
        """Extract embeddings for each audio chunk using batch processing.

        Uses batch processing to significantly improve performance:
        - Groups multiple chunks together for a single forward pass
        - Uses float16 precision on GPU/MPS
        - Reduces GPU/CPU data transfer overhead

        Args:
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            (n_chunks, embedding_dim) array of embeddings
        """
        if self._embeddings is not None:
            return self._embeddings

        import torch

        if progress_callback:
            progress_callback(0, 1, "loading_model")

        model, processor, device, use_half = self._load_model()

        # Calculate chunk parameters
        chunk_samples = int(self.chunk_duration * self.sr)
        n_chunks = len(self.audio) // chunk_samples

        if n_chunks == 0:
            # Audio too short, use entire audio as one chunk
            n_chunks = 1
            chunk_samples = len(self.audio)

        if progress_callback:
            progress_callback(0, n_chunks, "preparing_chunks")

        # Prepare all chunks first
        all_chunks = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = self.audio[start:end]

            # Pad if necessary (last chunk might be shorter)
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            all_chunks.append(chunk)

        embeddings = []
        total_batches = (n_chunks + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            # Process in batches
            for batch_idx, batch_start in enumerate(range(0, n_chunks, self.batch_size)):
                if progress_callback:
                    progress_callback(batch_idx, total_batches, "extracting_embeddings")

                batch_end = min(batch_start + self.batch_size, n_chunks)
                batch_chunks = all_chunks[batch_start:batch_end]

                # Process batch of chunks
                inputs = processor(
                    batch_chunks,
                    sampling_rate=self.sr,
                    return_tensors="pt",
                    padding=True,
                )

                # Move to device and convert to half precision if applicable
                if use_half:
                    inputs = {
                        k: v.to(device).half() if v.dtype == torch.float32 else v.to(device)
                        for k, v in inputs.items()
                    }
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get embeddings for entire batch
                outputs = model(**inputs)

                # Use mean pooling over sequence dimension
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                # Convert to float32 for numpy and move to CPU
                batch_embeddings = batch_embeddings.float().cpu().numpy()

                # Add each embedding in the batch
                for emb in batch_embeddings:
                    embeddings.append(emb)

        self._embeddings = np.array(embeddings)
        return self._embeddings

    def compute_similarity_matrix(self, progress_callback=None) -> np.ndarray:
        """Compute cosine similarity matrix between all chunk embeddings.

        Args:
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            (n_chunks, n_chunks) similarity matrix
        """
        if self._similarity_matrix is not None:
            return self._similarity_matrix

        embeddings = self.extract_embeddings(progress_callback)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Compute cosine similarity
        self._similarity_matrix = np.dot(normalized, normalized.T)

        return self._similarity_matrix

    def find_diagonal_patterns(
        self,
        min_duration: float = 10.0,
        threshold: float = 0.85,
        progress_callback=None,
    ) -> List[Tuple[int, int, float]]:
        """Find diagonal patterns in similarity matrix indicating loops.

        Args:
            min_duration: Minimum loop duration in seconds
            threshold: Minimum similarity threshold
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            List of (start_chunk, end_chunk, score) tuples
        """
        sim_matrix = self.compute_similarity_matrix(progress_callback)
        n_chunks = sim_matrix.shape[0]

        min_chunks = int(min_duration / self.chunk_duration)
        candidates = []

        # Scan diagonals (offset from main diagonal)
        for offset in range(min_chunks, n_chunks):
            diag = np.diag(sim_matrix, k=offset)

            if len(diag) == 0:
                continue

            # Find high-similarity regions
            above_thresh = diag > threshold

            # Find contiguous regions
            if not np.any(above_thresh):
                continue

            # Calculate average score for this diagonal
            high_vals = diag[above_thresh]
            if len(high_vals) > 0:
                avg_score = np.mean(high_vals)

                # Find best starting point
                best_idx = np.argmax(diag)
                candidates.append((best_idx, best_idx + offset, avg_score))

        # Also look for strong off-diagonal peaks
        for i in range(n_chunks):
            for j in range(i + min_chunks, n_chunks):
                if sim_matrix[i, j] > threshold:
                    candidates.append((i, j, sim_matrix[i, j]))

        # Remove duplicates and sort by score
        unique_candidates = {}
        for start, end, score in candidates:
            key = (start, end)
            if key not in unique_candidates or unique_candidates[key] < score:
                unique_candidates[key] = score

        sorted_candidates = [
            (start, end, score)
            for (start, end), score in unique_candidates.items()
        ]
        sorted_candidates.sort(key=lambda x: x[2], reverse=True)

        return sorted_candidates

    def find_diagonal_patterns_librosa(
        self,
        min_duration: float = 10.0,
        k: int = 7,             # nearest neighbors for recurrence matrix
        width: int = 3,         # recurrence width
        n_filters: int = 7,     # path_enhance filters
        filter_width: int = 51, # path_enhance filter width
        threshold: float = 0.3, # enhanced matrix threshold
        progress_callback=None,
    ) -> List[Tuple[int, int, float]]:
        """Find diagonal patterns using librosa recurrence_matrix and path_enhance.

        This method uses librosa's recurrence analysis which is specifically
        designed for music structure analysis. It provides cleaner diagonal
        patterns compared to raw similarity matrix analysis.

        Args:
            min_duration: Minimum loop duration in seconds
            k: Number of nearest neighbors for recurrence matrix
            width: Width of the recurrence band
            n_filters: Number of filters for path enhancement
            filter_width: Width of the path enhancement filter
            threshold: Minimum threshold for enhanced matrix values
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            List of (start_chunk, end_chunk, score) tuples
        """
        embeddings = self.extract_embeddings(progress_callback)
        n_chunks = embeddings.shape[0]

        if progress_callback:
            progress_callback(0, 1, "computing_recurrence")

        # Compute recurrence matrix using librosa
        # Using cosine similarity with affinity mode for soft edges
        rec = librosa.segment.recurrence_matrix(
            embeddings.T,  # librosa expects (features, time)
            k=k,
            width=width,
            metric='cosine',
            sym=True,
            mode='affinity',
            self=True
        )

        if progress_callback:
            progress_callback(0, 1, "enhancing_paths")

        # Apply path enhancement to make diagonal structures clearer
        rec_enhanced = librosa.segment.path_enhance(
            rec,
            n=filter_width,
            window='hann',
            n_filters=n_filters,
            zero_mean=True
        )

        # Normalize to 0-1 range
        if rec_enhanced.max() > rec_enhanced.min():
            rec_enhanced = (rec_enhanced - rec_enhanced.min()) / (rec_enhanced.max() - rec_enhanced.min())

        min_chunks = int(min_duration / self.chunk_duration)
        candidates = []

        # Scan diagonals (offset from main diagonal)
        for offset in range(min_chunks, n_chunks):
            diag = np.diag(rec_enhanced, k=offset)

            if len(diag) == 0:
                continue

            # Find regions above threshold
            above_thresh = diag > threshold

            if not np.any(above_thresh):
                continue

            # Find contiguous high-value regions
            high_vals = diag[above_thresh]
            if len(high_vals) > 0:
                avg_score = np.mean(high_vals)
                max_score = np.max(high_vals)

                # Find best starting point (maximum value position)
                best_idx = np.argmax(diag)

                # Weight by both average and max score
                combined_score = 0.6 * avg_score + 0.4 * max_score
                candidates.append((best_idx, best_idx + offset, combined_score))

        # Also look for strong off-diagonal peaks
        for i in range(n_chunks):
            for j in range(i + min_chunks, n_chunks):
                if rec_enhanced[i, j] > threshold * 1.2:  # Higher threshold for individual points
                    candidates.append((i, j, float(rec_enhanced[i, j])))

        # Remove duplicates and sort by score
        unique_candidates = {}
        for start, end, score in candidates:
            key = (start, end)
            if key not in unique_candidates or unique_candidates[key] < score:
                unique_candidates[key] = score

        sorted_candidates = [
            (start, end, score)
            for (start, end), score in unique_candidates.items()
        ]
        sorted_candidates.sort(key=lambda x: x[2], reverse=True)

        return sorted_candidates

    def chunk_to_samples(self, chunk_idx: int) -> int:
        """Convert chunk index to sample index."""
        chunk_samples = int(self.chunk_duration * self.sr)
        return chunk_idx * chunk_samples

    def _find_zero_crossing(self, sample_idx: int, window: int = 100) -> int:
        """Find nearest zero crossing to sample index."""
        start = max(0, sample_idx - window)
        end = min(len(self.audio), sample_idx + window)

        segment = self.audio[start:end]

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]

        if len(zero_crossings) == 0:
            return sample_idx

        # Find nearest to center
        center = sample_idx - start
        nearest_idx = zero_crossings[np.argmin(np.abs(zero_crossings - center))]

        return start + nearest_idx

    def refine_loop_points(
        self,
        start_sample: int,
        end_sample: int,
        search_window: Optional[int] = None,
    ) -> Tuple[int, int, float]:
        """Refine loop points using cross-correlation.

        Args:
            start_sample: Initial start sample
            end_sample: Initial end sample
            search_window: Search window in samples (defaults to ~chunk duration)

        Returns:
            (refined_start, refined_end, correlation_score)
        """
        if self.audio.size == 0 or self.sr <= 0:
            return int(start_sample), int(end_sample), 0.0

        start_sample = int(np.clip(start_sample, 0, self.audio.size - 1))
        end_sample = int(np.clip(end_sample, 0, self.audio.size - 1))

        if search_window is None:
            search_window = max(500, int(self.chunk_duration * self.sr))
        search_window = int(max(1, search_window))

        # Window size for comparison (about 50ms)
        window_size = max(32, int(0.05 * self.sr))
        half = window_size // 2

        ref_start = max(0, start_sample - half)
        ref_end = min(self.audio.size, start_sample + half)
        reference = self.audio[ref_start:ref_end]
        if reference.size < max(16, window_size // 4):
            return start_sample, end_sample, 0.0

        half_ref = int(reference.size // 2)
        ref0 = reference.astype(np.float64, copy=False) - float(np.mean(reference))
        ref_norm = float(np.linalg.norm(ref0))
        if not np.isfinite(ref_norm) or ref_norm < 1e-8:
            return start_sample, end_sample, 0.0

        search_start = max(0, end_sample - search_window)
        search_end = min(self.audio.size, end_sample + search_window + ref0.size)
        search_segment = self.audio[search_start:search_end]
        if search_segment.size < ref0.size:
            return start_sample, end_sample, 0.0

        y = search_segment.astype(np.float64, copy=False)
        numerators = np.correlate(y, ref0, mode="valid").astype(np.float64, copy=False)

        # Denominator per offset: ||ref0|| * ||y - mean(y)||
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
        best_offset = search_start + best_idx
        best_center = int(np.clip(best_offset + half_ref, 0, self.audio.size - 1))

        refined_start = self._find_zero_crossing(start_sample)
        refined_end = self._find_zero_crossing(best_center)

        return int(refined_start), int(refined_end), best_corr

    def find_loop_points(
        self,
        min_duration: float = 10.0,
        min_loop_fraction: float = 0.35,
        n_candidates: int = 20,
        similarity_threshold: float = 0.80,
        use_librosa_recurrence: bool = False,
        recurrence_k: int = 7,
        recurrence_width: int = 3,
        path_enhance_filters: int = 7,
        path_enhance_width: int = 51,
        progress_callback=None,
    ) -> List[DeepLoopCandidate]:
        """Find best loop points in audio using deep learning embeddings.

        Args:
            min_duration: Minimum loop duration in seconds
            min_loop_fraction: Minimum loop as fraction of total duration
            n_candidates: Number of candidates to return
            similarity_threshold: Minimum similarity threshold
            use_librosa_recurrence: Use librosa recurrence_matrix + path_enhance
            recurrence_k: k parameter for recurrence_matrix (nearest neighbors)
            recurrence_width: width parameter for recurrence_matrix
            path_enhance_filters: n_filters parameter for path_enhance
            path_enhance_width: filter_width parameter for path_enhance
            progress_callback: Optional callback(current, total, stage) for progress

        Returns:
            List of DeepLoopCandidate objects sorted by score
        """
        # Apply minimum fraction constraint
        min_duration = max(min_duration, self.duration * min_loop_fraction)

        # Choose pattern detection method
        if use_librosa_recurrence:
            # Use librosa recurrence_matrix + path_enhance
            patterns = self.find_diagonal_patterns_librosa(
                min_duration=min_duration,
                k=recurrence_k,
                width=recurrence_width,
                n_filters=path_enhance_filters,
                filter_width=path_enhance_width,
                threshold=0.3,
                progress_callback=progress_callback,
            )

            if not patterns:
                # Lower threshold and try again
                patterns = self.find_diagonal_patterns_librosa(
                    min_duration=min_duration,
                    k=recurrence_k,
                    width=recurrence_width,
                    n_filters=path_enhance_filters,
                    filter_width=path_enhance_width,
                    threshold=0.15,
                    progress_callback=progress_callback,
                )
            algorithm = "ast_librosa"
        else:
            # Use original diagonal pattern detection
            patterns = self.find_diagonal_patterns(
                min_duration=min_duration,
                threshold=similarity_threshold,
                progress_callback=progress_callback,
            )

            if not patterns:
                # Lower threshold and try again
                patterns = self.find_diagonal_patterns(
                    min_duration=min_duration,
                    threshold=similarity_threshold * 0.8,
                    progress_callback=progress_callback,
                )
            algorithm = "ast"

        if progress_callback:
            progress_callback(0, 1, "finding_patterns")

        candidates = []
        seen_ranges = []

        for start_chunk, end_chunk, sim_score in patterns[:n_candidates * 2]:
            # Convert to samples
            start_sample = self.chunk_to_samples(start_chunk)
            end_sample = self.chunk_to_samples(end_chunk)

            # Refine with correlation
            refined_start, refined_end, corr_score = self.refine_loop_points(
                start_sample, end_sample
            )

            # Skip if too similar to existing candidate
            is_duplicate = False
            for seen_start, seen_end in seen_ranges:
                if (abs(refined_start - seen_start) < self.sr * 0.5 and
                    abs(refined_end - seen_end) < self.sr * 0.5):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_ranges.append((refined_start, refined_end))

            # Combined score
            combined_score = sim_score * 0.7 + max(0, corr_score) * 0.3

            candidates.append(DeepLoopCandidate(
                loop_start=refined_start,
                loop_end=refined_end,
                score=combined_score,
                similarity_score=sim_score,
                algorithm=algorithm,
            ))

            if len(candidates) >= n_candidates:
                break

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def samples_to_time(self, samples: int) -> str:
        """Convert samples to MM:SS.mmm format."""
        seconds = samples / self.sr
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"


class BeatAlignedAnalyzer:
    """Beat/downbeat-aware loop analyzer using madmom.

    This analyzer detects downbeats (bar boundaries) and snaps loop points
    to musically meaningful positions. This is optional and requires madmom.
    """

    def __init__(self, file_path: str, sr: int = 44100):
        """Initialize beat analyzer.

        Args:
            file_path: Path to audio file
            sr: Target sample rate for analysis
        """
        self.file_path = file_path
        self.sr = sr
        self._downbeats: Optional[np.ndarray] = None
        self._bar_times: Optional[np.ndarray] = None
        self._madmom_available = self._check_madmom()

    @staticmethod
    def _check_madmom() -> bool:
        """Check if madmom is available."""
        try:
            import madmom
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if beat alignment is available."""
        return self._madmom_available

    def detect_downbeats(self, progress_callback=None) -> np.ndarray:
        """Detect downbeats using madmom's DBN beat tracker.

        Returns:
            Array of (time, beat_position) where beat_position 1 = downbeat
        """
        if self._downbeats is not None:
            return self._downbeats

        if not self._madmom_available:
            raise ImportError("madmom is required for beat detection")

        if progress_callback:
            progress_callback(0, 1, "detecting_beats")

        from madmom.features.downbeats import (
            RNNDownBeatProcessor,
            DBNDownBeatTrackingProcessor
        )

        # Use RNN + DBN for robust downbeat detection
        proc = RNNDownBeatProcessor()
        act = proc(self.file_path)

        # DBN processor with common time signatures
        dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4],  # 3/4 and 4/4 time
            fps=100
        )
        try:
            self._downbeats = dbn(act)
        except ValueError:
            # NumPy 2.x compatibility: madmom's DBNDownBeatTrackingProcessor
            # uses `np.asarray(results)` where `results` is a list of
            # (ndarray, float) tuples, which raises on newer NumPy.
            self._downbeats = self._process_dbn_safe(dbn, act)

        return self._downbeats

    @staticmethod
    def _process_dbn_safe(dbn, activations: np.ndarray) -> np.ndarray:
        """Fallback DBN decoding compatible with NumPy 2.x.

        Mirrors `madmom.features.downbeats.DBNDownBeatTrackingProcessor.process`
        but avoids `np.asarray(results)` on ragged sequences.
        """
        # Based on madmom 0.16.1 implementation
        first = 0
        if getattr(dbn, "threshold", None):
            idx = np.nonzero(activations >= dbn.threshold)[0]
            if idx.any():
                first = max(first, int(np.min(idx)))
                last = min(len(activations), int(np.max(idx)) + 1)
            else:
                last = first
            activations = activations[first:last]

        if not activations.any():
            return np.empty((0, 2))

        # Decode with each HMM and pick the best log-probability.
        results = [hmm.viterbi(activations) for hmm in dbn.hmms]
        best = max(range(len(results)), key=lambda i: float(results[i][1]))
        path, _ = results[best]

        st = dbn.hmms[best].transition_model.state_space
        om = dbn.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1

        if getattr(dbn, "correct", False):
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

        return np.vstack(((beats + first) / float(dbn.fps), beat_numbers[beats])).T

    def get_bar_boundaries(self, progress_callback=None) -> np.ndarray:
        """Get bar (measure) start times.

        Returns:
            Array of bar start times in seconds
        """
        if self._bar_times is not None:
            return self._bar_times

        downbeats = self.detect_downbeats(progress_callback)

        # Filter for downbeats only (beat_position == 1)
        self._bar_times = downbeats[downbeats[:, 1] == 1, 0]

        return self._bar_times

    def snap_to_bar(self, time_seconds: float, progress_callback=None) -> float:
        """Snap a time position to the nearest bar boundary.

        Args:
            time_seconds: Time in seconds to snap

        Returns:
            Nearest bar boundary time in seconds
        """
        bar_times = self.get_bar_boundaries(progress_callback)

        if len(bar_times) == 0:
            return time_seconds

        idx = np.argmin(np.abs(bar_times - time_seconds))
        return float(bar_times[idx])

    def snap_sample_to_bar(self, sample: int, sr: int, progress_callback=None) -> int:
        """Snap a sample position to the nearest bar boundary.

        Args:
            sample: Sample index to snap
            sr: Sample rate of the audio

        Returns:
            Nearest bar boundary as sample index
        """
        time_seconds = sample / sr
        snapped_time = self.snap_to_bar(time_seconds, progress_callback)
        return int(snapped_time * sr)


class BeatAlignedLoopAnalyzer(DeepLoopAnalyzer):
    """Extended DeepLoopAnalyzer with optional beat alignment."""

    def __init__(
        self,
        file_path: str,
        sr: int = 16000,
        chunk_duration: float = 2.0,
        batch_size: int = 16,
        use_beat_alignment: bool = False,
    ):
        """Initialize analyzer with optional beat alignment.

        Args:
            file_path: Path to audio file
            sr: Sample rate (16kHz for AST model)
            chunk_duration: Duration of each chunk in seconds
            batch_size: Number of chunks to process at once
            use_beat_alignment: Whether to use beat-aligned loop points
        """
        super().__init__(file_path, sr, chunk_duration, batch_size)
        self.use_beat_alignment = use_beat_alignment

        if use_beat_alignment:
            self._beat_analyzer = BeatAlignedAnalyzer(file_path)
            if not self._beat_analyzer.is_available:
                warnings.warn("madmom not available, beat alignment disabled")
                self.use_beat_alignment = False
        else:
            self._beat_analyzer = None

    def refine_loop_points_beat_aligned(
        self,
        start_sample: int,
        end_sample: int,
        original_sr: int,
        search_window: Optional[int] = None,
        progress_callback=None,
    ) -> Tuple[int, int, float]:
        """Refine loop points with beat alignment.

        First applies cross-correlation refinement, then snaps to bar boundaries.

        Args:
            start_sample: Initial start sample (in model sample rate)
            end_sample: Initial end sample (in model sample rate)
            original_sr: Original audio sample rate
            search_window: Search window for correlation refinement

        Returns:
            (refined_start, refined_end, correlation_score)
        """
        # First, do normal correlation-based refinement
        refined_start, refined_end, corr_score = self.refine_loop_points(
            start_sample, end_sample, search_window
        )

        # Then snap to bar boundaries if beat alignment is enabled
        if self.use_beat_alignment and self._beat_analyzer:
            # Convert from model SR to original SR for beat snapping
            scale = original_sr / self.sr

            start_original = int(refined_start * scale)
            end_original = int(refined_end * scale)

            # Snap to bars
            snapped_start = self._beat_analyzer.snap_sample_to_bar(
                start_original, original_sr, progress_callback
            )
            snapped_end = self._beat_analyzer.snap_sample_to_bar(
                end_original, original_sr, progress_callback
            )

            # Convert back to model SR
            refined_start = int(snapped_start / scale)
            refined_end = int(snapped_end / scale)

        return refined_start, refined_end, corr_score

    def find_loop_points(
        self,
        min_duration: float = 10.0,
        min_loop_fraction: float = 0.35,
        n_candidates: int = 20,
        similarity_threshold: float = 0.80,
        use_librosa_recurrence: bool = False,
        recurrence_k: int = 7,
        recurrence_width: int = 3,
        path_enhance_filters: int = 7,
        path_enhance_width: int = 51,
        progress_callback=None,
    ) -> List[DeepLoopCandidate]:
        """Find loop points with optional beat alignment.

        Same as parent class but uses beat-aligned refinement if enabled.
        """
        # If beat alignment not enabled, use parent implementation
        if not self.use_beat_alignment:
            return super().find_loop_points(
                min_duration=min_duration,
                min_loop_fraction=min_loop_fraction,
                n_candidates=n_candidates,
                similarity_threshold=similarity_threshold,
                use_librosa_recurrence=use_librosa_recurrence,
                recurrence_k=recurrence_k,
                recurrence_width=recurrence_width,
                path_enhance_filters=path_enhance_filters,
                path_enhance_width=path_enhance_width,
                progress_callback=progress_callback,
            )

        # Detect beats first
        if progress_callback:
            progress_callback(0, 1, "detecting_beats")

        try:
            self._beat_analyzer.detect_downbeats(progress_callback)
        except Exception as e:
            warnings.warn(f"Beat detection failed: {e}, falling back to normal analysis")
            return super().find_loop_points(
                min_duration=min_duration,
                min_loop_fraction=min_loop_fraction,
                n_candidates=n_candidates,
                similarity_threshold=similarity_threshold,
                use_librosa_recurrence=use_librosa_recurrence,
                recurrence_k=recurrence_k,
                recurrence_width=recurrence_width,
                path_enhance_filters=path_enhance_filters,
                path_enhance_width=path_enhance_width,
                progress_callback=progress_callback,
            )

        # Apply minimum fraction constraint
        min_duration = max(min_duration, self.duration * min_loop_fraction)

        # Choose pattern detection method
        if use_librosa_recurrence:
            patterns = self.find_diagonal_patterns_librosa(
                min_duration=min_duration,
                k=recurrence_k,
                width=recurrence_width,
                n_filters=path_enhance_filters,
                filter_width=path_enhance_width,
                threshold=0.3,
                progress_callback=progress_callback,
            )

            if not patterns:
                patterns = self.find_diagonal_patterns_librosa(
                    min_duration=min_duration,
                    k=recurrence_k,
                    width=recurrence_width,
                    n_filters=path_enhance_filters,
                    filter_width=path_enhance_width,
                    threshold=0.15,
                    progress_callback=progress_callback,
                )
            algorithm = "ast_librosa_beat"
        else:
            patterns = self.find_diagonal_patterns(
                min_duration=min_duration,
                threshold=similarity_threshold,
                progress_callback=progress_callback,
            )

            if not patterns:
                patterns = self.find_diagonal_patterns(
                    min_duration=min_duration,
                    threshold=similarity_threshold * 0.8,
                    progress_callback=progress_callback,
                )
            algorithm = "ast_beat"

        if progress_callback:
            progress_callback(0, 1, "finding_patterns")

        candidates = []
        seen_ranges = []

        # Get original sample rate for beat snapping
        # Load briefly to get SR
        _, original_sr = librosa.load(self.file_path, sr=None, duration=0.1)

        for start_chunk, end_chunk, sim_score in patterns[:n_candidates * 2]:
            start_sample = self.chunk_to_samples(start_chunk)
            end_sample = self.chunk_to_samples(end_chunk)

            # Use beat-aligned refinement
            refined_start, refined_end, corr_score = self.refine_loop_points_beat_aligned(
                start_sample, end_sample, original_sr, progress_callback=progress_callback
            )

            # Skip duplicates
            is_duplicate = False
            for seen_start, seen_end in seen_ranges:
                if (abs(refined_start - seen_start) < self.sr * 0.5 and
                    abs(refined_end - seen_end) < self.sr * 0.5):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_ranges.append((refined_start, refined_end))

            combined_score = sim_score * 0.7 + max(0, corr_score) * 0.3

            candidates.append(DeepLoopCandidate(
                loop_start=refined_start,
                loop_end=refined_end,
                score=combined_score,
                similarity_score=sim_score,
                algorithm=algorithm,
            ))

            if len(candidates) >= n_candidates:
                break

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates


def analyze_audio_deep(
    file_path: str,
    chunk_duration: float = 2.0,
    batch_size: int = 16,
    use_beat_alignment: bool = False,
    **kwargs
) -> List[DeepLoopCandidate]:
    """Convenience function to analyze audio file with deep learning.

    Args:
        file_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds (default: 2.0)
        batch_size: Number of chunks to process at once (default: 16)
        use_beat_alignment: Use beat-aligned loop detection (requires madmom)
        **kwargs: Arguments passed to find_loop_points

    Returns:
        List of DeepLoopCandidate objects
    """
    if use_beat_alignment:
        analyzer = BeatAlignedLoopAnalyzer(
            file_path,
            chunk_duration=chunk_duration,
            batch_size=batch_size,
            use_beat_alignment=True,
        )
    else:
        analyzer = DeepLoopAnalyzer(
            file_path,
            chunk_duration=chunk_duration,
            batch_size=batch_size,
        )
    return analyzer.find_loop_points(**kwargs)
