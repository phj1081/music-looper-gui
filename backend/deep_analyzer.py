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

    def __post_init__(self):
        # Ensure all values are Python native types for JSON serialization
        self.loop_start = int(self.loop_start)
        self.loop_end = int(self.loop_end)
        self.score = float(self.score)
        self.similarity_score = float(self.similarity_score)

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

    def extract_embeddings(self) -> np.ndarray:
        """Extract embeddings for each audio chunk using batch processing.

        Uses batch processing to significantly improve performance:
        - Groups multiple chunks together for a single forward pass
        - Uses float16 precision on GPU/MPS
        - Reduces GPU/CPU data transfer overhead

        Returns:
            (n_chunks, embedding_dim) array of embeddings
        """
        if self._embeddings is not None:
            return self._embeddings

        import torch

        model, processor, device, use_half = self._load_model()

        # Calculate chunk parameters
        chunk_samples = int(self.chunk_duration * self.sr)
        n_chunks = len(self.audio) // chunk_samples

        if n_chunks == 0:
            # Audio too short, use entire audio as one chunk
            n_chunks = 1
            chunk_samples = len(self.audio)

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

        with torch.no_grad():
            # Process in batches
            for batch_start in range(0, n_chunks, self.batch_size):
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

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute cosine similarity matrix between all chunk embeddings.

        Returns:
            (n_chunks, n_chunks) similarity matrix
        """
        if self._similarity_matrix is not None:
            return self._similarity_matrix

        embeddings = self.extract_embeddings()

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
    ) -> List[Tuple[int, int, float]]:
        """Find diagonal patterns in similarity matrix indicating loops.

        Args:
            min_duration: Minimum loop duration in seconds
            threshold: Minimum similarity threshold

        Returns:
            List of (start_chunk, end_chunk, score) tuples
        """
        sim_matrix = self.compute_similarity_matrix()
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
        search_window: int = 500,
    ) -> Tuple[int, int, float]:
        """Refine loop points using cross-correlation.

        Args:
            start_sample: Initial start sample
            end_sample: Initial end sample
            search_window: Search window in samples

        Returns:
            (refined_start, refined_end, correlation_score)
        """
        # Window size for comparison (about 50ms)
        window_size = int(0.05 * self.sr)

        # Get reference window at start
        ref_start = max(0, start_sample - window_size // 2)
        ref_end = min(len(self.audio), start_sample + window_size // 2)
        reference = self.audio[ref_start:ref_end]

        if len(reference) < window_size // 2:
            return start_sample, end_sample, 0.0

        # Search around end point
        best_corr = -1.0
        best_offset = end_sample

        search_start = max(0, end_sample - search_window)
        search_end = min(len(self.audio) - len(reference), end_sample + search_window)

        for offset in range(search_start, search_end):
            candidate = self.audio[offset:offset + len(reference)]
            if len(candidate) != len(reference):
                continue

            # Normalized cross-correlation
            corr = np.corrcoef(reference, candidate)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_offset = offset

        # Find zero crossing near the best points
        refined_start = self._find_zero_crossing(start_sample)
        refined_end = self._find_zero_crossing(best_offset)

        return refined_start, refined_end, best_corr

    def find_loop_points(
        self,
        min_duration: float = 10.0,
        min_loop_fraction: float = 0.35,
        n_candidates: int = 20,
        similarity_threshold: float = 0.80,
    ) -> List[DeepLoopCandidate]:
        """Find best loop points in audio using deep learning embeddings.

        Args:
            min_duration: Minimum loop duration in seconds
            min_loop_fraction: Minimum loop as fraction of total duration
            n_candidates: Number of candidates to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of DeepLoopCandidate objects sorted by score
        """
        # Apply minimum fraction constraint
        min_duration = max(min_duration, self.duration * min_loop_fraction)

        # Find diagonal patterns
        patterns = self.find_diagonal_patterns(
            min_duration=min_duration,
            threshold=similarity_threshold,
        )

        if not patterns:
            # Lower threshold and try again
            patterns = self.find_diagonal_patterns(
                min_duration=min_duration,
                threshold=similarity_threshold * 0.8,
            )

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


def analyze_audio_deep(
    file_path: str,
    chunk_duration: float = 2.0,
    batch_size: int = 16,
    **kwargs
) -> List[DeepLoopCandidate]:
    """Convenience function to analyze audio file with deep learning.

    Args:
        file_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds (default: 2.0)
        batch_size: Number of chunks to process at once (default: 16)
        **kwargs: Arguments passed to find_loop_points

    Returns:
        List of DeepLoopCandidate objects
    """
    analyzer = DeepLoopAnalyzer(
        file_path,
        chunk_duration=chunk_duration,
        batch_size=batch_size,
    )
    return analyzer.find_loop_points(**kwargs)
