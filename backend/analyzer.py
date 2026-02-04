"""Advanced Loop Detection using Self-Similarity Matrix (SSM).

This module provides an alternative loop detection algorithm that uses:
1. Self-Similarity Matrix (SSM) - finds repeating structures in audio
2. Diagonal detection - identifies loop candidates from SSM patterns
3. Cross-correlation refinement - precise sample-level alignment
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LoopCandidate:
    """Loop point candidate with scoring information."""
    loop_start: int  # in samples
    loop_end: int    # in samples
    score: float
    ssm_score: float
    correlation_score: float

    @property
    def duration_samples(self) -> int:
        return self.loop_end - self.loop_start


class SSMLoopAnalyzer:
    """Self-Similarity Matrix based loop finder."""

    def __init__(
        self,
        file_path: str,
        sr: Optional[int] = None,
        hop_length: int = 512,
    ):
        """Initialize analyzer with audio file.

        Args:
            file_path: Path to audio file
            sr: Sample rate (None for original)
            hop_length: Hop length for STFT analysis
        """
        self.file_path = file_path
        self.hop_length = hop_length

        # Load audio
        self.audio, self.sr = librosa.load(file_path, sr=sr, mono=True)
        self.duration = len(self.audio) / self.sr

        # Compute features lazily
        self._chroma = None
        self._mfcc = None
        self._ssm = None

    @property
    def chroma(self) -> np.ndarray:
        """Compute chroma features."""
        if self._chroma is None:
            self._chroma = librosa.feature.chroma_cqt(
                y=self.audio,
                sr=self.sr,
                hop_length=self.hop_length
            )
        return self._chroma

    @property
    def mfcc(self) -> np.ndarray:
        """Compute MFCC features."""
        if self._mfcc is None:
            self._mfcc = librosa.feature.mfcc(
                y=self.audio,
                sr=self.sr,
                n_mfcc=20,
                hop_length=self.hop_length
            )
        return self._mfcc

    @property
    def ssm(self) -> np.ndarray:
        """Compute Self-Similarity Matrix using combined features."""
        if self._ssm is None:
            # Combine chroma and MFCC for richer representation
            chroma_norm = librosa.util.normalize(self.chroma, axis=0)
            mfcc_norm = librosa.util.normalize(self.mfcc, axis=0)

            # Weight chroma more for harmonic content
            features = np.vstack([
                chroma_norm * 2.0,  # Harmonic content (weighted)
                mfcc_norm * 1.0     # Timbral content
            ])

            # Compute recurrence matrix (SSM)
            self._ssm = librosa.segment.recurrence_matrix(
                features,
                mode='affinity',
                metric='cosine',
                sparse=False
            )
        return self._ssm

    def frames_to_samples(self, frames: int) -> int:
        """Convert frame index to sample index."""
        return frames * self.hop_length

    def samples_to_frames(self, samples: int) -> int:
        """Convert sample index to frame index."""
        return samples // self.hop_length

    def find_diagonal_segments(
        self,
        min_duration: float = 5.0,
        max_duration: Optional[float] = None,
        threshold: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """Find diagonal segments in SSM that indicate repetition.

        Args:
            min_duration: Minimum loop duration in seconds
            max_duration: Maximum loop duration in seconds (None for no limit)
            threshold: Minimum similarity threshold

        Returns:
            List of (start_frame, end_frame, score) tuples
        """
        ssm = self.ssm
        n_frames = ssm.shape[0]

        min_frames = int(min_duration * self.sr / self.hop_length)
        max_frames = n_frames if max_duration is None else int(max_duration * self.sr / self.hop_length)

        segments = []

        # Scan diagonals (offset from main diagonal)
        for offset in range(min_frames, min(max_frames, n_frames)):
            # Get diagonal values
            diag = np.diag(ssm, k=offset)

            if len(diag) == 0:
                continue

            # Find regions above threshold
            above_thresh = diag > threshold

            # Find contiguous regions
            changes = np.diff(above_thresh.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if above_thresh[0]:
                starts = np.concatenate([[0], starts])
            if above_thresh[-1]:
                ends = np.concatenate([ends, [len(diag)]])

            # Score each segment
            for start, end in zip(starts, ends):
                segment_len = end - start
                if segment_len >= min_frames // 4:  # At least 1/4 of min duration
                    score = np.mean(diag[start:end])
                    # loop_start is at 'start', loop_end is at 'start + offset'
                    segments.append((start, start + offset, score))

        # Sort by score descending
        segments.sort(key=lambda x: x[2], reverse=True)

        return segments

    def refine_with_correlation(
        self,
        start_frame: int,
        end_frame: int,
        search_window: int = 100,
    ) -> Tuple[int, int, float]:
        """Refine loop points using cross-correlation.

        Args:
            start_frame: Initial start frame
            end_frame: Initial end frame
            search_window: Search window in samples for refinement

        Returns:
            (refined_start_sample, refined_end_sample, correlation_score)
        """
        start_sample = self.frames_to_samples(start_frame)
        end_sample = self.frames_to_samples(end_frame)

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
        best_offset = 0

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

    def find_loop_points(
        self,
        min_duration: float = 10.0,
        max_duration: Optional[float] = None,
        min_loop_fraction: float = 0.35,
        n_candidates: int = 20,
        ssm_threshold: float = 0.4,
    ) -> List[LoopCandidate]:
        """Find best loop points in audio.

        Args:
            min_duration: Minimum loop duration in seconds
            max_duration: Maximum loop duration in seconds
            min_loop_fraction: Minimum loop as fraction of total duration
            n_candidates: Number of candidates to return
            ssm_threshold: SSM similarity threshold

        Returns:
            List of LoopCandidate objects sorted by score
        """
        # Apply minimum fraction constraint
        min_duration = max(min_duration, self.duration * min_loop_fraction)

        # Find diagonal segments in SSM
        segments = self.find_diagonal_segments(
            min_duration=min_duration,
            max_duration=max_duration,
            threshold=ssm_threshold,
        )

        if not segments:
            # Fallback: lower threshold and try again
            segments = self.find_diagonal_segments(
                min_duration=min_duration,
                max_duration=max_duration,
                threshold=ssm_threshold * 0.5,
            )

        # Refine top candidates
        candidates = []
        seen_ranges = []  # To avoid duplicates

        for start_frame, end_frame, ssm_score in segments[:n_candidates * 2]:
            # Refine with correlation
            start_sample, end_sample, corr_score = self.refine_with_correlation(
                start_frame, end_frame
            )

            # Skip if too similar to existing candidate
            is_duplicate = False
            for seen_start, seen_end in seen_ranges:
                if (abs(start_sample - seen_start) < self.sr * 0.5 and
                    abs(end_sample - seen_end) < self.sr * 0.5):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_ranges.append((start_sample, end_sample))

            # Combined score (weighted)
            combined_score = ssm_score * 0.6 + max(0, corr_score) * 0.4

            candidates.append(LoopCandidate(
                loop_start=start_sample,
                loop_end=end_sample,
                score=combined_score,
                ssm_score=ssm_score,
                correlation_score=corr_score,
            ))

            if len(candidates) >= n_candidates:
                break

        # Sort by combined score
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def samples_to_time(self, samples: int) -> str:
        """Convert samples to MM:SS.mmm format."""
        seconds = samples / self.sr
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"


def analyze_audio(file_path: str, **kwargs) -> List[LoopCandidate]:
    """Convenience function to analyze audio file.

    Args:
        file_path: Path to audio file
        **kwargs: Arguments passed to find_loop_points

    Returns:
        List of LoopCandidate objects
    """
    analyzer = SSMLoopAnalyzer(file_path)
    return analyzer.find_loop_points(**kwargs)
