"""Music Looper Desktop App - Single file, no server."""

import io
import json
import os
import base64
import struct
import wave
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import webview
from mutagen.oggvorbis import OggVorbis
from webview.dom import DOMEventHandler

from deep_analyzer import DeepLoopAnalyzer

# Global window reference for API access
window = None


class Api:
    """Python API exposed to JavaScript via PyWebView bridge."""

    def __init__(self):
        self._deep_analyzer: DeepLoopAnalyzer | None = None
        self._current_file: str | None = None
        self._audio: np.ndarray | None = None
        self._sr: int | None = None

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
        """Analyze audio file for loop points using AST deep learning."""
        self._current_file = file_path

        try:
            return self._analyze(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze(self, file_path: str) -> dict:
        """Deep learning-based analysis (Audio Spectrogram Transformer)."""
        self._deep_analyzer = DeepLoopAnalyzer(file_path)
        candidates = self._deep_analyzer.find_loop_points()

        # Load audio for playback at original sample rate
        self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
        if len(self._audio.shape) == 1:
            self._audio = self._audio.reshape(-1, 1)
        else:
            self._audio = self._audio.T  # (samples, channels)

        # Scale sample indices from 16kHz (model) to original sample rate
        scale_factor = self._sr / self._deep_analyzer.sr

        loops = []
        for i, cand in enumerate(candidates):
            # Scale sample positions to original sample rate
            start_sample = int(cand.loop_start * scale_factor)
            end_sample = int(cand.loop_end * scale_factor)

            start_time = self._samples_to_ftime(start_sample, self._sr)
            end_time = self._samples_to_ftime(end_sample, self._sr)
            duration_samples = end_sample - start_sample
            duration_time = self._samples_to_ftime(duration_samples, self._sr)

            loops.append({
                "index": i,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_time,
                "score": float(cand.score),
                "similarity_score": float(cand.similarity_score),
            })

        duration = len(self._audio) / self._sr

        return {
            "success": True,
            "duration": duration,
            "sample_rate": self._sr,
            "loops": loops,
        }

    def _samples_to_ftime(self, samples: int, sr: int) -> str:
        """Convert samples to MM:SS.mmm format."""
        seconds = samples / sr
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"

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
            parts = [intro]
            for _ in range(loop_count):
                parts.append(loop)
            parts.append(outro)

            extended = np.concatenate(parts, axis=0)

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
