"""Audio loading utilities shared by all model wrappers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def ensure_ffmpeg() -> None:
    """Raise a clear error if FFmpeg is unavailable."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg is not installed or not found on the PATH. "
            "Install FFmpeg to decode audio files (WAV, MP3, M4A, FLAC)."
        )


def load_audio_16k_mono(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file as a 16 kHz mono float32 numpy array.

    Uses FFmpeg to decode any format Whisper/transformers can read, then
    optionally resamples with ``librosa`` if available.
    """
    ensure_ffmpeg()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Prefer librosa for format-agnostic loading + resampling.
    try:
        import librosa

        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass

    # Fallback: decode with FFmpeg to a temporary 16 kHz mono WAV.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", tmp.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        import soundfile as sf

        data, _ = sf.read(tmp.name, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Return duration in seconds, or None if it cannot be determined."""
    try:
        import librosa

        return float(librosa.get_duration(path=audio_path))
    except Exception:
        pass
    return None


def format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"