"""Shared contracts; model wrappers deliberately retain independent inference APIs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ASRResult:
    model: str
    text: str
    elapsed_seconds: float
    device: str
    detected_language: Optional[str] = None
    language_details: Optional[dict] = None

class ModelUnavailable(RuntimeError):
    """An actionable local setup, access, or capacity issue."""

class ASRModel:
    name: str
    def transcribe(self, audio_path: Path, language: Optional[str] = "kn") -> ASRResult:
        raise NotImplementedError
