"""Shared contracts; model wrappers deliberately retain independent inference APIs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .language_support import ENGLISH, SCHEDULED_LANGUAGES, WHISPER_ADDITIONAL_LANGUAGES

@dataclass
class ASRResult:
    model: str
    text: str
    elapsed_seconds: float
    device: str
    detected_language: Optional[str] = None
    language_details: Optional[dict] = None
    memory_usage_mb: Optional[float] = None

class ModelUnavailable(RuntimeError):
    """An actionable local setup, access, or capacity issue."""

class ASRModel:
    name: str
    supported_languages: frozenset[str] = frozenset()
    supports_auto_detection: bool = False
    supports_code_switching: bool = False
    def transcribe(self, audio_path: Path, language: Optional[str] = "kn") -> ASRResult:
        raise NotImplementedError
