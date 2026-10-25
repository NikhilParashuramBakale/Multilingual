"""Whisper model wrapper for the Kannada ASR benchmark.

Uses the official OpenAI Whisper implementation (``openai-whisper`` package).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from models._audio import load_audio_16k_mono
from models._base import ASRModel, ASRResult


class WhisperWrapper(ASRModel):
    """Wrapper around OpenAI Whisper via the ``openai-whisper`` package."""

    SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")

    def __init__(self, model_name: str = "medium") -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Whisper model '{model_name}'. "
                f"Choose from {self.SUPPORTED_MODELS}."
            )
        super().__init__(
            model_id=f"openai/whisper-{model_name}",
            display_name=f"Whisper {model_name}",
        )
        self.model_name = model_name

    def load(self, device: Optional[str] = None, **kwargs: Any) -> None:
        """Load the Whisper model. ``device`` may be 'cuda' or 'cpu'."""
        import whisper

        self._device = device or self.detect_device(prefer_cuda=True)
        try:
            self._model = whisper.load_model(self.model_name, device=self._device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_name}': {exc}"
            ) from exc
        self._loaded = True

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ASRResult:
        if not self._loaded:
            raise RuntimeError("Whisper model is not loaded. Call load() first.")
        start = time.perf_counter()
        try:
            result = self._model.transcribe(
                audio_path,
                language=language,
                fp16=self._device == "cuda",
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        text = (result.get("text") or "").strip()
        detected = result.get("language")
        return ASRResult(
            text=text,
            language=detected,
            processing_time=elapsed,
            device=self._device,
            model_name=self.display_name,
            raw_output=result,
        )