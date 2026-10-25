"""Wrapper for ``bodhan-ai/indic-transcribe-core`` via Hugging Face transformers.

This model is an Indic-specific Wav2Vec2 CTC model. It is loaded through the
Hugging Face ``transformers`` automatic-speech-recognition pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from models._audio import load_audio_16k_mono
from models._base import ASRModel, ASRResult


class IndicTranscribeCoreWrapper(ASRModel):
    """Wrapper around ``bodhan-ai/indic-transcribe-core``."""

    MODEL_ID = "bodhan-ai/indic-transcribe-core"
    from .language_support import SCHEDULED_LANGUAGES, ENGLISH
    supported_languages = frozenset(item.code for item in SCHEDULED_LANGUAGES + (ENGLISH,))
    supports_auto_detection = True
    supports_code_switching = True

    def __init__(self) -> None:
        super().__init__(
            model_id=self.MODEL_ID,
            display_name="Indic-Transcribe-core",
        )
        self._pipeline = None

    def load(self, device: Optional[str] = None, **kwargs: Any) -> None:
        """Load the model via the HF transformers pipeline."""
        from transformers import pipeline

        self._device = device or self.detect_device(prefer_cuda=True)
        try:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_ID,
                device=self._device if self._device == "cuda" else -1,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load '{self.MODEL_ID}': {exc}. "
                "Ensure you have accepted the model access conditions on Hugging Face "
                "and that transformers/torch are installed."
            ) from exc
        self._loaded = True

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ASRResult:
        if not self._loaded:
            raise RuntimeError("Indic-Transcribe-core is not loaded. Call load() first.")
        audio = load_audio_16k_mono(audio_path)
        start = time.perf_counter()
        try:
            result = self._pipeline(audio, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Indic-Transcribe-core transcription failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        text = (result.get("text") if isinstance(result, dict) else str(result)).strip()
        return ASRResult(
            text=text,
            language=None,  # model does not expose detected language
            processing_time=elapsed,
            device=self._device,
            model_name=self.display_name,
            raw_output=result if isinstance(result, dict) else {"text": result},
        )