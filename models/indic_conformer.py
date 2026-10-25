"""Optional wrapper for ``ai4bharat/indic-conformer-600m-multilingual``.

This is a multilingual Conformer model from AI4Bharat. It is loaded through the
Hugging Face ``transformers`` automatic-speech-recognition pipeline and supports
language codes via the ``language`` keyword argument.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from models._audio import load_audio_16k_mono
from models._base import ASRModel, ASRResult
from .language_support import ENGLISH, SCHEDULED_LANGUAGES


class IndicConformerWrapper(ASRModel):
    """Wrapper around ``ai4bharat/indic-conformer-600m-multilingual``."""

    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    from .language_support import SCHEDULED_LANGUAGES, ENGLISH
    supported_languages = frozenset(item.code for item in SCHEDULED_LANGUAGES + (ENGLISH,))
    supports_auto_detection = False
    supports_code_switching = True

    def __init__(self) -> None:
        super().__init__(
            model_id=self.MODEL_ID,
            display_name="Indic Conformer 600M",
        )
        self._pipeline = None

    def load(self, device: Optional[str] = None, **kwargs: Any) -> None:
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
            raise RuntimeError("Indic Conformer 600M is not loaded. Call load() first.")
        audio = load_audio_16k_mono(audio_path)
        # The pipeline accepts a language kwarg for multilingual models.
        call_kwargs: Dict[str, Any] = {}
        if language:
            call_kwargs["language"] = language
        call_kwargs.update(kwargs)
        start = time.perf_counter()
        try:
            result = self._pipeline(audio, **call_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Indic Conformer transcription failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        text = (result.get("text") if isinstance(result, dict) else str(result)).strip()
        return ASRResult(
            text=text,
            language=language,
            processing_time=elapsed,
            device=self._device,
            model_name=self.display_name,
            raw_output=result if isinstance(result, dict) else {"text": result},
        )


class IndicConformer:
    """Self-loading adapter using AI4Bharat's documented AutoModel call."""
    name = "ai4bharat/indic-conformer-600m-multilingual"
    supported_languages = frozenset(item.code for item in SCHEDULED_LANGUAGES + (ENGLISH,))
    supports_auto_detection = False
    supports_code_switching = True
    def __init__(self) -> None:
        from models.device import preferred_device
        self.model = None
        self.device = preferred_device(min_free_gib=5)
    def _load(self) -> None:
        if self.model is not None: return
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.name, trust_remote_code=True)
            if self.device == "cuda": self.model = self.model.to("cuda")
            self.model.eval()
        except Exception as exc:
            from models.base import ModelUnavailable
            raise ModelUnavailable("IndicConformer is gated: accept conditions at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual, run `hf auth login`, and install documented ONNX dependencies. Details: " + str(exc)) from exc
    def transcribe(self, audio_path, language="kn"):
        from models.base import ASRResult, ModelUnavailable
        self._load(); started = time.perf_counter()
        try:
            import torch, torchaudio
            wav, rate = torchaudio.load(str(audio_path)); wav = torch.mean(wav, dim=0, keepdim=True)
            if rate != 16000: wav = torchaudio.transforms.Resample(rate, 16000)(wav)
            with torch.inference_mode(): text = self.model(wav.to(self.device), language or "kn", "rnnt")
        except Exception as exc:
            raise ModelUnavailable(f"IndicConformer inference failed: {exc}") from exc
        return ASRResult(self.name, str(text).strip(), time.perf_counter()-started, self.device)
