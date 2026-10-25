from __future__ import annotations
import time
from pathlib import Path
from .base import ASRModel, ASRResult, ModelUnavailable
from .device import preferred_device
from .language_support import SCHEDULED_LANGUAGES, WHISPER_ADDITIONAL_LANGUAGES

class WhisperMedium(ASRModel):
    name = "openai/whisper-medium"
    supported_languages = frozenset(
        item.code for item in SCHEDULED_LANGUAGES + WHISPER_ADDITIONAL_LANGUAGES
    )
    supports_auto_detection = True
    supports_code_switching = True
    def __init__(self): self.model = None; self.device = preferred_device()
    def _load(self):
        if self.model is None:
            try:
                import whisper
                self.model = whisper.load_model("medium", device=self.device)
            except Exception as exc:
                raise ModelUnavailable(f"Whisper medium could not load: {exc}. Install FFmpeg and requirements.") from exc
    def transcribe(self, audio_path: Path, language: str | None = "kn") -> ASRResult:
        self._load(); started = time.perf_counter()
        try:
            result = self.model.transcribe(str(audio_path), task="transcribe", language=language, fp16=self.device == "cuda")
        except Exception as exc:
            raise ModelUnavailable(f"Whisper inference failed: {exc}") from exc
        return ASRResult(self.name, result["text"].strip(), time.perf_counter()-started, self.device,
                         result.get("language"))

    def translate_to_english(self, audio_path: Path, language: str | None = None) -> ASRResult:
        """Translate the audio locally with Whisper's separate translation task."""
        self._load(); started = time.perf_counter()
        try:
            result = self.model.transcribe(
                str(audio_path),
                task="translate",
                language=language,
                fp16=self.device == "cuda",
            )
        except Exception as exc:
            raise ModelUnavailable(f"Whisper English translation failed: {exc}") from exc
        return ASRResult(
            f"{self.name} English translation",
            result["text"].strip(),
            time.perf_counter() - started,
            self.device,
            result.get("language"),
        )
