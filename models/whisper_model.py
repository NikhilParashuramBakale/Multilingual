from __future__ import annotations
import time
from pathlib import Path
from .base import ASRModel, ASRResult, ModelUnavailable
from .device import preferred_device

class WhisperMedium(ASRModel):
    name = "openai/whisper-medium"
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
