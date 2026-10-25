from __future__ import annotations
import sys, time
from pathlib import Path
from .base import ASRModel, ASRResult, ModelUnavailable
from .device import preferred_device

class IndicTranscribeCore(ASRModel):
    """Uses the repository's documented IndicTranscribe class, not Whisper's API."""
    name = "bodhan-ai/indic-transcribe-core"
    repo_id = "bodhan-ai/indic-transcribe-core"
    def __init__(self): self.asr = None; self.device = preferred_device(min_free_gib=7)
    def _load(self):
        if self.asr is not None: return
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(self.repo_id)  # honors prior `hf auth login`; gated download otherwise fails clearly
            if model_dir not in sys.path: sys.path.insert(0, model_dir)
            from indic_transcribe import IndicTranscribe
            self.asr = IndicTranscribe.from_pretrained(model_dir)
        except Exception as exc:
            raise ModelUnavailable("Indic-Transcribe-core requires Hugging Face login and accepting its gated license at "
                "https://huggingface.co/bodhan-ai/indic-transcribe-core. Run `hf auth login`, then retry. Details: " + str(exc)) from exc
    def transcribe(self, audio_path: Path, language: str | None = "kn") -> ASRResult:
        self._load(); started = time.perf_counter()
        try:
            text, lid = self.asr.transcribe(str(audio_path), lang=language, return_lid=True)
        except Exception as exc:
            raise ModelUnavailable(f"Indic-Transcribe inference failed: {exc}") from exc
        return ASRResult(self.name, str(text).strip(), time.perf_counter()-started, self.device,
                         lid.get("lang") if isinstance(lid, dict) else None, lid if isinstance(lid, dict) else None)
