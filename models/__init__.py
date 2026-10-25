"""Registry and factory for ASR model wrappers."""

from __future__ import annotations

from typing import Dict, Optional

from models._base import ASRModel
from models.indic_conformer import IndicConformerWrapper
from models.indic_transcribe_core import IndicTranscribeCoreWrapper
from models.whisper import WhisperWrapper
from .base import ModelUnavailable
from .whisper_model import WhisperMedium
from .indic_transcribe import IndicTranscribeCore
from .indic_conformer import IndicConformer


MODEL_REGISTRY: Dict[str, ASRModel] = {
    "whisper_medium": WhisperWrapper(model_name="medium"),
    "whisper_small": WhisperWrapper(model_name="small"),
    "indic_transcribe_core": IndicTranscribeCoreWrapper(),
    "indic_conformer": IndicConformerWrapper(),
}

# App-facing wrappers retain each repository's documented inference entrypoint.
MODELS = {
    "Whisper medium": WhisperMedium,
    "Indic-Transcribe-core": IndicTranscribeCore,
    "IndicConformer 600M (optional)": IndicConformer,
}


def get_model(key: str) -> ASRModel:
    """Return the model wrapper instance for ``key``."""
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key]


def available_models() -> Dict[str, str]:
    """Return a mapping of model keys to human-readable names."""
    return {key: model.display_name for key, model in MODEL_REGISTRY.items()}
