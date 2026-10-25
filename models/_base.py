"""Base interface for ASR model wrappers used by the Kannada ASR benchmark."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ASRResult:
    """Structured output from any ASR model wrapper."""

    text: str
    language: Optional[str] = None
    processing_time: float = 0.0
    device: Optional[str] = None
    model_name: str = ""
    raw_output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "processing_time": self.processing_time,
            "device": self.device,
            "model_name": self.model_name,
            "raw_output": self.raw_output,
        }


class ASRModel(abc.ABC):
    """Abstract base class for all ASR model wrappers."""

    def __init__(self, model_id: str, display_name: str) -> None:
        self.model_id = model_id
        self.display_name = display_name
        self._model: Any = None
        self._device: Optional[str] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        return self._device

    @abc.abstractmethod
    def load(self, device: Optional[str] = None, **kwargs: Any) -> None:
        """Load model weights onto the specified device."""

    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs: Any) -> ASRResult:
        """Transcribe audio file at ``audio_path``."""

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False

    @staticmethod
    def detect_device(prefer_cuda: bool = True) -> str:
        """Return 'cuda' if available and preferred, else 'cpu'."""
        if prefer_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    @staticmethod
    def check_gpu_memory() -> Dict[str, Any]:
        """Return GPU memory info if CUDA is available."""
        info: Dict[str, Any] = {"cuda_available": False, "device_count": 0, "devices": []}
        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                        "allocated_gb": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2),
                    }
                )
        except Exception as exc:
            info["error"] = str(exc)
        return info