"""Local IndicTrans2 text-to-English adapter for native ASR transcripts."""

from __future__ import annotations

import time
from typing import Any

from .base import ModelUnavailable


MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
TARGET_LANGUAGE = "eng_Latn"
LANGUAGE_TAGS = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva", "doi": "doi_Deva",
    "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ks": "kas_Arab",
    "kok": "gom_Deva", "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Mtei",
    "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab",
}


class IndicTextTranslator:
    """Translate native Indic-Transcribe output locally with IndicTrans2."""

    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            self.processor = IndicProcessor(inference=True)
        except Exception as exc:
            raise ModelUnavailable(
                "IndicTrans2 text translation requires the local model download, "
                "IndicTransToolkit, transformers and an accepted model license. On Windows, "
                "IndicTransToolkit also needs Microsoft C++ Build Tools 14.0+ to compile "
                "its processor extension. Install the Visual Studio Desktop C++ workload, "
                "then rerun `pip install -r requirements.txt`. "
                f"Details: {exc}"
            ) from exc

    def translate(self, text: str, source_language: str) -> dict[str, Any]:
        source_tag = LANGUAGE_TAGS.get(source_language)
        if not source_tag:
            raise ModelUnavailable(
                f"IndicTrans2 text translation does not have a configured source tag for "
                f"language '{source_language}'. Choose a manual supported language."
            )
        if not text.strip():
            return {"text": "", "source_language": source_language, "device": self.device, "elapsed_seconds": 0.0}
        self._load()
        started = time.perf_counter()
        try:
            batch = self.processor.preprocess_batch(
                [text], src_lang=source_tag, tgt_lang=TARGET_LANGUAGE
            )
            inputs = self.tokenizer(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            with __import__("torch").inference_mode():
                generated = self.model.generate(**inputs, num_beams=5, max_length=256)
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            translated = self.processor.postprocess_batch(decoded, lang=TARGET_LANGUAGE)[0]
        except Exception as exc:
            raise ModelUnavailable(f"IndicTrans2 text translation failed: {exc}") from exc
        return {
            "model": MODEL_ID,
            "text": translated.strip(),
            "source_language": source_language,
            "device": self.device,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
