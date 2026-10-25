"""Local NLLB text-to-English adapter for native ASR transcripts."""

from __future__ import annotations

import time
from typing import Any

from .base import ModelUnavailable


MODEL_ID = "facebook/nllb-200-distilled-600M"
TARGET_LANGUAGE = "eng_Latn"
SOURCE_LANGUAGE_TAGS = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva", "doi": "doi_Deva",
    "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ks": "kas_Arab",
    "kok": "gom_Deva", "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Mtei",
    "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab", "en": "eng_Latn",
}


class NllbTextTranslator:
    """Translate native ASR text locally without IndicTransToolkit."""

    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, use_safetensors=True
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise ModelUnavailable(
                f"{MODEL_ID} could not load locally. The first use downloads its weights. "
                f"Details: {exc}"
            ) from exc

    def translate(self, text: str, source_language: str | None) -> dict[str, Any]:
        source_tag = SOURCE_LANGUAGE_TAGS.get(source_language or "")
        if not source_tag:
            raise ModelUnavailable(
                "NLLB translation needs a detected or manually selected source language."
            )
        if not text.strip():
            return {
                "model": MODEL_ID,
                "text": "",
                "source_language": source_language,
                "device": self.device,
                "elapsed_seconds": 0.0,
            }
        self._load()
        started = time.perf_counter()
        try:
            self.tokenizer.src_lang = source_tag
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with __import__("torch").inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
                    max_length=512,
                    num_beams=5,
                )
            translated = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0]
        except Exception as exc:
            raise ModelUnavailable(f"NLLB text translation failed: {exc}") from exc
        return {
            "model": MODEL_ID,
            "text": translated.strip(),
            "source_language": source_language,
            "device": self.device,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
