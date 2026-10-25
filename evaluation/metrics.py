"""Conservative, shared scoring normalization: punctuation/case only, never translation/transliteration."""
from __future__ import annotations
import re, unicodedata
from jiwer import wer, cer
from dataclasses import dataclass

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("C"))
    text = re.sub(r"[^\w\s\u0C80-\u0CFF]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def latin_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*", text or "")

def english_script_assessment(reference: str, prediction: str) -> dict:
    expected = latin_tokens(reference); actual = latin_tokens(prediction)
    retained = [word for word in expected if word.casefold() in {x.casefold() for x in actual}]
    missing = [word for word in expected if word.casefold() not in {x.casefold() for x in actual}]
    return {"reference_latin_terms": expected, "retained_latin_terms": retained, "not_retained_in_latin": missing,
            "status": "retained in Latin script" if expected and not missing else ("not applicable" if not expected else "not retained in Latin script; manually determine transliteration vs recognition error")}

def evaluate_prediction(reference: str, prediction: str, reviewed_code_switch: str = "", reviewed_clinical: str = "") -> dict:
    ref, pred = normalize_text(reference), normalize_text(prediction)
    if not ref: return {"wer": None, "cer": None, "normalized_reference": ref, "normalized_prediction": pred, "script": english_script_assessment(reference, prediction)}
    out = {"wer": round(wer(ref, pred), 4), "cer": round(cer(ref, pred), 4), "normalized_reference": ref, "normalized_prediction": pred, "script": english_script_assessment(reference, prediction)}
    for key, segment in (("code_switch", reviewed_code_switch), ("clinical_terms", reviewed_clinical)):
        # These are explicitly entered reviewed segments; no fabricated segment-level score.
        out[key] = {"reference": segment, "present_verbatim_after_normalization": bool(normalize_text(segment)) and normalize_text(segment) in pred} if segment else None
    return out

@dataclass(frozen=True)
class ErrorRates:
    reference: str; prediction: str; wer: float; cer: float
    substitutions: int; deletions: int; insertions: int; ref_words: int
    def to_dict(self) -> dict:
        return self.__dict__.copy()

def tokenize_words(text: str) -> list[str]:
    return normalize_text(text).split() if normalize_text(text) else []

def _edit_counts(reference: list[str], prediction: list[str]) -> tuple[int, int, int]:
    # Dynamic-programming backtrace so the report can expose S/D/I counts.
    table = [[(i, 0, i, 0) if j == 0 else (j, 0, 0, j) if i == 0 else (0, 0, 0, 0)
              for j in range(len(prediction)+1)] for i in range(len(reference)+1)]
    for i in range(1, len(reference)+1):
        for j in range(1, len(prediction)+1):
            if reference[i-1] == prediction[j-1]: table[i][j] = table[i-1][j-1]
            else:
                delete = table[i-1][j]; insert = table[i][j-1]; substitute = table[i-1][j-1]
                options = [(delete[0]+1, delete[1], delete[2]+1, delete[3]), (insert[0]+1, insert[1], insert[2], insert[3]+1), (substitute[0]+1, substitute[1]+1, substitute[2], substitute[3])]
                table[i][j] = min(options, key=lambda x: x[0])
    _, subs, dels, ins = table[-1][-1]; return subs, dels, ins

def compute_error_rates(reference: str, prediction: str) -> ErrorRates:
    ref, pred = normalize_text(reference), normalize_text(prediction)
    ref_words, pred_words = tokenize_words(ref), tokenize_words(pred)
    subs, dels, ins = _edit_counts(ref_words, pred_words)
    word_rate = (subs + dels + ins) / len(ref_words) if ref_words else 0.0
    char_rate = cer(ref, pred) if ref else 0.0
    return ErrorRates(ref, pred, word_rate, char_rate, subs, dels, ins, len(ref_words))

def aggregate_rates(rates: list[ErrorRates]) -> dict:
    total_words = sum(r.ref_words for r in rates); subs = sum(r.substitutions for r in rates); dels = sum(r.deletions for r in rates); ins = sum(r.insertions for r in rates)
    return {"wer": (subs+dels+ins)/total_words if total_words else 0.0, "total_words": total_words, "total_substitutions": subs, "total_deletions": dels, "total_insertions": ins}

def format_rate(rate: float) -> str:
    return f"{rate:.2%}"
