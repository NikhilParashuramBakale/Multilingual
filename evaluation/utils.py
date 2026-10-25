"""Evaluation utilities: WER, CER, normalization, and reporting helpers."""

from evaluation import (
    ErrorRates,
    aggregate_rates,
    compute_error_rates,
    format_rate,
    normalize_text,
    tokenize_words,
)


def load_reference(reference_path):
    """Load a reference transcript from a text file (one line per file)."""
    with open(reference_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_reference(reference_path, text):
    """Save a reference transcript to disk."""
    with open(reference_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def compare_models(
    audio_path,
    reference,
    wrappers,
    language=None,
):
    """Run every wrapper on the same audio and compute error rates.

    Args:
        audio_path: Path to the audio file.
        reference: Ground-truth transcript string.
        wrappers: Dict of {model_key: ASRModel} (already loaded).
        language: Optional language code passed to each model.

    Returns:
        Dict of {model_key: {"result": ASRResult, "rates": ErrorRates}}.
    """
    outputs = {}
    for key, wrapper in wrappers.items():
        result = wrapper.transcribe(audio_path, language=language)
        rates = compute_error_rates(reference, result.text)
        outputs[key] = {"result": result, "rates": rates}
    return outputs


def build_report(outputs, aggregate):
    """Build a serializable comparison report."""
    report = {
        "aggregate": aggregate,
        "per_model": {},
    }
    for key, data in outputs.items():
        report["per_model"][key] = {
            "model_name": data["result"].model_name,
            "text": data["result"].text,
            "language": data["result"].language,
            "processing_time": data["result"].processing_time,
            "device": data["result"].device,
            "rates": data["rates"].to_dict(),
        }
    return report


def report_to_csv_rows(report):
    """Flatten a report into CSV rows."""
    rows = []
    for key, data in report["per_model"].items():
        rates = data["rates"]
        rows.append({
            "model": data["model_name"],
            "wer": round(rates["wer"], 4),
            "cer": round(rates["cer"], 4),
            "substitutions": rates["substitutions"],
            "deletions": rates["deletions"],
            "insertions": rates["insertions"],
            "ref_words": rates["ref_words"],
            "processing_time": round(data["processing_time"], 3),
            "device": data["device"],
        })
    return rows