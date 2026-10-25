"""Unit tests for the evaluation utilities.

These tests do NOT require downloading any model weights.
"""

import pytest

from evaluation import (
    ErrorRates,
    aggregate_rates,
    compute_error_rates,
    format_rate,
    normalize_text,
    tokenize_words,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("hello    world") == "hello world"


def test_normalize_text_strips_control_chars() -> None:
    assert normalize_text("hello\x00world") == "helloworld"


def test_normalize_text_nfc() -> None:
    # Combining character sequence should be normalized
    assert normalize_text("e\u0301") == normalize_text("\u00e9")


def test_tokenize_words() -> None:
    assert tokenize_words("hello world") == ["hello", "world"]
    assert tokenize_words("") == []
    assert tokenize_words("   ") == []


# ---------------------------------------------------------------------------
# Error rates
# ---------------------------------------------------------------------------
def test_perfect_match() -> None:
    rates = compute_error_rates("hello world", "hello world")
    assert rates.wer == 0.0
    assert rates.cer == 0.0
    assert rates.substitutions == 0
    assert rates.deletions == 0
    assert rates.insertions == 0


def test_substitution() -> None:
    rates = compute_error_rates("hello", "hullo")
    assert rates.wer == pytest.approx(1.0)
    assert rates.substitutions == 1


def test_deletion() -> None:
    rates = compute_error_rates("hello world", "hello")
    assert rates.wer == pytest.approx(1.0 / 2.0)
    assert rates.deletions == 1


def test_insertion() -> None:
    rates = compute_error_rates("hello", "hello world")
    assert rates.wer == pytest.approx(1.0)
    assert rates.insertions == 1


def test_empty_reference_raises_zero() -> None:
    rates = compute_error_rates("", "hello")
    assert rates.ref_words == 0
    assert rates.wer == 0.0


def test_mixed_script_preserved() -> None:
    # Kannada-English code-switched text should be handled
    ref = "ನಮಸ्ते hello ಕನ್ನಡ"
    hyp = "ನಮಸ्ತे hello ಕನ್ನಡ"
    rates = compute_error_rates(ref, hyp)
    # These fixture strings genuinely differ by script code points.
    assert rates.wer == pytest.approx(0.25)


def test_to_dict() -> None:
    rates = compute_error_rates("a b", "a c")
    payload = rates.to_dict()
    assert "reference" in payload
    assert "wer" in payload
    assert "cer" in payload


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_rates_empty() -> None:
    agg = aggregate_rates([])
    assert agg["wer"] == 0.0
    assert agg["total_words"] == 0


def test_aggregate_rates_multiple() -> None:
    r1 = compute_error_rates("a b", "a b")
    r2 = compute_error_rates("a b c", "a x c")
    agg = aggregate_rates([r1, r2])
    assert agg["total_words"] == 5
    assert agg["total_substitutions"] == 1
    assert agg["wer"] == pytest.approx(1.0 / 5.0)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_format_rate() -> None:
    assert format_rate(0.1234) == "12.34%"
    assert format_rate(0.0) == "0.00%"
    assert format_rate(1.0) == "100.00%"


# ---------------------------------------------------------------------------
# Model wrapper smoke tests (no weights loaded)
# ---------------------------------------------------------------------------
def test_whisper_wrapper_validation() -> None:
    from models.whisper import WhisperWrapper

    with pytest.raises(ValueError):
        WhisperWrapper(model_name="invalid-model")


def test_indic_transcribe_core_wrapper_metadata() -> None:
    from models.indic_transcribe_core import IndicTranscribeCoreWrapper

    wrapper = IndicTranscribeCoreWrapper()
    assert wrapper.display_name == "Indic-Transcribe-core"
    assert wrapper.model_id == "bodhan-ai/indic-transcribe-core"
    assert not wrapper.is_loaded


def test_indic_conformer_wrapper_metadata() -> None:
    from models.indic_conformer import IndicConformerWrapper

    wrapper = IndicConformerWrapper()
    assert wrapper.display_name == "Indic Conformer 600M"
    assert wrapper.model_id == "ai4bharat/indic-conformer-600m-multilingual"
    assert not wrapper.is_loaded


def test_registry_has_expected_models() -> None:
    from models import available_models, get_model

    names = available_models()
    assert "whisper_medium" in names
    assert "indic_transcribe_core" in names
    assert "indic_conformer" in names
    assert get_model("whisper_medium").display_name == "Whisper medium"


def test_transcribe_without_load_raises() -> None:
    from models.indic_transcribe_core import IndicTranscribeCoreWrapper

    wrapper = IndicTranscribeCoreWrapper()
    with pytest.raises(RuntimeError):
        wrapper.transcribe("dummy.wav")
