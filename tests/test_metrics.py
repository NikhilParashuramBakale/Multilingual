from evaluation.metrics import normalize_text, evaluate_prediction, english_script_assessment

def test_normalization_preserves_kannada_and_latin_words():
    assert normalize_text("ನನಗೆ  Anxiety, ಇದೆ!") == "ನನಗೆ anxiety ಇದೆ"

def test_perfect_score_after_punctuation_normalization():
    result = evaluate_prediction("I have anxiety.", "i have anxiety")
    assert result["wer"] == 0 and result["cer"] == 0

def test_latin_retention_is_not_guessed_as_transliteration():
    assessment = english_script_assessment("ನನಗೆ anxiety ಇದೆ", "ನನಗೆ ಆಂಕ್ಸೈಟಿ ಇದೆ")
    assert assessment["not_retained_in_latin"] == ["anxiety"]
    assert "manually" in assessment["status"]

def test_empty_reference_is_not_scored():
    result = evaluate_prediction("", "some prediction")
    assert result["wer"] is None and result["cer"] is None
