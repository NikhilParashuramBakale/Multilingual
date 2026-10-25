from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation import aggregate_reports, build_report, evaluate_prediction, write_reports
from models import MODELS, ModelUnavailable
from models.indic_translation import IndicTextTranslator
from models.nllb_translation import NllbTextTranslator
from models.device import torch_info
from models.language_support import SCHEDULED_LANGUAGES, language_map

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "data" / "audio"
REF_DIR = ROOT / "data" / "references"
OUTPUT_DIR = ROOT / "outputs"
for directory in (AUDIO_DIR, REF_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def process_memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 2)
    except ImportError:
        return None


def model_languages(selected: list[str]) -> set[str]:
    if not selected:
        return set()
    supported = [MODELS[name].supported_languages for name in selected]
    return set(supported[0]).intersection(*supported[1:])


st.set_page_config(page_title="Multilingual Indian ASR Benchmark", layout="wide")
st.title("Multilingual Indian-language ASR Benchmark")
st.caption(
    "Local inference only. Audio and transcripts stay on this computer. "
    "Native transcription is preserved; optional English translation is a separate local task."
)

with st.expander("Hardware, licensing and model-access status", expanded=True):
    st.json(torch_info())
    import sys
    st.caption(f"Python environment: `{sys.executable}`")
    st.info(
        "Review each model card's current license, hardware requirements, dependencies and "
        "download/access conditions before loading weights. Gated Indic models require an "
        "accepted license and local `hf auth login`; no credentials or audio leave this device."
    )

selected = st.multiselect(
    "Models to run on the identical recording",
    list(MODELS),
    default=["Whisper medium", "Indic-Transcribe-core"],
)
common_codes = model_languages(selected)
languages = language_map(common_codes)
if selected and not common_codes:
    st.error("The selected models have no common documented languages. Select a compatible set.")

if common_codes:
    language_options = [("Automatic detection (where supported)", None)] + [
        (f"{languages[code].name} ({code})", code)
        for code in [item.code for item in SCHEDULED_LANGUAGES if item.code in common_codes]
    ]
    additional = [
        (f"{languages[code].name} ({code})", code)
        for code in sorted(common_codes - {item.code for item in SCHEDULED_LANGUAGES})
    ]
    language_options.extend(additional)
    selected_language_label = st.selectbox(
        "Reference / spoken language",
        language_options,
        format_func=lambda item: item[0],
        help="Manual override is sent to every selected adapter. Automatic detection is only offered when every selected model documents it.",
    )
    language = selected_language_label[1]
    if language is None and not all(MODELS[name].supports_auto_detection for name in selected):
        st.warning("At least one selected model has no documented automatic language detection; choose a manual language.")
    code_switch_options = [("No declared code-switch", None)]
    if "en" in common_codes:
        code_switch_options.append(("English (en)", "en"))
    code_switch_language = st.selectbox(
        "Declared code-switch language",
        code_switch_options,
        format_func=lambda item: item[0],
        help="This documents the test condition; it does not translate or silently switch models.",
    )[1]
else:
    language, code_switch_language = None, None

upload = st.file_uploader("Recording", type=["wav", "mp3", "flac"])
reference = st.text_area(
    "Verified reference transcript",
    help="Use the original-language transcript, including English words in code-mixed recordings. "
    "No WER/CER is reported without a verified reference.",
)
reviewed_switch = st.text_input("Manually reviewed code-switched segment (optional)")
reviewed_clinical = st.text_input("Manually reviewed clinically important terms (optional)")
translation_models = {
    "No text translation": None,
    "NLLB-200 (native text -> English)": NllbTextTranslator,
    "IndicTrans2 (Indic text -> English)": IndicTextTranslator,
}
translation_name = st.selectbox(
    "Text translation model for every ASR transcript",
    list(translation_models),
    help="The selected model translates each selected ASR model's native transcript. "
    "Whisper's audio translation task is intentionally not used for another model's text.",
)
st.caption(
    "Automatic language detection and code-switching accuracy vary by language, speaker, "
    "recording conditions and model. Results must be treated as language-specific evidence."
)

with st.expander("Documented capability matrix"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": name,
                    "auto_detection": model.supports_auto_detection,
                    "code_switching": model.supports_code_switching,
                    "documented_languages": len(model.supported_languages),
                }
                for name, model in MODELS.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

if "instances" not in st.session_state:
    st.session_state.instances = {}
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "history" not in st.session_state:
    st.session_state.history = []

auto_supported = bool(selected) and all(MODELS[name].supports_auto_detection for name in selected)
if upload and selected and (language is not None or auto_supported) and st.button("Run selected models", type="primary"):
    safe_name = Path(upload.name).name
    audio_path = AUDIO_DIR / safe_name
    audio_path.write_bytes(upload.getvalue())
    output_rows = []
    translations = {}
    for display_name in selected:
        try:
            if display_name not in st.session_state.instances:
                st.session_state.instances[display_name] = MODELS[display_name]()
            before = process_memory_mb()
            started = time.perf_counter()
            with st.spinner(f"Running {display_name} locally..."):
                result = st.session_state.instances[display_name].transcribe(audio_path, language)
            elapsed = time.perf_counter() - started
            after = process_memory_mb()
            metrics = evaluate_prediction(reference, result.text, reviewed_switch, reviewed_clinical)
            output_rows.append(
                {
                    "model": result.model,
                    "text": result.text,
                    "device": result.device,
                    "detected_language": result.detected_language,
                    "elapsed_seconds": round(elapsed, 2),
                    "memory_usage_mb": round(max(0, (after or 0) - (before or 0)), 2)
                    if before is not None and after is not None
                    else None,
                    "language": language,
                    "code_switch_language": code_switch_language,
                    **metrics,
                }
            )
        except ModelUnavailable as exc:
            st.error(f"{display_name}: {exc}")
    if translation_models[translation_name] is not None:
        try:
            translator_key = f"Text translator: {translation_name}"
            if translator_key not in st.session_state.instances:
                st.session_state.instances[translator_key] = translation_models[translation_name]()
            translator = st.session_state.instances[translator_key]
            for row in output_rows:
                try:
                    source_language = row["detected_language"] or language
                    with st.spinner(f"Translating {row['model']} output locally..."):
                        translations[row["model"]] = translator.translate(
                            row["text"], source_language
                        )
                except ModelUnavailable as exc:
                    st.error(f"{translation_name} for {row['model']} unavailable: {exc}")
        except ModelUnavailable as exc:
            st.error(f"{translation_name} unavailable: {exc}")
    if output_rows:
        report = build_report(
            safe_name, reference, output_rows, reviewed_switch, reviewed_clinical,
            language, code_switch_language,
        )
        report["text_translations"] = translations
        st.session_state.last_report = report
        st.session_state.history.append(report)
        if reference.strip():
            ref_path = REF_DIR / f"{Path(safe_name).stem}_{language or 'auto'}.json"
            ref_path.write_text(
                json.dumps({"recording": safe_name, "language": language, "reference": reference},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
elif upload and selected and language is None:
    st.info("Choose a manual language when a selected model does not document automatic detection.")

report = st.session_state.last_report
if report:
    st.subheader("Per-recording comparison")
    st.write({"recording": report["recording"], "language": report["language"],
              "code_switch_language": report["code_switch_language"]})
    columns = ("model", "text", "detected_language", "device", "elapsed_seconds",
               "memory_usage_mb", "wer", "cer")
    st.dataframe(pd.DataFrame([{key: row.get(key) for key in columns} for row in report["results"]]),
                 width="stretch")
    if report.get("text_translations"):
        st.subheader(f"Native transcript translations ({translation_name})")
        st.warning(
            "These translations are supplementary and may omit, mistranslate or normalize "
            "code-switched content. Use the native transcripts for ASR evaluation."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"asr_model": model, **translation}
                    for model, translation in report["text_translations"].items()
                ]
            ),
            width="stretch",
        )
    st.subheader("Script and code-switch review")
    for row in report["results"]:
        st.markdown(f"**{row['model']}** - {row['script']['status']}")
        st.json({"script": row["script"], "code_switch": row.get("code_switch")})
    st.caption("Missing Latin terms are not automatically labeled transliteration errors.")
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("Download language evaluation JSON", payload, "language_evaluation.json", "application/json")
    frame = pd.DataFrame([{key: row.get(key) for key in columns} for row in report["results"]])
    st.download_button("Download language evaluation CSV", frame.to_csv(index=False).encode("utf-8"),
                       "language_evaluation.csv", "text/csv")
    if st.button("Save evaluation report locally"):
        json_path, csv_path = write_reports(report, OUTPUT_DIR)
        st.success(f"Saved {json_path.name} and {csv_path.name}.")

if st.session_state.history:
    st.subheader("Aggregate results by model and language")
    st.caption("Only recordings with verified references contribute to WER/CER; untested languages have no fabricated result.")
    st.dataframe(pd.DataFrame(aggregate_reports(st.session_state.history)),
                 hide_index=True, width="stretch")

st.divider()
st.subheader("Delete local benchmark data")
if st.checkbox("I understand this permanently deletes local recordings, references and generated reports."):
    if st.button("Delete local test data and outputs"):
        for directory in (AUDIO_DIR, REF_DIR, OUTPUT_DIR):
            for item in directory.iterdir():
                if item.name != ".gitkeep":
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
        st.session_state.last_report = None
        st.session_state.history = []
        st.success("Local benchmark data deleted. Downloaded model caches were not removed.")
