from __future__ import annotations
import json, shutil
from pathlib import Path
import pandas as pd
import streamlit as st
from models import MODELS, ModelUnavailable
from models.device import torch_info
from evaluation import evaluate_prediction, build_report, write_reports, aggregate_reports

ROOT = Path(__file__).parent; AUDIO_DIR = ROOT / "data" / "audio"; REF_DIR = ROOT / "data" / "references"; OUTPUT_DIR = ROOT / "outputs"
for directory in (AUDIO_DIR, REF_DIR, OUTPUT_DIR): directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Kannada ASR Benchmark", layout="wide")
st.title("Kannada–English ASR Benchmark")
st.caption("Local inference only. Audio and transcripts stay on this computer; model weights download only from their official Hugging Face repositories after you choose to run a model.")

with st.expander("Hardware and model-access status", expanded=True):
    st.json(torch_info())
    st.warning("Indic-Transcribe-core (Bodhan Open Model License; ~4.6 GB) and optional IndicConformer (MIT; ~600M) are gated. Before running either, review/accept its Hugging Face access conditions and run `hf auth login`. The app never sends audio to an API.")

upload = st.file_uploader("Recording", type=["wav", "mp3", "flac"], help="A copy is stored locally in data/audio and ignored by Git.")
reference = st.text_area("Verified reference transcript", help="Do not enter a guessed transcript. WER/CER are shown only when this is provided.")
language = st.selectbox("Known spoken-language hint", [("Kannada (kn)", "kn"), ("Auto-detect where supported", None)], format_func=lambda x: x[0])[1]
reviewed_switch = st.text_input("Manually reviewed code-switched segment (optional)", help="Enter an exact reference segment selected by a reviewer.")
reviewed_clinical = st.text_input("Manually reviewed clinically important terms (optional)", help="Enter exact reference term(s); this is presence checking, not an invented accuracy statistic.")
selected = st.multiselect("Models", list(MODELS), default=["Whisper medium", "Indic-Transcribe-core"])

if "instances" not in st.session_state: st.session_state.instances = {}
if "last_report" not in st.session_state: st.session_state.last_report = None
if "history" not in st.session_state: st.session_state.history = []

if upload and selected and st.button("Run selected models", type="primary"):
    safe_name = Path(upload.name).name
    audio_path = AUDIO_DIR / safe_name
    audio_path.write_bytes(upload.getvalue())
    output_rows = []
    for display_name in selected:
        try:
            if display_name not in st.session_state.instances: st.session_state.instances[display_name] = MODELS[display_name]()
            with st.spinner(f"Running {display_name} locally…"):
                result = st.session_state.instances[display_name].transcribe(audio_path, language)
            metrics = evaluate_prediction(reference, result.text, reviewed_switch, reviewed_clinical)
            output_rows.append({"model": result.model, "text": result.text, "device": result.device, "detected_language": result.detected_language,
                                "elapsed_seconds": round(result.elapsed_seconds, 2), **metrics})
        except ModelUnavailable as exc: st.error(f"{display_name}: {exc}")
    if output_rows:
        st.session_state.last_report = build_report(safe_name, reference, output_rows, reviewed_switch, reviewed_clinical)
        st.session_state.history.append(st.session_state.last_report)

report = st.session_state.last_report
if report:
    st.subheader("Per-recording comparison")
    st.dataframe(pd.DataFrame([{k: r.get(k) for k in ("model", "text", "detected_language", "device", "elapsed_seconds", "wer", "cer")} for r in report["results"]]), use_container_width=True)
    st.subheader("English-in-Kannada script review")
    for r in report["results"]:
        script = r["script"]
        st.markdown(f"**{r['model']}** — {script['status']}")
        st.json(script)
    st.caption("‘Not retained in Latin script’ is intentionally not labeled as transliteration or an ASR error automatically; review the reference and prediction before making that distinction.")
    st.subheader("Error examples")
    for r in report["results"]:
        if r.get("wer") not in (None, 0): st.write({"model": r["model"], "reference": report["reference"], "prediction": r["text"]})
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("Download comparison JSON", payload, "comparison.json", "application/json")
    frame = pd.DataFrame([{k: r.get(k) for k in ("model", "text", "device", "detected_language", "elapsed_seconds", "wer", "cer")} for r in report["results"]])
    st.download_button("Download comparison CSV", frame.to_csv(index=False).encode("utf-8"), "comparison.csv", "text/csv")
    if st.button("Save report locally (outputs/)"):
        jpath, cpath = write_reports(report, OUTPUT_DIR); st.success(f"Saved {jpath.name} and {cpath.name}; both are excluded from Git.")
if st.session_state.history:
    st.subheader("Aggregate comparison (this session)")
    st.caption("Macro averages use only recordings with verified references; no unscored recordings are treated as zero-error.")
    st.dataframe(pd.DataFrame(aggregate_reports(st.session_state.history)), use_container_width=True)

st.divider(); st.subheader("Delete local benchmark data")
if st.checkbox("I understand this permanently deletes local recordings, references and generated reports."):
    if st.button("Delete local test data and outputs"):
        for directory in (AUDIO_DIR, REF_DIR, OUTPUT_DIR):
            for item in directory.iterdir():
                if item.name != ".gitkeep": shutil.rmtree(item) if item.is_dir() else item.unlink()
        st.session_state.last_report = None; st.session_state.history = []; st.success("Local audio, references, and outputs were deleted. Downloaded model caches were not removed.")
