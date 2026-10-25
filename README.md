---
noteId: "b32c9a60d04d11f1bffc2fe86bdf4348"
tags: []

---

# Multilingual Indian-language ASR Benchmark

Local, auditable comparison of **OpenAI Whisper medium**, **Bodhan AI Indic-Transcribe-core**, and optional **AI4Bharat IndicConformer 600M** on the *same* recordings. The interface covers the 22 scheduled Indian languages plus additional languages explicitly listed in each adapter's documented capability metadata. It is a standalone project; it does not alter or connect to Psychotherapy AI.

No accuracy claims are included. Add real recordings and verified per-language references, then run the same recording through each compatible model to create results.

## Privacy and access

Inference runs on the local device. The application never posts audio or transcripts to an external API. Native-language transcripts are preserved. After all selected ASR models transcribe the identical audio, a separately selected local text-translation model can translate every native transcript into English. Whisper's audio translation task is not used to translate another model's text. The first model load downloads weights directly from the official Hugging Face repository.

| Model | Official repository / setup | Access and output behavior |
| --- | --- | --- |
| Whisper medium | [OpenAI Whisper](https://github.com/openai/whisper) | Public MIT software; `medium` is multilingual. Language detection is reported. |
| Indic-Transcribe-core | [Bodhan AI model card](https://huggingface.co/bodhan-ai/indic-transcribe-core) | Gated: accept the Bodhan Open Model License and authenticate with Hugging Face. The card documents its native-script output, built-in language ID, and code-mixing support. |
| IndicConformer 600M | [AI4Bharat model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | Optional and gated: accept the Hub access condition and authenticate. MIT license; uses the repository's custom Transformers implementation and RNNT decoding. |
| NLLB-200 distilled 600M | [Meta model card](https://huggingface.co/facebook/nllb-200-distilled-600M) | Local text-to-English fallback for native ASR transcripts. Review the model card's CC-BY-NC-4.0 terms and hardware requirements before download. |

Review each repository's current license/access conditions yourself before download. Do not place credentials in this repository. `data/audio`, `data/references`, and `outputs` are ignored by Git.

## Setup — Windows

1. Install Python 3.10 or 3.11 and [FFmpeg](https://ffmpeg.org/download.html); make `ffmpeg` available on `PATH`. PyTorch 2.6 or newer is required for the local Transformers models; safetensors is preferred for model loading.
2. On Windows, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload. `IndicTransToolkit` compiles a native processor extension during installation.
3. In PowerShell:

   ```powershell
   cd G:\Projects_7thsem\Psychotherapy_AI\Kannada_ASR_Benchmark
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   # For NVIDIA GPU, first use the CUDA-matched command from pytorch.org.
   pip install -r requirements.txt
   hf auth login  # required only for gated Indic models, after accepting their pages
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

## Setup — Ubuntu

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
cd /path/to/Kannada_ASR_Benchmark
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# For NVIDIA GPU, install the CUDA-matched torch/torchaudio wheels from pytorch.org first.
pip install -r requirements.txt
hf auth login  # only after accepting gated model terms
streamlit run app.py
```

This project does not install drivers or system CUDA. It checks CUDA availability and free memory before its model wrappers choose GPU; CPU is used where feasible, but these large models can be slow or memory-limited. Always launch with `.\.venv\Scripts\python.exe -m streamlit run app.py`; using the parent project's `.venv` can select a different PyTorch version and cause the NLLB safety check or a second model download.

## Evaluation workflow

1. Upload one WAV, MP3, or FLAC file and select the same models for it.
2. Select a manual language override, or automatic detection when every selected model documents it. Unsupported model/language combinations are not silently replaced.
3. Declare an English code-switch condition when applicable.
4. Enter a human-verified original-language reference transcript. No reference means no WER/CER—not a zero score.
5. Select a text-translation model to translate every selected ASR model's native transcript. NLLB-200 is the local Transformers fallback; IndicTrans2 is also available when its native toolkit dependency is installed. Neither modifies the native transcripts.
6. Optionally enter exact reviewer-selected code-switch and clinically important segments. Their presence is reported separately; no segment accuracy is invented.
7. Inspect predictions side by side, script assessment, error examples, WER/CER, device, detected language, latency and memory usage.
8. Download CSV/JSON or explicitly save reports locally. The session aggregate is grouped by model and language and uses verified recordings only; it never treats an unverified or untested language as zero error.

Normalization is identical for every model: Unicode NFC, case folding, punctuation removal, and whitespace collapse. It does **not** translate, transliterate, remove negations, or discard symptom/English technical terms. Latin English terms are checked literally: if missing from model Latin output, the UI says that manual review is needed to distinguish transliteration from an ASR error. Automatic language detection and code-switching accuracy vary by language, speaker, recording conditions and model.

## Testing

```bash
pytest -q
```

Tests do not load or download model weights. At project creation there were no local recordings/reference transcripts, so no model benchmark or accuracy result was run or claimed.

## Troubleshooting

- **403 / gated repository**: visit the exact linked model card, accept terms, then run `hf auth login` in this virtual environment.
- **`ffmpeg` missing / MP3 fails**: install FFmpeg, restart the shell, and verify `ffmpeg -version`.
- **CUDA out of memory**: close GPU apps, choose one model at a time, or let it use CPU. Do not change drivers/CUDA automatically.
- **Indic model import / ONNX issue**: reinstall `requirements.txt`; the optional conformer pins the versions documented by AI4Bharat. The wrappers retain separate official inference paths rather than forcing a common interface.
- **Long, multi-speaker audio**: segment and review it first. Indic-Transcribe documents a single-speaker expectation and its own long-form chunking approach.

## Data deletion

Use the explicit “Delete local test data and outputs” control in the UI. It deletes only files inside `data/audio`, `data/references`, and `outputs`, preserving each `.gitkeep` placeholder. Downloaded Hugging Face model caches are deliberately not deleted by that control.
