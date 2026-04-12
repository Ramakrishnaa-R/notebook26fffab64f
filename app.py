"""
Messy Mashup — AST Genre Classifier
Streamlit deployment app for final-code.ipynb
Run: streamlit run app.py
"""

import os
import io
import random
import tempfile
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# ─────────────────────────── CONFIG ───────────────────────────────────────────

SR = 16000
CHUNK_DURATION = 10
OVERLAP_DURATION = 2
CHUNK_LEN = SR * CHUNK_DURATION
STEP = SR * (CHUNK_DURATION - OVERLAP_DURATION)
FOLDS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
IDX2GENRE = {i: g for i, g in enumerate(GENRES)}

GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠", "disco": "🪩",
    "hiphop": "🎤", "jazz": "🎷", "metal": "🤘", "pop": "🌟",
    "reggae": "🌴", "rock": "⚡",
}

GENRE_COLOR = {
    "blues": "#1E90FF", "classical": "#DAA520", "country": "#8B4513",
    "disco": "#FF69B4", "hiphop": "#9400D3", "jazz": "#20B2AA",
    "metal": "#B22222", "pop": "#FF6347", "reggae": "#228B22", "rock": "#FF8C00",
}

MODEL_DIR = os.environ.get("MODEL_DIR", ".")   # set env var or drop .pth files next to app.py

# ─────────────────────────── PAGE SETUP ───────────────────────────────────────

st.set_page_config(
    page_title="Messy Mashup | Genre AI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');

  * { margin: 0; padding: 0; }
  
  html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #0a0a0f;
    color: #e8e4d8;
  }

  .stApp { background-color: #0a0a0f; }

  h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700; }

  .headline-gradient {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 7vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.0;
    background: linear-gradient(to right, #f5e6c8, #e8b86d, #c97b2a);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }

  .subheader-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgb(107, 100, 86);
    margin-bottom: 2.5rem;
    font-weight: 400;
  }

  .framed-box {
    background-color: #111118;
    border: 1px solid rgba(30, 30, 42, 0.8);
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
  }

  .badge-genre {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
    text-align: center;
  }

  .progress-container {
    background-color: #1a1a24;
    border-radius: 4px;
    height: 7px;
    width: 100%;
    margin: 4px 0 10px 0;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  }

  .genre-entry {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: rgb(90, 84, 105);
    margin-bottom: 2px;
  }

  .status-indicator {
    display: inline-block;
    background-color: #1a1a24;
    border: 1px solid rgba(42, 42, 56, 0.7);
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.68rem;
    color: rgb(107, 100, 86);
    margin-right: 4px;
    font-weight: 500;
  }

  .stFileUploader > div { border-color: #2a2a38 !important; }
  .stFileUploader label { color: #e8e4d8 !important; font-weight: 500; }

  div[data-testid="stAudio"] { margin-top: 0.5rem; }

  footer { display: none; }
  #MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── HELPERS ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    return ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )


@st.cache_resource(show_spinner=False)
def load_models(model_dir: str, n_folds: int = FOLDS):
    """Load fine-tuned fold checkpoints. Falls back to zero-shot if not found."""
    feature_extractor = load_feature_extractor()
    models = []
    loaded_paths = []

    for fold in range(n_folds):
        path = os.path.join(model_dir, f"model_fold{fold}.pth")
        m = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=10,
            ignore_mismatched_sizes=True,
        )
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                m.load_state_dict(state)
                loaded_paths.append(f"fold{fold} ✓")
            except Exception as e:
                loaded_paths.append(f"fold{fold} ✗")
        else:
            loaded_paths.append(f"fold{fold} (base)")

        m.eval().to(DEVICE)
        models.append(m)

    return models, feature_extractor, loaded_paths


def load_waveform(file_bytes: bytes) -> torch.Tensor:
    """Load audio bytes → mono waveform at SR."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        waveform, sr = librosa.load(tmp_path, sr=SR, mono=True)
        waveform = torch.from_numpy(waveform).float()
    finally:
        os.unlink(tmp_path)
    return waveform


@torch.no_grad()
def predict(waveform: torch.Tensor, models, feature_extractor) -> dict:
    """Chunked ensemble inference → softmax probability dict."""
    if len(waveform) < CHUNK_LEN:
        waveform = F.pad(waveform, (0, CHUNK_LEN - len(waveform)))

    length = len(waveform)
    num_chunks = min(int(np.ceil((length - CHUNK_LEN) / STEP)) + 1, 10)
    chunks = []
    for i in range(num_chunks):
        start = i * STEP
        end = start + CHUNK_LEN
        chunk = waveform[start:end]
        if len(chunk) < CHUNK_LEN:
            chunk = F.pad(chunk, (0, CHUNK_LEN - len(chunk)))
        chunks.append(chunk)

    logits_sum = torch.zeros(1, len(GENRES))
    total = 0

    for chunk in chunks:
        inputs = feature_extractor(
            chunk.numpy(), sampling_rate=SR, return_tensors="pt"
        )
        xb = inputs["input_values"].to(DEVICE)
        for model in models:
            out = model(input_values=xb)
            logits_sum += out.logits.cpu()
            total += 1

    avg = logits_sum / total
    probs = torch.softmax(avg, dim=-1).squeeze().tolist()
    return {IDX2GENRE[i]: round(p * 100, 2) for i, p in enumerate(probs)}


def render_results(probs: dict):
    best_match = max(probs, key=probs.get)
    best_score = probs[best_match]
    match_color = GENRE_COLOR[best_match]
    match_emoji = GENRE_EMOJI[best_match]

    output_html = f"""
    <div class="framed-box" style="border-left: 4px solid {match_color}; text-align:center;">
      <div style="font-size:3.5rem; margin-bottom:0.2rem; margin-top: 0.5rem;">{match_emoji}</div>
      <div class="badge-genre" style="background:rgba({int(match_color[1:3], 16)}, {int(match_color[3:5], 16)}, {int(match_color[5:7], 16)}, 0.1); color:{match_color}; border:2px solid {match_color};">
        {best_match.upper()}
      </div>
      <div style="font-size:0.85rem; color:#7a7469; margin-top:0.8rem; font-family:'Space Mono',monospace; margin-bottom: 0.5rem;">
        {best_score:.1f}% match
      </div>
    </div>
    """
    st.markdown(output_html, unsafe_allow_html=True)

    st.markdown("##### Genre Breakdown")
    ranked_results = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for tag, score in ranked_results:
        tag_color = GENRE_COLOR[tag]
        display_html = f"""
        <div class="genre-entry"><span>{GENRE_EMOJI[tag]} {tag}</span><span>{score:.1f}%</span></div>
        <div class="progress-container">
          <div class="progress-fill" style="width:{score}%;background-color:{tag_color};"></div>
        </div>
        """
        st.markdown(display_html, unsafe_allow_html=True)


# ─────────────────────────── INTERFACE ───────────────────────────────────────────

st.markdown('<div class="headline-gradient">AUDIO GENRE IDENTIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Deep Learning · Multi-Model Ensemble · 10 Music Classes</div>', unsafe_allow_html=True)

# Initialize model stack
with st.spinner("Initializing neural networks…"):
    model_stack, feature_processor, status_indicators = load_models(MODEL_DIR)

status_badges = " ".join(f'<span class="status-indicator">{status}</span>' for status in status_indicators)
st.markdown(status_badges, unsafe_allow_html=True)
st.markdown("<br />", unsafe_allow_html=True)

# File input section
st.markdown('<div class="framed-box">', unsafe_allow_html=True)
sound_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Submit a music sample or audio clip (WAV format preferred)",
)
st.markdown("</div>", unsafe_allow_html=True)

if sound_file is not None:
    raw_data = sound_file.read()
    st.audio(raw_data, format=sound_file.type)

    with st.spinner("Processing audio through ensemble…"):
        try:
            audio_signal = load_waveform(raw_data)
            duration_seconds = len(audio_signal) / SR
            st.caption(
                f"📁 {sound_file.name} | ⏱ {duration_seconds:.1f}s | "
                f"📊 {len(audio_signal):,} samples | 🖥 {DEVICE.upper()}"
            )
            prediction_result = predict(audio_signal, model_stack, feature_processor)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            prediction_result = None

    if prediction_result:
        st.divider()
        render_results(prediction_result)

st.markdown("""
---
<div style="font-size:0.67rem;color:#4a4555;text-align:center;margin-top:1.2rem;font-family:'Space Mono',monospace;">
  ✨ Model checkpoints: place <code>model_fold0.pth</code> through <code>model_fold3.pth</code> in application directory, or configure <code>MODEL_DIR</code> | Powered by HuggingFace Transformers & Streamlit
</div>
""", unsafe_allow_html=True)
