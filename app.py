# app.py

import base64
import json
import time
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from tfrf_model import TFRFMulticlass

from preprocessing import preprocess


# =====================================================
# LOAD MODEL (hanya sekali saat app start)
# =====================================================

@st.cache_resource
def load_models():
    tfrf_model  = joblib.load("models/tfrf_model.pkl")
    svm_model   = joblib.load("models/svm_model.pkl")
    return tfrf_model, svm_model

tfrf_model, svm_model = load_models()

# =====================================================
# FUNGSI PREDIKSI
# =====================================================

def predict(judul: str, ringkasan: str):
    # 1. Preprocessing
    final_text = preprocess(judul, ringkasan)

    # 2. TF-RF Transform
    X = tfrf_model.transform([final_text])

    # 3. Prediksi kelas
    predicted_class = svm_model.predict(X)[0]

    # 4. Skor decision function (untuk confidence)
    scores      = svm_model.decision_function(X)[0]
    classes     = svm_model.classes_
    top3_idx    = np.argsort(scores)[::-1][:3]
    top3        = [(classes[i], scores[i]) for i in top3_idx]

    # Normalisasi skor ke persentase
    top3_scores = np.array([s for _, s in top3])
    top3_scores = top3_scores - top3_scores.min()
    total       = top3_scores.sum()
    top3_pct    = (top3_scores / total * 100) if total > 0 else top3_scores

    return predicted_class, [
        {"nama": cls, "persen": f"{pct:.1f}%"}
        for (cls, _), pct in zip(top3, top3_pct)
    ]

# =====================================================
# Panggil Label.json
# =====================================================

with open("models/ddc_labels.json", "r") as f:
    DDC_MAP = json.load(f)


# =====================================================
# UI — CONFIG & STYLE
# =====================================================

st.set_page_config(page_title="Sistem Klasifikasi Buku Perpustakaan Otomatis", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --brand: #2558df;
        --brand-dark: #1f4fc8;
        --text: #243447;
        --muted: #7b8794;
        --line: #d8dee9;
        --panel: #ffffff;
        --bg: #f7f9fc;
    }
    .stApp { background: var(--bg); }
    .stApp > header { background: transparent; }
    #MainMenu, header[data-testid="stHeader"], .stDeployButton, [data-testid="stToolbar"] {
        visibility: hidden; height: 0; min-height: 0;
    }
    .block-container {
        max-width: 100% !important;
        padding-top: 0rem; padding-left: 2rem;
        padding-right: 2rem; padding-bottom: 1.2rem;
    }
    .topbar-bleed {
        width: calc(100vw - 0px);
        margin-left: calc(50% - 50vw); margin-right: calc(50% - 50vw);
        margin-top: -0.99rem; margin-bottom: 2.15rem;
    }
    .topbar {
        position: sticky; top: 0; z-index: 20; margin: 0;
        padding: 1rem 1.4rem; background: var(--brand); color: white;
        display: flex; align-items: center; gap: 0.9rem;
        box-shadow: 0 1px 0 rgba(0,0,0,0.04); width: 100%; box-sizing: border-box;
    }
    .topbar-logo { width: 40px; height: 40px; object-fit: contain; flex: 0 0 auto; display: block; }
    .topbar-title { font-size: 1.35rem; font-weight: 800; letter-spacing: 0.2px; color: white; }
    .hero { text-align: center; margin: 3.8rem 0 1.7rem; }
    .hero h1 { font-size: 2.05rem; line-height: 1.2; margin: 0; color: var(--text); font-weight: 800; }
    .hero p { margin: 0.45rem 0 0; color: var(--muted); font-size: 1.02rem; }
    .section-divider { border-top: 1px solid rgba(216,222,233,0.8); margin: 1.2rem 0 1.8rem; }
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label { font-weight: 700; color: var(--text); margin-bottom: 0.25rem; }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 6px; border: 1px solid #d6dbe6;
        background: #fff; box-shadow: inset 0 1px 1px rgba(0,0,0,0.02);
    }
    div[data-testid="stTextArea"] textarea { min-height: 92px; }
    div[data-testid="stFormSubmitButton"] { margin-top: 26px; }
    div[data-testid="stFormSubmitButton"] button {
        height: 44px; width: 44px; margin-top: 0; padding: 0;
        border-radius: 10px; background: #a6b9ff; border: none;
        color: transparent; font-size: 18px; font-weight: 700;
        box-shadow: none; display: inline-grid; place-items: center;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'><path fill='white' fill-rule='evenodd' d='M12.323 13.383a5.5 5.5 0 1 1 1.06-1.06l2.897 2.897a.75.75 0 1 1-1.06 1.06l-2.897-2.897Zm.677-4.383a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z'/></svg>");
        background-repeat: no-repeat; background-position: center;
        background-size: 52%; transition: background-color 120ms ease, box-shadow 120ms ease;
        transform: none;
    }
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:focus {
        background-color: #3E73FF; color: transparent; border: none;
        box-shadow: none; transform: none; outline: none; -webkit-transform: none;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        background-color: #5f79e6; transform: none; box-shadow: none;
    }
    .results-title { font-size: 1.5rem; font-weight: 800; color: var(--text); margin: 1.4rem 0 0.55rem; }
    .panel {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 6px; padding: 2.2rem 5rem;
        box-shadow: 0 1px 0 rgba(0,0,0,0.01);
    }
    .placeholder-box {
        min-height: 156px; display: flex; align-items: center;
        justify-content: center; color: #667085;
        font-size: 1.02rem; text-align: center;
    }
    .result-row {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; font-size: 1.05rem; line-height: 1.3; color: #556070;
    }
    .result-row + .result-row { margin-top: 1.05rem; }
    .result-label { font-weight: 700; }
    .result-score { font-weight: 700; min-width: 60px; text-align: right; }
    .footer-note { text-align: center; color: #c1c7d0; font-size: 0.75rem; margin-top: 1.6rem; }
    .stAlert { margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# UI — TOPBAR
# =====================================================

logo_path = Path("logo/poliwangi.png")
logo_b64  = base64.b64encode(logo_path.read_bytes()).decode("utf-8") if logo_path.exists() else ""

st.markdown(
    f"""
    <div class="topbar-bleed">
        <div class="topbar">
            <img class="topbar-logo" src="data:image/png;base64,{logo_b64}" alt="Poliwangi logo" />
            <div class="topbar-title">Politeknik Negeri Banyuwangi</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# UI — HERO
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>Sistem Klasifikasi Buku Perpustakaan Otomatis</h1>
        <p>Masukkan judul dan deskripsi untuk klasifikasi DDC otomatis</p>
    </div>
    <div class="section-divider"></div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# UI — FORM INPUT
# =====================================================

with st.form(key="klasifikasi_form"):
    col_input, col_btn = st.columns([13, 1], vertical_alignment="center")

    with col_input:
        judul = st.text_input(
            "Judul Buku",
            placeholder="Masukkan judul buku (maksimal 100 kata) ...",
        )

    with col_btn:
        submit_button = st.form_submit_button("🔍", use_container_width=False)

    ringkasan = st.text_area(
        "Ringkasan Buku",
        placeholder="Masukkan Ringkasan Buku (Maksimal 250 kata) ...",
        height=95,
    )

# =====================================================
# UI — HASIL KLASIFIKASI
# =====================================================

st.markdown('<div class="results-title">Hasil Klasifikasi</div>', unsafe_allow_html=True)

placeholder_html = (
    '<div class="panel placeholder-box">'
    'Data akan ditampilkan setelah hasil pencarian selesai'
    '</div>'
)

if not submit_button:
    st.markdown(placeholder_html, unsafe_allow_html=True)
else:
    if judul.strip() == "" or ringkasan.strip() == "":
        st.warning("Mohon isi Judul Buku dan Ringkasan terlebih dahulu!")
        st.markdown(placeholder_html, unsafe_allow_html=True)
    else:
        with st.spinner("Sedang melakukan klasifikasi DDC..."):
            predicted_class, top3 = predict(judul, ringkasan)

        rows_html = "".join([
            f'<div class="result-row">'
            f'<div class="result-label">{DDC_MAP.get(item["nama"], "????")} - {item["nama"]}</div>'
            f'<div class="result-score">{item["persen"]}</div>'
            f'</div>'
            for item in top3
        ])

        st.markdown(
            f'<div class="panel">'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="footer-note">© 2026 Perpustakaan Politeknik Negeri Banyuwangi</div>', unsafe_allow_html=True)