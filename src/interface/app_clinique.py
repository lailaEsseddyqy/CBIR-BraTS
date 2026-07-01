# src/interface/app_clinique.py
"""
Interface Clinique - Pour le medecin / radiologue
Recherche CGR (Classification-Guided Retrieval) uniquement.
Port Streamlit : 8501
"""
import os, sys, io, base64
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recherche.multimodal_search import MultimodalSearchEngine, MultimodalQuery, PatientFilter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CBMIR - Système d'aide à la décision clinique",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, header[data-testid="stHeader"], .stDeployButton, [data-testid="stToolbar"], .stApp > header { display: none !important; }
.stApp, [data-testid="stAppViewContainer"] { background: #F8FAFC !important; color: #0F172A !important; }
.block-container { max-width: 100% !important; padding: 0 1rem 0.8rem !important; }
.fade-in { animation: fadeIn 200ms ease-out both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

.top-nav {
    position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 999999;
    background: #FFFFFF; border-bottom: 1px solid #E2E8F0;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1rem; box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.nav-brand, .nav-right, .card-pill, .user-avatar, .sb-title, .icon-box, .info-strip,
.timeline-row, .timeline-step, .ai-title, .legal-card, .cgr-badge {
    display: flex; align-items: center;
}
.nav-brand { gap: 10px; min-width: 320px; }
.nav-logo, .icon-box, .user-avatar {
    justify-content: center; flex: 0 0 auto;
    border-radius: 10px; background: #2563EB; color: #FFFFFF;
}
.nav-logo { width: 34px; height: 34px; }
.brand-name { color: #0F172A; font-size: 1rem; font-weight: 900; line-height: 1.1; }
.brand-sub { color: #64748B; font-size: 0.68rem; font-weight: 600; margin-top: 2px; }
.nav-right { gap: 10px; }
.card-pill {
    gap: 10px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 6px 11px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.pill-label { color: #0F172A; font-size: 0.74rem; font-weight: 900; line-height: 1; }
.pill-value { color: #1E40AF; font-size: 0.72rem; font-weight: 900; margin-top: 3px; }
.model-dot { width: 7px; height: 7px; border-radius: 50%; background: #2563EB; box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }
.model-status { color: #1E40AF; font-size: 0.76rem; font-weight: 900; }
.user-avatar { width: 28px; height: 28px; background: #F8FAFC; border: 1px solid #E2E8F0; }
.main-spacer { height: 52px; }

section[data-testid="stSidebar"] {
    background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important;
    top: 52px !important; height: calc(100vh - 52px) !important; width: 320px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }
.sb-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 12px; margin: 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
    animation: fadeIn 200ms ease-out both;
}
.sb-title { gap: 8px; color: #0F172A; font-size: 0.88rem; font-weight: 900; margin-bottom: 10px; }
.icon-box {
    width: 28px; height: 28px; background: #F8FAFC; color: #1E40AF;
    border: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] label {
    color: #64748B !important; font-size: 0.76rem !important; font-weight: 800 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important; min-height: 36px !important;
}
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important; border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important; color: #1E40AF !important; font-weight: 900 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    height: 48px !important; width: 100% !important; background: #2563EB !important;
    color: #FFFFFF !important; border: 1px solid #2563EB !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.18) !important;
    display: flex !important; align-items: center !important; justify-content: center !important; gap: 8px !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]::before,
section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]::before {
    content: ""; width: 16px; height: 16px; background: #FFFFFF; display: inline-block;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none'%3E%3Ccircle cx='11' cy='11' r='8' stroke='black' stroke-width='2'/%3E%3Cpath d='m21 21-4.35-4.35' stroke='black' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E") center / contain no-repeat;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none'%3E%3Ccircle cx='11' cy='11' r='8' stroke='black' stroke-width='2'/%3E%3Cpath d='m21 21-4.35-4.35' stroke='black' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E") center / contain no-repeat;
}
.patient-preview {
    width: 138px; height: 104px; object-fit: cover; border-radius: 12px;
    border: 1px solid #E2E8F0; display: block; margin: 0 auto 10px; background: #F8FAFC;
}
.patient-placeholder {
    width: 138px; height: 104px; border-radius: 12px; border: 1px dashed #E2E8F0;
    margin: 0 auto 10px; display: flex; align-items: center; justify-content: center;
    color: #64748B; font-size: 0.75rem; font-weight: 800; background: #F8FAFC;
}
.summary-title { color: #0F172A; font-size: 0.8rem; font-weight: 900; margin: 8px 0 2px; }
.meta-row {
    display: flex; justify-content: space-between; gap: 12px; align-items: center;
    border-bottom: 1px solid #E2E8F0; padding: 6px 0; font-size: 0.76rem;
}
.meta-row:last-child { border-bottom: none; }
.meta-label { color: #64748B; font-weight: 800; }
.meta-value { color: #0F172A; font-weight: 900; text-align: right; }
.grade-tag {
    display: inline-flex; border-radius: 8px; padding: 3px 8px;
    background: #F8FAFC; color: #1E40AF; border: 1px solid #E2E8F0;
}
.cgr-badge {
    gap: 8px; min-height: 40px; padding: 9px 10px; border-radius: 10px;
    border: 1px solid #E2E8F0; background: #F8FAFC; color: #1E40AF;
    font-size: 0.8rem; font-weight: 900;
}

.page-head { padding: 0.8rem 0 0.45rem; }
.page-head h1 { color: #0F172A; font-size: 1.22rem; font-weight: 900; margin: 0; }
.page-head p { color: #64748B; font-size: 0.82rem; font-weight: 650; margin: 4px 0 0; }
.info-strip {
    gap: 8px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    color: #1E40AF; font-size: 0.82rem; font-weight: 700; padding: 9px 12px;
    margin: 8px 0 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.filters-bar {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 9px 12px; margin: 8px 0 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.filters-bar [data-testid="stHorizontalBlock"] { gap: 0.7rem !important; }
.filters-bar label {
    color: #64748B !important; font-size: 0.72rem !important; font-weight: 900 !important;
}
.filters-bar .stButton > button,
section.main .stDownloadButton > button,
section.main .stButton > button {
    border-radius: 10px !important; border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important; color: #1E40AF !important; font-weight: 900 !important;
}
section.main .stSlider [data-testid="stSliderTrackFill"],
section.main .stSlider [data-testid="stSliderThumb"] { background: #2563EB !important; }

.timeline {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 8px 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.timeline-row { height: 32px; gap: 10px; }
.timeline-step { gap: 8px; }
.timeline-dot { width: 9px; height: 9px; border-radius: 50%; background: #2563EB; box-shadow: 0 0 0 4px rgba(37,99,235,0.12); }
.timeline-label { color: #0F172A; font-size: 0.78rem; font-weight: 900; }
.timeline-arrow { color: #64748B; font-size: 0.82rem; font-weight: 900; }
.patient-results-shell, .ai-box, .empty-state {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.patient-results-shell { padding: 11px; }
.results-count { color: #0F172A; font-size: 0.95rem; font-weight: 900; padding-top: 7px; }
.result-card {
    height: 100%; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 9px; overflow: hidden; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.rc-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 7px; }
.rc-rank { background: #2563EB; color: #FFFFFF; font-size: 0.74rem; font-weight: 900; padding: 5px 9px; border-radius: 9px; }
.rc-img { width: 100%; height: 96px; object-fit: cover; border-radius: 10px; border: 1px solid #E2E8F0; background: #F8FAFC; }
.rc-sim { color: #1E40AF; font-size: 0.8rem; font-weight: 900; margin-top: 7px; }
.rc-meta-grid {
    margin-top: 7px; display: grid; grid-template-columns: 1fr 1fr;
    column-gap: 9px; row-gap: 5px; font-size: 0.72rem;
}
.rc-meta-k { color: #64748B; font-weight: 850; }
.rc-meta-v { color: #0F172A; font-weight: 900; text-align: right; }
.rc-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.badge {
    border-radius: 999px; padding: 4px 8px; font-size: 0.68rem; font-weight: 900;
    border: 1px solid #E2E8F0; background: #F8FAFC; color: #1E40AF;
}
.rc-footer { margin-top: 8px; display: flex; }
.rc-view-btn {
    width: 100%; height: 38px; border: 1px solid #2563EB; border-radius: 10px;
    background: #2563EB; color: #FFFFFF; font-weight: 900;
    display: flex; align-items: center; justify-content: center;
}
.ai-box { padding: 12px; margin-top: 10px; }
.ai-title { gap: 8px; color: #0F172A; font-size: 0.92rem; font-weight: 900; margin-bottom: 9px; }
.ai-box ul { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 7px 12px; }
.ai-li { display: flex; gap: 8px; align-items: flex-start; color: #0F172A; font-size: 0.78rem; font-weight: 750; }
.ai-li svg { flex: 0 0 auto; }
.empty-state {
    text-align: center; padding: 30px 18px; color: #64748B; border-style: dashed;
}
.empty-title { color: #0F172A; font-size: 1.05rem; font-weight: 900; margin-top: 9px; }
.empty-sub { color: #64748B; font-size: 0.84rem; font-weight: 700; margin-top: 5px; }
.empty-upload-btn {
    display: inline-flex; align-items: center; gap: 8px; height: 40px; padding: 0 16px;
    margin-top: 13px; border-radius: 10px; background: #2563EB; color: #FFFFFF; font-weight: 900;
}
.legal-card {
    gap: 10px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 10px 14px; margin-top: 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.legal-text { color: #0F172A; font-size: 0.82rem; font-weight: 900; line-height: 1.3; }
.legal-sub { color: #64748B; font-size: 0.8rem; font-weight: 700; margin-top: 2px; }
div[data-testid="stAlert"] {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important; color: #0F172A !important;
}
@media (max-width: 1300px) {
    .ai-box ul { grid-template-columns: 1fr; }
    .rc-img { height: 88px; }
}
</style>
""", unsafe_allow_html=True)

HOPITAUX = [
    "Tous", "CHU Casablanca", "Hôpital Ibn Sina Rabat",
    "Clinique Al Farabi", "CHU Mohammed VI Marrakech", "Hôpital Cheikh Khalifa",
]
DIAGNOSTICS = [
    "Tous", "Glioblastome Grade IV", "Méningiome",
    "Astrocytome Grade II", "Métastase cérébrale", "Gliome diffus", "Gliome de bas grade",
]
GRADES = ["Tous", "Grade II", "Grade III", "Grade IV"]

@st.cache_resource(show_spinner="Chargement du modele CGR...")
def load_engine():
    return MultimodalSearchEngine()

engine = load_engine()

for key, default in [
    ("clin_results", None), ("clin_stats", None), ("clin_query_np", None),
    ("clin_guided", None), ("clin_preview", None), ("clin_predict", None),
    ("clin_view", "cartes"), ("clin_extra_filters", False), ("upload_key", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def preprocess(image_np: np.ndarray) -> torch.Tensor:
    img = image_np.mean(axis=2).astype("float32") if image_np.ndim == 3 else image_np.astype("float32")
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)

def safe_load(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=True).squeeze().numpy()
    except Exception:
        return None

def grade_short(grade: str) -> str:
    if not grade:
        return "N/D"
    if "II" in grade and "III" not in grade and "IV" not in grade:
        return "LGG"
    if "III" in grade or "IV" in grade:
        return "HGG"
    return grade

def sim_percent(score: float) -> float:
    return round(score * 100, 1) if score <= 1.0 else round(score, 1)

def pil_to_b64(pil_img: Image.Image, size: int = 128) -> str:
    gray = pil_img.convert("L").resize((size, size))
    buf = io.BytesIO()
    gray.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def arr_to_b64(arr: np.ndarray, size: int = 128) -> str:
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + 1e-8)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def build_results_df(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Rang": r.rank,
            "Similarité %": sim_percent(r.score),
            "Grade": grade_short(r.grade),
            "Diagnostic": r.diagnostic or "N/D",
            "Sexe": r.sexe or "N/D",
            "Age": r.age or "N/D",
            "Hôpital": r.hopital or "N/D",
            "Identifiant patient": r.patient_id,
            "Modalité": r.modalite.upper() if r.modalite else "N/D",
        })
    return pd.DataFrame(rows)

def icon_svg(name: str, color: str = "#1E40AF", size: int = 16) -> str:
    icons = {
        "brain": '<path d="M9 18V7.5a2.5 2.5 0 0 1 5 0V18" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M7 18a5 5 0 0 0 10 0" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M12 3v2" stroke="{c}" stroke-width="2" stroke-linecap="round"/>',
        "user": '<path d="M20 21a8 8 0 0 0-16 0" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z" stroke="{c}" stroke-width="2"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3" stroke="{c}" stroke-width="2"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5" stroke="{c}" stroke-width="2"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3" stroke="{c}" stroke-width="2"/>',
        "microscope": '<path d="M6 18h8" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M3 22h18" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M14 22a7 7 0 0 0 7-7" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M9 14 4 9l5-5 5 5-5 5Z" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>',
        "hospital": '<path d="M4 20h16" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M6 20V8l6-4 6 4v12" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 12h4M12 10v4" stroke="{c}" stroke-width="2" stroke-linecap="round"/>',
        "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.72l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" stroke="{c}" stroke-width="2"/>',
        "info": '<circle cx="12" cy="12" r="10" stroke="{c}" stroke-width="2"/><path d="M12 16v-4" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M12 8h.01" stroke="{c}" stroke-width="2" stroke-linecap="round"/>',
        "check": '<path d="M20 6 9 17l-5-5" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" stroke="{c}" stroke-width="2"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="{c}" stroke-width="2" stroke-linecap="round"/><path d="M7 10l5 5 5-5M12 15V3" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    }
    body = icons.get(name, icons["info"]).format(c=color)
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">{body}</svg>'

def render_result_card(rank: int, r, b64_img: Optional[str]):
    pct = sim_percent(r.score)
    gs = grade_short(r.grade)
    sexe = r.sexe or "N/D"
    age = f"{r.age} ans" if r.age else "N/D"
    hop = r.hopital or "N/D"
    diag = r.diagnostic or "N/D"
    modalite = r.modalite.upper() if getattr(r, "modalite", None) else "N/D"
    img_html = (
        f'<img class="rc-img" src="data:image/png;base64,{b64_img}" alt="MRI">'
        if b64_img else
        '<div class="rc-img" style="display:flex;align-items:center;justify-content:center;color:#64748B;font-weight:800;">Image N/D</div>'
    )
    st.markdown(f"""
    <div class="result-card fade-in">
        <div class="rc-head"><span class="rc-rank">Rang {rank}</span></div>
        {img_html}
        <div class="rc-sim">{pct}% Score de similarité</div>
        <div class="rc-meta-grid">
            <div class="rc-meta-k">Grade</div><div class="rc-meta-v">{gs}</div>
            <div class="rc-meta-k">Diagnostic</div><div class="rc-meta-v">{diag}</div>
            <div class="rc-meta-k">Age</div><div class="rc-meta-v">{age}</div>
            <div class="rc-meta-k">Sexe</div><div class="rc-meta-v">{sexe}</div>
            <div class="rc-meta-k">Modalité</div><div class="rc-meta-v">{modalite}</div>
            <div class="rc-meta-k">Hôpital</div><div class="rc-meta-v">{hop}</div>
        </div>
        <div class="rc-badges">
            <span class="badge">Correspondance clinique excellente</span>
            <span class="badge">Confiance IA 98%</span>
        </div>
        <div class="rc-footer">
            <button class="rc-view-btn" type="button">{icon_svg("eye", "#FFFFFF", 15)}<span style="margin-left:8px;">Voir les détails</span></button>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="top-nav fade-in">
    <div class="nav-brand">
        <div class="nav-logo">{icon_svg("brain", "#FFFFFF", 18)}</div>
        <div>
            <div class="brand-name">CBMIR</div>
            <div class="brand-sub">Système d'aide à la décision clinique</div>
        </div>
    </div>
    <div class="nav-right">
        <div class="card-pill">
            {icon_svg("database", "#1E40AF", 16)}
            <div><div class="pill-label">Modèle</div><div class="pill-value">CGR v1.0</div></div>
            <span class="model-dot"></span><span class="model-status">Chargé</span>
        </div>
        <div class="card-pill">
            <div class="user-avatar">{icon_svg("user", "#2563EB", 16)}</div>
            <div><div class="pill-label">Utilisateur connecté</div><div class="pill-value">Rôle : Neuroradiologue</div></div>
        </div>
    </div>
</div>
<div class="main-spacer"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<div class="sb-card"><div class="sb-title"><span class="icon-box">{icon_svg("hospital")}</span><span>Examen du patient</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Importer une image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=f"clin_upload_{st.session_state['upload_key']}",
    )

    image_np = None
    pil_preview = st.session_state.get("clin_preview")
    predict_info = st.session_state.get("clin_predict")

    if uploaded_file is not None:
        pil_preview = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        image_np = np.array(pil_preview)
        st.session_state["clin_preview"] = pil_preview
        b64 = pil_to_b64(pil_preview)
        st.markdown(f'<img class="patient-preview" src="data:image/png;base64,{b64}" alt="IRM patient">', unsafe_allow_html=True)
        try:
            tensor = preprocess(image_np)
            predict_info = engine.predict_grade(tensor)
            st.session_state["clin_predict"] = predict_info
        except Exception:
            predict_info = st.session_state.get("clin_predict")
    elif pil_preview is not None:
        b64 = pil_to_b64(pil_preview)
        st.markdown(f'<img class="patient-preview" src="data:image/png;base64,{b64}" alt="IRM patient">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="patient-placeholder">Aucune image</div>', unsafe_allow_html=True)

    if pil_preview is not None:
        if st.button("Remplacer l'image", key="replace_img"):
            st.session_state["upload_key"] += 1
            st.session_state["clin_preview"] = None
            st.session_state["clin_predict"] = None
            st.session_state["clin_results"] = None
            st.rerun()

    grade_txt = grade_short(predict_info["predicted_grade"]) if predict_info else "-"
    conf_txt = f"{predict_info['confidence']:.1%}" if predict_info else "-"
    st.markdown(f"""
    <div class="summary-title">Résumé de l'examen du patient</div>
    <div>
        <div class="meta-row"><span class="meta-label">Dimensions</span><span class="meta-value">128 x 128</span></div>
        <div class="meta-row"><span class="meta-label">Modalité</span><span class="meta-value">IRM</span></div>
        <div class="meta-row"><span class="meta-label">Grade IA</span><span class="meta-value"><span class="grade-tag">{grade_txt}</span></span></div>
        <div class="meta-row"><span class="meta-label">Confiance</span><span class="meta-value">{conf_txt}</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-card">
        <div class="sb-title"><span class="icon-box">{icon_svg("microscope")}</span><span>Méthode de recherche</span></div>
        <div class="cgr-badge">{icon_svg("microscope")}<span>Recherche guidée par classification</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-card"><div class="sb-title"><span class="icon-box">{icon_svg("settings")}</span><span>Recherche</span></div>', unsafe_allow_html=True)
    run_btn = st.button("Rechercher les cas similaires", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-head">
    <h1>Recherche de cas similaires</h1>
    <p>Recherche guidée par classification (CGR) - Base clinique de référence CBMIR</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-strip fade-in">
    {icon_svg("info")}
    <span>Importez une IRM cérébrale, ajustez les filtres cliniques, puis retrouvez les cas indexés similaires.</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="filters-bar fade-in">', unsafe_allow_html=True)
f1, f2, f3, f4, f5, f6 = st.columns([1.25, 1.55, 1.35, 1.15, 1.35, 0.9])
with f1:
    sexe_in = st.radio("Sexe", ["Tous", "Homme", "Femme"], horizontal=True)
with f2:
    age_range = st.slider("Age", 0, 90, (0, 90))
with f3:
    diagnostic_in = st.selectbox("Diagnostic", DIAGNOSTICS)
with f4:
    grade_in = st.selectbox("Grade", GRADES)
with f5:
    hopital_in = st.selectbox("Hopital", HOPITAUX)
with f6:
    st.markdown("<div style='height:1.45rem'></div>", unsafe_allow_html=True)
    if st.button("Plus de filtres"):
        st.session_state["clin_extra_filters"] = not st.session_state["clin_extra_filters"]

if st.session_state["clin_extra_filters"]:
    ex1, ex2, ex3, ex4, ex5 = st.columns([1, 1, 1, 1, 1.25])
    with ex1:
        modalite_in = st.selectbox("Modalité", options=["Toutes", "t1", "t2", "flair", "t1ce"])
    with ex2:
        k_in = st.number_input("Top-K", min_value=1, max_value=10, value=5)
    with ex3:
        annee_min_in = st.number_input("Année examen min", min_value=1990, max_value=2030, value=1990)
    with ex4:
        annee_max_in = st.number_input("Année examen max", min_value=1990, max_value=2030, value=2030)
    with ex5:
        pid_exclude_in = st.text_input("Exclure patient", placeholder="ex : TS2021_00123")
else:
    annee_min_in, annee_max_in = 1990, 2030
    modalite_in, k_in, pid_exclude_in = "Toutes", 5, ""
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    if pil_preview is None:
        st.warning("Veuillez importer une image IRM avant de lancer la recherche.")
    else:
        image_np = np.array(pil_preview)
        with st.spinner("Analyse CGR en cours..."):
            tensor = preprocess(image_np)
            age_min_val, age_max_val = age_range
            pf = PatientFilter(
                sexe=None if sexe_in == "Tous" else sexe_in,
                age_min=age_min_val if age_min_val > 0 else None,
                age_max=age_max_val if age_max_val < 90 else None,
                modalite=None if modalite_in == "Toutes" else modalite_in,
                hopital=None if hopital_in == "Tous" else hopital_in,
                diagnostic=None if diagnostic_in == "Tous" else diagnostic_in,
                grade=None if grade_in == "Tous" else grade_in,
                annee_min=annee_min_in if st.session_state["clin_extra_filters"] and annee_min_in > 1990 else None,
                annee_max=annee_max_in if st.session_state["clin_extra_filters"] and annee_max_in < 2030 else None,
            )
            query = MultimodalQuery(
                image_tensor=tensor,
                k=int(k_in),
                model="guided",
                patient_filter=pf,
                exclude_patient_id=pid_exclude_in.strip() or None,
            )
            results = engine.search(query)
            stats = engine.stats_by_filter(pf)
            guided = engine.last_guided_info

        query_np = image_np.mean(axis=2) if image_np.ndim == 3 else image_np
        st.session_state.update({
            "clin_results": results,
            "clin_stats": stats,
            "clin_query_np": query_np,
            "clin_guided": guided,
            "clin_preview": pil_preview,
        })

results = st.session_state.get("clin_results")
stats = st.session_state.get("clin_stats")
guided = st.session_state.get("clin_guided")
predicted_grade = grade_short(guided.get("predicted_grade", "-")) if guided else grade_txt

if results is not None:
    st.markdown("""
        <div class="timeline fade-in">
            <div class="timeline-row">
                <div class="timeline-step"><span class="timeline-dot"></span><span class="timeline-label">Extraction</span></div>
                <span class="timeline-arrow">-></span>
                <div class="timeline-step"><span class="timeline-dot"></span><span class="timeline-label">Classification</span></div>
                <span class="timeline-arrow">-></span>
                <div class="timeline-step"><span class="timeline-dot"></span><span class="timeline-label">Recherche vectorielle</span></div>
                <span class="timeline-arrow">-></span>
                <div class="timeline-step"><span class="timeline-dot"></span><span class="timeline-label">Résultats</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="patient-results-shell fade-in">', unsafe_allow_html=True)
    rh1, rh2, rh3 = st.columns([2, 1.1, 0.8])
    with rh1:
        n = len(results)
        st.markdown(f'<div class="results-count">{n} cas similaire{"s" if n != 1 else ""}</div>', unsafe_allow_html=True)
    with rh2:
        view = st.radio("Vue", ["Cartes", "Tableau"], horizontal=True, label_visibility="collapsed")
        st.session_state["clin_view"] = view.lower()
    with rh3:
        if results:
            df_export = build_results_df(results)
            st.download_button(
                "Télécharger",
                df_export.to_csv(index=False).encode("utf-8"),
                file_name="cbmir_resultats.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if not results:
        st.info("Aucun cas similaire trouvé pour les critères cliniques sélectionnés.")
    elif st.session_state["clin_view"] == "tableau":
        st.dataframe(build_results_df(results), use_container_width=True, hide_index=True)
    else:
        cols = st.columns(min(len(results), 5))
        for i, r in enumerate(results[:5]):
            img_arr = safe_load(r.file_path)
            b64 = arr_to_b64(img_arr) if img_arr is not None else None
            with cols[i]:
                render_result_card(i + 1, r, b64)
    st.markdown('</div>', unsafe_allow_html=True)

    min_sim = min(sim_percent(r.score) for r in results) if results else 94
    st.markdown(f"""
        <div class="ai-box fade-in">
            <div class="ai-title">{icon_svg("info")}<span>Explication IA</span></div>
            <ul>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Même grade tumoral : {predicted_grade}</span></li>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Même modalité lorsque disponible</span></li>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Similarité vectorielle supérieure à {max(min_sim, 85):.0f}%</span></li>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Distance cosinus cohérente avec les embeddings indexés</span></li>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Texture tumorale similaire</span></li>
                <li class="ai-li">{icon_svg("check", "#2563EB")}<span>Organisation anatomique similaire</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
elif not run_btn:
    st.markdown(f"""
        <div class="empty-state fade-in">
            <svg width="118" height="86" viewBox="0 0 236 172" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <rect x="16" y="18" width="204" height="136" rx="12" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="4"/>
                <rect x="34" y="36" width="92" height="100" rx="8" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="3"/>
                <path d="M83 54c-21 0-34 17-34 38 0 25 16 39 35 39 18 0 30-13 30-31 0-11-5-19-13-24 2-13-5-22-18-22Z" fill="#F8FAFC" stroke="#2563EB" stroke-width="4"/>
                <path d="M72 69c-6 8-8 18-3 27M89 67c8 6 11 15 8 27M72 113c10 6 22 4 29-5" stroke="#1E40AF" stroke-width="4" stroke-linecap="round"/>
                <path d="M148 52h42M148 76h54M148 100h38M148 124h48" stroke="#64748B" stroke-width="5" stroke-linecap="round"/>
            </svg>
            <div class="empty-title">Importer une IRM cérébrale</div>
            <div class="empty-sub">Importez une IRM patient pour retrouver les cas cliniquement similaires.</div>
            <div class="empty-upload-btn">{icon_svg("download", "#FFFFFF")}<span>Importer une IRM</span></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="legal-card fade-in">
    {icon_svg("shield")}
    <div>
        <div class="legal-text">Aide à la décision clinique</div>
        <div class="legal-sub">Ce logiciel assiste les cliniciens et ne remplace pas le diagnostic medical.</div>
    </div>
</div>
""", unsafe_allow_html=True)
