"""
CBMIR — Interface Technique (Streamlit)
Design : Dashboard analytique médical — Mode sombre natif
Sections :
  1. 🔍 Pipeline Explorer  — vecteur latent, latences, métriques, résultats visuels
  2. ⚖️ Comparaison Modèles — Baseline vs SupCon vs Guided (CGR)
  3. 📊 Évaluation (P@K)   — Precision@K par grade tumoral
  4. ⚙️ Architecture        — Spécifications techniques
Port : 8501
"""

import os, sys, time, warnings, html as html_module
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recherche.unified_engine import UnifiedSearchEngine
from src.evaluation.medical_metrics import (
    compute_all, filter_valid_results, is_valid_slice, SSIM_DISPLAY_MIN
)
from src.evaluation.precision_at_k import (
    run_evaluation, pick_query_slices_stratified, load_cache, K_VALUES, MODELS
)
from dotenv import load_dotenv

load_dotenv()

# ── Configuration Page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBMIR Platform — Tableau de bord de recherche",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Design Myhealth (dark dashboard) ────────────────────────────────────
st.markdown("""
<style>

/* ================================================
   GOOGLE FONT IMPORT
================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ================================================
   DESIGN TOKENS — Myhealth palette
================================================ */
:root {
    --bg:           #0A0E1A;
    --surface:      #111827;
    --surface-2:    #1A2235;
    --surface-3:    #1E293B;
    --border:       rgba(255,255,255,0.07);
    --border-hover: rgba(99,179,237,0.35);

    --accent:       #3B82F6;
    --accent-glow:  rgba(59,130,246,0.18);
    --cyan:         #06B6D4;
    --cyan-glow:    rgba(6,182,212,0.15);
    --success:      #10B981;
    --warning:      #F59E0B;
    --danger:       #EF4444;

    --text:         #F1F5F9;
    --text-2:       #CBD5E1;
    --muted:        #64748B;
    --muted-2:      #475569;

    --radius-sm:    10px;
    --radius-md:    16px;
    --radius-lg:    22px;

    --shadow:       0 4px 24px rgba(0,0,0,0.45);
    --shadow-card:  0 2px 12px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.04);
}

/* ================================================
   BASE RESET
================================================ */
*, *::before, *::after { box-sizing: border-box; }

html, body, .main, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 100% !important;
}

/* ================================================
   SCROLLBAR
================================================ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 3px; }

/* ================================================
   SIDEBAR — Myhealth left panel style
================================================ */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

.sidebar-header {
    padding: 22px 20px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.sidebar-logo {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(59,130,246,0.4);
}

.sidebar-brand-text .sidebar-title {
    color: var(--text);
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    line-height: 1;
    margin-bottom: 2px;
}

.sidebar-brand-text .sidebar-sub {
    color: var(--muted);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* Nav section label */
.nav-section-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    padding: 12px 20px 6px;
    margin-top: 4px;
}

/* Radio nav override */
section[data-testid="stSidebar"] .stRadio > label {
    display: none;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px !important;
    padding: 4px 10px;
}

section[data-testid="stSidebar"] .stRadio label[data-testid="stMarkdownContainer"] {
    display: flex !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    color: var(--text-2) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: var(--surface-3) !important;
    color: var(--text) !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-selected="true"],
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + label {
    background: var(--accent-glow) !important;
    color: #93C5FD !important;
}

/* Status chips in sidebar */
.status-section {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}

.status-section-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
}

.status-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-2);
    font-weight: 500;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px rgba(16,185,129,0.7);
    flex-shrink: 0;
}

/* ================================================
   HEADER BANNER
================================================ */
.dashboard-header {
    background: linear-gradient(135deg, #0F172A 0%, #131F38 50%, #0F172A 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px 28px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}

.dashboard-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.dashboard-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 30%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.header-left {}

.dashboard-title {
    font-size: 24px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    margin: 0 0 4px;
    line-height: 1.15;
}

.dashboard-title span {
    background: linear-gradient(90deg, #60A5FA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dashboard-subtitle {
    color: var(--muted);
    font-size: 12.5px;
    font-weight: 400;
    letter-spacing: 0.1px;
}

.header-badges {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-shrink: 0;
}

.header-badge {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-2);
    letter-spacing: 0.3px;
}

.header-badge.active {
    background: var(--accent-glow);
    border-color: rgba(59,130,246,0.4);
    color: #93C5FD;
}

.header-badge.cyan {
    background: var(--cyan-glow);
    border-color: rgba(6,182,212,0.35);
    color: #67E8F9;
}

/* ================================================
   SECTION EYEBROW (labels above sections)
================================================ */
.section-eyebrow {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.6px !important;
    color: var(--muted) !important;
    margin-bottom: 10px !important;
    margin-top: 0 !important;
}

/* ================================================
   KPI / STAT CARDS — Myhealth top cards
================================================ */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}

.kpi-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-1px);
}

.kpi-card-icon {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--surface-3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    margin-bottom: 14px;
    border: 1px solid var(--border);
}

.kpi-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 6px;
    letter-spacing: 0.2px;
}

.kpi-value {
    color: var(--text);
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
}

.kpi-delta {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--success);
}

.kpi-delta.up::before { content: '▲ '; font-size: 9px; }
.kpi-delta.down { color: var(--danger); }
.kpi-delta.down::before { content: '▼ '; font-size: 9px; }

.kpi-description {
    color: var(--muted);
    font-size: 11px;
    margin-top: 3px;
}

/* ================================================
   CONTENT CARDS
================================================ */
.custom-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    box-shadow: var(--shadow-card);
    margin-bottom: 14px;
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    letter-spacing: -0.2px;
}

.card-subtitle {
    font-size: 11.5px;
    color: var(--muted);
    margin-bottom: 16px;
}

/* ================================================
   PIPELINE FLOW BAR
================================================ */
.pipeline-flow {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 12px;
    color: var(--muted);
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    letter-spacing: 0.2px;
    margin-bottom: 18px;
    line-height: 1.6;
}

.pipeline-flow strong { color: #60A5FA; }

/* ================================================
   VECTOR BOX
================================================ */
.vector-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    font-size: 11.5px;
    color: #67E8F9;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ================================================
   SPEC CARD (Architecture page)
================================================ */
.spec-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    margin-bottom: 12px;
}

/* ================================================
   STREAMLIT METRIC OVERRIDE
================================================ */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 18px !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* ================================================
   BUTTONS
================================================ */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    height: 44px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.35) !important;
}

.stButton > button:hover {
    background: #2563EB !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ================================================
   FILE UPLOADER
================================================ */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed rgba(59,130,246,0.4) !important;
    border-radius: var(--radius-md) !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ================================================
   DATAFRAME / TABLE — conteneur visible
================================================ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: auto !important;
    border: 2px solid rgba(59,130,246,0.55) !important;
    background: #1E293B !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.15) !important;
}

/* ================================================
   HTML TABLES — toujours visibles (pas de canvas)
================================================ */
.cbmir-table-wrap {
    overflow-x: auto;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-md);
    background: var(--bg);
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    margin-bottom: 8px;
}
table.cbmir-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
}
table.cbmir-table thead th {
    background: var(--surface-2);
    color: #E5E7EB;
    font-weight: 600;
    font-size: 12px;
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    white-space: nowrap;
}
table.cbmir-table tbody td {
    color: #D1D5DB;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-weight: 500;
}
table.cbmir-table tbody tr.row-odd td  { background: var(--bg); }
table.cbmir-table tbody tr.row-even td { background: var(--bg); }
table.cbmir-table tbody tr:hover td    { background: var(--surface); }

/* ================================================
   TABS (Model comparison)
================================================ */
.stTabs [data-testid="stHorizontalBlock"] {
    gap: 4px;
}

.stTabs [role="tab"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 7px 16px !important;
    transition: all 0.15s !important;
}

.stTabs [role="tab"]:hover {
    border-color: var(--border-hover) !important;
    color: var(--text) !important;
}

.stTabs [role="tab"][aria-selected="true"] {
    background: var(--accent-glow) !important;
    border-color: rgba(59,130,246,0.5) !important;
    color: #93C5FD !important;
}

/* ================================================
   SELECTBOX / NUMBER INPUT / TEXT INPUT
================================================ */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-size: 13px !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stSlider label,
.stRadio label,
.stFileUploader label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text-2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ================================================
   DIVIDER
================================================ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 18px 0 !important;
}

/* ================================================
   EXPANDER
================================================ */
.streamlit-expander {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ================================================
   INFO / WARNING / ERROR BOXES
================================================ */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
}

/* ================================================
   SPINNER
================================================ */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ================================================
   PAGE TITLE h2
================================================ */
h2 {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.3px !important;
    margin-top: 0 !important;
    margin-bottom: 4px !important;
}

h3, h4 {
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}

/* ================================================
   CAPTION / SMALL TEXT
================================================ */
.stCaption, small, caption {
    color: var(--muted) !important;
    font-size: 11.5px !important;
}

/* ================================================
   ESPACE BLANC EN HAUT — suppression
================================================ */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stApp > header { display: none !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stVerticalBlock"] { background: transparent !important; }
[data-testid="column"] { background: transparent !important; }

/* ================================================
   FORCER TOUTES LES COULEURS DE TEXTE
================================================ */
.stMarkdown p,
.stMarkdown span,
.stMarkdown li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--text-2) !important;
}

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
}

[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 600 !important; }

/* Inputs */
input, textarea, select {
    color: var(--text) !important;
    background: var(--surface-2) !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] div { color: var(--text) !important; background: var(--surface-2) !important; }
[data-baseweb="popover"] { background: var(--surface-2) !important; }
[data-baseweb="menu"] li { color: var(--text) !important; background: var(--surface-2) !important; }
[data-baseweb="menu"] li:hover { background: var(--surface-3) !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderFileName"] { color: var(--text-2) !important; }

/* Expander */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: var(--text-2) !important; }

/* Slider / Radio */
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stRadio"] div[role="radiogroup"] label p { color: var(--text-2) !important; }

/* Alerts */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span { color: var(--text) !important; }

/* Caption */
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 11.5px !important; }

/* Divider */
[data-testid="stDivider"] hr { border-color: var(--border) !important; opacity: 1 !important; }

[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] .dvn-scroller {
    background: #1E293B !important;
    color: #F8FAFC !important;
}

</style>
""", unsafe_allow_html=True)

# ── Header Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div class="header-left">
        <div class="dashboard-title">CBMIR <span>Plateforme de recherche</span></div>
        <div class="dashboard-subtitle">
            IRM cérébrale · Recherche d'images par contenu · Espace d'évaluation technique
        </div>
    </div>
    <div class="header-badges">
        <span class="header-badge active">◎ En direct</span>
        <span class="header-badge cyan">BraTS 2021</span>
        <span class="header-badge">v1.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Initialisation du moteur (mis en cache) ────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles et connexion aux bases…")
def load_engine():
    return UnifiedSearchEngine()

engine = load_engine()


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation session_state
# ─────────────────────────────────────────────────────────────────────────────

# Page 1 — Pipeline Explorer
if "pipeline_image"   not in st.session_state: st.session_state["pipeline_image"]   = None
if "pipeline_results" not in st.session_state: st.session_state["pipeline_results"] = None

# Page 2 — Comparaison Modèles
if "cmp_image"        not in st.session_state: st.session_state["cmp_image"]        = None
if "cmp_results"      not in st.session_state: st.session_state["cmp_results"]      = None
if "cmp_fig_bytes"    not in st.session_state: st.session_state["cmp_fig_bytes"]    = None

# Page 3 — Evaluation P@K
if "eval_results"     not in st.session_state: st.session_state["eval_results"]     = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Styling tableaux — contraste élevé (lisible sur fond #0A0E1A)
_TABLE_CELL_PROPS = {
    "background-color": "#243044",
    "color":            "#F8FAFC",
    "border-color":     "#475569",
    "font-size":        "13px",
}
_TABLE_HEADER_STYLES = [
    {
        "selector": "thead th",
        "props": [
            ("background-color", "#1D4ED8"),
            ("color",            "#FFFFFF"),
            ("font-weight",      "700"),
            ("font-size",        "12px"),
            ("padding",          "10px 8px"),
            ("border",           "1px solid #3B82F6"),
        ],
    },
    {
        "selector": "tbody tr:nth-child(even) td",
        "props": [("background-color", "#2D3748"), ("color", "#F8FAFC")],
    },
    {
        "selector": "tbody tr:nth-child(odd) td",
        "props": [("background-color", "#243044"), ("color", "#F8FAFC")],
    },
]

def style_dark_table(df: pd.DataFrame):
    """Force des couleurs lisibles dans st.dataframe (thème sombre)."""
    return (
        df.style
        .set_properties(**_TABLE_CELL_PROPS)
        .set_table_styles(_TABLE_HEADER_STYLES, overwrite=False)
    )

def style_eval_table(df: pd.DataFrame):
    """Tableau P@K avec dégradé vert + texte toujours lisible."""
    return (
        df.style
        .background_gradient(subset=["Precision@K"], cmap="Greens", text_color_threshold=0.45)
        .set_properties(**{"color": "#F8FAFC", "font-size": "13px"})
        .set_table_styles(_TABLE_HEADER_STYLES, overwrite=False)
    )

def render_html_table(df: pd.DataFrame) -> None:
    """Tableau HTML statique — visible sans clic (contourne le canvas st.dataframe)."""
    headers = "".join(f"<th>{html_module.escape(str(col))}</th>" for col in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        zebra = "even" if i % 2 else "odd"
        cells = "".join(
            f"<td>{html_module.escape(str(row[col]))}</td>" for col in df.columns
        )
        rows.append(f'<tr class="row-{zebra}">{cells}</tr>')
    st.markdown(
        f'<div class="cbmir-table-wrap">'
        f'<table class="cbmir-table"><thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

_PIPELINE_COL_CONFIG = {
    "Rang":       st.column_config.NumberColumn("Rang",      width="small",  format="%d"),
    "Patient ID": st.column_config.TextColumn("Patient ID",  width="medium"),
    "Modalité":   st.column_config.TextColumn("Modalité",    width="small"),
    "Coupe z":    st.column_config.NumberColumn("Coupe z",   width="small",  format="%d"),
    "Cosinus":    st.column_config.NumberColumn("Cosinus",   width="medium", format="%.4f"),
    "SSIM":       st.column_config.NumberColumn("SSIM",      width="medium", format="%.4f"),
    "PSNR":       st.column_config.NumberColumn("PSNR",      width="small",  format="%.1f"),
}

def preprocess(image_np: np.ndarray) -> torch.Tensor:
    import torch.nn.functional as F
    img = image_np.mean(axis=2).astype("float32") \
          if image_np.ndim == 3 else image_np.astype("float32")
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


def transparent_fig():
    """Applique un fond transparent à la figure courante (compatible thème Streamlit)."""
    fig = plt.gcf()
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.tick_params(colors="#aaaaaa")
        ax.title.set_color("#dddddd")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
    return fig


def build_pipeline_figure(query_np, results) -> plt.Figure:
    n = min(len(results), 5)
    ncols = n + 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3.4))
    fig.patch.set_facecolor("none")

    ACCENT = {"query": "#4fc3f7", "high": "#34d399", "mid": "#fbbf24", "low": "#f87171"}

    def show(ax, img, title, color):
        ax.set_facecolor("#1a1d24")
        if img is not None:
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color="#64748b", fontsize=13, transform=ax.transAxes)
        ax.set_title(title, color=color, fontsize=7.5,
                     fontfamily="monospace", pad=5, fontweight="bold")
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    show(axes[0], query_np, "[ REQUÊTE ]", ACCENT["query"])
    for i, r in enumerate(results[:n]):
        img = safe_load(r.file_path)
        color = ACCENT["high"] if r.score > 0.65 else ACCENT["mid"] if r.score > 0.45 else ACCENT["low"]
        show(axes[i + 1], img,
             f"[ RANG {i+1} ] {r.score:.3f}\n{r.patient_id[-8:]}", color)

    plt.tight_layout(pad=0.8)
    return fig


def build_comparison_figure(query_np, results_b, results_s, results_g) -> plt.Figure:
    n = min(5, len(results_b), len(results_s), len(results_g))
    fig, axes = plt.subplots(4, n + 1, figsize=((n + 1) * 3.1, 11.5), squeeze=False)
    fig.patch.set_facecolor("none")

    MODEL_COLORS = {"baseline": "#3b82f6", "supcon": "#a78bfa", "guided": "#10b981"}

    def show(ax, img, title, color, bw=1.5):
        ax.set_facecolor("#1a1d24")
        if img is not None:
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/D", ha="center", va="center",
                    color="#475569", fontsize=11, transform=ax.transAxes)
        ax.set_title(title, color=color, fontsize=7.5, fontweight="bold", pad=3)
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(bw)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[0]:
        ax.set_facecolor("none")
        ax.axis("off")
    axes[0, 0].text(0.5, 0.5, "IMAGE\nREQUÊTE", ha="center", va="center",
                    color="#94a3b8", fontsize=9, fontweight="bold", transform=axes[0, 0].transAxes)
    for j in range(1, n + 1):
        axes[0, j].text(0.5, 0.5, f"Résultat #{j}", ha="center", va="center",
                        color="#64748b", fontsize=8.5, fontweight="bold", transform=axes[0, j].transAxes)

    for row_idx, (results, model_key) in enumerate(
        [(results_b, "baseline"), (results_s, "supcon"), (results_g, "guided")], start=1
    ):
        color_model = MODEL_COLORS[model_key]
        show(axes[row_idx, 0], query_np, "REQUÊTE", "#4fc3f7", 2.5)
        for i, r in enumerate(results[:n]):
            img = safe_load(r.file_path)
            score_color = "#34d399" if r.score > 0.65 else "#fbbf24"
            show(axes[row_idx, i + 1], img,
                 f"#{r.rank}  {r.score:.4f}\n{r.modalite.upper()} z={r.slice_z}", score_color)

    labels = [
        (0.66, "BASELINE\nNon-supervisé", "#3b82f6"),
        (0.40, "SUPCON\nContrastif",       "#a78bfa"),
        (0.15, "GUIDED\nCGR",             "#10b981"),
    ]
    for y, lbl, col in labels:
        fig.text(0.005, y, lbl, color=col, fontsize=9.5, fontweight="bold",
                 va="center", rotation=90)

    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions métier (identiques à Gradio, adaptées au flux Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def run_tech_analysis(image_np, model_choice, k, modalite, pid_exclude):
    t0 = time.time()
    tensor   = preprocess(image_np)
    query_np = tensor.squeeze().numpy()
    pid_excl = pid_exclude.strip() if pid_exclude else None

    model_map = {
        "Baseline": "baseline",
        "SupCon": "supcon",
        "Guided (CGR)": "guided",
    }
    selected_engine = model_map.get(model_choice, "baseline")

    t_enc_start = time.time()
    if selected_engine == "baseline":
        vecteur_brut = engine.encode_baseline(tensor)
    elif selected_engine == "guided":
        grade_info = engine.predict_grade(tensor)
        vecteur_brut = grade_info["vector"]
    else:
        vecteur_brut = engine.encode_supcon(tensor)
    t_enc_end = time.time()

    t_db_start = time.time()
    results = engine.search(
        image_tensor       = tensor,
        model              = selected_engine,
        k                  = int(k),
        modalite           = None if modalite == "Toutes" else modalite,
        exclude_patient_id = pid_excl,
    )
    t_db_end = time.time()

    if not results:
        return None, None, None, None, None, None, None

    t_met_start = time.time()
    n_before = len(results)
    results  = filter_valid_results(results, query_np, safe_load, ssim_min=SSIM_DISPLAY_MIN)
    n_after  = len(results)
    n_filtered = n_before - n_after
    t_met_end = time.time()
    t_total   = time.time() - t0

    if not results:
        return None, None, None, None, None, None, None

    # Payload top-1
    r0 = results[0]
    payload_top1 = {
        "rank"           : r0.rank,
        "slice_id"       : r0.slice_id,
        "distance_cosine": r0.score,
        "model"          : selected_engine,
        "embedding_dim"  : len(vecteur_brut),
        "patient"        : {
            "patient_id": r0.patient_id,
            "modalite"  : r0.modalite,
            "slice_z"   : r0.slice_z,
            "stats"     : r0.stats,
            "grade"     : getattr(r0, "grade", ""),
        },
        "metrics": r0.metrics,
    }
    if selected_engine == "guided" and engine.last_guided_info:
        payload_top1["guided"] = engine.last_guided_info

    latencies = {
        "encoding_ms"  : round((t_enc_end - t_enc_start) * 1000, 1),
        "db_search_ms" : round((t_db_end - t_db_start) * 1000, 1),
        "metrics_ms"   : round((t_met_end - t_met_start) * 1000, 1),
        "total_ms"     : round(t_total * 1000, 1),
    }

    vec_info = {
        "dim"    : len(vecteur_brut),
        "norm"   : float(np.linalg.norm(vecteur_brut)),
        "min"    : float(min(vecteur_brut)),
        "max"    : float(max(vecteur_brut)),
        "preview": (
            f"[{', '.join([f'{x:.4f}' for x in vecteur_brut[:8]])}  …  "
            f"{', '.join([f'{x:.4f}' for x in vecteur_brut[-8:]])}]"
        ),
    }

    # DataFrame résultats
    rows = []
    for r in results:
        m = r.metrics if hasattr(r, "metrics") and r.metrics else {}
        rows.append({
            "Rang"      : r.rank,
            "Patient ID": r.patient_id[-12:],
            "Modalité"  : r.modalite.upper(),
            "Coupe z"   : r.slice_z,
            "Grade OMS" : getattr(r, "grade", "") or "N/D",
            "Cosinus"   : round(float(r.score), 4),
            "SSIM"      : round(float(m.get("ssim", 0)), 4),
            "PSNR"      : round(float(m.get("psnr", 0)), 1),
        })
    df = pd.DataFrame(rows)

    fig = build_pipeline_figure(query_np, results)

    return fig, df, payload_top1, latencies, vec_info, n_filtered, len(set(r.patient_id for r in results))


def run_comparison(image_np, k, modalite, pid_exclude):
    tensor   = preprocess(image_np)
    query_np = tensor.squeeze().numpy()
    pid_excl = pid_exclude.strip() if pid_exclude else None
    mod      = None if modalite == "Toutes" else modalite

    def _avg_metric(results, metric_name):
        values = []
        for result in results or []:
            metrics = result.metrics if hasattr(result, "metrics") and result.metrics else {}
            values.append(float(metrics.get(metric_name, 0.0)))
        return round(float(np.mean(values)), 4) if values else 0.0

    def _avg_hist_normalized(results):
        values = []
        for result in results or []:
            metrics = result.metrics if hasattr(result, "metrics") and result.metrics else {}
            hist_value = float(metrics.get("hist", 0.0))
            values.append(np.clip((hist_value + 1.0) / 2.0, 0.0, 1.0))
        return round(float(np.mean(values)), 4) if values else 0.0

    def _avg_psnr_normalized(results):
        values = []
        for result in results or []:
            metrics = result.metrics if hasattr(result, "metrics") and result.metrics else {}
            psnr_value = float(metrics.get("psnr", 0.0))
            values.append(np.clip(psnr_value / 100.0, 0.0, 1.0))
        return round(float(np.mean(values)), 4) if values else 0.0

    t_b = time.time()
    results_b = engine.search(image_tensor=tensor, model="baseline", k=int(k), modalite=mod, exclude_patient_id=pid_excl)
    t_b = (time.time() - t_b) * 1000

    t_s = time.time()
    results_s = engine.search(image_tensor=tensor, model="supcon",   k=int(k), modalite=mod, exclude_patient_id=pid_excl)
    t_s = (time.time() - t_s) * 1000

    t_g = time.time()
    results_g = engine.search(image_tensor=tensor, model="guided",   k=int(k), modalite=mod, exclude_patient_id=pid_excl)
    t_g = (time.time() - t_g) * 1000

    results_b = filter_valid_results(results_b, query_np, safe_load, ssim_min=SSIM_DISPLAY_MIN)
    results_s = filter_valid_results(results_s, query_np, safe_load, ssim_min=SSIM_DISPLAY_MIN)
    results_g = filter_valid_results(results_g, query_np, safe_load, ssim_min=SSIM_DISPLAY_MIN)

    def avg_score(res):
        return round(float(np.mean([r.score for r in res])), 4) if res else 0.0

    def make_df(results):
        rows = []
        for r in results:
            m = r.metrics if hasattr(r, "metrics") and r.metrics else {}
            rows.append({
                "Rang"      : r.rank,
                "Patient ID": r.patient_id[-12:],
                "Mod"       : r.modalite.upper(),
                "z"         : r.slice_z,
                "Cosinus"   : round(float(r.score), 4),
                "SSIM"      : round(float(m.get("ssim", 0)), 4),
            })
        return pd.DataFrame(rows)

    fig = build_comparison_figure(query_np, results_b, results_s, results_g)

    radar = {
        "baseline": {
            "Cosinus": avg_score(results_b),
            "SSIM": _avg_metric(results_b, "ssim"),
            "HIST": _avg_hist_normalized(results_b),
            "Normalized PSNR": _avg_psnr_normalized(results_b),
        },
        "supcon": {
            "Cosinus": avg_score(results_s),
            "SSIM": _avg_metric(results_s, "ssim"),
            "HIST": _avg_hist_normalized(results_s),
            "Normalized PSNR": _avg_psnr_normalized(results_s),
        },
        "guided": {
            "Cosinus": avg_score(results_g),
            "SSIM": _avg_metric(results_g, "ssim"),
            "HIST": _avg_hist_normalized(results_g),
            "Normalized PSNR": _avg_psnr_normalized(results_g),
        },
    }

    scores = {
        "baseline": (avg_score(results_b), round(t_b, 0)),
        "supcon"  : (avg_score(results_s), round(t_s, 0)),
        "guided"  : (avg_score(results_g), round(t_g, 0)),
    }
    winner = max(scores.items(), key=lambda x: x[1][0])[0]

    return fig, make_df(results_b), make_df(results_s), make_df(results_g), scores, winner, radar


def run_precision_eval(n_queries, k_max, model_choice):
    k_vals = [k for k in K_VALUES if k <= int(k_max)]
    if not k_vals:
        k_vals = K_VALUES

    models_to_run = MODELS if model_choice == "Tous" else [
        m for m in MODELS if m == model_choice.lower()
    ]
    report = run_evaluation(
        n_per_grade=int(n_queries) // 3, # ✅ CORRECTI ON ICI
        k_values=k_vals,
        models=models_to_run,
        save_report=True,
    )

    if not report:
        return None, None

    results = report["results"]
    from src.evaluation.precision_at_k import build_report_figure
    fig = build_report_figure(results)

    rows = []
    for r in results:
        row = {"Modèle": r["model"].upper(), "K": r["k"], "Precision@K": round(r["global"], 4)}
        row.update({f"Grade {g}": round(v, 4) for g, v in r.get("by_grade", {}).items()})
        rows.append(row)

    return fig, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">◎</div>
        <div class="sidebar-brand-text">
            <p class="sidebar-title">CBMIR</p>
            <p class="sidebar-sub">Inspector · v1.0</p>
        </div>
    </div>
    <div class="nav-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["Exploration du pipeline", "Comparaison des modèles", "Évaluation (P@K)", "Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="status-section">
        <div class="status-section-label">Statut Système</div>
        <div class="status-chip"><span class="status-dot"></span>Qdrant — Connecté</div>
        <div class="status-chip"><span class="status-dot"></span>MongoDB — Connecté</div>
        <div class="status-chip"><span class="status-dot"></span>Modèles Chargés</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("BraTS 2021 · Projet de fin d'études")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Exploration du pipeline
# ─────────────────────────────────────────────────────────────────────────────

if page == "Exploration du pipeline":
    st.markdown("## Analyse de récupération IRM")
    st.markdown(
        "<div class='pipeline-flow'>"
        "Image IRM → Prétraitement [128×128] → Encodeur → Vecteur latent "
        "→ Qdrant ANN (cosinus) → <strong>Top-K patients</strong> "
        "→ Enrichissement MongoDB → Métriques de similarité"
        "</div>",
        unsafe_allow_html=True,
    )

    col_params, col_results = st.columns([1, 2], gap="large")

    with col_params:
        st.markdown("<p class='section-eyebrow'>Paramètres de recherche</p>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Image IRM (PNG / JPG / NPY)",
            type=["png", "jpg", "jpeg"],
            help="Importez une coupe IRM cérébrale au format image.",
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            image_np = np.array(img_pil)
            st.image(img_pil, caption="Image uploadée", use_container_width=True)
            st.session_state["pipeline_image"] = image_np
        else:
            image_np = st.session_state.get("pipeline_image", None)

        model_choice = st.selectbox(
            "Modèle extracteur",
            ["Baseline", "SupCon", "Guided (CGR)"],
            index=2,
        )
        col_k, col_mod = st.columns(2)
        k_val    = col_k.number_input("Top-K résultats", min_value=1, max_value=20, value=5, step=1)
        modalite = col_mod.selectbox("Modalité", ["Toutes", "t1", "t1ce", "t2", "flair"])
        pid_excl = st.text_input("Exclure Patient ID", placeholder="BraTS2021_00001",
                                 help="Optionnel — exclure le patient de la requête pour éviter l'auto-match.")

        run_btn = st.button("▶  Lancer le pipeline", type="primary", use_container_width=True)

    with col_results:
        # ── Lancement et sauvegarde en session_state ──────────────────────
        if run_btn:
            if image_np is None:
                st.warning("Veuillez importer une image IRM avant de lancer le pipeline.")
            else:
                with st.spinner("Encodage · Recherche vectorielle · Calcul des métriques…"):
                    fig, df, payload, latencies, vec_info, n_filtered, n_distinct = run_tech_analysis(
                        image_np, model_choice, k_val, modalite, pid_excl
                    )
                if fig is None:
                    st.session_state["pipeline_results"] = None
                    st.error("Aucun résultat valide retourné. Vérifiez les paramètres.")
                else:
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="none", dpi=120)
                    plt.close(fig)
                    buf.seek(0)
                    st.session_state["pipeline_results"] = {
                        "fig_bytes" : buf.read(),
                        "df"        : df,
                        "payload"   : payload,
                        "latencies" : latencies,
                        "vec_info"  : vec_info,
                        "n_filtered": n_filtered,
                        "n_distinct": n_distinct,
                        "k_val"     : int(k_val),
                    }

        # ── Affichage permanent depuis session_state ──────────────────────
        if st.session_state["pipeline_results"] is not None:
            res        = st.session_state["pipeline_results"]
            df         = res["df"]
            payload    = res["payload"]
            latencies  = res["latencies"]
            vec_info   = res["vec_info"]
            n_filtered = res["n_filtered"]
            n_distinct = res["n_distinct"]
            k_val_disp = res["k_val"]

            # ── Meilleur résultat (rang 1) ────────────────────────
            if payload:
                st.markdown("<p class='section-eyebrow'>Meilleur résultat (rang 1)</p>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score Cosinus",      f"{payload['distance_cosine']:.4f}")
                c2.metric("Dimension Vecteur",  f"{payload['embedding_dim']}D")
                c3.metric("Patients distincts", f"{n_distinct}/{k_val_disp}")
                c4.metric("Total Pipeline",     f"{latencies['total_ms']} ms")

            st.divider()

            # ── Vecteur latent ────────────────────────────────────
            with st.expander("🧬 Vecteur Latent Extrait", expanded=False):
                st.markdown(
                    f"<div class='vector-box'>"
                    f"dim = {vec_info['dim']}D  |  "
                    f"norm = {vec_info['norm']:.6f}  |  "
                    f"min = {vec_info['min']:.4f}  max = {vec_info['max']:.4f}\n\n"
                    f"{vec_info['preview']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Résultats visuels ─────────────────────────────────
            st.divider()
            st.markdown("<p class='section-eyebrow'>Résultats visuels</p>", unsafe_allow_html=True)
            st.image(res["fig_bytes"], use_container_width=True)

            # ── Tableau interactif ────────────────────────────────
            st.markdown("<p class='section-eyebrow'>Tableau des résultats</p>", unsafe_allow_html=True)
            render_html_table(df)

            if n_filtered > 0:
                st.warning(f"{n_filtered} coupe(s) filtrée(s) car SSIM < {SSIM_DISPLAY_MIN:.2f}.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Comparaison des modèles
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Comparaison des modèles":
    st.markdown("## Comparatif des modèles d'IA")
    st.info(
        "**Comparaison directe** — Modèle non supervisé (Baseline) "
        "vs. contrastif supervisé (SupCon) vs. Classification-Guided Retrieval (Guided).",
    )

    col_q, col_vis = st.columns([1, 2], gap="large")

    with col_q:
        st.markdown("<p class='section-eyebrow'>Image de requête</p>", unsafe_allow_html=True)
        uploaded_cmp = st.file_uploader(
            "Image IRM", type=["png", "jpg", "jpeg"], key="cmp_uploader"
        )
        if uploaded_cmp:
            img_pil = Image.open(uploaded_cmp).convert("RGB")
            image_np_cmp = np.array(img_pil)
            st.image(img_pil, use_container_width=True)
            st.session_state["cmp_image"] = image_np_cmp
        else:
            image_np_cmp = st.session_state.get("cmp_image", None)

        col_kc, col_mc = st.columns(2)
        k_cmp    = col_kc.number_input("Top-K", min_value=1, max_value=20, value=5, step=1, key="k_cmp")
        mod_cmp  = col_mc.selectbox("Filtre de modalité", ["Toutes", "t1", "t1ce", "t2", "flair"], key="mod_cmp")
        pid_cmp  = st.text_input("Exclure Patient ID", key="pid_cmp")
        run_cmp  = st.button("▶  Lancer la comparaison", type="primary", use_container_width=True)

    with col_vis:
        # ── Lancement et sauvegarde en session_state ──────────────────────
        if run_cmp:
            if image_np_cmp is None:
                st.warning("Importez une image IRM pour lancer la comparaison.")
            else:
                with st.spinner("Interrogation des trois moteurs…"):
                    fig_cmp, df_b, df_s, df_g, scores, winner, radar = run_comparison(
                        image_np_cmp, k_cmp, mod_cmp, pid_cmp
                    )
                import io
                buf = io.BytesIO()
                fig_cmp.savefig(buf, format="png", bbox_inches="tight",
                                facecolor="none", dpi=120)
                plt.close(fig_cmp)
                buf.seek(0)
                st.session_state["cmp_fig_bytes"] = buf.read()
                st.session_state["cmp_results"] = {
                    "df_b": df_b, "df_s": df_s, "df_g": df_g,
                    "scores": scores, "winner": winner, "radar": radar,
                }

        # ── Affichage permanent depuis session_state ──────────────────────
        if st.session_state["cmp_results"] is not None:
            res      = st.session_state["cmp_results"]
            df_b     = res["df_b"]
            df_s     = res["df_s"]
            df_g     = res["df_g"]
            scores   = res["scores"]
            winner   = res["winner"]
            radar    = res["radar"]

            score_b, lat_b = scores["baseline"]
            score_s, lat_s = scores["supcon"]
            score_g, lat_g = scores["guided"]

            st.markdown("<p class='section-eyebrow'>Synthèse comparative</p>", unsafe_allow_html=True)
            cb, cs, cg = st.columns(3)
            cb.metric("Baseline", f"{score_b:.4f}",
                      delta="Meilleur" if winner == "baseline" else f"{lat_b:.0f} ms")
            cs.metric("SupCon", f"{score_s:.4f}",
                      delta="Meilleur" if winner == "supcon" else f"{lat_s:.0f} ms")
            cg.metric("Guided (CGR)", f"{score_g:.4f}",
                      delta="Meilleur" if winner == "guided" else f"{lat_g:.0f} ms")

            st.divider()

            # ── Visualisation comparée ─────────────────────────────
            if "cmp_fig_bytes" in st.session_state:
                st.markdown("<p class='section-eyebrow'>Résultats visuels comparés</p>", unsafe_allow_html=True)
                st.image(st.session_state["cmp_fig_bytes"], use_container_width=True)

            st.divider()

            # ── Graphique radar — comparaison unifiée ──────────────────────
            st.markdown("<p class='section-eyebrow'>Comparaison radar — métriques moyennes</p>", unsafe_allow_html=True)

            def _rgba(hex_color, alpha):
                h = hex_color.lstrip("#")
                return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

            radar_models = [
                ("Baseline",    radar["baseline"], "#3B82F6"),
                ("SupCon",      radar["supcon"],   "#8B5CF6"),
                ("Guided (CGR)", radar["guided"],  "#10B981"),
            ]

            radar_theta = ["Cosinus", "SSIM", "HIST", "Normalized PSNR"]

            fig_radar = go.Figure()
            for name, metrics, color in radar_models:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[metrics[axis] for axis in radar_theta],
                    theta=radar_theta,
                    fill="toself",
                    fillcolor=_rgba(color, 0.25),
                    line=dict(color=color, width=2),
                    name=name,
                    opacity=0.85,
                ))

            fig_radar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                legend=dict(font=dict(color="#CBD5E1")),
                height=520,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tick0=0,
                        dtick=0.2,
                        gridcolor="#CBD5E1",
                        linecolor="#CBD5E1",
                        tickfont=dict(color="#CBD5E1"),
                    ),
                    angularaxis=dict(
                        rotation=90,
                        direction="clockwise",
                        gridcolor="#CBD5E1",
                        linecolor="#CBD5E1",
                        tickfont=dict(color="#CBD5E1"),
                    ),
                ),
                margin=dict(l=30, r=30, t=40, b=30),
            )

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.plotly_chart(fig_radar, use_container_width=True, theme=None)

            # ── Ranking visuel ─────────────────────────────────
            st.divider()
            st.markdown("<p class='section-eyebrow'>Classement par score cosinus moyen</p>", unsafe_allow_html=True)

            ranking = [
                ("Baseline",     score_b, lat_b, "#3B82F6", winner == "baseline"),
                ("SupCon",       score_s, lat_s, "#8B5CF6", winner == "supcon"),
                ("Guided (CGR)", score_g, lat_g, "#10B981", winner == "guided"),
            ]
            ranking.sort(key=lambda x: x[1], reverse=True)
            max_score = max(r[1] for r in ranking)

            for rank_i, (name, score, lat, color, is_win) in enumerate(ranking, 1):
                bar_pct = int((score / max_score) * 100) if max_score > 0 else 0
                win_badge = (
                    f'<span style="background:rgba(52,211,153,0.15);border:1px solid #34d399;'
                    f'color:#34d399;border-radius:20px;padding:2px 10px;font-size:10px;'
                    f'font-weight:700;margin-left:8px;">GAGNANT</span>'
                    if is_win else ""
                )
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px;padding:14px 18px;margin-bottom:8px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                        <span style="font-size:14px;font-weight:700;color:#F1F5F9;">
                            #{rank_i} {name}{win_badge}
                        </span>
                        <div style="text-align:right;">
                            <span style="font-size:22px;font-weight:800;color:{color};
                                         letter-spacing:-0.5px;">{score:.4f}</span>
                            <span style="font-size:11px;color:#64748B;margin-left:8px;">{lat:.0f} ms</span>
                        </div>
                    </div>
                    <div style="background:#1E293B;border-radius:6px;height:8px;overflow:hidden;">
                        <div style="background:{color};height:100%;width:{bar_pct}%;
                                    border-radius:6px;opacity:0.85;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Évaluation Precision@K
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Évaluation (P@K)":
    st.markdown("## Validation scientifique")
    st.info(
        "**Precision@K par grade tumoral OMS** — métrique de validation scientifique "
        "calculée sur un échantillon stratifié du corpus BraTS 2021.",
        icon="📊",
    )

    col_ev_params, col_ev_results = st.columns([1, 2], gap="large")

    with col_ev_params:
        st.markdown("<p class='section-eyebrow'>Paramètres d'évaluation</p>", unsafe_allow_html=True)
        n_queries  = st.slider("Nombre de requêtes (N)", min_value=10, max_value=100, value=50, step=5)
        k_max_eval = st.radio("K maximum évalué", options=[1, 3, 5, 10], index=3, horizontal=True)
        model_eval = st.selectbox(
            "Modèle(s)",
            ["Tous", "Baseline", "Supcon", "Guided"],
        )
        run_eval   = st.button("▶  Lancer Precision@K", type="primary", use_container_width=True)

        st.caption(
            f"L'évaluation porte sur **{n_queries} requêtes** "
            f"jusqu'à **K={k_max_eval}** avec stratification par grade tumoral."
        )

    with col_ev_results:
        if run_eval:
            with st.spinner(f"Évaluation en cours sur {n_queries} requêtes…"):
                try:
                    fig_eval, df_eval = run_precision_eval(n_queries, k_max_eval, model_eval)
                except Exception as e:
                    st.error(f"Erreur lors de l'évaluation : {e}")
                    fig_eval, df_eval = None, None

            if fig_eval is None:
                st.error("Aucune requête valide. Vérifiez le corpus et la configuration.")
            else:
                st.markdown("<p class='section-eyebrow'>Courbes Precision@K</p>", unsafe_allow_html=True)
                st.pyplot(fig_eval, use_container_width=True)
                plt.close(fig_eval)

                st.divider()
                st.markdown("<p class='section-eyebrow'>Tableau de résultats détaillé</p>", unsafe_allow_html=True)
                render_html_table(df_eval)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Architecture
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Architecture":
    st.markdown("## Architecture du système")
    st.info(
        "Spécifications techniques des trois modèles d'extraction de caractéristiques "
        "et de l'infrastructure de recherche vectorielle.",
    )

    st.subheader("Modèles d'extraction")
    col_a1, col_a2, col_a3 = st.columns(3, gap="medium")

    with col_a1:
        st.markdown("""
        <div class="spec-card">
                    <h4 style="color:#3b82f6;">Modèle 1 — Baseline</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Architecture** : BraTSAutoencoderLightning")
        st.markdown("**Encodeur** : Conv2D (1→16→32→64)")
        st.markdown("**Collection Qdrant** : `brats_embeddings`")
        st.metric("Dimension Embedding", "256D")

    with col_a2:
        st.markdown("""
        <div class="spec-card">
                    <h4 style="color:#a78bfa;">Modèle 2 — SupCon</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Architecture** : BraTSAutoencoderSupervised")
        st.markdown("**Supervision** : Supervised Contrastive Loss (grades OMS)")
        st.markdown("**Collection Qdrant** : `brats_supcon_embeddings`")
        st.metric("Dimension Embedding", "256D")

    with col_a3:
        st.markdown("""
        <div class="spec-card">
                    <h4 style="color:#10b981;">Modèle 3 — Guided (CGR)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Architecture** : SupCon gelé + MLP classifieur")
        st.markdown("**Stratégie** : Prédiction grade → filtre MongoDB → recherche SupCon")
        st.markdown("**Collection Qdrant** : `brats_supcon_embeddings` (filtrée)")
        st.metric("Dimension Embedding", "256D")

    st.divider()
    st.subheader("Infrastructure")

    col_i1, col_i2 = st.columns(2, gap="medium")
    with col_i1:
        st.markdown("#### Qdrant — Recherche vectorielle")
        infra_qdrant = pd.DataFrame({
            "Paramètre" : ["Métrique", "Algorithme ANN", "Collections", "Filtrage"],
            "Valeur"    : ["Cosine Similarity", "HNSW", "2 (baseline / supcon)", "Modalité MRI, Patient ID, Grade OMS"],
        })
        render_html_table(infra_qdrant)

    with col_i2:
        st.markdown("#### MongoDB — Enrichissement des métadonnées")
        infra_mongo = pd.DataFrame({
            "Paramètre" : ["Base de données", "Métadonnées", "Métriques calculées", "Dataset"],
            "Valeur"    : ["BraTS Atlas", "Patient ID, Modalité, Coupe z, Stats", "SSIM, PSNR, Histogramme", "BraTS 2021 — 1 251 patients"],
        })
        render_html_table(infra_mongo)

    st.divider()
    st.subheader("Flux de traitement")
    st.markdown("""
    ```
    Image IRM (PNG/NPY)
        ↓  Prétraitement : grayscale → normalisation → resize [128×128]
        ↓  Encodeur CNN → Vecteur latent (256D)
        ↓  Qdrant ANN Search (cosine, HNSW) → Top-K vecteurs
        ↓  Enrichissement MongoDB (métadonnées patient, modalité, slice)
        ↓  Filtrage SSIM (SSIM_min = {ssim_min})
        ↓  Résultats classés avec métriques (Cosinus, SSIM, PSNR)
    ```
    """.format(ssim_min=SSIM_DISPLAY_MIN))