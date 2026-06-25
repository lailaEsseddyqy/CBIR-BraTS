# src/interface/app_clinique_streamlit.py
"""
Interface Clinique — Pour le médecin / radiologue
Recherche de cas similaires à partir d'une image IRM avec filtres cliniques.
Utilise exclusivement le modèle SupCon.
Port Streamlit : 8501
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recherche.multimodal_search import MultimodalSearchEngine, MultimodalQuery, PatientFilter
from dotenv import load_dotenv

load_dotenv()

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="CBMIR — Aide à la Décision Clinique",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — thème clinique clair (inspiré mockup, widgets natifs Streamlit) ─────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: #F4F7FB !important; }

section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

.block-container { padding-top: 1.5rem !important; }

.sidebar-title {
    font-size: 1.1rem; font-weight: 700; color: #1E293B; margin: 0;
}
.sidebar-sub {
    font-size: 0.8rem; color: #64748B; margin-top: 2px;
}
.sidebar-model {
    background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 10px;
    padding: 12px 14px; margin-top: 1.5rem;
}
.sidebar-model-label {
    font-size: 0.7rem; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.sidebar-model-name { font-size: 0.95rem; font-weight: 700; color: #7C3AED; margin-top: 2px; }

.section-label {
    font-size: 0.78rem; font-weight: 600; color: #0066FF;
    text-transform: uppercase; letter-spacing: 0.6px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 5px; margin: 18px 0 10px 0;
}

[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}

.stButton > button[kind="primary"] {
    background: #0066FF !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}

.stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ───────────────────────────────────────────────────────────────
HOPITAUX = [
    "Tous", "CHU Casablanca", "Hôpital Ibn Sina Rabat",
    "Clinique Al Farabi", "CHU Mohammed VI Marrakech",
    "Hôpital Cheikh Khalifa",
]
DIAGNOSTICS = [
    "Tous", "Glioblastome Grade IV", "Méningiome",
    "Astrocytome Grade II", "Métastase cérébrale",
    "Gliome de bas grade",
]

# ── Initialisation moteur (mise en cache) ────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du moteur de recherche...")
def load_engine():
    return MultimodalSearchEngine()

engine = load_engine()

# ── Session state (affichage persistant après rerun) ──────────────────────────
if "clin_results"  not in st.session_state: st.session_state["clin_results"]  = None
if "clin_stats"    not in st.session_state: st.session_state["clin_stats"]    = None
if "clin_query_np" not in st.session_state: st.session_state["clin_query_np"] = None

# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess(image_np: np.ndarray) -> torch.Tensor:
    img = image_np.mean(axis=2).astype("float32") \
          if image_np.ndim == 3 else image_np.astype("float32")
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(128, 128), mode="bilinear",
                         align_corners=False).squeeze(0)

def safe_load(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=True).squeeze().numpy()
    except Exception:
        return None

def similarity_label(score: float):
    if score >= 0.65:
        return "Très similaire", "high"
    elif score >= 0.45:
        return "Similaire", "mid"
    else:
        return "Peu similaire", "low"

def build_results_figure(query_np, results) -> plt.Figure:
    n = min(len(results), 5)
    ncols = n + 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 3.4))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor("none")

    def show(ax, img, title, color):
        ax.set_facecolor("none")
        if img is not None:
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "Image\nindisponible", ha="center", va="center",
                    color="#9aa9bd", fontsize=8, transform=ax.transAxes)
        ax.set_title(title, color=color, fontsize=9, fontweight="bold", pad=8)
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(1.6)
        ax.set_xticks([])
        ax.set_yticks([])

    show(axes[0], query_np, "Image fournie", "#1d6fa8")
    for i, r in enumerate(results[:n]):
        img = safe_load(r.file_path)
        label, _ = similarity_label(r.score)
        color = "#2f9e8f" if r.score >= 0.65 else "#d6932a" if r.score >= 0.45 else "#c4564a"
        show(axes[i + 1], img, f"Cas {i+1} — {label}", color)

    plt.tight_layout(pad=1.0)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p class="sidebar-title">CBMIR</p>
    <p class="sidebar-sub">Aide à la décision clinique</p>
    <div class="sidebar-model">
        <div class="sidebar-model-label">Moteur actif</div>
        <div class="sidebar-model-name">SupCon</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Usage consultatif — ne remplace pas un diagnostic médical."
    )

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Recherche de cas IRM similaires")
st.caption(
    "Aide à la décision par imagerie comparative et données cliniques — Moteur SupCon"
)

st.info(
    "Importez une coupe IRM et définissez optionnellement le profil de votre patient. "
    "Le système croise la similarité visuelle de l'image avec vos critères cliniques "
    "pour identifier les antécédents les plus pertinents dans la base de référence.",
    icon=None,
)

# ── Mise en page principale ──────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    st.subheader("Paramètres de recherche")
    st.divider()

    uploaded_file = st.file_uploader(
        "Image IRM du patient",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Formats acceptés : PNG, JPEG, BMP, TIFF",
    )

    image_np = None
    if uploaded_file is not None:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        image_np = np.array(pil_img)
        st.image(pil_img, caption="Aperçu de l'image importée", use_container_width=True)

    st.markdown("<div class='section-label'>Filtres Cliniques</div>", unsafe_allow_html=True)

    sexe_in = st.radio(
        "Sexe du patient",
        options=["Tous", "Homme", "Femme"],
        horizontal=True,
    )

    col_age1, col_age2 = st.columns(2)
    with col_age1:
        age_min_in = st.number_input("Âge min", min_value=0, max_value=120, value=0,
                                     help="0 = sans filtre")
    with col_age2:
        age_max_in = st.number_input("Âge max", min_value=0, max_value=120, value=0,
                                     help="0 = sans filtre")

    hopital_in    = st.selectbox("Établissement", options=HOPITAUX)
    diagnostic_in = st.selectbox("Filtrer par diagnostic", options=DIAGNOSTICS)

    st.markdown("<div class='section-label'>Paramètres de la recherche</div>", unsafe_allow_html=True)

    modalite_in = st.selectbox(
        "Modalité IRM",
        options=["Toutes", "t1", "t2", "flair", "t1ce"],
    )
    k_in = st.slider("Nombre de cas à afficher", min_value=1, max_value=10, value=5)

    pid_exclude_in = st.text_input(
        "Exclure un identifiant patient",
        placeholder="ex : TS2021_00123",
    )

    st.divider()
    run_btn = st.button("Lancer la recherche clinique", type="primary", use_container_width=True)

# ── Zone des résultats ───────────────────────────────────────────────────────
with col_right:
    st.subheader("Dossiers similaires identifiés")
    st.divider()

    if run_btn:
        if image_np is None:
            st.warning("Veuillez importer une image IRM avant de lancer la recherche.")
        else:
            with st.spinner("Traitement en cours..."):
                tensor = preprocess(image_np)

                pf = PatientFilter(
                    sexe       = None if sexe_in == "Tous" else sexe_in,
                    age_min    = int(age_min_in) if age_min_in > 0 else None,
                    age_max    = int(age_max_in) if age_max_in > 0 else None,
                    modalite   = None if modalite_in == "Toutes" else modalite_in,
                    hopital    = None if hopital_in == "Tous" else hopital_in,
                    diagnostic = None if diagnostic_in == "Tous" else diagnostic_in,
                )

                query = MultimodalQuery(
                    image_tensor       = tensor,
                    k                  = int(k_in),
                    model              = "supcon",
                    patient_filter     = pf,
                    exclude_patient_id = pid_exclude_in.strip() or None,
                )

                results = engine.search(query)
                stats   = engine.stats_by_filter(pf)

            query_np = image_np.mean(axis=2) if image_np.ndim == 3 else image_np
            st.session_state["clin_results"]  = results
            st.session_state["clin_stats"]    = stats
            st.session_state["clin_query_np"] = query_np

    results  = st.session_state.get("clin_results")
    stats    = st.session_state.get("clin_stats")
    query_np = st.session_state.get("clin_query_np")

    if results is None and not run_btn:
        st.caption("Les résultats apparaîtront ici après le lancement de la recherche.")
    elif results is not None:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cas identifiés", len(results))
        mc2.metric("Dossiers éligibles", stats.get("total_eligible", "—") if stats else "—")
        mc3.metric("Total base", stats.get("total_collection", "—") if stats else "—")

        st.divider()

        if results and query_np is not None:
            fig = build_results_figure(query_np, results)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        elif not results:
            st.info("Aucun cas similaire trouvé pour ces critères.")

        if results:
            st.markdown(
                "<div class='section-label' style='margin-top:20px;'>Détail des dossiers</div>",
                unsafe_allow_html=True,
            )

            card_cols = st.columns(min(len(results), 5))
            for i, r in enumerate(results[:5]):
                label, _ = similarity_label(r.score)
                sexe_age = f"{r.sexe}, {r.age} ans" if r.sexe and r.age else "Infos patient N/D"
                with card_cols[i]:
                    with st.container(border=True):
                        st.caption(f"Cas {i + 1}")
                        st.markdown(f"**{label}** — score `{r.score:.4f}`")
                        st.markdown(f"**{r.diagnostic or 'Diagnostic inconnu'}**")
                        img = safe_load(r.file_path)
                        if img is not None:
                            st.image(img, use_container_width=True)
                        st.markdown(
                            f"**Patient** : {r.patient_id}  \n"
                            f"**Profil** : {sexe_age}  \n"
                            f"**Hôpital** : {r.hopital or 'N/D'}  \n"
                            f"**Modalité** : {r.modalite.upper()} (Z={r.slice_z})"
                        )

            with st.expander("Tableau récapitulatif des résultats"):
                df_rows = []
                for r in results:
                    label, _ = similarity_label(r.score)
                    df_rows.append({
                        "Rang"       : r.rank,
                        "Patient ID" : r.patient_id,
                        "Similarité" : label,
                        "Score"      : round(r.score, 4),
                        "Modalité"   : r.modalite.upper(),
                        "Coupe Z"    : r.slice_z,
                        "Diagnostic" : r.diagnostic or "N/D",
                        "Hôpital"    : r.hopital or "N/D",
                    })
                st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

# ── Avertissement légal ───────────────────────────────────────────────────────
st.divider()
st.warning(
    "Cet outil propose des cas d'imagerie visuellement similaires à partir d'une base de référence "
    "et des filtres sélectionnés. Les résultats sont indicatifs et doivent être interprétés "
    "par un professionnel de santé qualifié. Il ne constitue pas un acte de diagnostic médical.",
    icon=None,
)
