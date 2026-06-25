"""
Plateforme d'analyse d'imagerie médicale des plaies
Page d'accueil
"""

import streamlit as st

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="WoundAI — Analyse de Plaies",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Palette médicale sobre : blanc clinique, bleu acier, accent teal */
    :root {
        --primary: #1B4F72;
        --accent: #17A589;
        --warning: #E67E22;
        --bg-card: #F4F8FB;
        --text-muted: #7F8C8D;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4F72 0%, #154360 100%);
    }
    [data-testid="stSidebar"] * { color: #ECF0F1 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #BDC3C7 !important; }

    /* Hero */
    .hero-container {
        background: linear-gradient(135deg, #1B4F72 0%, #17A589 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 0.5rem 0;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.88;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Stat cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid #D5E8F0;
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .stat-card:hover { box-shadow: 0 4px 16px rgba(27,79,114,0.12); }
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B4F72;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.82rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.4rem;
    }
    .stat-delta {
        font-size: 0.78rem;
        color: #17A589;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #E8EEF3;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        border-left: 4px solid #1B4F72;
    }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
    .feature-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0.4rem;
    }
    .feature-desc { font-size: 0.83rem; color: var(--text-muted); line-height: 1.5; }

    /* Disclaimer */
    .disclaimer {
        background: #FEF9E7;
        border: 1px solid #F9E79F;
        border-left: 4px solid #E67E22;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.83rem;
        color: #7D6608;
    }

    /* Architecture diagram */
    .arch-box {
        background: #EAF4FB;
        border: 1.5px solid #AED6F1;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #1B4F72;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Mock : statistiques globales ────────────────────────────────────────────
# TODO: remplacer par de vraies valeurs chargées depuis le modèle et le dataset
def get_platform_stats() -> dict:
    """
    Retourne les statistiques globales de la plateforme.
    Backend attendu : charger depuis models/metadata.json et la base ChromaDB.
    """
    return {
        "nb_images": 1_547,
        "nb_classes": 6,
        "best_accuracy": 91.4,
        "best_architecture": "EfficientNet-B2",
        "nb_cases_db": 1_547,
        "mlflow_runs": 8,
    }


CLASSES = [
    "Brûlure",
    "Ulcère veineux",
    "Plaie diabétique",
    "Escarre",
    "Plaie chirurgicale",
    "Peau saine",
]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 WoundAI")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
    - 🏠 **Accueil** ← vous êtes ici
    - 📊 Dataset Explorer
    - 🔬 Entraînement
    - 🔍 Prédiction & Analyse
    - 🔥 Explicabilité (XAI)
    - 🤖 Assistant IA
    """)
    st.markdown("---")
    stats = get_platform_stats()
    st.markdown(f"**Modèle actif :** `{stats['best_architecture']}`")
    st.markdown(f"**Accuracy :** `{stats['best_accuracy']}%`")
    st.markdown(f"**Runs MLflow :** `{stats['mlflow_runs']}`")
    st.markdown("---")
    st.caption("v1.0 — Projet Deep Learning M2")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🏥 Aide au diagnostic clinique</div>
    <div class="hero-title">WoundAI Platform</div>
    <p class="hero-subtitle">
        Classification automatique de plaies cutanées par deep learning — 
        recherche par similarité visuelle — recommandations de traitement par IA.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Métriques clés ───────────────────────────────────────────────────────────
stats = get_platform_stats()

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, stats["nb_images"], "Images annotées", "dataset complet"),
    (c2, stats["nb_classes"], "Classes de plaies", "types distincts"),
    (c3, f"{stats['best_accuracy']}%", "Accuracy (best)", stats["best_architecture"]),
    (c4, stats["nb_cases_db"], "Cas dans la base", "recherche similarité"),
    (c5, stats["mlflow_runs"], "Runs MLflow", "expériences tracées"),
]
for col, number, label, delta in cards:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Fonctionnalités ──────────────────────────────────────────────────────────
st.markdown("#### Modules de la plateforme")

col1, col2, col3 = st.columns(3)

features = [
    (col1, [
        ("🔬", "Classification CNN", "Identification automatique du type de plaie parmi 6 classes via EfficientNet ou ResNet50, avec score de confiance et top-3."),
        ("🛡️", "Détection OOD", "Filtre hors-domaine via Autoencoder convolutif : rejette les images non médicales avant toute classification."),
    ]),
    (col2, [
        ("🔍", "Recherche par similarité", "Retrouvez les cas historiques visuellement proches via embeddings + ChromaDB. Utile pour comparer les traitements appliqués."),
        ("🔥", "Explicabilité Grad-CAM", "Heatmaps superposées à l'image originale pour visualiser les zones décisives du CNN — essentiel pour la confiance clinique."),
    ]),
    (col3, [
        ("🤖", "Assistant LLM (RAG)", "Recommandations de traitement contextualisées, générées par un LLM local (Ollama) enrichi par une base de connaissances médicales."),
        ("📈", "MLflow + Evidently", "Traçabilité complète des expériences d'entraînement et surveillance de la dérive des données en production."),
    ]),
]

for col, items in features:
    with col:
        for icon, title, desc in items:
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom:1rem">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Architecture pipeline ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Architecture du pipeline")

col_a, col_b, col_c, col_d, col_e = st.columns([2, 0.4, 2, 0.4, 2])

with col_a:
    st.markdown("""
    <div class="arch-box">📷 Image uploadée</div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">🛡️ Filtre OOD<br><small style="font-weight:400;color:#5D6D7E">Autoencoder</small></div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">🔬 CNN Classifieur<br><small style="font-weight:400;color:#5D6D7E">EfficientNet-B2</small></div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("<div style='margin-top:3.2rem;text-align:center;font-size:2rem;color:#AED6F1'>→</div>", unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="arch-box">📊 Diagnostic<br><small style="font-weight:400;color:#5D6D7E">Classe + confiance + top-3</small></div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">🔍 Embeddings<br><small style="font-weight:400;color:#5D6D7E">ChromaDB → top-K similaires</small></div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">🔥 Grad-CAM<br><small style="font-weight:400;color:#5D6D7E">Zones d'attention</small></div>
    """, unsafe_allow_html=True)

with col_d:
    st.markdown("<div style='margin-top:3.2rem;text-align:center;font-size:2rem;color:#AED6F1'>→</div>", unsafe_allow_html=True)

with col_e:
    st.markdown("""
    <div class="arch-box">🤖 RAG Pipeline<br><small style="font-weight:400;color:#5D6D7E">Retrieval + LLM local</small></div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">💊 Recommandation<br><small style="font-weight:400;color:#5D6D7E">Protocole de traitement</small></div>
    <div style="text-align:center;margin:0.4rem 0;color:#AED6F1;font-size:1.3rem;">↓</div>
    <div class="arch-box">📡 Langfuse<br><small style="font-weight:400;color:#5D6D7E">Traçabilité LLM</small></div>
    """, unsafe_allow_html=True)

# ─── Disclaimer médical ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Avertissement médical :</strong> Cette plateforme est un outil d'aide au diagnostic à usage pédagogique et de recherche uniquement.
    Elle ne remplace en aucun cas l'avis d'un professionnel de santé qualifié. Toute décision clinique doit être prise par un médecin habilité.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("""
    **Classes de plaies supportées**  
    """ + " · ".join([f"`{c}`" for c in CLASSES]))
with col_r:
    st.markdown("""
    **Stack technique**  
    `PyTorch` · `EfficientNet` · `ChromaDB` · `MLflow` · `Ollama` · `Langfuse` · `Evidently AI`
    """)
