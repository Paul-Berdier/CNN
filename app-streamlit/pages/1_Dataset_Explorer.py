"""
Page 1 — Exploration du dataset
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from pathlib import Path

st.set_page_config(page_title="Dataset Explorer — WoundAI", page_icon="📊", layout="wide")

# ─── CSS partagé (injecté sur chaque page) ────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1B4F72,#154360); }
    [data-testid="stSidebar"] * { color:#ECF0F1!important; }
    .section-title { font-size:1.35rem; font-weight:700; color:#1B4F72; margin:1.5rem 0 0.8rem; border-bottom:2px solid #AED6F1; padding-bottom:0.4rem; }
    .info-chip { display:inline-block; background:#EAF4FB; border:1px solid #AED6F1; border-radius:20px; padding:0.2rem 0.75rem; font-size:0.78rem; font-weight:600; color:#1B4F72; margin:0.15rem; }
    .metric-mini { background:#F4F8FB; border:1px solid #D5E8F0; border-radius:8px; padding:0.8rem 1rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ─── INTERFACE BACKEND ────────────────────────────────────────────────────────
# Les fonctions ci-dessous définissent le contrat attendu du backend.
# Remplacez le corps de chaque fonction par le vrai appel quand le backend est prêt.

def load_dataset_stats() -> dict:
    """
    Retourne les statistiques du dataset.
    BACKEND: lire depuis data/raw/ et compter les images par classe.
    Retour attendu: dict avec clés 'class_counts', 'total', 'splits', 'resolutions'
    """
    class_counts = {
        "Brûlure": 312,
        "Ulcère veineux": 198,
        "Plaie diabétique": 156,
        "Escarre": 287,
        "Plaie chirurgicale": 421,
        "Peau saine": 173,
    }
    return {
        "class_counts": class_counts,
        "total": sum(class_counts.values()),
        "splits": {"Train": 0.70, "Validation": 0.15, "Test": 0.15},
        "resolutions": {"min": "128×128", "max": "4032×3024", "cible": "224×224"},
        "is_balanced": False,
        "imbalance_ratio": 2.7,
    }


def get_sample_images(class_name: str, n: int = 6) -> list[dict]:
    """
    Retourne des exemples d'images pour une classe donnée.
    BACKEND: charger depuis data/raw/<class_name>/ et retourner PIL.Image + metadata.
    Retour attendu: list de dict {'image': PIL.Image, 'filename': str, 'resolution': str}
    """
    # Mock — génère des images de couleur avec du bruit
    samples = []
    palette = {
        "Brûlure": (180, 90, 60),
        "Ulcère veineux": (140, 100, 120),
        "Plaie diabétique": (160, 110, 90),
        "Escarre": (120, 100, 100),
        "Plaie chirurgicale": (200, 160, 140),
        "Peau saine": (220, 180, 150),
    }
    base = palette.get(class_name, (150, 130, 120))
    for i in range(n):
        noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
        img_array = np.clip(
            np.full((224, 224, 3), base, dtype=np.uint8) + noise - 20, 0, 255
        )
        samples.append({
            "array": img_array,
            "filename": f"{class_name.lower().replace(' ', '_')}_{i+1:03d}.jpg",
            "resolution": f"{random.choice([224, 384, 512, 1024])}×{random.choice([224, 384, 512, 768])}",
        })
    return samples


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 WoundAI")
    st.markdown("---")
    st.page_link("Home.py", label="🏠 Accueil")
    st.page_link("pages/1_Dataset_Explorer.py", label="📊 Dataset Explorer")
    st.page_link("pages/2_Training.py", label="🔬 Entraînement")
    st.page_link("pages/3_Prediction.py", label="🔍 Prédiction & Analyse")
    st.page_link("pages/4_Explainability.py", label="🔥 Explicabilité (XAI)")
    st.page_link("pages/5_AI_Assistant.py", label="🤖 Assistant IA")

# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Exploration du Dataset")
st.caption("Analyse de la distribution, des résolutions, et visualisation des exemples par classe.")

stats = load_dataset_stats()

# ─── Métriques rapides ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total images", f"{stats['total']:,}")
with c2:
    st.metric("Classes", len(stats["class_counts"]))
with c3:
    st.metric("Résolution cible", stats["resolutions"]["cible"])
with c4:
    ratio = stats["imbalance_ratio"]
    st.metric("Ratio déséquilibre", f"×{ratio:.1f}", delta="⚠️ déséquilibré" if not stats["is_balanced"] else "✅ équilibré",
              delta_color="inverse" if not stats["is_balanced"] else "normal")

# ─── Distribution des classes ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Distribution des classes</div>', unsafe_allow_html=True)

col_chart, col_info = st.columns([3, 1])
with col_chart:
    classes = list(stats["class_counts"].keys())
    counts = list(stats["class_counts"].values())
    total = stats["total"]

    colors = ["#1B4F72", "#2E86C1", "#17A589", "#27AE60", "#E67E22", "#CB4335"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(classes, counts, color=colors, edgecolor="white", height=0.65)

    # Annotations
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/total*100:.1f}%)",
                va="center", ha="left", fontsize=9.5, color="#2C3E50", fontweight="600")

    # Ligne médiane
    median_count = np.median(counts)
    ax.axvline(x=median_count, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(median_count + 3, -0.6, f"médiane ({int(median_count)})", color="#E74C3C", fontsize=8)

    ax.set_xlabel("Nombre d'images", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFBFC")
    fig.patch.set_facecolor("#FAFBFC")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_info:
    st.markdown("**Répartition train/val/test**")
    for split, ratio in stats["splits"].items():
        n = int(total * ratio)
        st.markdown(f"""<div class="metric-mini" style="margin-bottom:0.5rem">
            <strong style="color:#1B4F72">{split}</strong><br>
            <span style="font-size:1.2rem;font-weight:700">{n}</span>
            <span style="font-size:0.75rem;color:#7F8C8D"> ({ratio*100:.0f}%)</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Résolutions**")
    st.markdown(f"""
    - Min : `{stats['resolutions']['min']}`
    - Max : `{stats['resolutions']['max']}`
    - Cible : `{stats['resolutions']['cible']}`
    """)

# ─── Alerte déséquilibre ──────────────────────────────────────────────────────
if not stats["is_balanced"]:
    st.warning(
        f"⚠️ **Déséquilibre détecté** : ratio max/min = ×{stats['imbalance_ratio']:.1f}. "
        "Stratégie appliquée : **WeightedRandomSampler** + **data augmentation** sur les classes minoritaires."
    )

# ─── Visualisation par classe ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Exemples d\'images par classe</div>', unsafe_allow_html=True)

selected_class = st.selectbox(
    "Choisir une classe",
    options=list(stats["class_counts"].keys()),
    help="Sélectionnez une classe pour visualiser des exemples d'images."
)

n_examples = st.slider("Nombre d'exemples", 2, 8, 6)

with st.spinner(f"Chargement des exemples — {selected_class}…"):
    samples = get_sample_images(selected_class, n=n_examples)

cols = st.columns(min(n_examples, 6))
for i, sample in enumerate(samples):
    with cols[i % len(cols)]:
        st.image(sample["array"], caption=f"{sample['filename']}\n{sample['resolution']}", use_container_width=True)

# ─── Statistiques par classe ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Statistiques par classe</div>', unsafe_allow_html=True)

df_stats = pd.DataFrame([
    {
        "Classe": cls,
        "Images": count,
        "% du total": f"{count/total*100:.1f}%",
        "Train (~70%)": int(count * 0.70),
        "Val (~15%)": int(count * 0.15),
        "Test (~15%)": int(count * 0.15),
        "Statut": "✅ Majoritaire" if count > median_count else "⚠️ Minoritaire",
    }
    for cls, count in stats["class_counts"].items()
])

st.dataframe(
    df_stats,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Images": st.column_config.ProgressColumn("Images", min_value=0, max_value=max(counts)),
    }
)
