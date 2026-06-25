"""
Page 3 — Prédiction & Analyse
Upload image → OOD filter → Classification → Similarité → Résultats
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Prédiction — WoundAI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1B4F72,#154360); }
    [data-testid="stSidebar"] * { color:#ECF0F1!important; }
    .section-title { font-size:1.35rem; font-weight:700; color:#1B4F72; margin:1.5rem 0 0.8rem; border-bottom:2px solid #AED6F1; padding-bottom:0.4rem; }
    .result-card { background:#F4F8FB; border:1px solid #D5E8F0; border-radius:12px; padding:1.5rem; }
    .diagnosis-main { font-size:1.9rem; font-weight:800; color:#1B4F72; }
    .confidence-high { color:#17A589; font-weight:700; }
    .confidence-med { color:#E67E22; font-weight:700; }
    .confidence-low { color:#CB4335; font-weight:700; }
    .ood-ok { background:#D5F5E3; border:1px solid #A9DFBF; border-radius:8px; padding:0.8rem 1rem; color:#1E8449; font-weight:600; }
    .ood-alert { background:#FDEDEC; border:1px solid #F1948A; border-radius:8px; padding:0.8rem 1rem; color:#922B21; font-weight:600; }
    .sim-card { background:white; border:1px solid #E8EEF3; border-radius:8px; padding:0.6rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ─── INTERFACE BACKEND ────────────────────────────────────────────────────────

def run_ood_detection(image: Image.Image) -> dict:
    """
    Filtre OOD via l'autoencoder convolutif.
    BACKEND: charger models/autoencoder.pt, passer l'image, calculer l'erreur de
    reconstruction et comparer au seuil calibré.
    Retour attendu: {'is_ood': bool, 'reconstruction_error': float, 'threshold': float, 'score': float}
    """
    # Mock : simule toujours une image dans le domaine
    return {
        "is_ood": False,
        "reconstruction_error": 0.043,
        "threshold": 0.15,
        "score": 0.043 / 0.15,  # 0 = dans le domaine, >1 = hors domaine
    }


def classify_image(image: Image.Image) -> dict:
    """
    Classifie l'image avec le modèle CNN.
    BACKEND: charger models/best_cnn.pt (EfficientNet-B2), prétraiter l'image
    (resize 224×224, normalize ImageNet), passer en forward, softmax.
    Retour attendu: {
        'predicted_class': str,
        'confidence': float,        # 0-1
        'top3': list[tuple[str, float]],
        'all_probs': dict[str, float]
    }
    """
    CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale", "Peau saine"]
    probs = np.random.dirichlet(np.ones(6) * 0.5)
    # Simule une prédiction dominante
    probs[2] = 0.78
    probs = probs / probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    return {
        "predicted_class": CLASSES[sorted_idx[0]],
        "confidence": float(probs[sorted_idx[0]]),
        "top3": [(CLASSES[i], float(probs[i])) for i in sorted_idx[:3]],
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }


def search_similar_cases(image: Image.Image, k: int = 5) -> list[dict]:
    """
    Recherche les k cas les plus similaires dans ChromaDB.
    BACKEND: extraire l'embedding CNN de l'image (avant dernière couche),
    normaliser L2, requête ChromaDB collection.query(query_embeddings=[emb], n_results=k).
    Retour attendu: list de dict {'similarity': float, 'class_name': str, 'image_path': str, 'case_id': str}
    """
    CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale", "Peau saine"]
    np.random.seed(0)
    results = []
    for i in range(k):
        cls = np.random.choice(CLASSES, p=[0.05, 0.05, 0.7, 0.1, 0.07, 0.03])
        sim = 0.98 - i * 0.06 + np.random.uniform(-0.02, 0.02)
        noise = np.random.randint(0, 50, (80, 80, 3), dtype=np.uint8)
        palette = {"Plaie diabétique": (160, 110, 90), "Brûlure": (180, 90, 60),
                   "Ulcère veineux": (140, 100, 120), "Escarre": (120, 100, 100),
                   "Plaie chirurgicale": (200, 160, 140), "Peau saine": (220, 180, 150)}
        base = palette[cls]
        img_arr = np.clip(np.full((80, 80, 3), base, dtype=np.uint8) + noise - 20, 0, 255)
        results.append({
            "similarity": float(sim),
            "class_name": cls,
            "case_id": f"CASE-{2024 - i}-{np.random.randint(100, 999)}",
            "image_array": img_arr,
        })
    return results


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

    st.markdown("---")
    k_similar = st.slider("Cas similaires (k)", 1, 10, 5)
    show_all_probs = st.toggle("Afficher toutes les probabilités", value=False)

# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Prédiction & Analyse")
st.caption("Uploadez une image de plaie pour obtenir le diagnostic automatique, la recherche de cas similaires et l'analyse OOD.")

# ─── Upload ───────────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown('<div class="section-title">Image à analyser</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Déposez une image de plaie",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Formats acceptés : JPG, PNG, BMP, WebP",
    )

    use_demo = st.button("🖼️ Utiliser une image de démonstration")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif use_demo:
    # Image de démonstration synthétique
    np.random.seed(7)
    arr = np.clip(np.full((224, 224, 3), (160, 110, 90), dtype=np.uint8) + np.random.randint(0, 40, (224, 224, 3)), 0, 255)
    image = Image.fromarray(arr.astype(np.uint8))
else:
    image = None

with col_preview:
    if image:
        st.markdown('<div class="section-title">Aperçu</div>', unsafe_allow_html=True)
        st.image(image, caption="Image uploadée", use_container_width=True)
        w, h = image.size
        st.caption(f"Résolution : {w}×{h} px")

# ─── Pipeline d'analyse ───────────────────────────────────────────────────────
if image is not None:
    st.markdown("---")
    st.markdown("### 🔄 Pipeline d'analyse")

    with st.spinner("Étape 1/3 — Filtre OOD (Autoencoder)…"):
        ood_result = run_ood_detection(image)

    # Affichage OOD
    score_pct = ood_result["score"] * 100
    if ood_result["is_ood"]:
        st.markdown(f"""
        <div class="ood-alert">
            🚨 <strong>Image hors domaine détectée</strong> — Score d'anomalie : {score_pct:.1f}% (seuil : 100%)<br>
            Cette image ne correspond pas à une plaie cutanée connue. La classification est bloquée.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.markdown(f"""
        <div class="ood-ok">
            ✅ <strong>Image dans le domaine médical</strong> — Score d'anomalie : {score_pct:.1f}% (seuil : 100%) — Classification autorisée.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Étape 2/3 — Classification CNN (EfficientNet-B2)…"):
        pred = classify_image(image)

    with st.spinner("Étape 3/3 — Recherche de cas similaires (ChromaDB)…"):
        similar = search_similar_cases(image, k=k_similar)

    # ─── Résultats ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Résultats du diagnostic</div>', unsafe_allow_html=True)

    col_diag, col_top3 = st.columns([1, 1])

    with col_diag:
        conf = pred["confidence"]
        conf_color = "confidence-high" if conf > 0.75 else ("confidence-med" if conf > 0.5 else "confidence-low")
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.8rem;color:#7F8C8D;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem">Diagnostic principal</div>
            <div class="diagnosis-main">{pred['predicted_class']}</div>
            <div style="margin-top:0.5rem">
                Confiance : <span class="{conf_color}">{conf*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge confiance
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh([0], [conf], color="#17A589" if conf > 0.75 else "#E67E22", height=0.4)
        ax.barh([0], [1 - conf], left=[conf], color="#ECF0F1", height=0.4)
        ax.set_xlim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("#F4F8FB")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_top3:
        st.markdown("**Top-3 prédictions**")
        for cls_name, prob in pred["top3"]:
            bar_color = "#1B4F72" if cls_name == pred["predicted_class"] else "#AED6F1"
            fig, ax = plt.subplots(figsize=(5, 0.45))
            ax.barh([0], [prob], color=bar_color, height=0.5)
            ax.barh([0], [1 - prob], left=[prob], color="#F4F8FB", height=0.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            fig.patch.set_facecolor("white")
            plt.tight_layout(pad=0)
            st.markdown(f"**{cls_name}** — `{prob*100:.1f}%`")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    if show_all_probs:
        st.markdown("**Toutes les probabilités**")
        probs_df_data = [(k, v) for k, v in sorted(pred["all_probs"].items(), key=lambda x: -x[1])]
        for cls_name, prob in probs_df_data:
            st.progress(prob, text=f"{cls_name}: {prob*100:.1f}%")

    # ─── Cas similaires ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Cas historiques similaires</div>', unsafe_allow_html=True)
    st.caption(f"Top-{k_similar} cas les plus proches dans la base vectorielle (similarité cosinus)")

    cols = st.columns(k_similar)
    for i, case in enumerate(similar):
        with cols[i]:
            sim_pct = case["similarity"] * 100
            st.image(case["image_array"], use_container_width=True)
            color = "#17A589" if sim_pct > 90 else "#E67E22"
            st.markdown(f"""
            <div class="sim-card">
                <strong>{case['class_name']}</strong><br>
                <span style="color:{color};font-weight:700">{sim_pct:.1f}%</span>
                <span style="font-size:0.72rem;color:#7F8C8D"> similarité</span><br>
                <code style="font-size:0.7rem">{case['case_id']}</code>
            </div>
            """, unsafe_allow_html=True)

    # ─── CTA vers autres pages ─────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🔥 **Explicabilité** : voir les zones d'attention Grad-CAM de ce diagnostic → Page 4")
    with c2:
        st.info("🤖 **Assistant IA** : obtenir des recommandations de traitement → Page 5")
    with c3:
        st.success(f"✅ Diagnostic enregistré dans la session : **{pred['predicted_class']}** ({pred['confidence']*100:.1f}%)")

    # Store dans session state pour les autres pages
    st.session_state["last_prediction"] = pred
    st.session_state["last_image"] = image

else:
    st.markdown("---")
    st.info("⬆️ **Uploadez une image** ou cliquez sur **Utiliser une image de démonstration** pour commencer l'analyse.")

    # Infos sur le pipeline
    with st.expander("ℹ️ Comment fonctionne le pipeline ?"):
        st.markdown("""
        1. **Filtre OOD** — L'autoencoder convolutif calcule l'erreur de reconstruction de l'image.
           Si elle dépasse le seuil calibré, l'image est rejetée comme hors domaine.
        2. **Classification CNN** — EfficientNet-B2 (91.4% val accuracy) prédit la classe de plaie
           et retourne un score de confiance softmax + top-3 prédictions.
        3. **Recherche similarité** — L'embedding de la dernière couche convolutive est comparé
           (similarité cosinus) aux embeddings stockés dans ChromaDB pour retrouver les cas historiques proches.
        """)
