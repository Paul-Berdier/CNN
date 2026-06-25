"""
Page 4 — Explicabilité (XAI) — Grad-CAM
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

st.set_page_config(page_title="Explicabilité XAI — WoundAI", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1B4F72,#154360); }
    [data-testid="stSidebar"] * { color:#ECF0F1!important; }
    .section-title { font-size:1.35rem; font-weight:700; color:#1B4F72; margin:1.5rem 0 0.8rem; border-bottom:2px solid #AED6F1; padding-bottom:0.4rem; }
    .xai-note { background:#EAF4FB; border-left:4px solid #1B4F72; border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; font-size:0.87rem; color:#154360; }
</style>
""", unsafe_allow_html=True)


# ─── INTERFACE BACKEND ────────────────────────────────────────────────────────

def compute_gradcam(image: Image.Image, target_class: str | None = None) -> dict:
    """
    Calcule la heatmap Grad-CAM pour une image et une classe cible.
    BACKEND (PyTorch):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        model = load_model('models/best_cnn.pt')
        target_layers = [model.features[-1]]  # dernière couche conv
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(class_idx)] if target_class else None
        grayscale_cam = cam(input_tensor=preprocess(image), targets=targets)
        heatmap = show_cam_on_image(img_array/255.0, grayscale_cam[0], use_rgb=True)
    Retour attendu:
        {'heatmap_array': np.ndarray (H,W,3),
         'raw_cam': np.ndarray (H,W),
         'superimposed': np.ndarray (H,W,3),
         'target_class': str,
         'confidence': float}
    """
    # Mock : génère une heatmap gaussienne synthétique
    img_array = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    # Simule une attention sur la zone centrale (la plaie)
    h, w = 224, 224
    cx, cy = np.random.randint(80, 140), np.random.randint(80, 140)
    Y, X = np.ogrid[:h, :w]
    cam_map = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 35 ** 2))
    cam_map += 0.15 * np.random.rand(h, w)
    cam_map = np.clip(cam_map, 0, 1)

    # Heatmap colormap
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(cam_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Superposition
    alpha = 0.45
    superimposed = np.clip(
        alpha * heatmap_rgba[:, :, :3] + (1 - alpha) * img_array, 0, 1
    )
    superimposed = (superimposed * 255).astype(np.uint8)

    CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale", "Peau saine"]
    cls = target_class or "Plaie diabétique"
    return {
        "heatmap_array": heatmap_rgb,
        "raw_cam": (cam_map * 255).astype(np.uint8),
        "superimposed": superimposed,
        "target_class": cls,
        "confidence": np.random.uniform(0.72, 0.95),
    }


def get_batch_gradcam_examples() -> list[dict]:
    """
    Retourne des exemples Grad-CAM pré-calculés pour l'analyse de lot.
    BACKEND: charger des images du dataset test et calculer Grad-CAM pour chacune.
    """
    CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale"]
    examples = []
    np.random.seed(42)
    for i, cls in enumerate(CLASSES[:5]):
        palette = {
            "Brûlure": (180, 90, 60), "Ulcère veineux": (140, 100, 120),
            "Plaie diabétique": (160, 110, 90), "Escarre": (120, 100, 100), "Plaie chirurgicale": (200, 160, 140),
        }
        base = palette[cls]
        noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
        arr = np.clip(np.full((224, 224, 3), base, dtype=np.uint8) + noise - 20, 0, 255)
        img = Image.fromarray(arr)
        cam_data = compute_gradcam(img, cls)
        correct = i < 4  # 4 correctes, 1 mal classifiée (mock)
        examples.append({
            "image": arr,
            "true_class": cls,
            "predicted_class": cls if correct else "Brûlure",
            "confidence": cam_data["confidence"],
            "superimposed": cam_data["superimposed"],
            "correct": correct,
        })
    return examples


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
    st.markdown("**Méthode XAI**")
    xai_method = st.radio("", ["Grad-CAM", "Grad-CAM++", "Score-CAM"], label_visibility="collapsed")
    colormap_choice = st.selectbox("Colormap heatmap", ["jet", "inferno", "plasma", "viridis"])

# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown("## 🔥 Explicabilité — Grad-CAM")
st.caption("Visualisez les zones de l'image utilisées par le CNN pour prendre sa décision.")

st.markdown("""
<div class="xai-note">
    <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) calcule les gradients de la classe prédite
    par rapport aux feature maps de la dernière couche convolutive, puis pondère ces maps pour produire une heatmap
    indiquant les régions les plus influentes dans la décision du modèle.
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ Analyse d'une image", "📊 Analyse de lot"])

# ─── Onglet 1 : Image unique ──────────────────────────────────────────────────
with tab1:
    col_up, col_cfg = st.columns([2, 1])

    with col_up:
        st.markdown('<div class="section-title">Image à analyser</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Uploadez une image", type=["jpg", "jpeg", "png"], key="xai_upload")
        use_prev = False
        if "last_image" in st.session_state:
            use_prev = st.button("📥 Utiliser l'image de la page Prédiction")

    with col_cfg:
        CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale", "Peau saine"]
        target_class = st.selectbox("Classe cible pour Grad-CAM", ["Auto (classe prédite)"] + CLASSES)
        target = None if target_class == "Auto (classe prédite)" else target_class
        alpha_blend = st.slider("Opacité de la heatmap", 0.2, 0.8, 0.45, 0.05)

    # Chargement image
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif use_prev and "last_image" in st.session_state:
        image = st.session_state["last_image"]
    else:
        # Image de démo
        np.random.seed(9)
        arr = np.clip(np.full((224, 224, 3), (160, 110, 90), dtype=np.uint8) + np.random.randint(0, 35, (224, 224, 3)), 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))
        st.info("ℹ️ Image de démonstration utilisée. Uploadez votre propre image ou utilisez la page Prédiction.")

    with st.spinner(f"Calcul Grad-CAM ({xai_method})…"):
        cam_result = compute_gradcam(image, target)

    # ─── Affichage triptyque ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Résultats Grad-CAM</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image.resize((224, 224)), caption="Image originale", use_container_width=True)
    with c2:
        st.image(cam_result["heatmap_array"], caption="Heatmap Grad-CAM", use_container_width=True)
    with c3:
        st.image(cam_result["superimposed"], caption="Superposition", use_container_width=True)

    # Légende colorbar
    fig, ax = plt.subplots(figsize=(6, 0.4))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="jet")
    ax.set_yticks([])
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(["Faible contribution", "Moyenne", "Forte contribution"], fontsize=8)
    ax.set_title("Échelle d'importance", fontsize=8, pad=4)
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Méta
    st.markdown(f"""
    **Classe analysée :** `{cam_result['target_class']}` &nbsp;|&nbsp;
    **Confiance :** `{cam_result['confidence']*100:.1f}%` &nbsp;|&nbsp;
    **Méthode :** `{xai_method}` &nbsp;|&nbsp;
    **Architecture :** `EfficientNet-B2`
    """)

    with st.expander("💡 Comment interpréter cette heatmap ?"):
        st.markdown("""
        - **Zones rouges/chaudes** : régions qui ont le plus contribué à la décision du modèle.
        - **Zones bleues/froides** : régions peu ou pas utilisées.
        - **Bon signe clinique** : les zones chaudes devraient se concentrer sur la plaie elle-même,
          pas sur l'arrière-plan, la peau saine environnante, ou des artefacts (gants, règle graduée...).
        - **Signal d'alerte** : si la chaleur se concentre sur des éléments périphériques,
          le modèle utilise des corrélations spurieuses du dataset.
        """)

# ─── Onglet 2 : Analyse de lot ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Analyse Grad-CAM sur plusieurs images</div>', unsafe_allow_html=True)
    st.caption("5 images correctement classifiées + cas d'erreur, avec heatmaps Grad-CAM.")

    with st.spinner("Calcul des heatmaps sur le lot…"):
        examples = get_batch_gradcam_examples()

    for ex in examples:
        status = "✅" if ex["correct"] else "❌"
        label_color = "#17A589" if ex["correct"] else "#CB4335"
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
            with c1:
                st.markdown(f"**{status} {ex['true_class']}**")
                st.caption(f"Prédit : **{ex['predicted_class']}**")
                conf_color = "green" if ex["correct"] else "red"
                st.markdown(f"<span style='color:{label_color};font-weight:700'>{ex['confidence']*100:.1f}%</span>", unsafe_allow_html=True)
            with c2:
                st.image(ex["image"], caption="Original", use_container_width=True)
            with c3:
                st.image(ex["superimposed"], caption="Grad-CAM superposé", use_container_width=True)
            with c4:
                if not ex["correct"]:
                    st.warning(f"⚠️ **Erreur** : la vraie classe est `{ex['true_class']}` mais le modèle a prédit `{ex['predicted_class']}`. "
                               "La heatmap peut révéler que le modèle a focalisé sur des zones non pertinentes.")
                else:
                    st.success("✅ Classification correcte. Vérifiez que la heatmap se concentre bien sur la plaie.")
            st.markdown("---")
