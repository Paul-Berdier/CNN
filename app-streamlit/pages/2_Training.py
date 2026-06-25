"""
Page 2 — Entraînement des modèles
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Entraînement — WoundAI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1B4F72,#154360); }
    [data-testid="stSidebar"] * { color:#ECF0F1!important; }
    .section-title { font-size:1.35rem; font-weight:700; color:#1B4F72; margin:1.5rem 0 0.8rem; border-bottom:2px solid #AED6F1; padding-bottom:0.4rem; }
    .run-card { background:#F4F8FB; border:1px solid #D5E8F0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
    .run-badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.75rem; font-weight:700; }
    .badge-best { background:#D5F5E3; color:#1E8449; }
    .badge-run { background:#EAF4FB; color:#1B4F72; }
</style>
""", unsafe_allow_html=True)


# ─── INTERFACE BACKEND ────────────────────────────────────────────────────────

def get_mlflow_runs() -> list[dict]:
    """
    Retourne la liste des runs MLflow enregistrés.
    BACKEND: appeler mlflow.search_runs(experiment_ids=[...]) ou l'API REST MLflow.
    """
    return [
        {"name": "efficientnet-b2-lr0.001-aug-v2", "arch": "EfficientNet-B2", "lr": 0.001,
         "batch": 32, "epochs": 50, "val_acc": 91.4, "f1_macro": 0.898, "best": True, "status": "FINISHED"},
        {"name": "resnet50-lr0.001-aug-v1", "arch": "ResNet50", "lr": 0.001,
         "batch": 32, "epochs": 40, "val_acc": 87.2, "f1_macro": 0.851, "best": False, "status": "FINISHED"},
        {"name": "efficientnet-b2-lr0.0001-noaug", "arch": "EfficientNet-B2", "lr": 0.0001,
         "batch": 16, "epochs": 30, "val_acc": 84.7, "f1_macro": 0.823, "best": False, "status": "FINISHED"},
        {"name": "resnet50-scratch-lr0.01", "arch": "ResNet50 (scratch)", "lr": 0.01,
         "batch": 64, "epochs": 60, "val_acc": 72.1, "f1_macro": 0.689, "best": False, "status": "FINISHED"},
        {"name": "autoencoder-conv-v1", "arch": "Autoencoder", "lr": 0.001,
         "batch": 32, "epochs": 40, "val_acc": None, "f1_macro": None, "best": False, "status": "FINISHED"},
    ]


def simulate_training_curves(arch: str, lr: float, epochs: int, augment: bool) -> dict:
    """
    Simule des courbes d'entraînement.
    BACKEND: remplacer par un entraînement réel via subprocess ou thread, avec
    callback Streamlit pour mise à jour en temps réel.
    Retour attendu: dict avec 'train_loss', 'val_loss', 'train_acc', 'val_acc' (listes)
    """
    np.random.seed(42)
    base_final = {"EfficientNet-B2": 0.914, "ResNet50": 0.872, "VGG16": 0.791}.get(arch, 0.85)
    if not augment:
        base_final -= 0.05

    def smooth_curve(target, n, noise=0.015):
        raw = [target * (1 - np.exp(-4 * i / n)) + np.random.normal(0, noise) for i in range(1, n + 1)]
        return [min(max(v, 0), 1) for v in raw]

    train_acc = smooth_curve(base_final + 0.03, epochs)
    val_acc = smooth_curve(base_final, epochs)
    train_loss = [1.5 * np.exp(-3.5 * i / epochs) + np.random.normal(0, 0.02) for i in range(1, epochs + 1)]
    val_loss = [1.6 * np.exp(-3.2 * i / epochs) + np.random.normal(0, 0.025) for i in range(1, epochs + 1)]

    return {
        "train_acc": train_acc, "val_acc": val_acc,
        "train_loss": [max(0, v) for v in train_loss],
        "val_loss": [max(0, v) for v in val_loss],
    }


def launch_training(config: dict) -> bool:
    """
    Lance l'entraînement avec la configuration donnée.
    BACKEND: appeler le script d'entraînement (core/model_utils.py) via subprocess.
    Config attendue: {'arch', 'lr', 'batch_size', 'epochs', 'augment', 'freeze_layers'}
    Retour: True si succès, False sinon.
    """
    # Mock : simule une attente
    time.sleep(0.5)
    return True


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
st.markdown("## 🔬 Entraînement des Modèles")
st.caption("Configurez et lancez un entraînement CNN, visualisez les courbes en temps réel, comparez les runs dans MLflow.")

tab1, tab2, tab3 = st.tabs(["⚙️ Configurer & Lancer", "📈 Runs MLflow", "📊 Comparer les architectures"])

# ─── Onglet 1 : Configuration ─────────────────────────────────────────────────
with tab1:
    col_cfg, col_curves = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="section-title">Hyperparamètres</div>', unsafe_allow_html=True)

        arch = st.selectbox("Architecture", ["EfficientNet-B2", "ResNet50", "VGG16", "DenseNet121"])
        lr = st.select_slider("Learning rate", options=[0.1, 0.01, 0.001, 0.0001, 0.00001], value=0.001)
        batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
        epochs = st.slider("Epochs", 10, 100, 50)
        augment = st.toggle("Data augmentation", value=True)
        freeze = st.toggle("Transfer learning (ImageNet)", value=True)

        st.markdown("**Stratégie de fine-tuning**")
        finetune = st.radio("", ["Toutes les couches gelées (feature extractor)", "Gel partiel + dégel progressif", "Entraînement from scratch"], label_visibility="collapsed")

        optimizer = st.selectbox("Optimiseur", ["Adam", "AdamW", "SGD + momentum"])
        scheduler = st.selectbox("LR Scheduler", ["Cosine Annealing", "Step Decay", "ReduceLROnPlateau", "Aucun"])

        run_name = f"{arch.lower().replace(' ', '_').replace('-', '')}-lr{lr}-{'aug' if augment else 'noaug'}"
        st.markdown(f"**Nom du run MLflow :** `{run_name}`")

    with col_curves:
        st.markdown('<div class="section-title">Aperçu des courbes (simulation)</div>', unsafe_allow_html=True)
        st.caption("Les courbes ci-dessous sont une simulation. Elles seront remplacées par les vraies métriques lors de l'entraînement.")

        curves = simulate_training_curves(arch, lr, epochs, augment)
        ep = list(range(1, epochs + 1))

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, (y1, y2, ylabel) in zip(axes, [
            (curves["train_acc"], curves["val_acc"], "Accuracy"),
            (curves["train_loss"], curves["val_loss"], "Loss"),
        ]):
            ax.plot(ep, y1, color="#1B4F72", linewidth=2, label="Train")
            ax.plot(ep, y2, color="#17A589", linewidth=2, linestyle="--", label="Validation")
            ax.fill_between(ep, y1, y2, alpha=0.08, color="#1B4F72")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("#FAFBFC")

        fig.patch.set_facecolor("#FAFBFC")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_val = max(curves["val_acc"])
        best_epoch = curves["val_acc"].index(best_val) + 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Meilleure val_acc", f"{best_val*100:.1f}%")
        c2.metric("Epoch optimale", best_epoch)
        c3.metric("Val loss finale", f"{curves['val_loss'][-1]:.4f}")

    st.markdown("---")
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        launch = st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True)

    if launch:
        with st.spinner(f"Lancement du run `{run_name}` — entraînement en cours…"):
            config = {"arch": arch, "lr": lr, "batch_size": batch_size,
                      "epochs": epochs, "augment": augment, "freeze": freeze}
            success = launch_training(config)

        if success:
            st.success(f"✅ Run **{run_name}** enregistré dans MLflow. Consultez l'interface ci-dessous.")
        else:
            st.error("❌ Échec du lancement. Vérifiez les logs.")

    st.info("💡 **Tip :** Ouvrez l'interface MLflow pour comparer vos runs en temps réel → `mlflow ui --port 5000`")
    if st.button("🔗 Ouvrir MLflow UI", help="Ouvre localhost:5000 dans un nouvel onglet"):
        st.markdown('<meta http-equiv="refresh" content="0;url=http://localhost:5000">', unsafe_allow_html=True)

# ─── Onglet 2 : Runs MLflow ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Runs enregistrés</div>', unsafe_allow_html=True)

    runs = get_mlflow_runs()
    for run in runs:
        badge = '<span class="run-badge badge-best">⭐ MEILLEUR</span>' if run["best"] else '<span class="run-badge badge-run">RUN</span>'
        acc_str = f"{run['val_acc']}%" if run["val_acc"] else "—"
        f1_str = f"{run['f1_macro']:.3f}" if run["f1_macro"] else "—"

        st.markdown(f"""
        <div class="run-card">
            {badge} &nbsp; <strong>{run['name']}</strong><br>
            <span style="font-size:0.83rem;color:#5D6D7E">
                Architecture : <code>{run['arch']}</code> &nbsp;|&nbsp;
                LR : <code>{run['lr']}</code> &nbsp;|&nbsp;
                Batch : <code>{run['batch']}</code> &nbsp;|&nbsp;
                Epochs : <code>{run['epochs']}</code><br>
                Val Accuracy : <strong style="color:#1B4F72">{acc_str}</strong> &nbsp;|&nbsp;
                F1 Macro : <strong style="color:#17A589">{f1_str}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

# ─── Onglet 3 : Comparaison ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Comparaison des architectures</div>', unsafe_allow_html=True)

    import pandas as pd
    runs_df = pd.DataFrame([r for r in runs if r["val_acc"] is not None])
    runs_df = runs_df[["name", "arch", "lr", "batch", "epochs", "val_acc", "f1_macro"]].copy()
    runs_df.columns = ["Run", "Architecture", "LR", "Batch", "Epochs", "Val Acc (%)", "F1 Macro"]
    runs_df = runs_df.sort_values("Val Acc (%)", ascending=False)

    st.dataframe(
        runs_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Val Acc (%)": st.column_config.ProgressColumn("Val Acc (%)", min_value=0, max_value=100),
            "F1 Macro": st.column_config.ProgressColumn("F1 Macro", min_value=0, max_value=1),
        }
    )

    # Bar chart comparaison
    fig, ax = plt.subplots(figsize=(8, 3.5))
    archs = runs_df["Architecture"].tolist()
    accs = runs_df["Val Acc (%)"].tolist()
    colors = ["#17A589" if a == max(accs) else "#1B4F72" for a in accs]
    bars = ax.bar(archs, accs, color=colors, edgecolor="white", width=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{acc}%", ha="center", fontsize=10, fontweight="700", color="#2C3E50")
    ax.set_ylabel("Val Accuracy (%)", fontsize=10)
    ax.set_ylim(0, max(accs) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFBFC")
    fig.patch.set_facecolor("#FAFBFC")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
