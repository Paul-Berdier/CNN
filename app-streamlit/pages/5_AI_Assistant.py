"""
Page 5 — Assistant IA (RAG + LLM + Langfuse)
"""

import streamlit as st
import time
import json
from datetime import datetime

st.set_page_config(page_title="Assistant IA — WoundAI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1B4F72,#154360); }
    [data-testid="stSidebar"] * { color:#ECF0F1!important; }
    .section-title { font-size:1.35rem; font-weight:700; color:#1B4F72; margin:1.5rem 0 0.8rem; border-bottom:2px solid #AED6F1; padding-bottom:0.4rem; }
    .reco-card { background:#F4F8FB; border:1px solid #D5E8F0; border-radius:12px; padding:1.5rem 1.8rem; }
    .reco-section { margin:1rem 0; }
    .reco-section-title { font-size:0.88rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#1B4F72; margin-bottom:0.4rem; }
    .doc-card { background:white; border:1px solid #E8EEF3; border-left:3px solid #17A589; border-radius:0 8px 8px 0; padding:0.7rem 1rem; margin-bottom:0.6rem; font-size:0.84rem; }
    .trace-card { background:#FAFBFC; border:1px solid #E0E6EB; border-radius:8px; padding:0.8rem 1rem; font-size:0.82rem; font-family:monospace; }
    .disclaimer { background:#FEF9E7; border:1px solid #F9E79F; border-left:4px solid #E67E22; border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; font-size:0.83rem; color:#7D6608; margin-top:1rem; }
    .history-item { background:#F8FAFB; border:1px solid #EAF0F5; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.6rem; }
</style>
""", unsafe_allow_html=True)


# ─── INTERFACE BACKEND ────────────────────────────────────────────────────────

def retrieve_medical_documents(class_name: str, n_docs: int = 3) -> list[dict]:
    """
    Recherche sémantique dans la base de connaissances médicales (ChromaDB).
    BACKEND:
        from sentence_transformers import SentenceTransformer
        import chromadb
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("medical_knowledge")
        query_emb = encoder.encode([class_name]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=n_docs)
    Retour attendu: list de dict {'id': str, 'title': str, 'content': str, 'source': str}
    """
    kb = {
        "Plaie diabétique": [
            {"id": "proto-diab-01", "title": "Protocole de traitement des plaies du pied diabétique",
             "content": "Nettoyage à l'eau stérile ou sérum physiologique. Débridement des tissus nécrotiques si présents. Application d'un pansement hydrocelloïde ou hydrofibre selon l'exsudat. Décharge du pied obligatoire. Surveillance biquotidienne de la glycémie. Antibiothérapie si signes d'infection (rougeur, chaleur, écoulement purulent).",
             "source": "Protocole CHU - Diabétologie 2024"},
            {"id": "proto-diab-02", "title": "Soins infirmiers — Pied diabétique",
             "content": "Évaluation vasculaire (index de pression systolique). Mesure de la plaie (longueur × largeur × profondeur). Photo de suivi à chaque pansement. Éducation du patient sur l'inspection quotidienne. Orientation podologue et diabétologue dans les 48h.",
             "source": "Guide Soins IDE 2023"},
            {"id": "proto-diab-03", "title": "Critères d'alerte — Orientation urgences",
             "content": "Cellulite extensive, signes systémiques d'infection (fièvre, hypotension), gangrène, déformation osseuse (pied de Charcot), douleur ischémique de repos. Hospitalisation immédiate requise dans ces cas.",
             "source": "SFDIA Recommandations 2024"},
        ],
        "Brûlure": [
            {"id": "proto-burn-01", "title": "Prise en charge des brûlures cutanées",
             "content": "Refroidissement immédiat à l'eau tiède (15-20°C) pendant 20 minutes. Ne pas utiliser de glace. Évaluation de la surface brûlée (règle des 9 de Wallace). Pansement gras ou hydrocolloïde pour brûlures superficielles. Antalgie adaptée.",
             "source": "Protocole Urgences Brûlés 2024"},
            {"id": "proto-burn-02", "title": "Classification et orientation des brûlures",
             "content": "1er degré : traitement ambulatoire. 2ème degré superficiel : centre de soins. 2ème degré profond et 3ème degré : centre des brûlés. Brûlures > 10% SCB chez adulte, > 5% chez enfant : hospitalisation systématique.",
             "source": "SFETB Guidelines 2023"},
        ],
        "Escarre": [
            {"id": "proto-esc-01", "title": "Protocole de soins des escarres",
             "content": "Nettoyage à l'eau et savon doux ou sérum physiologique. Débridement enzymatique si nécrose. Pansements adaptés au stade (stade I: film transparent, stade II: hydrocolloïde, stade III-IV: hydrofibre + alginate). Repositionnement toutes les 2h obligatoire.",
             "source": "HAS Recommandations Escarres 2024"},
        ],
    }
    default = [
        {"id": "proto-gen-01", "title": f"Soins généraux — {class_name}",
         "content": f"Évaluation clinique complète de la plaie. Nettoyage antiseptique doux. Pansement adapté à l'exsudat. Traçabilité photographique. Orientation spécialiste selon évolution.",
         "source": "Protocole Soins de Plaies 2024"},
    ]
    return kb.get(class_name, default)[:n_docs]


def build_rag_prompt(class_name: str, confidence: float, n_similar: int, documents: list[dict]) -> str:
    """
    Construit le prompt RAG augmenté par le contexte médical.
    """
    context = "\n\n".join([
        f"[{i+1}] **{doc['title']}** (source: {doc['source']})\n{doc['content']}"
        for i, doc in enumerate(documents)
    ])
    return f"""Tu es un assistant médical spécialisé dans le traitement des plaies.
Un modèle d'intelligence artificielle a analysé une image de plaie et a produit le diagnostic suivant :

Diagnostic : {class_name}
Confiance : {confidence*100:.1f}%
Cas similaires retrouvés : {n_similar} cas historiques

Voici les protocoles de traitement pertinents issus de la base de connaissances :
{context}

En te basant UNIQUEMENT sur les protocoles ci-dessus, fournis une réponse structurée avec :
1. Un résumé du diagnostic
2. Les recommandations de traitement immédiates
3. Les soins de suivi recommandés
4. Les critères d'alerte nécessitant une consultation spécialisée

IMPORTANT : Ces recommandations sont fournies à titre indicatif uniquement et ne remplacent pas l'avis d'un professionnel de santé qualifié."""


def call_llm(prompt: str, model: str = "llama3.2") -> dict:
    """
    Appelle le LLM local (Ollama) et enregistre la trace dans Langfuse.
    BACKEND:
        import ollama
        from langfuse import Langfuse
        langfuse = Langfuse()
        trace = langfuse.trace(name="rag-recommendation", input=prompt)
        start = time.time()
        response = ollama.generate(model=model, prompt=prompt)
        latency = time.time() - start
        trace.update(output=response["response"])
        trace.generation(name="llm-generation", model=model,
                         input=prompt, output=response["response"],
                         usage={"input": len(prompt.split()), "output": len(response["response"].split())})
        langfuse.flush()
    Retour attendu: {'response': str, 'latency_ms': int, 'tokens_in': int, 'tokens_out': int, 'trace_id': str}
    """
    # Mock : réponse simulée
    time.sleep(1.5)  # simule la latence LLM
    mock_response = f"""## Résumé du diagnostic

Le modèle d'IA a identifié une **plaie diabétique** avec une confiance de {'{conf}%'.replace('{conf}', '78')}. Ce type de plaie est fréquent chez les patients diabétiques et nécessite une prise en charge multidisciplinaire rapide.

## Recommandations de traitement immédiates

- **Nettoyage** : lavage à l'eau stérile ou sérum physiologique, sans friction excessive.
- **Débridement** : retrait des tissus nécrotiques et fibrineux si présents, par un professionnel habilité.
- **Pansement** : hydrofibre ou hydrocolloïde adapté au niveau d'exsudat. Changer toutes les 48-72h ou si saturé.
- **Décharge** : mise en décharge immédiate du pied (orthèse, fauteuil roulant) — **indispensable**.
- **Glycémie** : surveillance renforcée, objectif < 1.8 g/L.

## Soins de suivi recommandés

- Mesure et photographie de la plaie à chaque pansement (traçabilité).
- Bilan vasculaire périphérique (IPS) dans les 48h.
- Orientation podologue et diabétologue dans les 48-72h.
- Éducation du patient sur l'auto-inspection quotidienne des pieds.
- Contrôle HbA1c et ajustement du traitement si > 8%.

## Critères d'alerte — Consultation spécialisée urgente

- Rougeur, chaleur ou œdème s'étendant au-delà de la plaie (cellulite).
- Écoulement purulent ou odeur nauséabonde.
- Fièvre > 38.5°C ou frissons.
- Douleur ischémique de repos, pied froid et dépigmenté.
- Déformation osseuse (suspicion pied de Charcot) : IRM urgente.
- Aggravation en 48h malgré les soins.

---
⚠️ **Avertissement** : Ces recommandations sont générées par un système d'IA à titre indicatif et pédagogique uniquement. Elles ne remplacent pas l'évaluation clinique d'un médecin ou infirmier habilité."""

    return {
        "response": mock_response,
        "latency_ms": 1523,
        "tokens_in": len(prompt.split()),
        "tokens_out": len(mock_response.split()),
        "trace_id": f"trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}-mock",
        "model": model,
    }


def log_to_langfuse(prompt: str, response: str, metadata: dict) -> str:
    """
    Enregistre la trace dans Langfuse.
    BACKEND: utiliser le client Langfuse instancié globalement.
    Retour: trace_id (str)
    """
    # Mock
    return f"lf-trace-{datetime.now().strftime('%Y%m%d%H%M%S')}"


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
    llm_model = st.selectbox("Modèle LLM", ["llama3.2", "mistral", "qwen2.5"])
    n_docs = st.slider("Documents RAG", 1, 5, 3)
    n_similar = st.slider("Cas similaires", 0, 10, 5)

# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Assistant IA — Recommandations de traitement")
st.caption("Pipeline RAG : diagnostic CNN → recherche base de connaissances → LLM local (Ollama) → recommandation contextualisée.")

tab1, tab2, tab3 = st.tabs(["💊 Générer une recommandation", "📡 Traces Langfuse", "📋 Historique"])

# ─── Onglet 1 : Générer ───────────────────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown('<div class="section-title">Paramètres</div>', unsafe_allow_html=True)

        # Pré-remplir depuis session state si disponible
        CLASSES = ["Brûlure", "Ulcère veineux", "Plaie diabétique", "Escarre", "Plaie chirurgicale", "Peau saine"]
        default_cls = "Plaie diabétique"
        default_conf = 0.78
        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            default_cls = pred["predicted_class"] if pred["predicted_class"] in CLASSES else "Plaie diabétique"
            default_conf = pred["confidence"]
            st.success(f"✅ Diagnostic importé depuis la page Prédiction : **{default_cls}**")

        selected_class = st.selectbox("Diagnostic CNN", CLASSES, index=CLASSES.index(default_cls))
        confidence = st.slider("Score de confiance (%)", 0, 100, int(default_conf * 100))

        generate = st.button("🤖 Générer la recommandation", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("**Pipeline RAG**")
        st.markdown(f"""
        1. Diagnostic : `{selected_class}` ({confidence}%)
        2. Recherche : `{n_docs}` documents dans ChromaDB
        3. Prompt augmenté → `{llm_model}`
        4. Trace → Langfuse
        """)

    with col_out:
        st.markdown('<div class="section-title">Recommandation générée</div>', unsafe_allow_html=True)

        if generate:
            # Step 1 : retrieval
            with st.spinner("🔍 Recherche dans la base de connaissances médicales…"):
                docs = retrieve_medical_documents(selected_class, n_docs)

            st.markdown("**Documents retrouvés (contexte RAG)**")
            for doc in docs:
                st.markdown(f"""
                <div class="doc-card">
                    <strong>{doc['title']}</strong><br>
                    <span style="color:#7F8C8D;font-size:0.78rem">{doc['source']}</span>
                </div>
                """, unsafe_allow_html=True)

            # Step 2 : prompt
            prompt = build_rag_prompt(selected_class, confidence / 100, n_similar, docs)

            # Step 3 : LLM
            with st.spinner(f"🤖 Génération par `{llm_model}`…"):
                llm_result = call_llm(prompt, model=llm_model)

            # Affichage
            st.markdown("""<div class="reco-card">""", unsafe_allow_html=True)
            st.markdown(llm_result["response"])
            st.markdown("""</div>""", unsafe_allow_html=True)

            st.markdown("""<div class="disclaimer">
            ⚠️ <strong>Avertissement médical :</strong> Ces recommandations sont générées automatiquement à titre pédagogique uniquement.
            Elles ne remplacent en aucun cas l'évaluation d'un professionnel de santé qualifié.
            </div>""", unsafe_allow_html=True)

            # Métriques LLM
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latence", f"{llm_result['latency_ms']} ms")
            c2.metric("Tokens input", llm_result["tokens_in"])
            c3.metric("Tokens output", llm_result["tokens_out"])
            c4.metric("Trace ID", llm_result["trace_id"][:12] + "…")

            # Sauvegarde historique
            if "rag_history" not in st.session_state:
                st.session_state["rag_history"] = []
            st.session_state["rag_history"].append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "class": selected_class,
                "confidence": confidence,
                "model": llm_model,
                "latency_ms": llm_result["latency_ms"],
                "trace_id": llm_result["trace_id"],
                "response_preview": llm_result["response"][:120] + "…",
            })

        else:
            st.info("👆 Sélectionnez un diagnostic et cliquez sur **Générer la recommandation** pour lancer le pipeline RAG.")

# ─── Onglet 2 : Traces Langfuse ───────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Traces Langfuse</div>', unsafe_allow_html=True)
    st.caption("Chaque appel au LLM est tracé dans Langfuse : prompt, réponse, latence, tokens.")

    mock_traces = [
        {"trace_id": "lf-20240615-143201-abc", "name": "rag-recommendation", "class": "Plaie diabétique",
         "model": "llama3.2", "latency_ms": 1523, "tokens_in": 487, "tokens_out": 312, "status": "SUCCESS"},
        {"trace_id": "lf-20240615-142845-def", "name": "rag-recommendation", "class": "Brûlure",
         "model": "llama3.2", "latency_ms": 1891, "tokens_in": 412, "tokens_out": 289, "status": "SUCCESS"},
        {"trace_id": "lf-20240615-141200-ghi", "name": "rag-recommendation", "class": "Escarre",
         "model": "mistral", "latency_ms": 2103, "tokens_in": 398, "tokens_out": 341, "status": "SUCCESS"},
        {"trace_id": "lf-20240615-140532-jkl", "name": "rag-recommendation", "class": "Ulcère veineux",
         "model": "llama3.2", "latency_ms": 1647, "tokens_in": 455, "tokens_out": 298, "status": "SUCCESS"},
        {"trace_id": "lf-20240615-135918-mno", "name": "rag-recommendation", "class": "Plaie chirurgicale",
         "model": "qwen2.5", "latency_ms": 987, "tokens_in": 421, "tokens_out": 275, "status": "SUCCESS"},
    ]

    # Ajouter les traces de session
    if "rag_history" in st.session_state:
        for h in st.session_state["rag_history"]:
            mock_traces.insert(0, {
                "trace_id": h["trace_id"],
                "name": "rag-recommendation",
                "class": h["class"],
                "model": h["model"],
                "latency_ms": h["latency_ms"],
                "tokens_in": 450,
                "tokens_out": 300,
                "status": "SUCCESS",
            })

    for trace in mock_traces[:7]:
        st.markdown(f"""
        <div class="trace-card">
            <span style="color:#17A589;font-weight:700">✓ {trace['status']}</span> &nbsp;
            <strong>{trace['name']}</strong> &nbsp;
            <span style="color:#7F8C8D">({trace['trace_id'][:24]}…)</span><br>
            Class: <code>{trace['class']}</code> | Model: <code>{trace['model']}</code> |
            Latence: <strong>{trace['latency_ms']} ms</strong> |
            Tokens: <strong>{trace['tokens_in']} in / {trace['tokens_out']} out</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔗 Ouvrir le dashboard Langfuse"):
        st.markdown('[Ouvrir Langfuse Cloud →](https://cloud.langfuse.com)', unsafe_allow_html=True)

    # Statistiques agrégées
    st.markdown('<div class="section-title">Statistiques d\'utilisation</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total traces", len(mock_traces))
    avg_lat = sum(t["latency_ms"] for t in mock_traces) / len(mock_traces)
    c2.metric("Latence moyenne", f"{avg_lat:.0f} ms")
    total_tokens = sum(t["tokens_in"] + t["tokens_out"] for t in mock_traces)
    c3.metric("Tokens consommés", f"{total_tokens:,}")
    c4.metric("Taux succès", "100%")

# ─── Onglet 3 : Historique ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Historique des consultations (session)</div>', unsafe_allow_html=True)

    if "rag_history" not in st.session_state or not st.session_state["rag_history"]:
        st.info("Aucune recommandation générée dans cette session. Utilisez l'onglet **Générer**.")
    else:
        for entry in reversed(st.session_state["rag_history"]):
            st.markdown(f"""
            <div class="history-item">
                <strong>{entry['timestamp']}</strong> — <code>{entry['class']}</code> ({entry['confidence']}%) —
                Modèle : <code>{entry['model']}</code> — Latence : <strong>{entry['latency_ms']} ms</strong><br>
                <span style="color:#5D6D7E;font-size:0.82rem">{entry['response_preview']}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Effacer l'historique"):
            st.session_state["rag_history"] = []
            st.rerun()
