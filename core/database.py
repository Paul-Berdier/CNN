# -*- coding: utf-8 -*-
"""database.py — Connexion et opérations sur la base vectorielle (ChromaDB).

Ce module gère DEUX collections distinctes dans une même base ChromaDB :

  1. `wound_images`            -> embeddings d'IMAGES (similarité visuelle, Partie 2).
                                  Les vecteurs (sortie du CNN) sont fournis à la main.
  2. `connaissances_medicales` -> documents TEXTE des protocoles (RAG, Partie 5).
                                  Le texte est embeddé automatiquement (sentence-transformers).

Les deux collections partagent le même client / le même dossier sur disque,
mais restent séparées (embeddings de natures différentes).

Dépendances : pip install chromadb sentence-transformers
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from core.config import settings


# ─────────────────────────────────────────────
# 0. Client ChromaDB (partagé par les deux collections)
# ─────────────────────────────────────────────
def get_chroma_client(persist_dir: str | None = None) -> chromadb.PersistentClient:
    """Crée ou ouvre la base ChromaDB persistante (un dossier sur disque).

    Le mode persistant garantit que les embeddings survivent entre deux
    exécutions. Par défaut, on utilise le chemin défini dans la config, afin
    que les images ET les documents médicaux vivent dans la même base.
    """
    persist_dir = persist_dir or settings.chroma_path
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_client_chromadb() -> chromadb.PersistentClient:
    """Alias conservé pour la partie 5 — pointe sur le même client/DB."""
    return get_chroma_client()


# ═════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Collection d'IMAGES (similarité visuelle)
# ═════════════════════════════════════════════════════════════════════════
def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "wound_images",
) -> chromadb.Collection:
    """Récupère (ou crée) la collection d'embeddings d'images.

    Métrique cosinus, adaptée aux embeddings normalisés L2 produits par le CNN.
    (Pas de fonction d'embedding attachée : on fournit les vecteurs nous-mêmes.)
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{collection_name}' : {collection.count()} embeddings existants")
    return collection


def insert_embeddings(
    collection: chromadb.Collection,
    records: list[dict],
    batch_size: int = 100,
) -> None:
    """Insère une liste d'embeddings d'images dans ChromaDB, par batch.

    Chaque record : {"id", "path", "class", "embedding"} (embedding = np.ndarray).
    `upsert` = insert si nouveau, update sinon (relançable sans doublon).
    """
    total = len(records)
    inserted = 0
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        ids = [r["id"] for r in batch]
        embeddings = [r["embedding"].tolist() for r in batch]  # ChromaDB veut des listes Python
        metadatas = [{"path": r["path"], "class": r["class"]} for r in batch]
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        inserted += len(batch)
        print(f"  Inséré {inserted}/{total} embeddings...")
    print(f"Insertion terminée : {total} embeddings dans la collection")


def search_similar(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    k: int = 5,
    exclude_id: Optional[str] = None,
) -> list[dict]:
    """Recherche les K images les plus similaires à un embedding requête.

    `exclude_id` : id à exclure (évite que l'image requête se retourne elle-même).
    Renvoie une liste de {"id", "path", "class", "similarity"} (similarity ∈ [0, 1]).
    """
    n_results = k + 1 if exclude_id else k
    actual_n = min(n_results, collection.count())
    if actual_n == 0:
        return []

    raw = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=actual_n,
        include=["metadatas", "distances"],
    )

    results = []
    ids = raw["ids"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]  # distance cosinus ∈ [0, 2]

    for id_, meta, dist in zip(ids, metadatas, distances):
        if exclude_id and id_ == exclude_id:
            continue
        # distance cosinus ∈ [0, 2] -> similarité ∈ [0, 1]
        similarity = round(1 - (dist / 2), 4)
        results.append({
            "id": id_,
            "path": meta["path"],
            "class": meta["class"],
            "similarity": similarity,
        })
    return results[:k]


def get_collection_stats(collection: chromadb.Collection) -> dict:
    """Statistiques basiques sur la collection (utile pour la page d'accueil Streamlit)."""
    count = collection.count()
    if count > 0:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        class_counts: dict = {}
        for meta in all_meta:
            c = meta.get("class", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1
    else:
        class_counts = {}
    return {"total_images": count, "class_distribution": class_counts}


def reset_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "wound_images",
) -> None:
    """Supprime et recrée la collection d'images. /!\\ Irréversible (dev uniquement)."""
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' supprimée.")


# ═════════════════════════════════════════════════════════════════════════
# PARTIE 5 — Collection de CONNAISSANCES MÉDICALES (RAG)
# ═════════════════════════════════════════════════════════════════════════
def get_embedding_function():
    """Fonction d'embedding sentence-transformers, attachée à la collection médicale.

    Indexation et requêtes utilisent ainsi automatiquement le MÊME modèle.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model,
    )


def get_collection_medical(client=None):
    """Récupère (ou crée) la collection des documents médicaux, en cosinus."""
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_medical,
        embedding_function=get_embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"description": "Protocoles de traitement des plaies (RAG - Partie 5)"},
    )


def _load_jsonl(jsonl_path: str) -> list[dict]:
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def index_knowledge_base(jsonl_path: str = settings.jsonl_path, reset: bool = False) -> int:
    """Indexe la base JSONL des protocoles dans ChromaDB. Idempotent (upsert)."""
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(settings.collection_medical)
        except Exception:
            pass

    col = get_collection_medical(client)
    docs = _load_jsonl(jsonl_path)

    ids = [d["id"] for d in docs]
    documents = [d["contenu"] for d in docs]
    metadatas = [
        {
            "type_plaie": d["type_plaie"],
            "categorie": d["categorie"],
            "titre": d["titre"],
            "source": d["source"],
            "mots_cles": ", ".join(d.get("mots_cles", [])),
        }
        for d in docs
    ]

    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return col.count()


def search_kb(query: str, type_plaie: Optional[str] = None, k: int = 3) -> list[dict]:
    """Recherche sémantique dans la base médicale. Filtre optionnel par type de plaie."""
    col = get_collection_medical()
    where = {"type_plaie": type_plaie} if type_plaie else None
    res = col.query(query_texts=[query], n_results=k, where=where)

    resultats = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]
        resultats.append(
            {
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": distance,
                "similarite": 1 - distance,
            }
        )
    return resultats


# ─────────────────────────────────────────────
# Exécution directe : indexe la base médicale + démo de recherche
#   $ python -m core.database
# ─────────────────────────────────────────────
if __name__ == "__main__":
    n = index_knowledge_base(reset=True)
    print(f"[OK] {n} documents indexés dans la collection '{settings.collection_medical}'.\n")

    print("--- Requête libre : 'plaie qui ne cicatrise pas, faut-il consulter ?' ---")
    for r in search_kb("plaie qui ne cicatrise pas, faut-il consulter un spécialiste ?", k=3):
        m = r["metadata"]
        print(f"  [{r['similarite']:.3f}] {m['titre']} — {m['type_plaie']}/{m['categorie']}")

    print("\n--- Diagnostic CNN = 'venous_ulcers' : protocole de traitement ---")
    for r in search_kb("traitement et prise en charge", type_plaie="venous_ulcers", k=3):
        m = r["metadata"]
        print(f"  [{r['similarite']:.3f}] {m['titre']} — {m['categorie']}")