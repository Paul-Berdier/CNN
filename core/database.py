"""
database.py
-----------
Interface avec ChromaDB pour stocker et rechercher les embeddings d'images.

Principe (analogie bio) :
ChromaDB c'est comme une mémoire à long terme organisée par "similarité de forme".
Au lieu de chercher un mot-clé exact, on cherche "ce qui ressemble à ça".
C'est exactement ce dont le clinicien a besoin : pas "trouve-moi un ulcère veineux"
mais "trouve-moi des plaies qui ressemblent visuellement à celle-ci".

Dépendance : chromadb  →  pip install chromadb
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# 1. Connexion à ChromaDB
# ─────────────────────────────────────────────

def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Crée ou ouvre une base ChromaDB persistante sur disque.

    Le mode "persistant" est important : les embeddings survivent
    entre deux exécutions du programme. Sinon tout est perdu au redémarrage.

    Args:
        persist_dir : dossier où ChromaDB stocke ses données

    Returns:
        client ChromaDB prêt à l'emploi
    """
    # Création du dossier si nécessaire
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    print(f"ChromaDB connecté → {persist_dir}")
    return client


# ─────────────────────────────────────────────
# 2. Création / récupération de la collection
# ─────────────────────────────────────────────

def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "wound_images"
) -> chromadb.Collection:
    """
    Récupère la collection d'embeddings d'images si elle existe,
    la crée sinon.

    Une "collection" dans ChromaDB = une table dans une BDD classique,
    mais optimisée pour la recherche par similarité vectorielle.

    Args:
        client          : client ChromaDB
        collection_name : nom de la collection (on en aura une autre pour le RAG)

    Returns:
        collection ChromaDB
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        # Métrique de distance : cosinus
        # Adapté aux embeddings normalisés L2 (ce qu'on produit dans image_similarity.py)
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Collection '{collection_name}' : {collection.count()} embeddings existants")
    return collection


# ─────────────────────────────────────────────
# 3. Insertion des embeddings dans la base
# ─────────────────────────────────────────────

def insert_embeddings(
    collection: chromadb.Collection,
    records: list[dict],
    batch_size: int = 100
) -> None:
    """
    Insère une liste d'embeddings dans ChromaDB.

    On insère par batch pour éviter de surcharger la mémoire
    si le dataset est grand.

    Args:
        collection : collection ChromaDB cible
        records    : liste de dicts produite par extract_all_embeddings()
                     Chaque dict doit avoir : "id", "path", "class", "embedding"
        batch_size : nombre d'embeddings insérés par appel
    """
    total = len(records)
    inserted = 0

    # Traitement par batch
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]

        ids         = [r["id"] for r in batch]
        embeddings  = [r["embedding"].tolist() for r in batch]  # ChromaDB veut des listes Python
        metadatas   = [{"path": r["path"], "class": r["class"]} for r in batch]

        # upsert = insert si nouveau, update si l'id existe déjà
        # Pratique pour relancer le script sans dupliquer
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

        inserted += len(batch)
        print(f"  Inséré {inserted}/{total} embeddings...")

    print(f"Insertion terminée : {total} embeddings dans la collection")


# ─────────────────────────────────────────────
# 4. Recherche des K plus proches voisins
# ─────────────────────────────────────────────

def search_similar(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    k: int = 5,
    exclude_id: str = None
) -> list[dict]:
    """
    Recherche les K images les plus similaires à un embedding de requête.

    Args:
        collection      : collection ChromaDB
        query_embedding : vecteur numpy normalisé de l'image requête
        k               : nombre de résultats à retourner
        exclude_id      : ID à exclure des résultats (utile si l'image
                          requête est déjà dans la base — évite de se
                          retourner soi-même comme "meilleur résultat")

    Returns:
        results : liste de dicts avec les clés :
                  - "id"         : identifiant de l'image
                  - "path"       : chemin vers l'image
                  - "class"      : classe de la plaie
                  - "similarity" : score entre 0 et 1 (1 = identique)
    """
    # On demande k+1 résultats si on doit en exclure un
    n_results = k + 1 if exclude_id else k

    actual_n = min(n_results, collection.count())
    if actual_n == 0:
        return []

    raw = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=actual_n,
        include=["metadatas", "distances"]
    )

    results = []

    ids        = raw["ids"][0]           # liste d'IDs
    metadatas  = raw["metadatas"][0]     # liste de métadonnées
    distances  = raw["distances"][0]     # liste de distances cosinus (entre 0 et 2)

    for id_, meta, dist in zip(ids, metadatas, distances):

        # On saute l'image requête si elle est dans la base
        if exclude_id and id_ == exclude_id:
            continue

        # Conversion distance cosinus → similarité entre 0 et 1
        # Distance cosinus ∈ [0, 2] → similarité = 1 - (dist / 2)
        similarity = round(1 - (dist / 2), 4)

        results.append({
            "id":         id_,
            "path":       meta["path"],
            "class":      meta["class"],
            "similarity": similarity
        })

    # On s'assure de ne retourner que k résultats
    return results[:k]


# ─────────────────────────────────────────────
# 5. Utilitaires
# ─────────────────────────────────────────────

def get_collection_stats(collection: chromadb.Collection) -> dict:
    """
    Retourne des statistiques basiques sur la collection.
    Utile pour la page d'accueil Streamlit (P4).
    """
    count = collection.count()

    # Récupère tous les métadonnées pour compter par classe
    if count > 0:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        class_counts = {}
        for meta in all_meta:
            c = meta.get("class", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1
    else:
        class_counts = {}

    return {
        "total_images": count,
        "class_distribution": class_counts
    }


def reset_collection(client: chromadb.PersistentClient, collection_name: str = "wound_images") -> None:
    """
    Supprime et recrée la collection.
    À utiliser uniquement en développement pour repartir de zéro.

    /!\ Irréversible — tous les embeddings sont perdus.
    """
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' supprimée.")