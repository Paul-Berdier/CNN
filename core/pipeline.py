"""
pipeline.py
-----------
Pipeline complet de recherche par similarité visuelle.
C'est le seul fichier que P4 (Streamlit) a besoin d'importer.

Utilisation typique (côté P4) :
    from core.pipeline import SimilarityPipeline

    pipeline = SimilarityPipeline(
        model_path="models/best_model.h5",
        chroma_dir="./chroma_db"
    )

    # Phase d'indexation (à faire une seule fois)
    pipeline.index_dataset("data/raw/")

    # Phase de recherche (appelée à chaque upload dans Streamlit)
    results = pipeline.search("path/to/new_image.jpg", k=5)

Dépendances : image_similarity.py, database.py
"""

import numpy as np
from pathlib import Path

from core.image_similarity import (
    load_feature_extractor,
    extract_embedding,
    extract_all_embeddings,
)
from core.database import (
    get_chroma_client,
    get_or_create_collection,
    insert_embeddings,
    search_similar,
    get_collection_stats,
)


# ─────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────

class SimilarityPipeline:
    """
    Encapsule tout le pipeline de recherche par similarité.

    Cycle de vie :
        1. Instanciation  → charge le modèle et ouvre la base
        2. Indexation     → extrait et stocke les embeddings du dataset (une fois)
        3. Recherche      → pour chaque nouvelle image uploadée
    """

    def __init__(
        self,
        model_path: str,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "wound_images",
        target_size: tuple = (224, 224),
        layer_name: str = None,
    ):
        """
        Initialise le pipeline.

        Args:
            model_path      : chemin vers le modèle .h5 ou .keras de P1
            chroma_dir      : dossier de persistance ChromaDB
            collection_name : nom de la collection ChromaDB
            target_size     : taille d'entrée du CNN (doit correspondre à P1)
            layer_name      : couche de sortie pour les embeddings.
                              None = auto (avant-dernière couche)
        """
        print("Initialisation du pipeline de similarité...")

        # 1. Chargement de l'extracteur d'embeddings (CNN tronqué)
        self.extractor   = load_feature_extractor(model_path, layer_name)
        self.target_size = target_size

        # 2. Connexion à ChromaDB
        self.client     = get_chroma_client(chroma_dir)
        self.collection = get_or_create_collection(self.client, collection_name)

        print("Pipeline prêt.")


    # ─────────────────────────────────────────
    # Indexation du dataset
    # ─────────────────────────────────────────

    def index_dataset(self, dataset_dir: str, force_reindex: bool = False) -> None:
        """
        Extrait les embeddings de toutes les images du dataset
        et les stocke dans ChromaDB.

        À appeler UNE SEULE FOIS (ou si le dataset change).
        Si la base est déjà remplie, on saute l'indexation
        sauf si force_reindex=True.

        Args:
            dataset_dir   : chemin vers data/raw/
            force_reindex : si True, réindexe même si la base n'est pas vide
        """
        current_count = self.collection.count()

        if current_count > 0 and not force_reindex:
            print(f"Base déjà indexée ({current_count} embeddings). "
                  f"Utilisez force_reindex=True pour réindexer.")
            return

        print(f"Indexation du dataset : {dataset_dir}")

        # Extraction de tous les embeddings
        records = extract_all_embeddings(
            extractor=self.extractor,
            dataset_dir=dataset_dir,
            target_size=self.target_size
        )

        if not records:
            print("[ATTENTION] Aucune image trouvée dans le dataset.")
            return

        # Stockage dans ChromaDB
        insert_embeddings(self.collection, records)
        print(f"Indexation terminée : {len(records)} images dans la base.")


    # ─────────────────────────────────────────
    # Recherche de similarité
    # ─────────────────────────────────────────

    def search(self, img_path: str, k: int = 5) -> list[dict]:
        """
        Trouve les K images les plus similaires à une image requête.

        C'est la fonction principale appelée par P4 à chaque upload.

        Args:
            img_path : chemin vers l'image uploadée par le clinicien
            k        : nombre de résultats à retourner

        Returns:
            Liste de dicts triés par similarité décroissante :
            [
                {
                    "id"        : "classe_1/img_042",
                    "path"      : "/data/raw/classe_1/img_042.jpg",
                    "class"     : "venous_ulcer",
                    "similarity": 0.94       ← entre 0 et 1
                },
                ...
            ]

        Raises:
            FileNotFoundError : si l'image n'existe pas
            RuntimeError      : si la base est vide (pas encore indexée)
        """
        # Vérifications préalables
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image introuvable : {img_path}")

        if self.collection.count() == 0:
            raise RuntimeError(
                "La base est vide. Appelez d'abord index_dataset() "
                "pour indexer le dataset."
            )

        # Extraction de l'embedding de l'image requête
        query_embedding = extract_embedding(
            extractor=self.extractor,
            img_path=img_path,
            target_size=self.target_size
        )

        # Recherche dans ChromaDB
        results = search_similar(
            collection=self.collection,
            query_embedding=query_embedding,
            k=k,
            # On passe le nom du fichier comme exclude_id
            # au cas où l'image est déjà dans la base
            exclude_id=str(Path(img_path).stem)
        )

        return results


    # ─────────────────────────────────────────
    # Utilitaires exposés à P4
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Retourne les stats de la base vectorielle.
        Appelable depuis la page d'accueil Streamlit.

        Returns:
            {
                "total_images": 1240,
                "class_distribution": {
                    "venous_ulcer": 312,
                    "burn": 289,
                    ...
                }
            }
        """
        return get_collection_stats(self.collection)

    def is_indexed(self) -> bool:
        """
        Retourne True si la base contient au moins un embedding.
        Utile pour afficher un warning dans Streamlit si l'indexation
        n'a pas encore été faite.
        """
        return self.collection.count() > 0


# ─────────────────────────────────────────────
# Script d'indexation standalone
# ─────────────────────────────────────────────
# Permet de lancer l'indexation en ligne de commande :
#   python -m core.pipeline
# Sans avoir besoin de lancer Streamlit.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexation du dataset dans ChromaDB")
    parser.add_argument("--model",      required=True, help="Chemin vers le modèle .h5 de P1")
    parser.add_argument("--dataset",    required=True, help="Chemin vers data/raw/")
    parser.add_argument("--chroma_dir", default="./chroma_db", help="Dossier ChromaDB")
    parser.add_argument("--reindex",    action="store_true", help="Forcer la réindexation")
    args = parser.parse_args()

    pipeline = SimilarityPipeline(
        model_path=args.model,
        chroma_dir=args.chroma_dir
    )

    pipeline.index_dataset(args.dataset, force_reindex=args.reindex)

    stats = pipeline.get_stats()
    print(f"\nStats finales : {stats['total_images']} images indexées")
    for cls, count in stats["class_distribution"].items():
        print(f"  {cls} : {count} images")