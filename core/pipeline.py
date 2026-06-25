"""
pipeline.py
-----------
Pipeline complet de recherche par similarité visuelle.
C'est le seul fichier que P4 (Streamlit) a besoin d'importer.

Utilisation typique (côté P4) :
    from core.pipeline import SimilarityPipeline

    pipeline = SimilarityPipeline(chroma_dir="./chroma_db")

    # Phase d'indexation — une seule fois
    pipeline.index_dataset("data/raw/")

    # Phase de recherche — à chaque upload dans Streamlit
    results = pipeline.search("path/to/image.jpg", k=5)

Pourquoi encapsuler dans une classe ?
    Le notebook solution est pensé pour Colab (script linéaire, tout dans l'ordre).
    Dans Streamlit, la recherche est appelée à chaque upload utilisateur.
    Sans encapsulation, P4 devrait gérer elle-même le chargement du modèle,
    la connexion ChromaDB et l'état entre les appels.
    La classe SimilarityPipeline isole tout ça : P4 ne voit qu'une méthode search().
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


class SimilarityPipeline:
    """
    Encapsule tout le pipeline de recherche par similarité.

    Cycle de vie :
        1. Instanciation  → charge ResNet50 et ouvre ChromaDB
        2. Indexation     → extrait et stocke les embeddings (une fois)
        3. Recherche      → pour chaque nouvelle image uploadée
    """

    def __init__(
        self,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "wound_images",
    ):
        """
        Initialise le pipeline.

        Plus besoin de model_path : le FeatureExtractor charge
        directement les poids ImageNet via torchvision.

        Args:
            chroma_dir      : dossier de persistance ChromaDB
            collection_name : nom de la collection ChromaDB
        """
        print("Initialisation du pipeline de similarité...")

        # 1. Chargement du FeatureExtractor ResNet50
        self.extractor = load_feature_extractor()

        # 2. Connexion à ChromaDB
        self.client     = get_chroma_client(chroma_dir)
        self.collection = get_or_create_collection(self.client, collection_name)

        print("Pipeline prêt.")

    # ─────────────────────────────────────────
    # Indexation du dataset
    # ─────────────────────────────────────────

    def index_dataset(self, dataset_dir: str, force_reindex: bool = False) -> None:
        """
        Extrait les embeddings de toutes les images et les stocke dans ChromaDB.
        À appeler UNE SEULE FOIS (ou si le dataset change).

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

        records = extract_all_embeddings(
            extractor=self.extractor,
            dataset_dir=dataset_dir
        )

        if not records:
            print("[ATTENTION] Aucune image trouvée dans le dataset.")
            return

        insert_embeddings(self.collection, records)
        print(f"Indexation terminée : {len(records)} images dans la base.")

    # ─────────────────────────────────────────
    # Recherche de similarité
    # ─────────────────────────────────────────

    def search(self, img_path: str, k: int = 5) -> list[dict]:
        """
        Trouve les K images les plus similaires à une image requête.
        Fonction principale appelée par P4 à chaque upload.

        Args:
            img_path : chemin vers l'image uploadée
            k        : nombre de résultats

        Returns:
            [
                {
                    "id"        : "drive_burn_img_042.jpg",
                    "path"      : "/data/raw/burn/img_042.jpg",
                    "class"     : "burn",
                    "similarity": 0.94
                },
                ...
            ]

        Raises:
            FileNotFoundError : si l'image n'existe pas
            RuntimeError      : si la base n'est pas encore indexée
        """
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image introuvable : {img_path}")

        if self.collection.count() == 0:
            raise RuntimeError(
                "La base est vide. Appelez d'abord index_dataset()."
            )

        query_embedding = extract_embedding(self.extractor, img_path)

        results = search_similar(
            collection=self.collection,
            query_embedding=query_embedding,
            k=k,
            exclude_id=f"drive_{Path(img_path).parent.name}_{Path(img_path).name}"
        )

        return results

    # ─────────────────────────────────────────
    # Utilitaires pour P4
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Stats de la base — pour la page d'accueil Streamlit."""
        return get_collection_stats(self.collection)

    def is_indexed(self) -> bool:
        """True si la base contient au moins un embedding."""
        return self.collection.count() > 0


# ─────────────────────────────────────────────
# Script d'indexation standalone
# ─────────────────────────────────────────────
# python -m core.pipeline --dataset data/raw/

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexation du dataset dans ChromaDB")
    parser.add_argument("--dataset",    required=True,          help="Chemin vers data/raw/")
    parser.add_argument("--chroma_dir", default="./chroma_db",  help="Dossier ChromaDB")
    parser.add_argument("--reindex",    action="store_true",    help="Forcer la réindexation")
    args = parser.parse_args()

    pipeline = SimilarityPipeline(chroma_dir=args.chroma_dir)
    pipeline.index_dataset(args.dataset, force_reindex=args.reindex)

    stats = pipeline.get_stats()
    print(f"\nStats : {stats['total_images']} images indexées")
    for cls, count in stats["class_distribution"].items():
        print(f"  {cls} : {count} images")