"""
test_similarity.py
------------------
Tests unitaires pour la Partie 2 — Recherche par similarité.

Stratégie : on ne teste PAS le modèle CNN de P1 ici (c'est le rôle de test_model.py).
On teste uniquement la logique de notre code :
    - normalisation des embeddings
    - insertion et recherche dans ChromaDB
    - comportement du pipeline (erreurs, cas limites)

On utilise des embeddings FACTICES (vecteurs aléatoires) pour être
totalement indépendants du modèle de P1 et du dataset réel.

Lancement :
    pytest tests/test_similarity.py -v
    pytest tests/test_similarity.py -v --tb=short   ← plus lisible en cas d'erreur
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────
# Fixtures — données réutilisables entre les tests
# ─────────────────────────────────────────────
# Une "fixture" pytest = une fonction qui prépare des données
# et les injecte automatiquement dans les tests qui en ont besoin.

@pytest.fixture
def random_embedding():
    """Embedding factice de dimension 2048 (comme ResNet50), normalisé L2."""
    vec = np.random.rand(2048).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # normalisation L2
    return vec


@pytest.fixture
def random_embedding_small():
    """Embedding factice de dimension 4 — pour les tests rapides."""
    vec = np.random.rand(4).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


@pytest.fixture
def chroma_collection(tmp_path):
    """
    Collection ChromaDB temporaire créée dans un dossier temp.
    Détruite automatiquement après chaque test.

    tmp_path est une fixture pytest intégrée qui fournit
    un dossier temporaire unique par test.
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))
    collection = client.get_or_create_collection(
        name="test_wounds",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


@pytest.fixture
def populated_collection(chroma_collection):
    """
    Collection ChromaDB déjà remplie avec 10 embeddings factices.
    Utile pour tester la recherche sans avoir à insérer dans chaque test.
    """
    classes = ["burn", "venous_ulcer", "diabetic_wound", "pressure_ulcer", "surgical"]

    for i in range(10):
        vec = np.random.rand(4).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        chroma_collection.upsert(
            ids=[f"img_{i:03d}"],
            embeddings=[vec.tolist()],
            metadatas=[{
                "path": f"/data/raw/{classes[i % len(classes)]}/img_{i:03d}.jpg",
                "class": classes[i % len(classes)]
            }]
        )

    return chroma_collection


@pytest.fixture
def fake_image(tmp_path):
    """
    Crée une vraie image PNG minimaliste (3x3 pixels) sur disque.
    Nécessaire pour tester les fonctions qui lisent un fichier image.
    """
    from PIL import Image
    img_path = tmp_path / "test_wound.png"
    # Image RGB 3x3 pixels avec des valeurs aléatoires
    arr = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_path))
    return str(img_path)


# ─────────────────────────────────────────────
# Tests — normalisation L2
# ─────────────────────────────────────────────

class TestNormalization:
    """
    On teste que nos embeddings sont bien normalisés.
    Un vecteur normalisé L2 a une norme de 1.0 (± epsilon numérique).
    """

    def test_normalized_embedding_has_unit_norm(self, random_embedding):
        """La norme d'un embedding normalisé doit être ≈ 1.0."""
        norm = np.linalg.norm(random_embedding)
        assert abs(norm - 1.0) < 1e-5, f"Norme attendue ≈ 1.0, obtenue {norm}"

    def test_zero_vector_not_normalized(self):
        """Un vecteur nul ne doit pas provoquer une division par zéro."""
        from core.image_similarity import extract_embedding

        # On mocke extractor.predict() pour retourner un vecteur nul
        mock_extractor = MagicMock()
        mock_extractor.predict.return_value = np.zeros((1, 2048))

        # On mocke aussi preprocess_image pour ne pas avoir besoin d'une vraie image
        with patch("core.image_similarity.preprocess_image", return_value=np.zeros((1, 224, 224, 3))):
            embedding = extract_embedding(mock_extractor, "fake_path.jpg")

        # Le vecteur nul reste nul (pas de NaN, pas d'erreur)
        assert not np.any(np.isnan(embedding)), "Le vecteur nul ne doit pas produire de NaN"
        assert np.all(embedding == 0), "Le vecteur nul doit rester nul"

    def test_embedding_dimension_preserved_after_normalization(self, random_embedding):
        """La normalisation ne doit pas changer la dimension du vecteur."""
        assert random_embedding.shape == (2048,)


# ─────────────────────────────────────────────
# Tests — insertion dans ChromaDB
# ─────────────────────────────────────────────

class TestDatabaseInsertion:
    """Tests sur l'insertion des embeddings dans ChromaDB."""

    def test_insert_single_record(self, chroma_collection):
        """Un embedding inséré doit être retrouvable dans la collection."""
        from core.database import insert_embeddings

        vec = np.random.rand(4).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        records = [{
            "id": "test_img_001",
            "path": "/data/raw/burn/img_001.jpg",
            "class": "burn",
            "embedding": vec
        }]

        insert_embeddings(chroma_collection, records)

        # Vérification : la collection doit contenir 1 élément
        assert chroma_collection.count() == 1

    def test_insert_multiple_records(self, chroma_collection):
        """L'insertion de N enregistrements doit donner N éléments dans la base."""
        from core.database import insert_embeddings

        n = 15
        records = []
        for i in range(n):
            vec = np.random.rand(4).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            records.append({
                "id": f"img_{i:03d}",
                "path": f"/data/raw/burn/img_{i:03d}.jpg",
                "class": "burn",
                "embedding": vec
            })

        insert_embeddings(chroma_collection, records, batch_size=5)  # test du batching
        assert chroma_collection.count() == n

    def test_upsert_does_not_duplicate(self, chroma_collection):
        """
        Insérer deux fois le même ID ne doit pas créer de doublon.
        C'est le comportement upsert (update + insert).
        """
        from core.database import insert_embeddings

        vec = np.random.rand(4).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        record = [{
            "id": "same_id",
            "path": "/data/raw/burn/img.jpg",
            "class": "burn",
            "embedding": vec
        }]

        insert_embeddings(chroma_collection, record)
        insert_embeddings(chroma_collection, record)  # deuxième fois

        # Doit toujours être 1, pas 2
        assert chroma_collection.count() == 1


# ─────────────────────────────────────────────
# Tests — recherche par similarité
# ─────────────────────────────────────────────

class TestSimilaritySearch:
    """Tests sur la recherche des K plus proches voisins."""

    def test_search_returns_k_results(self, populated_collection, random_embedding_small):
        """La recherche doit retourner exactement k résultats."""
        from core.database import search_similar

        results = search_similar(populated_collection, random_embedding_small, k=3)
        assert len(results) == 3

    def test_search_results_have_required_keys(self, populated_collection, random_embedding_small):
        """Chaque résultat doit contenir les clés attendues par P4 (Streamlit)."""
        from core.database import search_similar

        results = search_similar(populated_collection, random_embedding_small, k=1)

        assert len(results) > 0
        result = results[0]

        assert "id"         in result, "Clé 'id' manquante"
        assert "path"       in result, "Clé 'path' manquante"
        assert "class"      in result, "Clé 'class' manquante"
        assert "similarity" in result, "Clé 'similarity' manquante"

    def test_similarity_score_between_0_and_1(self, populated_collection, random_embedding_small):
        """Les scores de similarité doivent être dans [0, 1]."""
        from core.database import search_similar

        results = search_similar(populated_collection, random_embedding_small, k=5)

        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0, (
                f"Score hors bornes : {r['similarity']}"
            )

    def test_identical_vector_has_max_similarity(self, chroma_collection):
        """
        Un vecteur identique à celui stocké doit avoir une similarité ≈ 1.0.
        C'est le test le plus important : vérifie que la métrique cosinus fonctionne.
        """
        from core.database import insert_embeddings, search_similar

        # On insère un vecteur connu
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # déjà normalisé
        insert_embeddings(chroma_collection, [{
            "id": "known_vec",
            "path": "/data/raw/burn/known.jpg",
            "class": "burn",
            "embedding": vec
        }])

        # On recherche avec le même vecteur
        results = search_similar(chroma_collection, vec, k=1)

        assert len(results) > 0
        assert results[0]["similarity"] > 0.99, (
            f"Similarité d'un vecteur identique devrait être ≈ 1.0, "
            f"obtenu {results[0]['similarity']}"
        )

    def test_search_excludes_query_image(self, populated_collection):
        """
        Si l'image requête est dans la base, elle ne doit pas apparaître
        dans les résultats (on ne veut pas se retourner soi-même).
        """
        from core.database import search_similar

        # Récupère l'embedding du premier élément de la base
        first = populated_collection.get(ids=["img_000"], include=["embeddings"])
        vec = np.array(first["embeddings"][0])

        results = search_similar(
            populated_collection,
            vec,
            k=3,
            exclude_id="img_000"  # on s'exclut
        )

        ids_returned = [r["id"] for r in results]
        assert "img_000" not in ids_returned, "L'image requête ne doit pas apparaître dans les résultats"

    def test_search_on_empty_collection_returns_empty(self, chroma_collection, random_embedding_small):
        """Une recherche sur une base vide doit retourner une liste vide (pas crasher)."""
        from core.database import search_similar

        # chroma_collection est vide (fixture de base, sans populated_collection)
        results = search_similar(chroma_collection, random_embedding_small, k=5)
        assert results == []


# ─────────────────────────────────────────────
# Tests — pipeline complet
# ─────────────────────────────────────────────

class TestSimilarityPipeline:
    """
    Tests d'intégration sur la classe SimilarityPipeline.
    On mocke le modèle CNN pour ne pas dépendre de P1.
    """

    def _make_mock_pipeline(self, tmp_path):
        """
        Crée un SimilarityPipeline avec un extracteur CNN mocké.
        Retourne le pipeline prêt à l'emploi.
        """
        from core.pipeline import SimilarityPipeline

        # Mock du modèle CNN : predict() retourne un vecteur aléatoire de dim 4
        mock_extractor = MagicMock()
        mock_extractor.predict.return_value = np.random.rand(1, 4).astype(np.float32)
        mock_extractor.input = MagicMock()

        with patch("core.pipeline.load_feature_extractor", return_value=mock_extractor):
            pipeline = SimilarityPipeline(
                model_path="fake_model.h5",
                chroma_dir=str(tmp_path / "chroma"),
                target_size=(224, 224)
            )

        return pipeline

    def test_pipeline_initializes_correctly(self, tmp_path):
        """Le pipeline doit s'initialiser sans erreur."""
        pipeline = self._make_mock_pipeline(tmp_path)
        assert pipeline is not None
        assert pipeline.collection is not None

    def test_is_indexed_returns_false_on_empty_base(self, tmp_path):
        """is_indexed() doit retourner False si la base est vide."""
        pipeline = self._make_mock_pipeline(tmp_path)
        assert pipeline.is_indexed() is False

    def test_search_raises_if_image_not_found(self, tmp_path):
        """search() doit lever FileNotFoundError si l'image n'existe pas."""
        pipeline = self._make_mock_pipeline(tmp_path)

        with pytest.raises(FileNotFoundError):
            pipeline.search("image_qui_nexiste_pas.jpg", k=3)

    def test_search_raises_if_base_empty(self, tmp_path, fake_image):
        """search() doit lever RuntimeError si la base n'est pas indexée."""
        pipeline = self._make_mock_pipeline(tmp_path)

        with pytest.raises(RuntimeError, match="base est vide"):
            pipeline.search(fake_image, k=3)

    def test_search_returns_results_after_indexing(self, tmp_path, fake_image):
        """
        Après indexation manuelle, search() doit retourner des résultats.
        On insère directement dans la collection pour simuler l'indexation.
        """
        pipeline = self._make_mock_pipeline(tmp_path)

        # Simulation : on insère 5 embeddings à la main
        for i in range(5):
            vec = np.random.rand(4).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            pipeline.collection.upsert(
                ids=[f"img_{i}"],
                embeddings=[vec.tolist()],
                metadatas=[{"path": f"/data/img_{i}.jpg", "class": "burn"}]
            )

        # Mock de preprocess_image pour éviter de charger une vraie image CNN
        with patch("core.pipeline.extract_embedding") as mock_extract:
            mock_extract.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results = pipeline.search(fake_image, k=3)

        assert len(results) == 3
        assert all("similarity" in r for r in results)

    def test_get_stats_returns_correct_structure(self, tmp_path):
        """get_stats() doit retourner un dict avec les bonnes clés."""
        pipeline = self._make_mock_pipeline(tmp_path)
        stats = pipeline.get_stats()

        assert "total_images" in stats
        assert "class_distribution" in stats
        assert isinstance(stats["total_images"], int)
        assert isinstance(stats["class_distribution"], dict)