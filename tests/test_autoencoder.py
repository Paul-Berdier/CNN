"""
test_autoencoder.py
-------------------
Tests unitaires pour core/autoencoder.py.

Stratégie : on ne teste PAS la qualité de reconstruction
(ça dépend des données et des epochs — ça appartient à l'évaluation).
On teste uniquement la logique du code :
    - architecture (dimensions input/output)
    - dataset PyTorch (chargement, format)
    - boucle d'entraînement (loss diminue, fichier sauvegardé)
    - seuil OOD (type, valeur cohérente)
    - inférence (format de sortie, comportement OOD)

Lancement :
    pytest tests/test_autoencoder.py -v
"""

import pytest
import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from PIL import Image
from unittest.mock import patch


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def model():
    """Instance fraîche de WoundAutoencoder sur CPU."""
    from core.autoencoder import WoundAutoencoder
    return WoundAutoencoder().to("cpu")


@pytest.fixture
def fake_image_dir(tmp_path):
    """
    Crée un mini dataset factice :
        tmp/
            Burns/
                img_001.png  (3x224x224 pixels aléatoires)
                img_002.png
            Abrasions/
                img_003.png
    """
    classes = {"Burns": 2, "Abrasions": 1}

    for class_name, n_imgs in classes.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(n_imgs):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(str(class_dir / f"img_{i:03d}.png"))

    return str(tmp_path)


@pytest.fixture
def fake_dataframe(fake_image_dir):
    """
    DataFrame avec colonne 'path' pointant vers les images factices.
    Format attendu par build_dataloader().
    """
    rows = []
    for class_dir in Path(fake_image_dir).iterdir():
        for img_file in class_dir.iterdir():
            rows.append({
                "Class":    class_dir.name,
                "Name_img": img_file.name,
                "path":     str(img_file)
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_dataloader(fake_dataframe):
    """DataLoader minimaliste pour les tests d'entraînement."""
    from core.autoencoder import build_dataloader
    return build_dataloader(fake_dataframe, batch_size=2, shuffle=False)


@pytest.fixture
def trained_model(model, tiny_dataloader, tmp_path):
    """
    Autoencoder entraîné sur 2 epochs avec le mini dataset.
    Utilisé pour les tests qui nécessitent un modèle entraîné.
    """
    from core.autoencoder import train_autoencoder
    save_path = str(tmp_path / "models" / "test_autoencoder.pth")
    train_autoencoder(model, tiny_dataloader, epochs=2, lr=0.001, save_path=save_path)
    return model


# ─────────────────────────────────────────────
# Tests — Architecture
# ─────────────────────────────────────────────

class TestArchitecture:
    """
    On vérifie que l'autoencoder produit des tenseurs
    de la bonne forme à chaque étape.
    """

    def test_output_shape_matches_input(self, model):
        """
        La sortie du décodeur doit avoir la même forme que l'entrée.
        Si ce n'est pas le cas, MSELoss(output, input) crasherait.
        """
        x = torch.zeros(1, 3, 224, 224)  # batch=1, RGB, 224x224
        with torch.no_grad():
            output = model(x)
        assert output.shape == x.shape, (
            f"Shape attendue {x.shape}, obtenue {output.shape}"
        )

    def test_output_shape_with_batch(self, model):
        """Vérifie que le batching fonctionne (batch=4)."""
        x = torch.zeros(4, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 3, 224, 224)

    def test_output_values_between_0_and_1(self, model):
        """
        La dernière couche est Sigmoid → pixels dans [0, 1].
        Important pour la cohérence avec MSELoss sur pixels non normalisés.
        """
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.min() >= 0.0, f"Valeur min négative : {output.min()}"
        assert output.max() <= 1.0, f"Valeur max > 1 : {output.max()}"

    def test_encoder_output_shape(self, model):
        """
        L'espace latent (sortie encodeur) doit être (batch, 64, 56, 56).
        Si P2 change l'archi, ce test détecte les régressions.
        """
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            latent = model.encoder(x)
        assert latent.shape == (1, 64, 56, 56), (
            f"Shape latent attendue (1, 64, 56, 56), obtenue {latent.shape}"
        )

    def test_model_has_encoder_and_decoder(self, model):
        """L'autoencoder doit avoir les deux attributs encoder et decoder."""
        assert hasattr(model, "encoder"), "Attribut 'encoder' manquant"
        assert hasattr(model, "decoder"), "Attribut 'decoder' manquant"


# ─────────────────────────────────────────────
# Tests — Dataset et DataLoader
# ─────────────────────────────────────────────

class TestDataset:
    """Tests sur WoundDataset et build_dataloader()."""

    def test_dataset_length_matches_dataframe(self, fake_dataframe):
        """Le dataset doit avoir autant d'items que de lignes dans le DataFrame."""
        from core.autoencoder import WoundDataset, AE_TRANSFORMS
        dataset = WoundDataset(fake_dataframe, transform=AE_TRANSFORMS)
        assert len(dataset) == len(fake_dataframe)

    def test_dataset_item_shape(self, fake_dataframe):
        """
        Chaque item doit être un tuple (image, image) de shape (3, 224, 224).
        L'entrée et la cible sont identiques pour un autoencoder.
        """
        from core.autoencoder import WoundDataset, AE_TRANSFORMS
        dataset = WoundDataset(fake_dataframe, transform=AE_TRANSFORMS)
        img, target = dataset[0]

        assert img.shape    == (3, 224, 224), f"Shape image : {img.shape}"
        assert target.shape == (3, 224, 224), f"Shape target : {target.shape}"

    def test_input_equals_target(self, fake_dataframe):
        """
        Pour un autoencoder, entrée et cible sont la même image.
        Si ce n'est pas le cas, le modèle n'apprend pas à se reconstruire.
        """
        from core.autoencoder import WoundDataset, AE_TRANSFORMS
        dataset = WoundDataset(fake_dataframe, transform=AE_TRANSFORMS)
        img, target = dataset[0]
        assert torch.equal(img, target), "Entrée et cible doivent être identiques"

    def test_image_values_in_range(self, fake_dataframe):
        """
        AE_TRANSFORMS utilise ToTensor() sans normalisation ImageNet
        → pixels dans [0, 1].
        """
        from core.autoencoder import WoundDataset, AE_TRANSFORMS
        dataset = WoundDataset(fake_dataframe, transform=AE_TRANSFORMS)
        img, _ = dataset[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_dataloader_batch_shape(self, tiny_dataloader):
        """Un batch du DataLoader doit avoir la bonne forme."""
        batch_imgs, batch_targets = next(iter(tiny_dataloader))
        assert batch_imgs.shape[1:]    == (3, 224, 224)
        assert batch_targets.shape[1:] == (3, 224, 224)
        assert batch_imgs.shape[0]     == batch_targets.shape[0]  # même batch size


# ─────────────────────────────────────────────
# Tests — Entraînement
# ─────────────────────────────────────────────

class TestTraining:
    """Tests sur la boucle d'entraînement."""

    def test_training_returns_loss_history(self, model, tiny_dataloader, tmp_path):
        """train_autoencoder() doit retourner une liste de pertes par epoch."""
        from core.autoencoder import train_autoencoder
        save_path    = str(tmp_path / "models" / "ae.pth")
        loss_history = train_autoencoder(model, tiny_dataloader, epochs=2, save_path=save_path)

        assert isinstance(loss_history, list)
        assert len(loss_history) == 2  # autant d'éléments que d'epochs

    def test_loss_is_positive(self, model, tiny_dataloader, tmp_path):
        """La MSE loss doit être positive (c'est une erreur quadratique)."""
        from core.autoencoder import train_autoencoder
        save_path    = str(tmp_path / "models" / "ae.pth")
        loss_history = train_autoencoder(model, tiny_dataloader, epochs=2, save_path=save_path)

        for loss in loss_history:
            assert loss > 0, f"Loss négative ou nulle : {loss}"

    def test_loss_decreases_over_epochs(self, model, tiny_dataloader, tmp_path):
        """
        Sur plusieurs epochs, la loss doit globalement diminuer.
        Test sur 5 epochs pour avoir une tendance claire.

        Note : pas garanti à 100% sur 2 images (trop peu de données),
        mais sur 5 epochs avec lr=0.001 c'est quasi-certain.
        """
        from core.autoencoder import train_autoencoder
        save_path    = str(tmp_path / "models" / "ae.pth")
        loss_history = train_autoencoder(model, tiny_dataloader, epochs=5, save_path=save_path)

        # La loss finale doit être inférieure à la loss initiale
        assert loss_history[-1] < loss_history[0], (
            f"La loss n'a pas diminué : {loss_history[0]:.5f} → {loss_history[-1]:.5f}"
        )

    def test_model_saved_after_training(self, model, tiny_dataloader, tmp_path):
        """Le fichier .pth doit exister après l'entraînement."""
        from core.autoencoder import train_autoencoder
        save_path = str(tmp_path / "models" / "ae_test.pth")
        train_autoencoder(model, tiny_dataloader, epochs=2, save_path=save_path)

        assert os.path.exists(save_path), f"Fichier modèle non trouvé : {save_path}"

    def test_saved_model_loadable(self, model, tiny_dataloader, tmp_path):
        from core.autoencoder import train_autoencoder, load_autoencoder
        save_path = str(tmp_path / "models" / "ae_load_test.pth")
        train_autoencoder(model, tiny_dataloader, epochs=2, save_path=save_path)

        loaded_model = load_autoencoder(save_path)
        assert loaded_model is not None

        # Les deux modèles doivent être en eval() pour que BatchNorm
        # se comporte de façon identique
        model.eval()
        loaded_model.eval()

        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out_original = model(x)
            out_loaded   = loaded_model(x)

        assert torch.allclose(out_original, out_loaded, atol=1e-5), (
            "Le modèle rechargé produit des outputs différents"
        )


# ─────────────────────────────────────────────
# Tests — Seuil OOD
# ─────────────────────────────────────────────

class TestOODThreshold:
    """Tests sur le calcul du seuil OOD."""

    def test_threshold_is_float(self, trained_model, tiny_dataloader):
        """Le seuil doit être un float."""
        from core.autoencoder import compute_ood_threshold
        threshold = compute_ood_threshold(trained_model, tiny_dataloader, percentile=95)
        assert isinstance(threshold, float)

    def test_threshold_is_positive(self, trained_model, tiny_dataloader):
        """Le seuil MSE doit être positif."""
        from core.autoencoder import compute_ood_threshold
        threshold = compute_ood_threshold(trained_model, tiny_dataloader, percentile=95)
        assert threshold > 0, f"Seuil négatif ou nul : {threshold}"

    def test_higher_percentile_gives_higher_threshold(self, trained_model, tiny_dataloader):
        """
        Un percentile plus élevé doit donner un seuil plus élevé.
        Percentile 99 > Percentile 50 — si ce n'est pas le cas,
        le calcul de percentile est inversé.
        """
        from core.autoencoder import compute_ood_threshold
        threshold_50 = compute_ood_threshold(trained_model, tiny_dataloader, percentile=50)
        threshold_99 = compute_ood_threshold(trained_model, tiny_dataloader, percentile=99)
        assert threshold_99 >= threshold_50, (
            f"Percentile 99 ({threshold_99:.5f}) devrait être ≥ percentile 50 ({threshold_50:.5f})"
        )


# ─────────────────────────────────────────────
# Tests — Inférence OOD
# ─────────────────────────────────────────────

class TestOODInference:
    """Tests sur verify_image_domain()."""

    def test_returns_dict_with_required_keys(self, trained_model, tmp_path):
        """Le résultat doit contenir les clés attendues par Streamlit (P4)."""
        from core.autoencoder import verify_image_domain

        # Création d'une image de test
        img_path = str(tmp_path / "test.png")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = verify_image_domain(img_path, trained_model, threshold=0.1)

        assert "is_valid"             in result
        assert "reconstruction_error" in result
        assert "threshold"            in result
        assert "verdict"              in result

    def test_reconstruction_error_is_positive(self, trained_model, tmp_path):
        """L'erreur de reconstruction doit être un float positif."""
        from core.autoencoder import verify_image_domain

        img_path = str(tmp_path / "test.png")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = verify_image_domain(img_path, trained_model, threshold=0.1)
        assert result["reconstruction_error"] > 0

    def test_high_threshold_accepts_image(self, trained_model, tmp_path):
        """
        Avec un seuil très élevé, toute image doit être acceptée.
        Seuil = 9999 → aucune image ne peut avoir une erreur aussi haute.
        """
        from core.autoencoder import verify_image_domain

        img_path = str(tmp_path / "test.png")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = verify_image_domain(img_path, trained_model, threshold=9999.0)
        assert result["is_valid"] is True

    def test_zero_threshold_rejects_image(self, trained_model, tmp_path):
        """
        Avec un seuil de 0, toute image doit être rejetée.
        L'erreur de reconstruction est toujours > 0.
        """
        from core.autoencoder import verify_image_domain

        img_path = str(tmp_path / "test.png")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = verify_image_domain(img_path, trained_model, threshold=0.0)
        assert result["is_valid"] is False

    def test_verdict_is_string(self, trained_model, tmp_path):
        """Le verdict doit être une chaîne lisible (pour Streamlit)."""
        from core.autoencoder import verify_image_domain

        img_path = str(tmp_path / "test.png")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = verify_image_domain(img_path, trained_model, threshold=0.1)
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0