"""
image_similarity.py
-------------------
Extraction des embeddings visuels avec PyTorch + ResNet50 pré-entraîné ImageNet.

Deux différences avec le notebook solution :
1. PyTorch au lieu de TensorFlow/Keras
2. ResNet50 standalone (pas le modèle fine-tuné de P1) — avantage :
   P3 peut travailler sans attendre P1. Les embeddings ImageNet sont
   suffisamment riches pour la similarité visuelle même sur des images médicales.

Pourquoi ResNet50 ImageNet marche quand même sur des plaies ?
   (analogie bio) : c'est comme un naturaliste formé sur toutes les espèces du monde.
   Même s'il n'a jamais vu CETTE espèce précise, il reconnaît les textures,
   les formes, les patterns — et peut dire "ça ressemble à ça".
   Les couches profondes de ResNet50 détectent des features génériques
   (bords, textures, gradients de couleur) qui sont utiles dans n'importe quel domaine.

Dépendances : torch, torchvision, Pillow
"""

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────

IMAGE_SIZE = 224  # Taille d'entrée standard de ResNet50

# Normalisation ImageNet — valeurs standards pour tous les modèles pré-entraînés torchvision
# Ces valeurs (mean/std par canal RGB) correspondent à la distribution d'ImageNet
IMAGENET_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Device : GPU si disponible, CPU sinon
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 1. Définition du FeatureExtractor
# ─────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    ResNet50 pré-entraîné dont on a retiré la dernière couche (classification).

    Architecture ResNet50 originale :
        Conv → ... → GlobalAvgPool → [FC(2048 → 1000)]  ← on retire ça
                                   ↑
                              on sort ici → vecteur 2048 dims

    Directement repris du notebook solution, sans modification.
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Chargement de ResNet50 avec poids ImageNet
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # On coupe la dernière couche FC (classification 1000 classes ImageNet)
        # list(base_model.children())[:-1] = tout sauf le dernier module
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # shape (batch, 2048)
        return x


def load_feature_extractor() -> FeatureExtractor:
    """
    Instancie et retourne le FeatureExtractor prêt à l'emploi.

    Pas besoin de chemin de modèle — les poids ImageNet sont
    téléchargés automatiquement par torchvision au premier appel.

    Returns:
        extractor : FeatureExtractor en mode eval sur le bon device
    """
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()  # désactive dropout et batch norm stochastique
    print(f"FeatureExtractor chargé sur {DEVICE}")
    return extractor


# ─────────────────────────────────────────────
# 2. Prétraitement d'une image
# ─────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Charge et prépare une image pour l'inférence PyTorch.

    Args:
        img_path : chemin vers l'image (jpg, png, bmp...)

    Returns:
        tensor : shape (1, 3, 224, 224), normalisé ImageNet, sur DEVICE
    """
    image = Image.open(img_path).convert("RGB")  # force 3 canaux (évite les PNG RGBA)
    tensor = IMAGENET_TRANSFORMS(image)           # shape (3, 224, 224)
    tensor = tensor.unsqueeze(0).to(DEVICE)       # shape (1, 3, 224, 224)
    return tensor


# ─────────────────────────────────────────────
# 3. Extraction d'un embedding pour une image
# ─────────────────────────────────────────────

def extract_embedding(extractor: FeatureExtractor, img_path: str) -> np.ndarray:
    """
    Retourne l'embedding normalisé L2 d'une image.

    Pourquoi normaliser en L2 ?
        La similarité cosinus mesure l'angle entre deux vecteurs.
        Si les vecteurs ont des longueurs très différentes (une image très
        lumineuse vs une sombre), les produits scalaires internes de ChromaDB
        deviennent numériquement instables.
        En ramenant tous les vecteurs à longueur 1, on garantit que la
        similarité reflète uniquement le contenu visuel, pas l'intensité
        globale d'activation — c'est plus robuste et plus précis.

    Args:
        extractor : FeatureExtractor chargé
        img_path  : chemin vers l'image

    Returns:
        embedding : vecteur numpy 1D de dim 2048, normalisé L2
    """
    tensor = preprocess_image(img_path)

    with torch.no_grad():  # pas de calcul de gradient → plus rapide, moins de mémoire
        embedding = extractor(tensor)             # shape (1, 2048)

    embedding = embedding.cpu().numpy().flatten()  # shape (2048,) en numpy

    # Normalisation L2 : on divise par la norme du vecteur
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ─────────────────────────────────────────────
# 4. Extraction en batch sur tout le dataset
# ─────────────────────────────────────────────

def extract_all_embeddings(
    extractor: FeatureExtractor,
    dataset_dir: str
) -> list[dict]:
    """
    Parcourt tout le dataset et extrait les embeddings de chaque image.

    Structure attendue du dataset :
        dataset_dir/
            classe_1/
                img_001.jpg
            classe_2/
                ...

    Args:
        extractor   : FeatureExtractor chargé
        dataset_dir : chemin vers data/raw/

    Returns:
        records : liste de dicts avec les clés :
                  "id", "path", "class", "embedding"
    """
    dataset_path = Path(dataset_dir)
    records = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        img_files = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]

        print(f"  Classe '{class_name}' : {len(img_files)} images")

        for img_file in img_files:
            try:
                embedding = extract_embedding(extractor, str(img_file))

                records.append({
                    # ID cohérent avec le notebook : "classe_nomfichier"
                    "id":        f"drive_{class_name}_{img_file.name}",
                    "path":      str(img_file.absolute()),
                    "class":     class_name,
                    "embedding": embedding
                })

            except Exception as e:
                print(f"  [ERREUR] {img_file.name} : {e}")

    print(f"\nTotal : {len(records)} embeddings extraits")
    return records