"""
image_similarity.py
-------------------
Extraction des embeddings visuels à partir du CNN entraîné par P1.

Principe (analogie bio) :
Un CNN, c'est comme un système nerveux visuel — les premières couches détectent
des bords, les couches du milieu des textures, et la dernière couche classifie.
On coupe juste avant la classification pour récupérer la "représentation mentale"
de l'image : c'est l'embedding.

Dépendance : le modèle sauvegardé par P1 (fichier .h5 ou .keras)
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ─────────────────────────────────────────────
# 1. Chargement du modèle et création de l'extracteur
# ─────────────────────────────────────────────

def load_feature_extractor(model_path: str, layer_name: str = None) -> Model:
    """
    Charge le modèle CNN de P1 et retourne une version tronquée
    qui produit des embeddings au lieu de classifications.

    Args:
        model_path  : chemin vers le fichier .h5 ou .keras de P1
        layer_name  : nom de la couche de sortie souhaitée.
                      Si None, on prend l'avant-dernière couche
                      (avant la couche Dense de classification).

    Returns:
        extractor : un Model Keras dont la sortie est un vecteur d'embedding
    """
    # Chargement du modèle complet de P1
    full_model = load_model(model_path)

    # Si aucune couche précisée, on prend l'avant-dernière
    # (la dernière est en général un Dense(nb_classes) + softmax)
    if layer_name is None:
        # On cherche la dernière couche qui n'est pas Dense/classification
        # Typiquement : GlobalAveragePooling2D ou Flatten pour ResNet/EfficientNet
        for layer in reversed(full_model.layers):
            if "dense" not in layer.name.lower() and "softmax" not in layer.name.lower():
                layer_name = layer.name
                break

    print(f"Couche de sortie pour les embeddings : {layer_name}")

    # Création du modèle tronqué : même entrée, sortie = couche choisie
    extractor = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(layer_name).output
    )

    return extractor


# ─────────────────────────────────────────────
# 2. Prétraitement d'une image unique
# ─────────────────────────────────────────────

def preprocess_image(img_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Charge et prépare une image pour l'inférence.

    Args:
        img_path    : chemin vers l'image
        target_size : taille d'entrée du CNN (doit correspondre à ce que P1 a utilisé)

    Returns:
        img_array : tableau numpy de shape (1, H, W, 3), normalisé
    """
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)           # shape (H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)       # shape (1, H, W, 3)
    img_array = preprocess_input(img_array)             # normalisation ResNet
    return img_array


# ─────────────────────────────────────────────
# 3. Extraction d'un embedding pour une image
# ─────────────────────────────────────────────

def extract_embedding(extractor: Model, img_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Retourne l'embedding normalisé (L2) d'une image.

    La normalisation L2 est importante : elle permet d'utiliser
    la similarité cosinus, qui est plus fiable que la distance
    euclidienne pour des vecteurs de haute dimension.

    Args:
        extractor   : modèle tronqué produit par load_feature_extractor()
        img_path    : chemin vers l'image
        target_size : taille d'entrée attendue

    Returns:
        embedding : vecteur numpy 1D normalisé
    """
    img_array = preprocess_image(img_path, target_size)
    embedding = extractor.predict(img_array, verbose=0)  # shape (1, dim)
    embedding = embedding.flatten()                       # shape (dim,)

    # Normalisation L2 : on divise par la norme du vecteur
    # → tous les vecteurs ont une longueur de 1
    # → la similarité cosinus devient juste un produit scalaire
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ─────────────────────────────────────────────
# 4. Extraction en batch sur tout le dataset
# ─────────────────────────────────────────────

def extract_all_embeddings(
    extractor: Model,
    dataset_dir: str,
    target_size: tuple = (224, 224)
) -> list[dict]:
    """
    Parcourt tout le dataset et extrait les embeddings de chaque image.

    Structure attendue du dataset (celle du projet) :
        dataset_dir/
            classe_1/
                img_001.jpg
                img_002.jpg
            classe_2/
                ...

    Args:
        extractor   : modèle tronqué
        dataset_dir : chemin vers le dossier racine du dataset
        target_size : taille d'entrée du CNN

    Returns:
        records : liste de dicts avec les clés :
                  - "id"        : identifiant unique (chemin relatif sans extension)
                  - "path"      : chemin absolu vers l'image
                  - "class"     : nom de la classe (nom du sous-dossier)
                  - "embedding" : vecteur numpy normalisé
    """
    dataset_path = Path(dataset_dir)
    records = []

    # Extensions d'images supportées
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # On parcourt chaque sous-dossier (= chaque classe)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        img_files = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]

        print(f"  Classe '{class_name}' : {len(img_files)} images")

        for img_file in img_files:
            try:
                embedding = extract_embedding(extractor, str(img_file), target_size)

                records.append({
                    "id": str(img_file.relative_to(dataset_path).with_suffix("")),
                    "path": str(img_file.absolute()),
                    "class": class_name,
                    "embedding": embedding
                })

            except Exception as e:
                # On log l'erreur mais on continue — une image corrompue
                # ne doit pas bloquer tout le batch
                print(f"  [ERREUR] {img_file.name} : {e}")

    print(f"\nTotal : {len(records)} embeddings extraits")
    return records