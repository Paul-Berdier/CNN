"""Chargement d'un modèle entraîné (checkpoint scripts/train.py) et prédiction sur une image."""

import json
import os

import torch
from PIL import Image

from core.autoencoder import AE_TRANSFORMS, WoundAutoencoder
from core.data_processing import preprocess_img
from core.model_utils import build_resnet50, build_vgg16, build_efficientnet_b0

BUILDERS = {
    "resnet50": build_resnet50,
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
}

# Ordre alphabétique standard du projet (cf. core.dataset.load_datasets), utilisé par
# défaut pour les checkpoints qui n'ont pas de class_to_idx sauvegardé.
DEFAULT_CLASSES = ["Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound"]


def load_model_raw(weights_path, arch, class_to_idx=None, device=None):
    """Charge un state_dict brut (torch.save(model.state_dict(), ...)) sans config.json associé.

    Cas des checkpoints produits hors de scripts/train.py (ex. models/Resnet_weights.pth,
    models/Efficientnet_weights.pth, issus de notebooks/Entrainnement_modèle*.ipynb) :
    pas de class_to_idx ni d'hyperparamètres sauvegardés. On suppose l'ordre alphabétique
    standard du projet (DEFAULT_CLASSES) sauf si class_to_idx est fourni explicitement,
    et le dropout par défaut de chaque builder (cf. core/model_utils.py), qui correspond
    à ce que ces notebooks utilisaient (pas de dropout pour ResNet50, 0.2 pour EfficientNetB0).

    Retourne (model, class_to_idx).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = class_to_idx or {c: i for i, c in enumerate(DEFAULT_CLASSES)}

    model = BUILDERS[arch](len(class_to_idx), freeze_base=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    return model, class_to_idx


def load_model(checkpoint_prefix, device=None):
    """Charge les poids + la config sauvegardés par scripts/train.py.

    checkpoint_prefix : ex. 'models/resnet50-lr0.001-frozen' (sans extension).
    Retourne (model, class_to_idx).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{checkpoint_prefix}_config.json", encoding="utf-8") as f:
        config = json.load(f)
    class_to_idx = config["class_to_idx"]
    args = config["args"]

    model = BUILDERS[args["arch"]](len(class_to_idx), freeze_base=False, dropout=args["dropout"])
    model.load_state_dict(torch.load(f"{checkpoint_prefix}.pt", map_location=device))
    model.to(device).eval()

    return model, class_to_idx


def load_ood_filter(models_dir="models", device=None):
    """Charge l'autoencoder OOD + son seuil calibré (scripts/train_autoencoder.py).

    Retourne (ood_model, threshold).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(models_dir, "ood_threshold.json"), encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    ood_model = WoundAutoencoder().to(device)
    weights_path = os.path.join(models_dir, "wound_autoencoder.pth")
    ood_model.load_state_dict(torch.load(weights_path, map_location=device))
    ood_model.eval()

    return ood_model, threshold


def predict_image(model, image, class_to_idx, device=None, top_k=3, ood_model=None, ood_threshold=None):
    """Prédit la classe d'une image (PIL.Image ou np.ndarray).

    Si ood_model/ood_threshold sont fournis (cf. load_ood_filter), l'image est d'abord
    passée par le filtre hors-domaine (Exercice 1.7) : si son erreur de reconstruction
    dépasse le seuil, la classification n'est pas effectuée et la fonction retourne
    ("hors_domaine", erreur_reconstruction, []).

    Retourne (classe_predite, confiance, top_k) où top_k est une liste
    [(classe, probabilité), ...] triée par probabilité décroissante.
    """
    device = device or next(model.parameters()).device

    if ood_model is not None and ood_threshold is not None:
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        ae_tensor = AE_TRANSFORMS(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = ood_model(ae_tensor)
            reconstruction_error = torch.mean((recon - ae_tensor) ** 2).item()
        if reconstruction_error > ood_threshold:
            return "hors_domaine", reconstruction_error, []

    idx_to_class = {i: c for c, i in class_to_idx.items()}

    tensor = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    top_probs, top_idx = probs.topk(top_k)
    top = [(idx_to_class[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]
    predicted_class, confidence = top[0]

    return predicted_class, confidence, top
