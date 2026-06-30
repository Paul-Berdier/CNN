"""Chargement d'un modèle entraîné (checkpoint scripts/train.py) et prédiction sur une image."""

import json

import torch

from core.data_processing import preprocess_img
from core.model_utils import build_resnet50, build_vgg16, build_efficientnet_b0

BUILDERS = {
    "resnet50": build_resnet50,
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
}


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


def predict_image(model, image, class_to_idx, device=None, top_k=3):
    """Prédit la classe d'une image (PIL.Image ou np.ndarray).

    Retourne (classe_predite, confiance, top_k) où top_k est une liste
    [(classe, probabilité), ...] triée par probabilité décroissante.
    """
    device = device or next(model.parameters()).device
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    tensor = preprocess_img(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    top_probs, top_idx = probs.topk(top_k)
    top = [(idx_to_class[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]
    predicted_class, confidence = top[0]

    return predicted_class, confidence, top
