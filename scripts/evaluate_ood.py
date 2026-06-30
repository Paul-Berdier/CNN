"""Teste le filtre OOD (core/autoencoder.py) avec des images hors-domaine et in-domain.

Exemple :
    python scripts/evaluate_ood.py
"""

import json
import os

import pandas as pd

from core.autoencoder import DEVICE, WoundAutoencoder, verify_image_domain

MODELS_DIR = "models"
OOD_DIR = "data/ood_samples"
RAW_DIR = "data/raw"
N_IN_DOMAIN_SAMPLES = 20


def list_in_domain_sample(raw_dir, n):
    """Échantillonne n images in-domain (plaies) parmi data/raw/<Classe>/*."""
    paths = []
    for entry in sorted(os.scandir(raw_dir), key=lambda e: e.name):
        if entry.is_dir():
            for fichier in sorted(os.listdir(entry.path))[:3]:  # quelques images par classe
                chemin = os.path.join(entry.path, fichier)
                if os.path.isfile(chemin):
                    paths.append(chemin)
    return paths[:n]


def main():
    config_path = os.path.join(MODELS_DIR, "ood_threshold.json")
    weights_path = os.path.join(MODELS_DIR, "wound_autoencoder.pth")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]

    model = WoundAutoencoder().to(DEVICE)
    import torch
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    print(f"Seuil OOD utilisé : {threshold:.5f}\n")

    rows = []

    print("=== Images hors-domaine (data/ood_samples/) ===")
    for fichier in sorted(os.listdir(OOD_DIR)):
        chemin = os.path.join(OOD_DIR, fichier)
        result = verify_image_domain(chemin, model, threshold)
        rows.append({"image": fichier, "attendu": "OOD", "is_valid": result["is_valid"],
                      "reconstruction_error": result["reconstruction_error"]})

    print("\n=== Échantillon d'images in-domain (data/raw/) ===")
    for chemin in list_in_domain_sample(RAW_DIR, N_IN_DOMAIN_SAMPLES):
        result = verify_image_domain(chemin, model, threshold)
        rows.append({"image": os.path.basename(chemin), "attendu": "in-domain", "is_valid": result["is_valid"],
                      "reconstruction_error": result["reconstruction_error"]})

    df = pd.DataFrame(rows)

    ood_rows = df[df["attendu"] == "OOD"]
    in_domain_rows = df[df["attendu"] == "in-domain"]

    ood_detected = (~ood_rows["is_valid"]).sum()  # is_valid=False = détecté comme OOD, correct
    in_domain_correct = in_domain_rows["is_valid"].sum()  # is_valid=True = accepté à raison

    print("\n=== Résumé ===")
    print(f"Images OOD correctement détectées comme hors-domaine : {ood_detected}/{len(ood_rows)}")
    print(f"Images in-domain correctement acceptées : {in_domain_correct}/{len(in_domain_rows)}")

    output_path = os.path.join(MODELS_DIR, "ood_evaluation.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDétail sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
