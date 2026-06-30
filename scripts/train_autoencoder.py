"""
train_autoencoder.py
--------------------
Script d'entraînement de l'autoencoder OOD.

À lancer UNE SEULE FOIS depuis la racine du projet :
    python scripts/train_autoencoder.py

    Options :
    --csv        chemin vers data.csv          (défaut: data/data.csv)
    --dataset    chemin vers data/raw/         (défaut: data/raw)
    --epochs     nombre d'epochs               (défaut: 10)
    --batch      taille des batchs             (défaut: 32)
    --lr         learning rate                 (défaut: 0.001)
    --percentile percentile pour seuil OOD     (défaut: 95)
    --augmented  utiliser le dataset augmenté  (flag, pas de valeur)

Produit :
    models/wound_autoencoder.pth   ← poids du modèle
    models/ood_threshold.json      ← seuil OOD calibré

Note : attendre le dataset augmenté de P1 avant de lancer en production.
En attendant, ce script tourne sur data/raw/ pour tester que tout fonctionne.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from core.autoencoder import (
    WoundAutoencoder,
    build_dataloader,
    train_autoencoder,
    compute_ood_threshold,
    DEVICE,
)


# ─────────────────────────────────────────────
# 1. Chargement et préparation du DataFrame
# ─────────────────────────────────────────────

def prepare_dataframe(csv_path: str, dataset_dir: str) -> pd.DataFrame:
    """
    Charge le CSV et construit la colonne 'path' vers les images réelles.

    Args:
        csv_path    : chemin vers data.csv
        dataset_dir : chemin vers data/raw/ (ou data/augmented/ si P1 a fini)

    Returns:
        DataFrame filtré avec colonne 'path' valide
    """
    df = pd.read_csv(csv_path)

    # Construction du chemin complet : dataset_dir/Class/Name_img
    df["path"] = df.apply(
        lambda row: os.path.join(dataset_dir, row["Class"], row["Name_img"]),
        axis=1
    )

    # Filtrage des images manquantes sur disque
    avant = len(df)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    apres = len(df)

    if avant != apres:
        print(f"[ATTENTION] {avant - apres} images manquantes ignorées")

    print(f"Dataset prêt : {apres} images")
    print("\nDistribution des classes :")
    print(df["Class"].value_counts().to_string())

    return df


# ─────────────────────────────────────────────
# 2. Courbe de loss
# ─────────────────────────────────────────────

def plot_loss_curve(loss_history: list[float], save_path: str = "models/autoencoder_loss.png"):
    """
    Trace et sauvegarde la courbe de loss d'entraînement.
    Utile pour vérifier que le modèle converge bien.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.title("Loss d'entraînement — Autoencoder OOD")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Courbe de loss sauvegardée → {save_path}")


# ─────────────────────────────────────────────
# 3. Script principal
# ─────────────────────────────────────────────

def main(args):

    print(f"Device utilisé : {DEVICE}")
    print("─" * 50)

    # ── Préparation des données ──────────────
    df = prepare_dataframe(args.csv, args.dataset)

    # Split train / validation (80% / 20%)
    # stratify=Class pour garder la distribution dans les deux splits
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Class"]
    )
    print(f"\nSplit : {len(train_df)} train / {len(val_df)} validation")

    # ── DataLoaders ──────────────────────────
    train_loader = build_dataloader(train_df, batch_size=args.batch, shuffle=True)
    val_loader   = build_dataloader(val_df,   batch_size=args.batch, shuffle=False)

    # ── Modèle ───────────────────────────────
    model = WoundAutoencoder().to(DEVICE)
    print(f"\nArchitecture autoencoder prête sur {DEVICE}")

    # ── Entraînement ─────────────────────────
    print("\n" + "─" * 50)
    loss_history = train_autoencoder(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_path="models/wound_autoencoder.pth"
    )

    # Courbe de loss
    os.makedirs("models", exist_ok=True)
    plot_loss_curve(loss_history)

    # ── Seuil OOD ────────────────────────────
    print("\n" + "─" * 50)
    print("Calibration du seuil OOD sur le jeu de validation...")
    threshold = compute_ood_threshold(
        model=model,
        val_loader=val_loader,
        percentile=args.percentile
    )

    # Sauvegarde du seuil dans un fichier JSON
    # → utilisé par Streamlit (P4) et verify_image_domain()
    threshold_data = {
        "threshold":  threshold,
        "percentile": args.percentile,
        "n_val":      len(val_df),
        "dataset":    args.dataset
    }
    with open("models/ood_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=4)
    print(f"Seuil OOD sauvegardé → models/ood_threshold.json")

    # ── Résumé final ─────────────────────────
    print("\n" + "═" * 50)
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"  Modèle      : models/wound_autoencoder.pth")
    print(f"  Seuil OOD   : {threshold:.5f} (percentile {args.percentile})")
    print(f"  Loss finale : {loss_history[-1]:.5f}")
    print("═" * 50)


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de l'autoencoder OOD")

    parser.add_argument("--csv",        default="data/data.csv",  help="Chemin vers data.csv")
    parser.add_argument("--dataset",    default="data/raw",       help="Chemin vers le dossier images")
    parser.add_argument("--epochs",     type=int,   default=10,   help="Nombre d'epochs")
    parser.add_argument("--batch",      type=int,   default=32,   help="Taille des batchs")
    parser.add_argument("--lr",         type=float, default=0.001,help="Learning rate")
    parser.add_argument("--percentile", type=int,   default=95,   help="Percentile seuil OOD")

    args = parser.parse_args()
    main(args)