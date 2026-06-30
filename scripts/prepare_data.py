"""Pipeline de préparation des données : lecture du brut, augmentation puis prétraitement."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from core.data_processing import augmented_img, preprocess_img

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
AUGMENTED_IMG_DIR = os.path.join(PROCESSED_DIR, "data_augmented")


def load_raw_dataset(raw_dir):
    """Construit le DataFrame brut à partir des images classées par dossier (data/raw/<Classe>/*.jpg).

    Reprend la logique de notebooks/Exploration_des_données.ipynb (recup_fichier),
    pour ne plus dépendre d'un data.pkl pré-généré et absent du repo.
    """
    records = []
    for entry in sorted(os.scandir(raw_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        class_name = entry.name
        for fichier in sorted(os.listdir(entry.path)):
            chemin_complet = os.path.join(entry.path, fichier)
            if not os.path.isfile(chemin_complet):
                continue
            with Image.open(chemin_complet) as img:
                img_array = np.array(img.convert("RGB"))
            h, w = img_array.shape[:2]
            records.append({
                "Class": class_name,
                "Nom_img": fichier,
                "img": img_array,
                "Hauteur": h,
                "Largeur": w,
            })

    return pd.DataFrame(records)


def data_augmentation(df, nb_row_by_class):
    os.makedirs(AUGMENTED_IMG_DIR, exist_ok=True)

    df_augmented = []
    for class_name in df["Class"].unique().tolist():
        sub_df = df[df["Class"] == class_name]
        nb_row = sub_df.shape[0]
        if nb_row >= nb_row_by_class:
            continue

        class_dir = os.path.join(AUGMENTED_IMG_DIR, f"data_augmented_{class_name}")
        os.makedirs(class_dir, exist_ok=True)

        nb_generate = nb_row_by_class - nb_row
        for e in range(nb_generate):
            img_random = np.random.choice(sub_df.index, size=1)[0]
            img_augmented = augmented_img(df["img"][img_random])
            df_augmented.append({
                "Class": class_name,
                "Nom_img": f"{class_name}_data_augmented_{e}",
                "img": img_augmented,
                "Hauteur": 640,
                "Largeur": 640,
            })
            img_augmented.save(os.path.join(class_dir, f"augmented_{class_name}_{e}.jpeg"))

    return pd.DataFrame(df_augmented)


def save_distribution_comparison(counts_before, counts_after, output_path):
    """Sauvegarde un bar chart comparant la répartition des classes avant/après augmentation."""
    classes = sorted(set(counts_before.index) | set(counts_after.index))
    before = [counts_before.get(c, 0) for c in classes]
    after = [counts_after.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, before, width, label="Avant augmentation", color="steelblue")
    ax.bar(x + width / 2, after, width, label="Après augmentation", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Répartition des classes avant/après augmentation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_raw_dataset(RAW_DIR)
    counts_before = df["Class"].value_counts()

    df_augmented = data_augmentation(df, nb_row_by_class=122)
    df_raw = pd.concat([df, df_augmented], ignore_index=True)
    counts_after = df_raw["Class"].value_counts()

    print(df.shape)
    print(df_augmented.shape)
    print(df_raw.shape)

    print("\nRépartition avant augmentation :")
    print(counts_before)
    print("\nRépartition après augmentation :")
    print(counts_after)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    comparison_path = os.path.join(PROCESSED_DIR, "class_distribution_before_after.png")
    save_distribution_comparison(counts_before, counts_after, comparison_path)
    print(f"\nGraphique de comparaison sauvegardé : {comparison_path}")

    df_raw.to_pickle(os.path.join(PROCESSED_DIR, "data_with_augmentation.pkl"))

    df_raw["img_preprocess"] = df_raw["img"].apply(preprocess_img)
    df_preprocess = df_raw[["Class", "img_preprocess"]]
    print(df_preprocess.describe())

    df_preprocess.to_pickle(os.path.join(PROCESSED_DIR, "data_preprocess.pkl"))


if __name__ == "__main__":
    main()
