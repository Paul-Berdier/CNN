"""Pipeline de préparation des données : augmentation puis prétraitement."""

import os

import numpy as np
import pandas as pd

from core.data_processing import augmented_img, preprocess_img

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
AUGMENTED_IMG_DIR = os.path.join(PROCESSED_DIR, "data_augmented")


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


def main():
    df = pd.read_pickle(os.path.join(RAW_DIR, "data.pkl"))
    df_augmented = data_augmentation(df, nb_row_by_class=122)
    df_raw = pd.concat([df, df_augmented], ignore_index=True)

    print(df.shape)
    print(df_augmented.shape)
    print(df_raw.shape)

    df_raw.to_pickle(os.path.join(PROCESSED_DIR, "data_with_augmentation.pkl"))

    df_raw["img_preprocess"] = df_raw["img"].apply(preprocess_img)
    df_preprocess = df_raw[["Class", "img_preprocess"]]
    print(df_preprocess.describe())

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_preprocess.to_pickle(os.path.join(PROCESSED_DIR, "data_preprocess.pkl"))


if __name__ == "__main__":
    main()
