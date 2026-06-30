"""Dataset PyTorch et split train/val/test pour la classification des plaies."""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# NOTE: data/processed/data_preprocess.pkl a été sérialisé avec une pandas 3.0
# pré-release (StringDtype 'python'/na_value=nan). Avec pandas 2.3.3 (version du
# projet), pd.read_pickle lève NotImplementedError. Voir échange d'équipe pour
# décider de l'alignement de version avant d'utiliser ce module en l'état.


class WoundDataset(Dataset):
    """Dataset PyTorch lisant un DataFrame (colonnes 'Class' et 'img_preprocess')."""

    def __init__(self, df, class_to_idx):
        self.images = df["img_preprocess"].tolist()
        self.labels = [class_to_idx[c] for c in df["Class"].tolist()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_datasets(pkl_path, test_size=0.15, val_size=0.15, random_state=42):
    """Charge le pickle prétraité et le découpe en train/val/test (split stratifié).

    Retourne (train_dataset, val_dataset, test_dataset, class_to_idx).
    """
    df = pd.read_pickle(pkl_path)

    classes = sorted(df["Class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    df_train, df_temp = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df["Class"],
        random_state=random_state,
    )
    relative_test_size = test_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_size,
        stratify=df_temp["Class"],
        random_state=random_state,
    )

    train_dataset = WoundDataset(df_train, class_to_idx)
    val_dataset = WoundDataset(df_val, class_to_idx)
    test_dataset = WoundDataset(df_test, class_to_idx)

    print(f"Classes ({len(classes)}) : {classes}")
    print(f"Train : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_to_idx
