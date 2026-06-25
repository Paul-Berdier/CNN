import os
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # Flip horizontal aléatoire
    transforms.RandomRotation(degrees=15),    # Rotation aléatoire
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Variation de couleur
    transforms.RandomResizedCrop(size=640, scale=(0.8, 1.0)),  # Recadrage aléatoire
])

def augmented_img(image):
    img_pil = Image.fromarray(image)
    img_augmented = train_transforms(img_pil)
    return img_augmented

def data_augmentation(df,nb_row_by_class):
    output_dir = "data_augmented"
    

    # Boucle sur les classe
    df_augmented = []
    for i in df['Class'].unique().tolist():
        print(i)
        sub_df = df[df['Class'] == i]
        nb_row = sub_df.shape[0]
        os.makedirs(f"{output_dir}/data_augmented_{i}", exist_ok=True)
        if nb_row < nb_row_by_class :
            nb_generate = nb_row_by_class - nb_row
            for e in range(nb_generate):
                img_random = np.random.choice(sub_df.index, size=1)[0]
                img_augmented = augmented_img(df['img'][img_random])
                df_augmented.append({
                'Class': i,
                'Nom_img': f"{i}_data_augmented_{e}",
                'img': img_augmented,
                'Hauteur': 640,
                'Largeur': 640,
                })

                img_augmented.save(f"{output_dir}/data_augmented_{i}/augmented_{i}_{e}.jpeg") 
    df_augmented = pd.DataFrame(df_augmented)
    return df_augmented


df = pd.read_pickle('data.pkl')
df_augmented = data_augmentation(df,122)

df_raw = pd.concat([df, df_augmented], ignore_index=True)

print(df.shape)
print(df_augmented.shape)
print(df_raw.shape)

df_raw.to_pickle('data_with_augmentation.pkl')
