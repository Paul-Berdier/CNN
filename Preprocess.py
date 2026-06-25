import os
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

df = pd.read_pickle('data_with_augmentation.pkl')
print(df['Class'].value_counts())

train_transforms = transforms.Compose([
    transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(), # Transform le format en Tenseur (donc plus numpy ni Pil.Image)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_img(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_augmented = train_transforms(image)
    return img_augmented



def visualize_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        img = image

    elif isinstance(image, torch.Tensor):
        # C x H x W -> H x W x C
        img = image.permute(1, 2, 0).numpy()

        # dénormalisation si Normalize((0.5,), (0.5,))
        img = (img * 0.5) + 0.5

        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)

    else:
        raise ValueError(
            f"Type non supporté : {type(image)}"
        )

    img.show()

df['img_preprocess'] = df['img'].apply(preprocess_img)

df_preprocess = df[['Class','img_preprocess']]
df_preprocess.to_pickle('data_preprocess.pkl')
# visualize_image(df['img_preprocess'][800])
print(df_preprocess.describe())
