"""Chargement et prétraitement des images."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # Flip horizontal aléatoire
    transforms.RandomRotation(degrees=15),    # Rotation aléatoire
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Variation de couleur
    transforms.RandomResizedCrop(size=640, scale=(0.8, 1.0)),  # Recadrage aléatoire
])

preprocess_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),  # Transforme le format en Tenseur (donc plus numpy ni PIL.Image)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def augmented_img(image):
    img_pil = Image.fromarray(image)
    return augmentation_transforms(img_pil)


def preprocess_img(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return preprocess_transforms(image)


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
        raise ValueError(f"Type non supporté : {type(image)}")

    img.show()
