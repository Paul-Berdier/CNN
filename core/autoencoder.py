"""
autoencoder.py
--------------
Autoencoder convolutif pour la détection d'images hors-domaine (OOD).

Rôle dans le pipeline :
    ÉTAPE 1 (ce fichier) : filtre OOD — l'image est-elle une plaie ?
    ÉTAPE 2 (pipeline.py) : si oui → recherche de similarité + CNN P1

Principe :
    L'autoencoder apprend à reconstruire les images de plaies.
    Face à une image inconnue (photo de chat, radiographie...),
    il reconstruit mal → erreur de reconstruction élevée → alerte OOD.

    Analogie bio : c'est comme un anticorps — formé sur les antigènes connus,
    il reconnaît les étrangers par leur mauvaise complémentarité.

Architecture (reprise du notebook solution) :
    Encodeur : image (3, 224, 224) → (64, 56, 56)   [compression ×16]
    Décodeur : (64, 56, 56) → image reconstruite (3, 224, 224)

Dépendances : torch, torchvision, Pillow, numpy, pandas
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms pour l'autoencoder : PAS de normalisation ImageNet
# On veut des pixels dans [0, 1] pour que MSELoss soit cohérente
# (MSELoss sur des valeurs normalisées négatives n'a pas de sens physique)
AE_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()           # pixel [0, 255] → float [0.0, 1.0]
])


# ─────────────────────────────────────────────
# 1. Architecture de l'autoencoder
# ─────────────────────────────────────────────

class WoundAutoencoder(nn.Module):
    """
    Autoencoder convolutif pour images de plaies.

    Repris directement du notebook solution.

    Flux des données :
        Encodeur :
            (3, 224, 224) → Conv2d stride=2 → (32, 112, 112)
                         → Conv2d stride=2 → (64,  56,  56)   ← espace latent

        Décodeur :
            (64, 56, 56) → ConvTranspose2d → (32, 112, 112)
                        → ConvTranspose2d → (3,  224, 224)   ← reconstruction

    BatchNorm2d : stabilise l'entraînement (réduit la covariate shift)
    Sigmoid en sortie : force les pixels reconstruits dans [0, 1]
    """

    def __init__(self):
        super(WoundAutoencoder, self).__init__()

        # Compression progressive de l'image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # → (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64,  56,  56)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Reconstruction inverse
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # → (3,  224, 224)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────
# 2. Dataset PyTorch pour l'entraînement
# ─────────────────────────────────────────────

class WoundDataset(Dataset):
    """
    Dataset PyTorch qui charge les images de plaies depuis un DataFrame pandas.

    Pour un autoencoder, l'entrée ET la cible sont la même image :
    on veut que le modèle apprenne à se reconstruire lui-même.

    Args:
        df        : DataFrame avec une colonne 'path' (chemin vers l'image)
        transform : pipeline de transformations torchvision
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Cible = entrée (c'est le principe de l'autoencoder)
        return image, image


def build_dataloader(df, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Crée un DataLoader à partir d'un DataFrame de chemins d'images.

    Args:
        df         : DataFrame avec colonne 'path'
        batch_size : taille des batchs
        shuffle    : mélanger les données à chaque epoch

    Returns:
        DataLoader prêt à l'emploi
    """
    dataset = WoundDataset(df, transform=AE_TRANSFORMS)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"DataLoader créé : {len(dataset)} images, batch_size={batch_size}")
    return loader


# ─────────────────────────────────────────────
# 3. Entraînement
# ─────────────────────────────────────────────

def train_autoencoder(
    model: WoundAutoencoder,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    save_path: str = "models/wound_autoencoder.pth"
) -> list[float]:
    """
    Entraîne l'autoencoder et sauvegarde les poids.

    Loss utilisée : MSELoss (erreur quadratique moyenne pixel par pixel)
    Optimizer : Adam

    Args:
        model      : instance de WoundAutoencoder
        dataloader : DataLoader produit par build_dataloader()
        epochs     : nombre d'epochs
        lr         : learning rate
        save_path  : où sauvegarder les poids (.pth)

    Returns:
        loss_history : liste des pertes par epoch (utile pour tracer la courbe)
    """
    criterion    = nn.MSELoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print(f"Entraînement de l'autoencoder sur {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, inputs)  # cible = entrée
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        epoch_loss = total_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        print(f"  Epoch [{epoch+1}/{epochs}] — MSE Loss : {epoch_loss:.5f}")

    # Sauvegarde des poids
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Autoencoder sauvegardé → {save_path}")

    return loss_history


# ─────────────────────────────────────────────
# 4. Calibration du seuil OOD
# ─────────────────────────────────────────────

def compute_ood_threshold(
    model: WoundAutoencoder,
    val_loader: DataLoader,
    percentile: int = 95
) -> float:
    """
    Calcule le seuil OOD à partir des erreurs de reconstruction
    sur le jeu de validation (images de plaies connues).

    Stratégie du percentile :
        On calcule l'erreur de reconstruction pour toutes les images connues.
        Le seuil = Xème percentile de ces erreurs.
        → 95ème percentile : on accepte 95% des images connues,
          on rejette les 5% les plus "difficiles" + toutes les images OOD.

        Compromis :
        - Percentile élevé (99) → peu de faux positifs, plus de faux négatifs
        - Percentile bas   (90) → plus de faux positifs, moins de faux négatifs

    Args:
        model      : autoencoder entraîné
        val_loader : DataLoader sur le jeu de validation
        percentile : seuil percentile (défaut 95)

    Returns:
        threshold : valeur de seuil MSE
    """
    model.eval()

    # reduction='none' : on veut l'erreur par image, pas la moyenne globale
    criterion = nn.MSELoss(reduction="none")
    errors    = []

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, inputs)

            # Moyenne sur les dimensions (C, H, W) → un scalaire par image
            sample_losses = loss.mean(dim=[1, 2, 3]).cpu().numpy()
            errors.extend(sample_losses)

    threshold = float(np.percentile(errors, percentile))
    print(f"Seuil OOD (percentile {percentile}) : {threshold:.5f}")
    return threshold


# ─────────────────────────────────────────────
# 5. Inférence OOD sur une image
# ─────────────────────────────────────────────

def verify_image_domain(
    image_path: str,
    model: WoundAutoencoder,
    threshold: float
) -> dict:
    """
    Vérifie si une image appartient au domaine des plaies connues.

    Args:
        image_path : chemin vers l'image à tester
        model      : autoencoder entraîné
        threshold  : seuil calculé par compute_ood_threshold()

    Returns:
        {
            "is_valid"            : True/False
            "reconstruction_error": float
            "threshold"           : float
            "verdict"             : str  ← message lisible pour Streamlit (P4)
        }
    """
    model.eval()

    image  = Image.open(image_path).convert("RGB")
    tensor = AE_TRANSFORMS(image).unsqueeze(0).to(DEVICE)

    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstruction       = model(tensor)
        reconstruction_error = criterion(reconstruction, tensor).item()

    is_valid = reconstruction_error <= threshold

    verdict = (
        "[VALIDE] Image médicale de plaie détectée. Traitement autorisé."
        if is_valid else
        "[ALERTE OOD] Contenu non conforme. Image hors domaine médical."
    )

    print(f"\n--- Vérification OOD : {os.path.basename(image_path)} ---")
    print(f"Erreur de reconstruction : {reconstruction_error:.5f} (seuil : {threshold:.5f})")
    print(f"Verdict : {verdict}")

    return {
        "is_valid":             is_valid,
        "reconstruction_error": reconstruction_error,
        "threshold":            threshold,
        "verdict":              verdict
    }


# ─────────────────────────────────────────────
# 6. Chargement d'un modèle sauvegardé
# ─────────────────────────────────────────────

def load_autoencoder(model_path: str = "models/wound_autoencoder.pth") -> WoundAutoencoder:
    """
    Charge un autoencoder depuis ses poids sauvegardés.
    À utiliser dans Streamlit (P4) pour l'inférence, sans réentraîner.

    Args:
        model_path : chemin vers le fichier .pth

    Returns:
        model : WoundAutoencoder en mode eval, prêt pour l'inférence
    """
    model = WoundAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Autoencoder chargé depuis {model_path}")
    return model