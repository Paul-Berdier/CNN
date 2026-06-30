"""Définition et entraînement des modèles CNN."""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes, freeze_base=True, dropout=0.0):
    """Construit un ResNet50 pré-entraîné ImageNet, tête remplacée pour num_classes.

    Couche Grad-CAM (dernière couche convolutive) : model.layer4[-1] (dernier bloc
    Bottleneck), vérifié sur torchvision 0.19.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    head = nn.Linear(model.fc.in_features, num_classes)
    if dropout > 0:
        head = nn.Sequential(nn.Dropout(dropout), head)
    model.fc = head  # toujours entraînable
    return model


def build_vgg16(num_classes, freeze_base=True, dropout=0.5):
    """Construit un VGG16 pré-entraîné ImageNet, tête remplacée pour num_classes.

    Couche Grad-CAM (dernière couche convolutive) : model.features[28] (dernier
    Conv2d ; ATTENTION model.features[-1] est un MaxPool2d, pas une convolution),
    vérifié sur torchvision 0.19.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Les deux Dropout déjà présents dans le classifier (p=0.5 par défaut) sont repris à 'dropout'
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # toujours entraînable
    return model


def build_efficientnet_b0(num_classes, freeze_base=True, dropout=0.2):
    """Construit un EfficientNetB0 pré-entraîné ImageNet, tête remplacée pour num_classes.

    Repris du notebook notebooks/Entrainnement_modèle.ipynb (architecture évaluée par
    un autre membre de l'équipe, jamais committée), intégré ici avec la même interface
    que build_resnet50/build_vgg16.

    Couche Grad-CAM (dernière couche convolutive) : model.features[-1] (bloc
    Conv2dNormActivation se terminant par un Conv2d 1x1), vérifié sur torchvision 0.19.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # classifier[0] est déjà un Dropout (p=0.2 par défaut) ; classifier[1] la tête à remplacer
    model.classifier[0].p = dropout
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # toujours entraînable
    return model


def unfreeze_last_layers(model, n_layers):
    """Dégèle les n_layers derniers blocs du backbone pour un fine-tuning progressif.

    ResNet50 : les blocs sont layer1..layer4 (n_layers compte parmi ces 4 blocs).
    VGG16 / EfficientNetB0 : les blocs sont les couches individuelles de model.features.
    """
    if hasattr(model, "layer4"):  # ResNet
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif hasattr(model, "features"):  # VGG / EfficientNet
        # Seules les couches avec paramètres comptent (on ignore ReLU/MaxPool, sans poids)
        blocks = [m for m in model.features.children() if any(True for _ in m.parameters())]
    else:
        raise ValueError("Architecture non supportée pour le dégel progressif.")

    for block in blocks[-n_layers:]:
        for param in block.parameters():
            param.requires_grad = True

    return model
