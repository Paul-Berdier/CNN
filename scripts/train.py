"""Entraînement ResNet50 / VGG16 / EfficientNetB0 sur la classification des plaies, avec suivi MLflow.

Exemple :
    python scripts/train.py --arch resnet50 --lr 1e-3 --batch-size 16 --epochs 30
    python scripts/train.py --arch vgg16 --lr 1e-4 --batch-size 16 --epochs 30 --unfreeze-n 2
    python scripts/train.py --arch efficientnet_b0 --lr 1e-3 --batch-size 16 --epochs 30
"""

import argparse
import json
import os

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from core.dataset import load_datasets
from core.model_utils import build_resnet50, build_vgg16, build_efficientnet_b0, unfreeze_last_layers

DATA_PATH = "data/processed/data_preprocess.pkl"
MODELS_DIR = "models"
EXPERIMENT_NAME = "wound-classification"

BUILDERS = {
    "resnet50": build_resnet50,
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(arch, num_classes, dropout, unfreeze_n):
    if arch not in BUILDERS:
        raise ValueError(f"Architecture inconnue : {arch}")
    model = BUILDERS[arch](num_classes, freeze_base=True, dropout=dropout)

    if unfreeze_n > 0:
        unfreeze_last_layers(model, unfreeze_n)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def evaluate_test(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes)
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["resnet50", "vgg16", "efficientnet_b0"], required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)  # dataset petit (~854 images) : batch modeste
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)  # early stopping sur val_loss
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--unfreeze-n", type=int, default=0)  # 0 = tête seule (frozen)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, class_to_idx = load_datasets(DATA_PATH)
    classes = sorted(class_to_idx, key=class_to_idx.get)
    # Classes parfaitement équilibrées après augmentation (122/classe, cf. vérification
    # Étape 0) : pas de WeightedRandomSampler ni de class_weight nécessaire.

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args.arch, len(classes), args.dropout, args.unfreeze_n).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    run_name = args.run_name or f"{args.arch}-lr{args.lr}-{'frozen' if args.unfreeze_n == 0 else f'unfreeze{args.unfreeze_n}'}"

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "architecture": args.arch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "unfreeze_n": args.unfreeze_n,
        })

        os.makedirs(MODELS_DIR, exist_ok=True)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_path = os.path.join(MODELS_DIR, f"{run_name}.pt")

        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)
            print(f"[{epoch+1}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping à l'epoch {epoch+1} (val_loss sans amélioration depuis {args.patience} epochs)")
                    break

        # Réévaluation avec les meilleurs poids (val_loss minimale)
        model.load_state_dict(torch.load(best_path))
        report, cm = evaluate_test(model, test_loader, device, classes)
        print(report)

        report_path = os.path.join(MODELS_DIR, f"{run_name}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        cm_path = os.path.join(MODELS_DIR, f"{run_name}_confusion_matrix.png")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation=45)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        config_path = os.path.join(MODELS_DIR, f"{run_name}_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"class_to_idx": class_to_idx, "args": vars(args)}, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(best_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(config_path)


if __name__ == "__main__":
    main()
