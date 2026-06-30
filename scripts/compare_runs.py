"""Compare les architectures entraînées (scripts/train.py) à partir des artefacts sauvegardés.

Lit chaque models/{run_name}_config.json + models/{run_name}_classification_report.txt
et produit un tableau comparatif (accuracy, F1 macro, F1 par classe).

Exemple :
    python scripts/compare_runs.py
"""

import glob
import json
import os
import re

import pandas as pd

MODELS_DIR = "models"


def parse_classification_report(text):
    """Extrait accuracy, recall/F1 macro et F1 par classe d'un classification_report sklearn (texte)."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    per_class_f1 = {}

    for line in lines[1:]:
        parts = line.split()
        if line.strip().startswith("accuracy"):
            accuracy = float(parts[-2])
        elif line.strip().startswith("macro avg"):
            macro_precision, macro_recall, macro_f1 = (float(p) for p in parts[-4:-1])
        elif line.strip().startswith("weighted avg"):
            weighted_f1 = float(parts[-2])
        else:
            # ligne "classe precision recall f1-score support"
            *name_parts, precision, recall, f1, support = parts
            class_name = " ".join(name_parts)
            per_class_f1[class_name] = float(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
    }


def main():
    rows = []
    for config_path in sorted(glob.glob(os.path.join(MODELS_DIR, "*_config.json"))):
        run_name = os.path.basename(config_path)[: -len("_config.json")]
        report_path = os.path.join(MODELS_DIR, f"{run_name}_classification_report.txt")
        if not os.path.exists(report_path):
            continue

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        with open(report_path, encoding="utf-8") as f:
            metrics = parse_classification_report(f.read())

        row = {
            "run_name": run_name,
            "architecture": config["args"]["arch"],
            "accuracy": metrics["accuracy"],
            "recall_macro": metrics["macro_recall"],
            "precision_macro": metrics["macro_precision"],
            "f1_macro": metrics["macro_f1"],
            "f1_weighted": metrics["weighted_f1"],
        }
        row.update({f"f1_{c}": v for c, v in metrics["per_class_f1"].items()})
        rows.append(row)

    if not rows:
        print("Aucun run avec config + classification_report trouvé dans models/.")
        return

    # Trié par recall_macro : en contexte médical on priorise la minimisation des faux
    # négatifs (cf. justification dans notebooks/MLFlow.ipynb).
    df = pd.DataFrame(rows).sort_values("recall_macro", ascending=False)
    print(df.to_string(index=False))

    output_path = os.path.join(MODELS_DIR, "runs_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nTableau sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
