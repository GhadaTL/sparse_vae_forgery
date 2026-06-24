"""
evaluation/metrics.py
Calcul des métriques d'évaluation de détection d'anomalies.
"""
import json
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)


def compute_all_metrics(scores: np.ndarray,
                        labels: np.ndarray,
                        threshold: float = None) -> dict:
    """
    Calcule toutes les métriques de détection.

    Args:
        scores    : (N,) scores d'anomalie (plus haut = plus suspect)
        labels    : (N,) labels vrais (0=authentique, 1=forgé)
        threshold : seuil θ (si None, utilise le meilleur F1)

    Returns:
        dict avec AUC-ROC, Precision, Recall, F1, Threshold, Confusion Matrix
    """
    # AUC-ROC
    auc = roc_auc_score(labels, scores)

    # Seuil optimal (meilleur F1 si non fourni)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        f1s = [f1_score(labels, (scores >= t).astype(int), zero_division=0)
               for t in thresholds]
        best_idx = int(np.argmax(f1s))
        threshold = float(thresholds[best_idx])

    preds = (scores >= threshold).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    cm        = confusion_matrix(labels, preds).tolist()

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    metrics = {
        "auc_roc":   float(auc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1_score":  float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "n_forged":    int(labels.sum()),
        "n_authentic": int((labels == 0).sum()),
    }
    return metrics


def print_metrics(metrics: dict):
    """Affiche les métriques de façon lisible."""
    print("\n" + "=" * 50)
    print("  RÉSULTATS D'ÉVALUATION FINALE")
    print("=" * 50)
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1_score']:.4f}")
    print(f"  Threshold  : {metrics['threshold']:.4f}")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"\n  Matrice de confusion :")
    cm = metrics["confusion_matrix"]
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("=" * 50 + "\n")


def save_scores_csv(scores: np.ndarray,
                    labels: np.ndarray,
                    paths: list,
                    output_path: str):
    """
    Sauvegarde les scores d'anomalie de chaque image dans un CSV.

    Colonnes : path, true_label, anomaly_score, predicted_label
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "true_label", "anomaly_score", "predicted_label"]
        )
        writer.writeheader()
        for path, label, score in zip(paths, labels, scores):
            writer.writerow({
                "path":             path,
                "true_label":       int(label),
                "anomaly_score":    f"{score:.6f}",
                "predicted_label":  int(score > 0),   # provisoire, sera recalibré
            })
    print(f"[Scores] Sauvegardé dans {output_path}")


def save_metrics_json(metrics: dict, output_path: str):
    """Sauvegarde les métriques en JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Métriques] Sauvegardé dans {output_path}")
