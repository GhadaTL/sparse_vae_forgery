"""
evaluate.py
Évaluation finale sur un ensemble balancé :
  - 500 images forgées   (FMIDV-2022)  tirées avec seed fixe
  - 500 images authentiques (MIDV-2020) tirées avec seed fixe

Compatible Google Colab.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ── Résolution des chemins (compatible Colab + local) ──────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# ───────────────────────────────────────────────────────────────────────────

from utils.helpers import set_seed, load_config, load_checkpoint
from data.datasets import get_midv2020_loaders, get_eval_transform
from models.full_model import SparseVAE
from evaluation.anomaly_scorer import AnomalyScorer
from evaluation.metrics import (compute_all_metrics, print_metrics,
                                 save_scores_csv, save_metrics_json)


# =============================================================================
# Dataset balancé unifié
# =============================================================================

class BalancedEvalDataset(Dataset):
    """
    Dataset d'évaluation balancé :
      - n_samples images authentiques depuis MIDV-2020  (label=0)
      - n_samples images forgées      depuis FMIDV-2022 (label=1)

    Tirage reproductible avec seed fixe.
    """

    def __init__(self, midv_root: str, fmidv_root: str,
                 image_size: int = 224, n_samples: int = 500, seed: int = 42):

        self.transform = get_eval_transform(image_size)
        self.samples   = []   # liste de (path, label)
        rng = np.random.RandomState(seed)

        # ── Authentiques depuis MIDV-2020 ──────────────────────────────────
        midv_path = Path(midv_root)
        auth_imgs = sorted([
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for p in midv_path.rglob(ext)
        ])
        if len(auth_imgs) == 0:
            raise ValueError(f"Aucune image authentique trouvée dans {midv_root}")

        n_auth = min(n_samples, len(auth_imgs))
        chosen = rng.choice(len(auth_imgs), size=n_auth, replace=False)
        for idx in chosen:
            self.samples.append((str(auth_imgs[idx]), 0))

        # ── Forgées depuis FMIDV-2022/forged/ ─────────────────────────────
        forg_path = Path(fmidv_root) / "forged"
        if not forg_path.exists():
            # Fallback : cherche un sous-dossier contenant "forg" dans le nom
            candidates = [d for d in Path(fmidv_root).iterdir()
                          if d.is_dir() and "forg" in d.name.lower()]
            forg_path = candidates[0] if candidates else Path(fmidv_root)

        forg_imgs = sorted([
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for p in forg_path.rglob(ext)
        ])
        if len(forg_imgs) == 0:
            raise ValueError(
                f"Aucune image forgée trouvée dans {fmidv_root}/forged/\n"
                "Vérifiez que le dossier 'forged/' existe dans fmidv2022_root."
            )

        n_forg = min(n_samples, len(forg_imgs))
        chosen = rng.choice(len(forg_imgs), size=n_forg, replace=False)
        for idx in chosen:
            self.samples.append((str(forg_imgs[idx]), 1))

        print(f"[BalancedEvalDataset] {n_auth} authentiques + {n_forg} forgées "
              f"= {n_auth + n_forg} images  (seed={seed})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "label": label, "path": path}


# =============================================================================
# Évaluation principale
# =============================================================================

def evaluate(cfg_path:   str = "configs/default.yaml",
             checkpoint: str = None,
             output_dir: str = "results",
             n_samples:  int = 500):

    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Évaluation] Device : {device}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Charger le modèle ──────────────────────────────────────────────────
    model     = SparseVAE(cfg).to(device)
    ckpt_path = checkpoint or cfg.training.best_model_path
    print(f"[Évaluation] Chargement de {ckpt_path}")
    state = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state["model"])
    model.k_controller.load_state_dict(state["k_ctrl"])
    model.eval()
    print(f"  → epoch={state['epoch']}  val_loss={state['val_loss']:.4f}  K={model.current_k}")

    # ── Calibration sur val authentiques (MIDV-2020) ───────────────────────
    print("\n[Évaluation] Calibration du scorer sur val authentiques...")
    _, val_loader = get_midv2020_loaders(cfg)

    scorer = AnomalyScorer(
        weights=(
            cfg.scoring.weights.recon,
            cfg.scoring.weights.kl,
            cfg.scoring.weights.heatmap,
            cfg.scoring.weights.latent,
        ),
        threshold_percentile=cfg.scoring.threshold_percentile,
        sigma=cfg.scoring.heatmap_sigma,
        heatmap_size=cfg.scoring.heatmap_size,
    )
    scorer.fit(model, val_loader, device)

    # ── Dataset balancé 500 authentiques + 500 forgées ────────────────────
    test_dataset = BalancedEvalDataset(
        midv_root  = cfg.data.midv2020_root,
        fmidv_root = cfg.data.fmidv2022_root,
        image_size = cfg.data.image_size,
        n_samples  = n_samples,
        seed       = cfg.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = 2,          # 2 workers recommandé sur Colab
        pin_memory  = True,
    )

    # ── Inférence ─────────────────────────────────────────────────────────
    print(f"\n[Évaluation] Inférence sur {len(test_dataset)} images...")
    results = scorer.predict_dataset(model, test_loader, device)
    results["preds"] = (results["scores"] >= scorer.threshold).astype(int)

    # ── Métriques ─────────────────────────────────────────────────────────
    metrics = compute_all_metrics(
        scores    = results["scores"],
        labels    = results["labels"],
        threshold = scorer.threshold,
    )
    print_metrics(metrics)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    scores_path  = str(Path(output_dir) / "scores_anomalie.csv")
    metrics_path = str(Path(output_dir) / "metrics.json")

    save_scores_csv(
        scores      = results["scores"],
        labels      = results["labels"],
        paths       = results["paths"],
        output_path = scores_path,
    )
    save_metrics_json(metrics, metrics_path)

    print(f"\n[Résultats sauvegardés dans '{output_dir}/']")
    print(f"  scores_anomalie.csv  ← score de chaque image")
    print(f"  metrics.json         ← AUC-ROC, F1, Precision, Recall...")
    return metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluation Sparse VAE — Dataset Balancé (Colab-compatible)"
    )
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint à utiliser (défaut : best_model_path du config)")
    parser.add_argument("--output",     type=str, default="results")
    parser.add_argument("--n_samples",  type=int, default=500,
                        help="Nombre d'images par classe (défaut : 500)")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.n_samples)