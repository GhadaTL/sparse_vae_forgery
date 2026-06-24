"""
ablation.py
Étude d'ablation — obligatoire pour publication.

Configurations testées :
  1. Baseline         : S_recon seul
  2. +KL              : S_recon + S_kl
  3. +Heatmap         : S_recon + S_heatmap
  4. Complet          : S_recon + S_kl + S_heatmap + S_latent

Pour chaque config, calcule AUC-ROC sur FMIDV-2022 et log les résultats.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.helpers import set_seed, load_config, load_checkpoint, prepare_multiscale_targets
from data.datasets import get_midv2020_loaders, get_fmidv2022_loader
from models.full_model import SparseVAE
from evaluation.heatmap import compute_heatmap_stats, compute_multiscale_heatmap
from evaluation.metrics import print_metrics, save_metrics_json

import torch.nn.functional as F


# =============================================================================
# Calcul des scores bruts pour tout un dataset
# =============================================================================

@torch.no_grad()
def collect_all_raw_scores(model, loader, heatmap_stats, device, cfg):
    """
    Collecte les 4 scores bruts pour chaque image du loader.
    """
    model.eval()
    all_recon, all_kl, all_heatmap, all_latent, all_labels = [], [], [], [], []

    for batch in tqdm(loader, desc="  Collecte scores"):
        images = batch["image"].to(device)
        labels = batch["label"]

        outputs = model(images)
        _, _, x_f = prepare_multiscale_targets(images, device)

        # S_recon
        s_recon = F.mse_loss(outputs["x_hat_fine"], x_f,
                             reduction="none").mean(dim=[1, 2, 3]).cpu().numpy()

        # S_kl
        kl_bd = -0.5 * (1 + outputs["logvar"]
                        - outputs["mu"].pow(2)
                        - outputs["logvar"].exp())
        s_kl = kl_bd.sum(dim=1).cpu().numpy()

        # S_latent
        k = max(1, outputs["mask"].sum(dim=1).float().mean().item())
        s_latent = (outputs["z_sparse"].norm(dim=1) / k).cpu().numpy()

        # S_heatmap
        s_heatmap = []
        for i in range(images.shape[0]):
            H = compute_multiscale_heatmap(
                model, images[i:i+1], heatmap_stats, device,
                sigma=cfg.scoring.heatmap_sigma,
                heatmap_size=cfg.scoring.heatmap_size,
            )
            s_heatmap.append(float(np.percentile(H, 95)))

        all_recon.extend(s_recon.tolist())
        all_kl.extend(s_kl.tolist())
        all_heatmap.extend(s_heatmap)
        all_latent.extend(s_latent.tolist())
        all_labels.extend(labels.numpy().tolist())

    return {
        "recon":   np.array(all_recon),
        "kl":      np.array(all_kl),
        "heatmap": np.array(all_heatmap),
        "latent":  np.array(all_latent),
        "labels":  np.array(all_labels),
    }


def zscore(arr, mean, std):
    return (arr - mean) / (std + 1e-8)


# =============================================================================
# Ablation principale
# =============================================================================

def ablation(cfg_path: str = "configs/default.yaml",
             checkpoint: str = None,
             output_dir: str = "results/ablation"):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Modèle ---
    model = SparseVAE(cfg).to(device)
    ckpt_path = checkpoint or cfg.training.best_model_path
    state = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[Ablation] Modèle chargé (epoch={state['epoch']})")

    # --- Loaders ---
    _, val_loader  = get_midv2020_loaders(cfg)
    test_loader    = get_fmidv2022_loader(cfg)

    # --- Stats heatmap et normalisation (sur val authentiques) ---
    heatmap_stats = compute_heatmap_stats(model, val_loader, device,
                                          cfg.scoring.heatmap_size)
    print("\n[Ablation] Collecte des scores sur val (authentiques)...")
    val_raw = collect_all_raw_scores(model, val_loader, heatmap_stats, device, cfg)

    # Stats de normalisation
    norm_stats = {}
    for k in ["recon", "kl", "heatmap", "latent"]:
        norm_stats[k] = {
            "mean": float(val_raw[k].mean()),
            "std":  float(val_raw[k].std() + 1e-8),
        }

    print("[Ablation] Collecte des scores sur test (FMIDV-2022)...")
    test_raw = collect_all_raw_scores(model, test_loader, heatmap_stats, device, cfg)
    labels = test_raw["labels"]

    # Scores normalisés
    z_recon   = zscore(test_raw["recon"],   norm_stats["recon"]["mean"],   norm_stats["recon"]["std"])
    z_kl      = zscore(test_raw["kl"],      norm_stats["kl"]["mean"],      norm_stats["kl"]["std"])
    z_heatmap = zscore(test_raw["heatmap"], norm_stats["heatmap"]["mean"], norm_stats["heatmap"]["std"])
    z_latent  = zscore(test_raw["latent"],  norm_stats["latent"]["mean"],  norm_stats["latent"]["std"])

    # --- Configurations d'ablation ---
    configs = {
        "1_baseline":    z_recon,
        "2_plus_kl":     0.67 * z_recon + 0.33 * z_kl,
        "3_plus_heatmap":0.57 * z_recon + 0.43 * z_heatmap,
        "4_complet":     0.4 * z_recon + 0.2 * z_kl + 0.3 * z_heatmap + 0.1 * z_latent,
    }

    # --- Résultats ---
    print("\n" + "=" * 55)
    print("  ABLATION STUDY — AUC-ROC sur FMIDV-2022")
    print("=" * 55)

    ablation_results = {}
    for name, scores in configs.items():
        auc = roc_auc_score(labels, scores)
        print(f"  {name:<25} AUC = {auc:.4f}")
        ablation_results[name] = float(auc)

    print("=" * 55 + "\n")

    save_metrics_json(ablation_results, f"{output_dir}/ablation_auc.json")
    print(f"[Ablation] Résultats sauvegardés dans {output_dir}/ablation_auc.json")
    return ablation_results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study — Sparse VAE")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output",     type=str, default="results/ablation")
    args = parser.parse_args()
    ablation(args.config, args.checkpoint, args.output)
