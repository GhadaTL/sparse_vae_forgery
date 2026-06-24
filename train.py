"""
train.py
Script d'entraînement principal.

Protocole en 3 phases :
  Phase 1 (epochs 1-30)   : β=0, reconstruction pure, sélection best sur val recon loss
  Phase 2 (epochs 31-130) : warm-up linéaire β 0→4, pas de sélection
  Phase 3 (epochs 131-300): β=4.0 fixe, sélection best sur val total loss

K est adapté automatiquement à la fin de chaque époque (Phase 2 et 3).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Ajout du répertoire racine au path (compatible Colab + local)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # S'assurer que le CWD est bien la racine du projet

from utils.helpers import (set_seed, load_config, get_beta, get_phase,
                            compute_latent_diagnostics, prepare_multiscale_targets,
                            save_checkpoint, load_checkpoint, Logger)
from data.datasets import get_midv2020_loaders
from models.full_model import SparseVAE
from losses.loss import total_loss


# =============================================================================
# Boucle d'entraînement — une époque
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, beta, device, logger, epoch):
    model.train()
    model.dinov2.eval()  # DINOv2 toujours en eval

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    n_batches      = 0

    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        x_c, x_m, x_f = prepare_multiscale_targets(images, device)

        optimizer.zero_grad()
        outputs = model(images)
        losses  = criterion(outputs, x_c, x_m, x_f, beta)

        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += losses["loss_total"].item()
        recon_loss_sum += losses["loss_recon"].item()
        kl_loss_sum    += losses["loss_kl"].item()
        n_batches      += 1

        pbar.set_postfix({
            "total": f"{losses['loss_total'].item():.4f}",
            "recon": f"{losses['loss_recon'].item():.4f}",
            "kl":    f"{losses['loss_kl'].item():.4f}",
            "K":     model.current_k,
            "β":     f"{beta:.2f}",
        })

    return {
        "train/loss_total": total_loss_sum / n_batches,
        "train/loss_recon": recon_loss_sum / n_batches,
        "train/loss_kl":    kl_loss_sum    / n_batches,
    }


# =============================================================================
# Boucle de validation — une époque
# =============================================================================

@torch.no_grad()
def validate(model, loader, criterion, beta, device):
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    n_batches      = 0

    # Pour l'Adaptive K Controller
    all_kl_per_dim = []
    all_masks      = []
    all_mu         = []
    all_logvar     = []

    for batch in tqdm(loader, desc="  Val", leave=False):
        images = batch["image"].to(device)
        x_c, x_m, x_f = prepare_multiscale_targets(images, device)

        outputs = model(images)
        losses  = criterion(outputs, x_c, x_m, x_f, beta)

        total_loss_sum += losses["loss_total"].item()
        recon_loss_sum += losses["loss_recon"].item()
        kl_loss_sum    += losses["loss_kl"].item()
        n_batches      += 1

        all_kl_per_dim.append(outputs["kl_per_dim"].detach())
        all_masks.append(outputs["mask"].detach())
        all_mu.append(outputs["mu"].detach())
        all_logvar.append(outputs["logvar"].detach())

    avg_kl_per_dim = torch.stack(all_kl_per_dim).mean(dim=0)  # (D,)
    all_masks_cat  = torch.cat(all_masks,  dim=0)
    all_mu_cat     = torch.cat(all_mu,     dim=0)
    all_logvar_cat = torch.cat(all_logvar, dim=0)

    diag = compute_latent_diagnostics(all_mu_cat, all_logvar_cat,
                                      all_masks_cat, avg_kl_per_dim)

    return {
        "val/loss_total": total_loss_sum / n_batches,
        "val/loss_recon": recon_loss_sum / n_batches,
        "val/loss_kl":    kl_loss_sum    / n_batches,
        "val/active_ratio":  diag["active_ratio"],
        "val/sparsity_rate": diag["sparsity_rate"],
        "val/kl_mean":       diag["kl_mean"],
        "val/dead_dims":     diag["dead_dims"],
        "val/posterior_collapse": diag["posterior_collapse"],
        "val/kl_explosion":       diag["kl_explosion"],
    }, avg_kl_per_dim


# =============================================================================
# Boucle principale
# =============================================================================

def train(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Entraînement] Device : {device}")

    # --- DataLoaders ---
    train_loader, val_loader = get_midv2020_loaders(cfg)

    # --- Modèle ---
    model = SparseVAE(cfg).to(device)

    # --- Optimiseur (uniquement les paramètres entraînables) ---
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # --- Criterion ---
    criterion = total_loss(cfg)

    # --- Logger ---
    logger = Logger(
        use_wandb=cfg.logging.use_wandb,
        project=cfg.logging.project,
    )

    # --- État de l'entraînement ---
    best_val_loss = float("inf")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    print(f"\n[Entraînement] Début — {cfg.training.epochs} epochs\n")
    print(f"  Phase 1 : epochs 1–{cfg.beta.phase1_end}       β=0, reconstruction pure")
    print(f"  Phase 2 : epochs {cfg.beta.phase1_end+1}–{cfg.beta.phase2_end}  warm-up β 0→{cfg.beta.beta_final}")
    print(f"  Phase 3 : epochs {cfg.beta.phase2_end+1}–{cfg.training.epochs}   β={cfg.beta.beta_final} fixe\n")

    for epoch in range(1, cfg.training.epochs + 1):
        beta  = get_beta(epoch, cfg)
        phase = get_phase(epoch, cfg)

        print(f"Époque {epoch}/{cfg.training.epochs} | Phase {phase} | β={beta:.3f} | K={model.current_k}")

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, beta, device, logger, epoch
        )

        # --- Val ---
        val_metrics, avg_kl_per_dim = validate(
            model, val_loader, criterion, beta, device
        )

        # --- Affichage ---
        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch,
                       "beta": beta, "K": model.current_k, "phase": phase}
        logger.log(all_metrics, step=epoch)

        # --- Diagnostics ---
        if val_metrics["val/posterior_collapse"]:
            print(f"  ⚠ POSTERIOR COLLAPSE détecté (KL_mean={val_metrics['val/kl_mean']:.4f})")
        if val_metrics["val/kl_explosion"]:
            print(f"  ⚠ KL EXPLOSION détectée (KL_mean={val_metrics['val/kl_mean']:.4f})")

        # --- Adaptive K (phases 2 et 3) ---
        if phase >= 2:
            new_k = model.update_k(avg_kl_per_dim, val_metrics["val/loss_recon"])
            print(f"  → K adaptatif : {model.current_k} (EMA)")

        # --- Checkpoint last ---
        save_checkpoint({
            "epoch":    epoch,
            "model":    model.state_dict(),
            "optim":    optimizer.state_dict(),
            "k_ctrl":   model.k_controller.state_dict(),
            "beta":     beta,
            "phase":    phase,
            "val_loss": val_metrics["val/loss_total"],
        }, cfg.training.last_model_path)

        # --- Checkpoint best ---
        if phase == 1:
            # Sélection sur val reconstruction loss
            monitor = val_metrics["val/loss_recon"]
        elif phase == 3:
            # Sélection sur val total loss
            monitor = val_metrics["val/loss_total"]
        else:
            # Phase 2 : pas de sélection
            monitor = None

        if monitor is not None and monitor < best_val_loss:
            best_val_loss = monitor
            save_checkpoint({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optim":    optimizer.state_dict(),
                "k_ctrl":   model.k_controller.state_dict(),
                "beta":     beta,
                "phase":    phase,
                "val_loss": monitor,
            }, cfg.training.best_model_path)
            print(f"  ✓ Nouveau best model sauvegardé (loss={monitor:.4f})")

    print(f"\n[Entraînement terminé] Best val loss = {best_val_loss:.4f}")
    print(f"  Modèles sauvegardés :")
    print(f"    best : {cfg.training.best_model_path}")
    print(f"    last : {cfg.training.last_model_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Sparse VAE Forgery Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()
    train(args.config)