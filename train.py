"""
train.py
========
Training loop for the Sparse VAE forgery detector (CDC §5.4).

3-phase protocol (phases lues depuis configs/default.yaml):
    Phase 1  epochs 1–phase1_epochs          : β=0   pure reconstruction
    Phase 2  phase1_epochs–phase1+phase2     : β 0→beta_final  KL warm-up
    Phase 3  phase1+phase2+                  : β=beta_final    full sparse VAE

Usage:
    python train.py                         # uses configs/default.yaml
    python train.py --config path/to.yaml
    python train.py --no_adaptive           # disable K/β controllers
"""

import os
import random
import yaml
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.full_model     import SparseVAE
from losses.total_loss     import total_loss, compute_sparsity_metrics
from utils.k_controller    import KController
from utils.beta_controller import BetaController
from data.dataset          import MIDV2020Dataset

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def get_phases(cfg: dict):
    """Retourne (p1, p2) lus depuis cfg."""
    p1 = cfg["training"]["phase1_epochs"]
    p2 = p1 + cfg["training"]["phase2_epochs"]
    return p1, p2


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, device, adaptive, k_ctrl, cfg):
    """
    Run one training epoch.

    Returns metric dict incluant :
        loss, l_recon, l_kl, sparsity, beta — scalaires moyennés
        kl_per_dim                          — array (latent_dim,) moyenné sur tous les batches
    """
    model.set_train_mode()

    # K appliqué une seule fois avant tous les batches
    if adaptive:
        model.sparse_latent.k = k_ctrl.get_k()

    p1, p2 = get_phases(cfg)

    totals     = dict(loss=0.0, l_recon=0.0, l_kl=0.0, sparsity=0.0)
    kl_dim_acc = None   # accumulateur pour kl_per_dim (latent_dim,)
    n          = 0
    loss_dict  = {}

    for batch in loader:
        images = batch["image"].to(device) if isinstance(batch, dict) \
                 else batch[0].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss_dict = total_loss(
            outputs,
            epoch,
            alpha      = cfg["loss"]["alpha"],
            gamma      = cfg["loss"]["gamma"],
            lam_c      = cfg["loss"]["lambda_coarse"],
            lam_m      = cfg["loss"]["lambda_medium"],
            lam_f      = cfg["loss"]["lambda_fine"],
            beta_max   = cfg["model"]["beta_final"],
            phase1_end = p1,
            phase2_end = p2,
        )

        sp = compute_sparsity_metrics(outputs["z_sparse"])

        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(),
            cfg["training"]["grad_clip_norm"],
        )
        optimizer.step()

        totals["loss"]     += loss_dict["loss"].item()
        totals["l_recon"]  += loss_dict["l_recon"].item()
        totals["l_kl"]     += loss_dict["l_kl"].item()
        totals["sparsity"] += sp["sparsity_rate"]

        # ── accumulation kl_per_dim ───────────────────────────
        # outputs["kl_per_dim"] : (B, latent_dim) ou (latent_dim,)
        # On moyenne sur le batch puis on accumule sur les batches.
        kl_dim_batch = outputs["kl_per_dim"]
        if isinstance(kl_dim_batch, torch.Tensor):
            kl_dim_batch = kl_dim_batch.detach().cpu().numpy()
        if kl_dim_batch.ndim == 2:
            kl_dim_batch = kl_dim_batch.mean(axis=0)   # (latent_dim,)

        kl_dim_acc = kl_dim_batch if kl_dim_acc is None \
                     else kl_dim_acc + kl_dim_batch
        n += 1

    avg            = {k: v / n for k, v in totals.items()}
    avg["beta"]    = loss_dict.get("beta", 0.0)
    avg["kl_per_dim"] = kl_dim_acc / n   # (latent_dim,) moyenné sur l'epoch

    return avg


# ──────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, epoch, device, cfg):
    model.set_eval_mode()

    p1, p2 = get_phases(cfg)

    totals = dict(loss=0.0, l_recon=0.0, l_kl=0.0, active_ratio=0.0)
    n = 0

    for batch in loader:
        images = batch["image"].to(device) if isinstance(batch, dict) \
                 else batch[0].to(device)

        outputs   = model(images)
        loss_dict = total_loss(
            outputs,
            epoch,
            alpha      = cfg["loss"]["alpha"],
            gamma      = cfg["loss"]["gamma"],
            lam_c      = cfg["loss"]["lambda_coarse"],
            lam_m      = cfg["loss"]["lambda_medium"],
            lam_f      = cfg["loss"]["lambda_fine"],
            beta_max   = cfg["model"]["beta_final"],
            phase1_end = p1,
            phase2_end = p2,
        )

        totals["loss"]         += loss_dict["loss"].item()
        totals["l_recon"]      += loss_dict["l_recon"].item()
        totals["l_kl"]         += loss_dict["l_kl"].item()
        totals["active_ratio"] += outputs["mask"].float().mean().item()
        n += 1

    return {k: v / n for k, v in totals.items()}


# ──────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────

def check_diagnostics(val_metrics: dict, epoch: int, cfg: dict):
    _, p2 = get_phases(cfg)
    kl = val_metrics["l_kl"]
    ar = val_metrics["active_ratio"]

    if epoch > p2:
        if kl < cfg["diagnostics"]["kl_collapse_threshold"]:
            print(f"  [WARN] KL={kl:.2f} < {cfg['diagnostics']['kl_collapse_threshold']}"
                  f" — possible posterior collapse.")
        elif kl > cfg["diagnostics"]["kl_explode_threshold"]:
            print(f"  [WARN] KL={kl:.2f} > {cfg['diagnostics']['kl_explode_threshold']}"
                  f" — latent space not regularised.")

    if ar < cfg["diagnostics"]["active_ratio_min"]:
        print(f"  [WARN] active_ratio={ar:.3f} < {cfg['diagnostics']['active_ratio_min']}"
              f" — too many dead dims.")
    elif ar > cfg["diagnostics"]["active_ratio_max"]:
        print(f"  [WARN] active_ratio={ar:.3f} > {cfg['diagnostics']['active_ratio_max']}"
              f" — sparsity too low.")


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def train(cfg: dict, adaptive: bool = True):
    set_seed(cfg["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  adaptive={adaptive}")

    if _WANDB:
        wandb.init(
            project = cfg.get("wandb", {}).get("project", "sparse-vae-forgery"),
            config  = cfg,
        )

    # ── data ──────────────────────────────────────────────────
    train_ds = MIDV2020Dataset(cfg["data"]["midv2020_path"], split="train",
                               image_size=cfg["data"]["image_size"])
    val_ds   = MIDV2020Dataset(cfg["data"]["midv2020_path"], split="val",
                               image_size=cfg["data"]["image_size"])

    train_loader = DataLoader(train_ds,
                              batch_size  = cfg["training"]["batch_size"],
                              shuffle     = True,
                              num_workers = cfg["data"]["num_workers"],
                              pin_memory  = True)
    val_loader   = DataLoader(val_ds,
                              batch_size  = cfg["training"]["batch_size"],
                              shuffle     = False,
                              num_workers = cfg["data"]["num_workers"],
                              pin_memory  = True)

    # ── model ─────────────────────────────────────────────────
    model = SparseVAE(
        latent_dim = cfg["model"]["latent_dim"],
        k          = cfg["model"]["k"],
    ).to(device)

    params = model.count_parameters()
    print(f"[train] trainable params : {params['trainable']:,}")
    print(f"[train] frozen params    : {params['frozen']:,}")

    # ── optimizer ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg["training"]["epochs"],
        eta_min = cfg["training"]["lr_min"],
    )

    # ── adaptive controllers ───────────────────────────────────
    k_ctrl = KController(
        latent_dim   = cfg["model"]["latent_dim"],
        initial_k    = cfg["model"]["k"],
        min_k        = cfg["adaptive"]["k_min"],
        max_k        = cfg["adaptive"]["k_max"],
        tau_fraction = cfg["adaptive"]["k_tau_fraction"],
        tau_min      = cfg["adaptive"]["k_tau_min"],
        ema_alpha    = cfg["adaptive"]["k_ema_alpha"],
        ratio_min    = cfg["adaptive"]["k_ratio_min"],
        ratio_max    = cfg["adaptive"]["k_ratio_max"],
        recon_loss_low  = cfg["adaptive"]["k_recon_loss_low"],
        recon_loss_high = cfg["adaptive"]["k_recon_loss_high"],
        recon_step      = cfg["adaptive"]["k_recon_step"],
    )
    beta_ctrl = BetaController(
        beta_max        = cfg["model"]["beta_final"],
        target_sparsity = cfg["adaptive"]["target_sparsity"],
        step            = cfg["adaptive"]["beta_step"],
        warmup_epoch    = cfg["model"]["beta_warmup_epochs"],
    )

    early_stop = EarlyStopping(
        patience  = cfg["training"]["early_stopping_patience"],
        min_delta = cfg["training"]["early_stopping_min_delta"],
    )

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")

    p1, p2 = get_phases(cfg)

    # ── training loop ─────────────────────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):

        if epoch <= p1:
            phase = "Phase 1 — reconstruction only"
        elif epoch <= p2:
            phase = "Phase 2 — KL warm-up"
        else:
            phase = "Phase 3 — full sparse VAE"

        tr = train_epoch(model, train_loader, optimizer, epoch,
                         device, adaptive, k_ctrl, cfg)

        # ── update controllers APRÈS l'epoch ──────────────────
        if adaptive:
            new_k = k_ctrl.update(
                kl_per_dim = tr["kl_per_dim"],   # array (latent_dim,)
                recon_loss = tr["l_recon"],
            )
            new_beta = beta_ctrl.update(tr["sparsity"], epoch)
        else:
            new_k    = model.sparse_latent.k
            new_beta = tr["beta"]

        # ── validate ──────────────────────────────────────────
        vl = val_epoch(model, val_loader, epoch, device, cfg)
        scheduler.step()

        check_diagnostics(vl, epoch, cfg)

        log = {
            "epoch":            epoch,
            "train/loss":       tr["loss"],
            "train/recon":      tr["l_recon"],
            "train/kl":         tr["l_kl"],
            "train/sparsity":   tr["sparsity"],
            "val/loss":         vl["loss"],
            "val/recon":        vl["l_recon"],
            "val/kl":           vl["l_kl"],
            "val/active_ratio": vl["active_ratio"],
            "latent/k":         new_k,
            "latent/beta":      new_beta,
            "lr":               scheduler.get_last_lr()[0],
        }
        if _WANDB:
            wandb.log(log)

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | {phase}"
            f" | train={tr['loss']:.4f}  val={vl['loss']:.4f}"
            f" | K={new_k}  β={new_beta:.2f}"
            f" | active_ratio={vl['active_ratio']:.3f}"
        )

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "config":      cfg,
            }, ckpt_path)
            print(f"  ✓ best model saved  (val_loss={best_val_loss:.4f})")

        torch.save(model.state_dict(),
                   os.path.join(cfg["output"]["checkpoint_dir"], "last_model.pth"))

        if early_stop.step(vl["loss"]):
            print(f"[EarlyStopping] stopped at epoch {epoch}")
            break

    print(f"\n[train] Done. Best val loss: {best_val_loss:.4f}")
    if _WANDB:
        wandb.finish()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--no_adaptive", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, adaptive=not args.no_adaptive)