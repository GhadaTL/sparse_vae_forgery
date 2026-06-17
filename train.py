"""
train.py
========
Training loop for the Sparse VAE forgery detector (CDC §5.4).

3-phase protocol:
    Phase 1  epochs 1–20  : β=0   pure reconstruction
    Phase 2  epochs 21–50 : β 0→4 KL warm-up
    Phase 3  epochs 51+   : β=4   full sparse VAE

Usage:
    python train.py                         # uses configs/default.yaml
    python train.py --config path/to.yaml
    python train.py --no_adaptive           # disable K/β controllers
"""

import os
import sys
import yaml
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────
from models.full_model  import SparseVAE               # fixed: was FullModel
from losses.total_loss  import total_loss, compute_sparsity_metrics  # fixed: was total_loss_fn
from utils.k_controller    import KController
from utils.beta_controller import BetaController
from data.dataset          import MIDV2020Dataset

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ──────────────────────────────────────────────────────────────
# Reproducibility (CDC p.22 — mandatory for publication)
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
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
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, device,
                adaptive: bool, k_ctrl, beta_ctrl, cfg):
    """Run one training epoch. Returns metric dict."""
    model.set_train_mode()

    totals = dict(loss=0, l_recon=0, l_kl=0, sparsity=0)
    n = 0

    for batch in loader:
        images = batch["image"].to(device) if isinstance(batch, dict) \
                 else batch[0].to(device)

        optimizer.zero_grad()

        # ── forward ───────────────────────────────────────────
        outputs = model(images)

        # ── dynamic k (adaptive mode) ─────────────────────────
        if adaptive:
            current_k = k_ctrl.get_k()
            if current_k != model.sparse_latent.k:
                model.sparse_latent.k = current_k
                # re-run forward with updated k
                outputs = model(images)

        # ── loss ──────────────────────────────────────────────
        loss_dict = total_loss(
            outputs, epoch,
            alpha    = cfg["loss"]["alpha"],
            gamma    = cfg["loss"]["gamma"],
            lam_c    = cfg["loss"]["lambda_coarse"],
            lam_m    = cfg["loss"]["lambda_medium"],
            lam_f    = cfg["loss"]["lambda_fine"],
            beta_max = cfg["model"]["beta_final"],
        )

        # ── sparsity metrics for controllers ──────────────────
        sp = compute_sparsity_metrics(outputs["z_sparse"])

        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        totals["loss"]     += loss_dict["loss"].item()
        totals["l_recon"]  += loss_dict["l_recon"].item()
        totals["l_kl"]     += loss_dict["l_kl"].item()
        totals["sparsity"] += sp["sparsity_rate"]
        n += 1

    avg = {k: v / n for k, v in totals.items()}
    avg["beta"] = loss_dict["beta"]   # same for whole epoch
    return avg


# ──────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, epoch, device, cfg):
    model.set_eval_mode()

    totals = dict(loss=0, l_recon=0, l_kl=0, active_ratio=0)
    n = 0

    for batch in loader:
        images = batch["image"].to(device) if isinstance(batch, dict) \
                 else batch[0].to(device)

        outputs   = model(images)
        loss_dict = total_loss(outputs, epoch,
                               alpha    = cfg["loss"]["alpha"],
                               gamma    = cfg["loss"]["gamma"],
                               lam_c    = cfg["loss"]["lambda_coarse"],
                               lam_m    = cfg["loss"]["lambda_medium"],
                               lam_f    = cfg["loss"]["lambda_fine"],
                               beta_max = cfg["model"]["beta_final"])

        totals["loss"]         += loss_dict["loss"].item()
        totals["l_recon"]      += loss_dict["l_recon"].item()
        totals["l_kl"]         += loss_dict["l_kl"].item()
        totals["active_ratio"] += outputs["mask"].float().mean().item()
        n += 1

    return {k: v / n for k, v in totals.items()}


# ──────────────────────────────────────────────────────────────
# Diagnostics (CDC §2.6 and §3.4)
# ──────────────────────────────────────────────────────────────

def check_diagnostics(val_metrics: dict, epoch: int):
    kl = val_metrics["l_kl"]
    ar = val_metrics["active_ratio"]

    if epoch > 50:    # only after Phase 3 starts
        if kl < 2:
            print(f"  [WARN] KL={kl:.2f} < 2 — possible posterior collapse. "
                  f"Try reducing lr or β.")
        elif kl > 10:
            print(f"  [WARN] KL={kl:.2f} > 10 — latent space not regularised. "
                  f"Try increasing β warmup.")

    if ar < 0.15:
        print(f"  [WARN] active_ratio={ar:.3f} < 0.15 — too many dead dims. "
              f"Try reducing K or β.")
    elif ar > 0.85:
        print(f"  [WARN] active_ratio={ar:.3f} > 0.85 — sparsity too low. "
              f"Try increasing K.")


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def train(cfg: dict, adaptive: bool = True):
    set_seed(cfg["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  adaptive={adaptive}")

    # ── W&B ───────────────────────────────────────────────────
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
        eta_min = 1e-6,
    )

    # ── adaptive controllers ───────────────────────────────────
    k_ctrl   = KController(
        initial_k     = cfg["model"]["k"],
        min_k         = 1,
        max_k         = cfg["model"]["latent_dim"],
        step          = 1,
        window        = 5,
        threshold_pct = 5.0,
    )
    beta_ctrl = BetaController(
        beta_max         = cfg["model"]["beta_final"],
        target_sparsity  = 0.75,   # 75% zeros ≈ 25% active (k=16/64)
        step             = 0.1,
        warmup_epoch     = cfg["model"]["beta_warmup_epochs"],
    )

    early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # ── training loop ─────────────────────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):

        # Phase label
        p1 = cfg["training"]["phase1_epochs"]
        p2 = p1 + cfg["training"]["phase2_epochs"]
        if epoch <= p1:
            phase = f"Phase 1 — reconstruction only (β=0)"
        elif epoch <= p2:
            phase = f"Phase 2 — KL warm-up"
        else:
            phase = f"Phase 3 — full sparse VAE"

        # ── train ─────────────────────────────────────────────
        tr = train_epoch(model, train_loader, optimizer, epoch,
                         device, adaptive, k_ctrl, beta_ctrl, cfg)

        # ── update controllers after epoch ────────────────────
        if adaptive:
            new_k    = k_ctrl.update(tr["l_recon"])
            new_beta = beta_ctrl.update(tr["sparsity"], epoch)
        else:
            new_k    = model.sparse_latent.k
            new_beta = tr["beta"]

        # ── validate ──────────────────────────────────────────
        vl = val_epoch(model, val_loader, epoch, device, cfg)
        scheduler.step()

        # ── diagnostics ───────────────────────────────────────
        check_diagnostics(vl, epoch)

        # ── logging ───────────────────────────────────────────
        log = {
            "epoch":              epoch,
            "train/loss":         tr["loss"],
            "train/recon":        tr["l_recon"],
            "train/kl":           tr["l_kl"],
            "train/sparsity":     tr["sparsity"],
            "val/loss":           vl["loss"],
            "val/recon":          vl["l_recon"],
            "val/kl":             vl["l_kl"],
            "val/active_ratio":   vl["active_ratio"],
            "latent/k":           new_k,
            "latent/beta":        new_beta,
            "lr":                 scheduler.get_last_lr()[0],
        }
        if _WANDB:
            wandb.log(log)

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | {phase[:30]}"
            f" | train={tr['loss']:.4f}  val={vl['loss']:.4f}"
            f" | K={new_k}  β={new_beta:.2f}"
            f" | active_ratio={vl['active_ratio']:.3f}"
        )

        # ── checkpoint ────────────────────────────────────────
        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "config":      cfg,
            }, "checkpoints/best_model.pth")
            print(f"  ✓ best model saved  (val_loss={best_val_loss:.4f})")

        # Always save last
        torch.save(model.state_dict(), "checkpoints/last_model.pth")

        # ── early stopping ────────────────────────────────────
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
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Disable K/β adaptive controllers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, adaptive=not args.no_adaptive)
