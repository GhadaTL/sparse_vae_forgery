# train.py
# CDC §5.4 — protocole 3 phases
# Entraîne UN run (K, β) sur MIDV-2020 authentiques uniquement
# Sauvegarde : checkpoint + JSON (sans auc_roc)

import argparse
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from models.full_model      import SparseVAE
from losses.total_loss      import total_loss, beta_annealing
from evaluation.metrics     import validate_projection_head


# ── Reproducibilité ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """CDC §reproducibilité — obligatoire pour publication."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Cibles multi-échelle ─────────────────────────────────────────────────────

def prepare_multiscale_targets(x: torch.Tensor):
    """
    CDC §4.4 — redimensionner x aux 3 résolutions cibles.
    x : (B, 3, 224, 224)
    """
    x_c = F.interpolate(x, (16, 16), mode="bilinear", align_corners=False)
    x_m = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
    x_f = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
    return x_c, x_m, x_f


# ── Critères latents ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_latent_criteria(model, val_loader, device) -> dict:
    """
    Calcule les critères K et β sur val set authentiques.
    CDC §2.6, §3.4.
    Appelé sur le best checkpoint après entraînement.
    Ne touche pas aux images forgées.
    """
    model.eval()

    all_mu, all_logvar = [], []
    all_sparsity, all_ratio = [], []

    for x, _ in val_loader:
        x           = x.to(device)
        tokens      = model.dinov2(x)
        mu, log_var = model.projection_head(tokens)
        z_sparse, _, mask = model.sparse_latent(mu, log_var)

        all_mu.append(mu.cpu())
        all_logvar.append(log_var.cpu())
        all_sparsity.append((z_sparse == 0).float().mean().item())
        all_ratio.append(mask.float().mean().item())

    mu_all     = torch.cat(all_mu,     dim=0)   # (N, 64)
    logvar_all = torch.cat(all_logvar, dim=0)   # (N, 64)

    # ── Critères K ───────────────────────────────────────────
    sparsity_rate = float(np.mean(all_sparsity))
    active_ratio  = float(np.mean(all_ratio))
    var_dims      = mu_all.var(dim=0)            # (64,)
    n_collapsed   = int((var_dims < 0.01).sum().item())

    # ── Critère β ────────────────────────────────────────────
    final_kl = float(
        -0.5 * torch.sum(
            1 + logvar_all - mu_all.pow(2) - logvar_all.exp(),
            dim=1
        ).mean().item()
    )

    return {
        "sparsity_rate": sparsity_rate,
        "active_ratio":  active_ratio,
        "n_collapsed":   n_collapsed,
        "final_kl":      final_kl,
    }


# ── Entraînement d'un run ────────────────────────────────────────────────────

def train_one_run(k: int,
                  beta_max: float,
                  train_loader,
                  val_loader,
                  cfg: dict,
                  device: torch.device) -> dict:
    """
    Entraîne un run complet pour une combinaison (K, β).
    CDC §5.4 — protocole 3 phases.

    Phase 1 (1–20)   : β = 0  — reconstruction pure
    Phase 2 (21–50)  : β 0→4  — KL warm-up
    Phase 3 (51–100) : β = 4  — entraînement complet

    Retourne dict avec critères latents + val_loss.
    auc_roc = None — sera complété par evaluate.py.
    """
    run_name        = f"k{k}_beta{int(beta_max)}"
    checkpoint_path = Path(f"checkpoints/{run_name}_best.pt")
    result_path     = Path(f"results/ablation/{run_name}.json")

    # ── Skip si déjà fait ────────────────────────────────────
    if result_path.exists():
        print(f"  [skip] {run_name} — déjà entraîné")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'─'*55}")
    print(f"  Run : K={k}, β={beta_max}  →  {run_name}")
    print(f"{'─'*55}")

    set_seed(42)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results/ablation").mkdir(parents=True, exist_ok=True)

    # ── Modèle ───────────────────────────────────────────────
    model = SparseVAE(
        latent_dim = cfg["latent_dim"],
        k          = k,
        dropout    = cfg["dropout"],
    ).to(device)

    # Optimiseur — ProjectionHead + Decoder uniquement
    # DINOv2 est frozen — jamais dans l'optimiseur
    optimizer = optim.AdamW(
        list(model.projection_head.parameters()) +
        list(model.decoder.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    stopped_early    = False

    for epoch in range(1, cfg["epochs"] + 1):

        # ────────────────────────────────────────────────────
        # TRAIN
        # ────────────────────────────────────────────────────
        model.train()

        for x, _ in train_loader:
            x = x.to(device)
            x_c, x_m, x_f = prepare_multiscale_targets(x)

            tokens                     = model.dinov2(x)
            mu, log_var                = model.projection_head(tokens)
            z_sparse, kl, mask         = model.sparse_latent(mu, log_var)
            x_hat_c, x_hat_m, x_hat_f = model.decoder(z_sparse)

            metrics = total_loss(
                x_c, x_hat_c,
                x_m, x_hat_m,
                x_f, x_hat_f,
                kl_loss  = kl,
                epoch    = epoch,
                beta_max = beta_max,
            )
            metrics["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # ────────────────────────────────────────────────────
        # VALIDATION §2.6 — fin Phase 1 uniquement (epoch 20)
        # Forward partiel : s'arrête après ProjectionHead
        # ────────────────────────────────────────────────────
        if epoch == 20:
            print(f"\n  ── Validation §2.6 (fin Phase 1) ──")
            result_26 = validate_projection_head(
                model, val_loader, device
            )
            if not result_26["ok_phase1"]:
                print(f"  ⚠️  Phase 1 non validée — run arrêté")
                # Retourner résultat invalide
                return {
                    "k":             k,
                    "beta":          beta_max,
                    "val_loss":      float("inf"),
                    "sparsity_rate": 0.0,
                    "active_ratio":  0.0,
                    "n_collapsed":   64,
                    "final_kl":      0.0,
                    "auc_roc":       None,
                    "checkpoint":    None,
                    "valid":         False,
                }
            print(f"  ✅ Phase 1 validée")
            model.train()

        # ────────────────────────────────────────────────────
        # VALIDATION LOSS — forward complet
        # ────────────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_c, x_m, x_f = prepare_multiscale_targets(x)

                tokens                     = model.dinov2(x)
                mu, log_var                = model.projection_head(tokens)
                z_sparse, kl, _            = model.sparse_latent(mu, log_var)
                x_hat_c, x_hat_m, x_hat_f = model.decoder(z_sparse)

                m = total_loss(
                    x_c, x_hat_c,
                    x_m, x_hat_m,
                    x_f, x_hat_f,
                    kl_loss  = kl,
                    epoch    = epoch,
                    beta_max = beta_max,
                )
                val_losses.append(m["loss"].item())

        val_loss = sum(val_losses) / len(val_losses)
        beta_cur = beta_annealing(epoch, beta_max=beta_max)

        print(f"  Epoch {epoch:03d} | "
              f"val_loss={val_loss:.5f} | "
              f"β={beta_cur:.2f}")

        # ────────────────────────────────────────────────────
        # EARLY STOPPING + SAUVEGARDE CHECKPOINT
        # CDC §5.2 — patience=15
        # ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping epoch {epoch}")
                stopped_early = True
                break

        model.train()

    # ── Critères latents sur best checkpoint ─────────────────
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    criteria = compute_latent_criteria(model, val_loader, device)

    print(f"\n  Résultat {run_name} :")
    print(f"    val_loss      = {best_val_loss:.5f}")
    print(f"    sparsity_rate = {criteria['sparsity_rate']:.2f}")
    print(f"    active_ratio  = {criteria['active_ratio']:.2f}")
    print(f"    n_collapsed   = {criteria['n_collapsed']}")
    print(f"    final_kl      = {criteria['final_kl']:.3f}")

    # ── Sauvegarde JSON (sans auc_roc) ────────────────────────
    result = {
        "k":             k,
        "beta":          beta_max,
        "val_loss":      best_val_loss,
        "sparsity_rate": criteria["sparsity_rate"],
        "active_ratio":  criteria["active_ratio"],
        "n_collapsed":   criteria["n_collapsed"],
        "final_kl":      criteria["final_kl"],
        "auc_roc":       None,   # complété par evaluate.py
        "checkpoint":    str(checkpoint_path),
        "valid":         True,
        "stopped_early": stopped_early,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Main (un seul run) ────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int,   required=True)
    parser.add_argument("--beta_max", type=float, required=True)
    parser.add_argument("--config",   type=str,
                        default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    # Dataloaders — MIDV-2020 authentiques uniquement
    from data.dataset import MIDV2020Dataset
    from torch.utils.data import DataLoader

    train_dataset = MIDV2020Dataset(split="train")
    val_dataset   = MIDV2020Dataset(split="val")
    train_loader  = DataLoader(train_dataset,
                               batch_size=cfg["batch_size"],
                               shuffle=True,
                               num_workers=4)
    val_loader    = DataLoader(val_dataset,
                               batch_size=cfg["batch_size"],
                               shuffle=False,
                               num_workers=4)

    train_one_run(
        k            = args.k,
        beta_max     = args.beta_max,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
    )