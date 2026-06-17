"""
losses/total_loss.py
====================
Step ⑤ — Total loss
MSE + SSIM multi-scale reconstruction + β-KL annealing (CDC §5)

L_total = L_recon + β(t) × L_KL
L_recon = 0.1×L_coarse + 0.3×L_medium + 0.6×L_fine
L_x     = 0.8×MSE(x, x̂) + 0.2×(1 − SSIM(x, x̂))

β schedule (3-phase — CDC §5.4):
    Phase 1  epochs 1–20  : β = 0      (pure reconstruction)
    Phase 2  epochs 21–50 : β 0 → 4    (KL warm-up)
    Phase 3  epochs 51+   : β = 4      (full training)
"""

import torch
import torch.nn.functional as F

try:
    from kornia.losses import ssim_loss as kornia_ssim
    _KORNIA = True
except ImportError:
    _KORNIA = False


# ──────────────────────────────────────────────────────────────
# Per-scale helpers
# ──────────────────────────────────────────────────────────────

def _mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x, reduction="mean")


def _ssim_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """1 − SSIM via kornia. Falls back to MSE if kornia is absent."""
    if _KORNIA:
        return kornia_ssim(x_hat, x, window_size=7, reduction="mean")
    return _mse(x, x_hat)


def scale_loss(x: torch.Tensor, x_hat: torch.Tensor,
               alpha: float = 0.8, gamma: float = 0.2) -> torch.Tensor:
    """L_x = alpha × MSE + gamma × (1 − SSIM)"""
    return alpha * _mse(x, x_hat) + gamma * _ssim_loss(x, x_hat)


# ──────────────────────────────────────────────────────────────
# β schedule
# ──────────────────────────────────────────────────────────────

def beta_annealing(epoch: int,
                   phase1_end: int   = 20,
                   phase2_end: int   = 50,
                   beta_max:   float = 4.0) -> float:
    """
    3-phase β schedule (CDC §5.4).

    epoch < phase1_end              : β = 0
    phase1_end ≤ epoch < phase2_end : β linearly 0 → beta_max
    epoch ≥ phase2_end              : β = beta_max
    """
    if epoch < phase1_end:
        return 0.0
    if epoch < phase2_end:
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        return beta_max * progress
    return beta_max


# ──────────────────────────────────────────────────────────────
# Main loss function
# ──────────────────────────────────────────────────────────────

def total_loss(outputs: dict, epoch: int,
               alpha: float = 0.8, gamma: float = 0.2,
               lam_c: float = 0.1, lam_m: float = 0.3, lam_f: float = 0.6,
               beta_max: float = 4.0) -> dict:
    """
    Compute total loss from the output dict of SparseVAE.forward().

    Args
        outputs  : dict returned by SparseVAE.forward()
        epoch    : current epoch (for β scheduling)
        alpha    : MSE weight inside each scale loss
        gamma    : SSIM weight inside each scale loss
        lam_c/m/f: scale weights (coarse / medium / fine)
        beta_max : maximum β value

    Returns dict with:
        loss        — total loss (backward on this)
        l_recon     — reconstruction loss
        l_coarse    — coarse scale loss
        l_medium    — medium scale loss
        l_fine      — fine scale loss
        l_kl        — raw KL divergence
        beta        — current β value
    """
    l_c = scale_loss(outputs["x_coarse"], outputs["x_hat_coarse"], alpha, gamma)
    l_m = scale_loss(outputs["x_medium"], outputs["x_hat_medium"], alpha, gamma)
    l_f = scale_loss(outputs["x_fine"],   outputs["x_hat_fine"],   alpha, gamma)

    l_recon = lam_c * l_c + lam_m * l_m + lam_f * l_f

    beta   = beta_annealing(epoch, beta_max=beta_max)
    l_kl   = outputs["kl"]
    loss   = l_recon + beta * l_kl

    return {
        "loss":     loss,
        "l_recon":  l_recon,
        "l_coarse": l_c,
        "l_medium": l_m,
        "l_fine":   l_f,
        "l_kl":     l_kl,
        "beta":     beta,
    }


# ──────────────────────────────────────────────────────────────
# Sparsity metrics (CDC §3.4 — for controller feedback)
# ──────────────────────────────────────────────────────────────

def compute_sparsity_metrics(z_sparse: torch.Tensor) -> dict:
    """
    Compute sparsity diagnostics on a batch of z_sparse vectors.
    Feed these into KController and BetaController.

    Args
        z_sparse : (B, latent_dim)

    Returns dict with:
        sparsity_rate        — fraction of zero entries across batch
        active_ratio         — fraction of non-zero entries
        n_collapsed          — dims with zero activity across ALL samples
        n_active             — dims active at least once in the batch
        mean_active_per_sample — mean active dims per sample
    """
    total_zeros    = (z_sparse == 0).float().sum().item()
    total_elements = z_sparse.numel()
    sparsity_rate  = total_zeros / total_elements
    active_ratio   = 1.0 - sparsity_rate

    dims_active  = (z_sparse.abs() > 1e-7).any(dim=0)   # (latent_dim,)
    n_collapsed  = int((~dims_active).sum().item())
    n_active     = int(dims_active.sum().item())

    mean_active  = (z_sparse.abs() > 1e-7).sum(dim=1).float().mean().item()

    return {
        "sparsity_rate":         float(sparsity_rate),
        "active_ratio":          float(active_ratio),
        "n_collapsed":           n_collapsed,
        "n_active":              n_active,
        "mean_active_per_sample": float(mean_active),
    }


# Backward-compatible alias — old code that called total_loss_fn still works
def total_loss_fn(x, x_hat, beta=0.0, z_sparse=None):
    """
    Compatibility shim for old train.py calls.
    NOTE: the new API uses total_loss(outputs, epoch) instead.
    """
    l = F.mse_loss(x_hat, x)
    sparsity = 0.0
    if z_sparse is not None:
        sparsity = compute_sparsity_metrics(z_sparse)["sparsity_rate"]
    return l, sparsity
