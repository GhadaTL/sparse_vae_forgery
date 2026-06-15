import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.losses import ssim_loss  # CDC §stack technique
except ImportError:
    ssim_loss = None


def _mse_per_scale(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x, reduction="mean")


def _ssim_per_scale(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, via kornia. Fallback MSE si kornia absent."""
    if ssim_loss is not None:
        return ssim_loss(x_hat, x, window_size=7, reduction="mean")
    return _mse_per_scale(x, x_hat)


def reconstruction_loss(
    x_c: torch.Tensor, x_hat_c: torch.Tensor,
    x_m: torch.Tensor, x_hat_m: torch.Tensor,
    x_f: torch.Tensor, x_hat_f: torch.Tensor,
    alpha: float = 0.8,
    gamma: float = 0.2,
    lam_c: float = 0.1,
    lam_m: float = 0.3,
    lam_f: float = 0.6,
) -> dict:
    """
    CDC §5.2 :
      L_x = α·MSE(x, x̂) + γ·(1 - SSIM(x, x̂))
      L_recon = λ_c·L_c + λ_m·L_m + λ_f·L_f
    """
    def scale_loss(x, xh):
        return alpha * _mse_per_scale(x, xh) + gamma * _ssim_per_scale(x, xh)

    l_c = scale_loss(x_c, x_hat_c)
    l_m = scale_loss(x_m, x_hat_m)
    l_f = scale_loss(x_f, x_hat_f)

    l_recon = lam_c * l_c + lam_m * l_m + lam_f * l_f

    return {
        "l_recon": l_recon,
        "l_coarse": l_c,
        "l_medium": l_m,
        "l_fine":   l_f,
    }


def beta_annealing(epoch: int,
                   warmup_start: int = 20,
                   warmup_end: int   = 50,
                   beta_max: float   = 4.0) -> float:
    """
    CDC §5.2 et §5.4 — protocole 3 phases :
      Phase 1 (epochs 1–20)  : β = 0  (reconstruction pure)
      Phase 2 (epochs 21–50) : β 0→4  (KL warm-up linéaire)
      Phase 3 (epochs 51+)   : β = 4  (entraînement complet)
    """
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return beta_max * progress
    else:
        return beta_max


def total_loss(
    x_c, x_hat_c,
    x_m, x_hat_m,
    x_f, x_hat_f,
    kl_loss: torch.Tensor,
    epoch:   int,
) -> dict:
    """
    CDC §5.2 :
      L_total = L_recon + β(t) × L_KL
    """
    recon = reconstruction_loss(x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f)
    beta  = beta_annealing(epoch)
    loss  = recon["l_recon"] + beta * kl_loss

    return {
        "loss":     loss,
        "l_recon":  recon["l_recon"],
        "l_coarse": recon["l_coarse"],
        "l_medium": recon["l_medium"],
        "l_fine":   recon["l_fine"],
        "l_kl":     kl_loss,
        "beta":     beta,
    }