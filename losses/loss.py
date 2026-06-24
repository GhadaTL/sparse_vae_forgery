"""
losses/total_loss.py
Loss totale : reconstruction multi-échelle (MSE + SSIM) + β × KL.

L_scale  = α × MSE + γ × (1 - SSIM)
L_recon  = 0.1×L_coarse + 0.3×L_medium + 0.6×L_fine
L_total  = L_recon + β(t) × L_KL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.losses import ssim_loss as kornia_ssim_loss
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


# =============================================================================
# SSIM (avec fallback sans kornia)
# =============================================================================

def compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Retourne 1 - SSIM(pred, target), scalaire dans [0, 1].
    Utilise kornia si disponible, sinon implémentation simplifiée.
    """
    if KORNIA_AVAILABLE:
        # kornia attend window_size impair et images ∈ [0,1]
        window = min(pred.shape[-1], 11)
        if window % 2 == 0:
            window -= 1
        return kornia_ssim_loss(pred, target, window_size=window)

    # --- Fallback SSIM simplifié ---
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(pred,   3, 1, 1)
    mu2 = F.avg_pool2d(target, 3, 1, 1)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred,     3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
    sigma12   = F.avg_pool2d(pred * target,   3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# =============================================================================
# Loss par échelle
# =============================================================================

def scale_loss(pred: torch.Tensor,
               target: torch.Tensor,
               alpha: float = 0.8,
               gamma: float = 0.2) -> torch.Tensor:
    """
    L_scale = α × MSE + γ × (1 - SSIM)

    Args:
        pred   : (B, 3, H, W) — reconstruction
        target : (B, 3, H, W) — cible (downsampleée à la même résolution)
        alpha  : poids MSE (0.8)
        gamma  : poids SSIM (0.2)
    """
    mse  = F.mse_loss(pred, target)
    ssim = compute_ssim_loss(pred, target)
    return alpha * mse + gamma * ssim


# =============================================================================
# Loss de reconstruction multi-échelle
# =============================================================================

def multiscale_recon_loss(x_hat_coarse: torch.Tensor,
                          x_hat_medium: torch.Tensor,
                          x_hat_fine:   torch.Tensor,
                          x_coarse:     torch.Tensor,
                          x_medium:     torch.Tensor,
                          x_fine:       torch.Tensor,
                          alpha:  float = 0.8,
                          gamma:  float = 0.2,
                          lam_c:  float = 0.1,
                          lam_m:  float = 0.3,
                          lam_f:  float = 0.6) -> tuple:
    """
    L_recon = λ_c × L_coarse + λ_m × L_medium + λ_f × L_fine

    Returns:
        l_recon  : scalaire — loss totale pondérée
        l_coarse : scalaire — loss échelle coarse
        l_medium : scalaire — loss échelle medium
        l_fine   : scalaire — loss échelle fine
    """
    l_coarse = scale_loss(x_hat_coarse, x_coarse, alpha, gamma)
    l_medium = scale_loss(x_hat_medium, x_medium, alpha, gamma)
    l_fine   = scale_loss(x_hat_fine,   x_fine,   alpha, gamma)

    l_recon = lam_c * l_coarse + lam_m * l_medium + lam_f * l_fine

    return l_recon, l_coarse, l_medium, l_fine


# =============================================================================
# Loss totale
# =============================================================================

class total_loss(nn.Module):
    """
    Loss complète du Sparse VAE.

    L_total = L_recon + β × L_KL

    Args:
        cfg : configuration (DotDict)
    """

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.loss.alpha_mse
        self.gamma = cfg.loss.gamma_ssim
        self.lam_c = cfg.loss.lambda_coarse
        self.lam_m = cfg.loss.lambda_medium
        self.lam_f = cfg.loss.lambda_fine

    def forward(self,
                outputs: dict,
                x_coarse: torch.Tensor,
                x_medium: torch.Tensor,
                x_fine:   torch.Tensor,
                beta:     float) -> dict:
        """
        Args:
            outputs  : dict retourné par SparseVAE.forward()
            x_coarse : (B, 3, 16, 16) — cible coarse
            x_medium : (B, 3, 32, 32) — cible medium
            x_fine   : (B, 3, 64, 64) — cible fine
            beta     : β courant (phase scheduler)

        Returns:
            dict avec toutes les losses pour logging
        """
        l_recon, l_coarse, l_medium, l_fine = multiscale_recon_loss(
            x_hat_coarse=outputs["x_hat_coarse"],
            x_hat_medium=outputs["x_hat_medium"],
            x_hat_fine=outputs["x_hat_fine"],
            x_coarse=x_coarse,
            x_medium=x_medium,
            x_fine=x_fine,
            alpha=self.alpha,
            gamma=self.gamma,
            lam_c=self.lam_c,
            lam_m=self.lam_m,
            lam_f=self.lam_f,
        )

        kl_loss = outputs["kl_loss"]
        l_total = l_recon + beta * kl_loss

        return {
            "loss_total":  l_total,
            "loss_recon":  l_recon,
            "loss_coarse": l_coarse,
            "loss_medium": l_medium,
            "loss_fine":   l_fine,
            "loss_kl":     kl_loss,
        }
