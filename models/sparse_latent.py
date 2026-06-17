import torch
import torch.nn as nn


# =========================================================
# TOP-K + STRAIGHT-THROUGH ESTIMATOR
# =========================================================

class TopKStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, k):
        """
        z: latent vector (B, D)
        k: number of active dimensions
        """
        abs_z = z.abs()
        topk_vals, topk_idx = torch.topk(abs_z, k, dim=-1)

        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)

        z_sparse = z * mask
        ctx.save_for_backward(mask)
        return z_sparse

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-Through Estimator : gradient passe comme une identité."""
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None


# =========================================================
# SPARSE LATENT MODULE  — avec reparameterization + KL
# =========================================================

class SparseLatent(nn.Module):
    """
    Étape ③ du pipeline :
        (mu, logvar) → reparameterization → Top-K sparsity → (z_sparse, kl_loss)

    Le KL est calculé analytiquement sur mu/logvar AVANT le masquage Top-K,
    conformément à la pratique standard VAE (le masque est non-différentiable
    et le STE ne propage pas de gradient vers logvar via le masque).
    """

    def __init__(self, latent_dim: int = 64, k: int = 16):
        super().__init__()
        self.latent_dim  = latent_dim
        self.k_default   = k
        self.topk_fn     = TopKStraightThrough

    # ------------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * std  (eps ~ N(0,I))."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # déterministe à l'inférence

    # ------------------------------------------------------------------
    def kl_loss(self, mu: torch.Tensor,
                logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z|x) || p(z)) analytique pour prior N(0,I).
        = -0.5 * sum(1 + logvar - mu² - exp(logvar))
        Retourne la moyenne sur le batch (scalaire).
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()   # (B,D) → (B,) → scalaire

    # ------------------------------------------------------------------
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor,
                k: int = None):
        """
        Args:
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)
            k      : nombre de dimensions actives (override k_default si fourni)

        Returns:
            z_sparse : (B, latent_dim)  — vecteur latent clairsemé
            kl       : scalaire          — perte KL
            mask     : (B, latent_dim)  — masque binaire Top-K
        """
        if k is None:
            k = self.k_default

        # sécurité : k dans [1, latent_dim]
        k = int(max(1, min(k, self.latent_dim)))

        # 1. reparameterization trick
        z = self.reparameterize(mu, logvar)

        # 2. KL avant masquage
        kl = self.kl_loss(mu, logvar)

        # 3. Top-K sparsity avec STE
        z_sparse = self.topk_fn.apply(z, k)

        # 4. reconstruire le masque (pour logging / contrôleurs)
        mask = (z_sparse != 0).float()

        return z_sparse, kl, mask
