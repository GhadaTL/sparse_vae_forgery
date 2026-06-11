"""
models/sparse_latent.py
=======================
Étape ③ — Sparse Latent Space
Reparametrization trick + Top-K hard sparsity + KL divergence.
Cœur de la contribution du pipeline.
"""

import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                   training: bool = True, n_samples: int = 10) -> torch.Tensor:
    """
    Reparametrization trick : z = mu + sigma * epsilon.
    Différentiable par rapport à mu et logvar.

    Entraînement : un seul échantillon (efficace).
    Inférence    : moyenne de n_samples pour stabilité.
    """
    std = torch.exp(0.5 * logvar)
    if training:
        eps = torch.randn_like(std)
        return mu + std * eps
    else:
        z_samples = [mu + std * torch.randn_like(std) for _ in range(n_samples)]
        return torch.stack(z_samples).mean(dim=0)


def top_k_sparsity(z: torch.Tensor, k: int = 16):
    """
    Top-K hard sparsity : garde seulement les K dimensions
    de plus forte magnitude, met les autres à zéro.

    Gradient : straight-through estimator (backward non bloqué).

    Args:
        z : (B, latent_dim)
        k : nombre de dimensions actives

    Returns:
        z_sparse : (B, latent_dim) — seulement K dims ≠ 0
        mask     : (B, latent_dim) — binaire 0/1
    """
    z_abs = z.abs()

    # Seuil = K-ième plus grande valeur par sample
    threshold, _ = z_abs.topk(k, dim=1)          # (B, K)
    threshold = threshold[:, -1:]                  # (B, 1)

    # Masque binaire
    mask = (z_abs >= threshold).float()            # (B, latent_dim)

    # Application du masque (straight-through : gradient passe entier)
    z_sparse = z * mask

    return z_sparse, mask


class SparseLatentLayer(nn.Module):
    """
    Couche latente sparse d'un β-VAE.

    Étapes :
        1. Reparametrization trick → z
        2. Top-K sparsity         → z_sparse, mask
        3. KL divergence          → kl_loss (sur z original)
    """

    def __init__(self, latent_dim: int = 64, k: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.k = k

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Args:
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)

        Returns:
            z_sparse : (B, latent_dim)  — K dims actives
            kl_loss  : scalaire         — KL(q||p) moyenne sur batch
            mask     : (B, latent_dim)  — pour monitoring LEA
        """
        # 1. Échantillonnage différentiable
        z = reparameterize(mu, logvar, training=self.training)

        # 2. Sparsité Top-K
        z_sparse, mask = top_k_sparsity(z, self.k)

        # 3. KL divergence (calculée sur z avant sparsité)
        #    KL(N(μ,σ²) || N(0,I)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        return z_sparse, kl_loss, mask

    # ------------------------------------------------------------------ #
    #  Métriques LEA (Latent Efficiency Analysis) — pour monitoring W&B   #
    # ------------------------------------------------------------------ #

    def active_ratio(self, mask: torch.Tensor) -> float:
        """Fraction moyenne de dimensions actives sur le batch. Cible : ~0.25"""
        return mask.float().mean().item()

    def sparsity_rate(self, mask: torch.Tensor) -> float:
        """Fraction de dimensions NON actives (zéro). Cible : ~0.75"""
        return 1.0 - self.active_ratio(mask)

    def latent_entropy(self, mask: torch.Tensor) -> float:
        """Entropie sur les activations — mesure de diversité."""
        p = mask.float().mean(dim=0).clamp(1e-8, 1 - 1e-8)  # (latent_dim,)
        return (-p * p.log() - (1 - p) * (1 - p).log()).mean().item()

    def variance_per_dim(self, z_sparse: torch.Tensor) -> torch.Tensor:
        """Variance par dimension sur le batch. Toutes > 0.001 = pas de dim morte."""
        return z_sparse.var(dim=0)   # (latent_dim,)