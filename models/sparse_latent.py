"""
models/sparse_latent.py
=======================
Step ③ — Sparse Latent Space
Reparametrization trick + Top-K hard sparsity + KL divergence
Conforms to CDC §3

Three operations in sequence:
    1. Reparametrization : z = mu + sigma * eps   (differentiable sampling)
    2. Top-K sparsity    : keep only K largest |z| dimensions, zero the rest
    3. KL divergence     : -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# 1. Reparametrization trick
# ──────────────────────────────────────────────────────────────

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor,
                  training: bool = True, n_samples: int = 10) -> torch.Tensor:
    """
    z = mu + sigma * eps,   eps ~ N(0, I)

    Differentiable w.r.t. mu and logvar via the reparametrization trick.
    At inference: average of n_samples draws for stability (CDC §2.4).

    Args
        mu      : (B, latent_dim)
        logvar  : (B, latent_dim)  — log sigma^2, clamped in [-4, 4]
        training: True during training, False at inference
        n_samples: number of MC samples averaged at inference

    Returns
        z : (B, latent_dim)
    """
    std = torch.exp(0.5 * logvar)          # sigma = exp(0.5 * log sigma^2)

    if training:
        eps = torch.randn_like(std)
        return mu + std * eps
    else:
        # Average n_samples for lower variance estimate at inference
        samples = [mu + std * torch.randn_like(std) for _ in range(n_samples)]
        return torch.stack(samples).mean(dim=0)


# ──────────────────────────────────────────────────────────────
# 2. Top-K sparsity with straight-through estimator
# ──────────────────────────────────────────────────────────────

class _TopKStraightThrough(torch.autograd.Function):
    """
    Top-K hard sparsity with straight-through gradient estimator (CDC §3.2).

    Forward  : keep K dims with largest |z|, zero the rest
    Backward : gradient passes as if no mask was applied (STE)
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, k: int):
        abs_z = z.abs()                                    # (B, D)
        topk_vals, topk_idx = torch.topk(abs_z, k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)                  # binary {0,1}
        ctx.save_for_backward(mask)
        return z * mask, mask

    @staticmethod
    def backward(ctx, grad_z_sparse, grad_mask):
        mask, = ctx.saved_tensors
        # STE: gradient flows only through active dims
        return grad_z_sparse * mask, None


def top_k_sparsity(z: torch.Tensor, k: int):
    """
    Apply Top-K hard sparsity to z.

    Args
        z : (B, latent_dim)
        k : number of active dimensions to keep

    Returns
        z_sparse : (B, latent_dim)  — k dims active, rest = 0
        mask     : (B, latent_dim)  — binary float {0.0, 1.0}
    """
    k = max(1, min(int(k), z.size(-1)))
    return _TopKStraightThrough.apply(z, k)


# ──────────────────────────────────────────────────────────────
# 3. KL divergence
# ──────────────────────────────────────────────────────────────

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Analytical KL between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I).

    KL = -0.5 * sum_i (1 + logvar_i - mu_i^2 - exp(logvar_i))

    Computed on the ORIGINAL z (before Top-K), as per CDC §3.3.

    Returns
        kl : scalar — mean over batch
    """
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )
    return kl_per_sample.mean()


# ──────────────────────────────────────────────────────────────
# 4. Full sparse latent layer (drop-in module)
# ──────────────────────────────────────────────────────────────

class SparseLatentLayer(nn.Module):
    """
    Complete sparse latent layer (CDC §3.5).

    Chains:
        reparametrize(mu, logvar)  → z
        top_k_sparsity(z, k)       → z_sparse, mask
        kl_divergence(mu, logvar)  → kl   (on z, not z_sparse)

    Args
        latent_dim : latent space dimension (default 64)
        k          : number of active dimensions (default 16)
        n_samples  : MC samples at inference (default 10)
    """

    def __init__(self, latent_dim: int = 64, k: int = 16, n_samples: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.k          = k
        self.n_samples  = n_samples

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Args
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)

        Returns
            z_sparse : (B, latent_dim)  sparse latent vector
            kl       : scalar           KL loss term
            mask     : (B, latent_dim)  binary Top-K mask
        """
        # 1. Sample z via reparametrization
        z = reparametrize(mu, logvar, training=self.training,
                          n_samples=self.n_samples)

        # 2. Apply Top-K sparsity  (STE in backward)
        z_sparse, mask = top_k_sparsity(z, self.k)

        # 3. KL on z (not z_sparse — CDC §3.3)
        kl = kl_divergence(mu, logvar)

        return z_sparse, kl, mask

    # ── LEA diagnostics (CDC §3.4) ────────────────────────────

    def active_ratio(self, mask: torch.Tensor) -> float:
        """Fraction of dimensions active on average. Target: ~0.25 (k/latent_dim)."""
        return mask.float().mean().item()

    def sparsity_rate(self, mask: torch.Tensor) -> float:
        """Fraction of dimensions zeroed."""
        return 1.0 - self.active_ratio(mask)

    def dead_dimensions(self, z_sparse: torch.Tensor,
                        threshold: float = 0.001) -> int:
        """Number of dimensions with variance below threshold across the batch."""
        return (z_sparse.var(dim=0) < threshold).sum().item()

    def lea_metrics(self, z_sparse: torch.Tensor,
                    mask: torch.Tensor) -> dict:
        """All LEA metrics in one call — log to W&B after each val epoch."""
        return {
            "latent/active_ratio":  self.active_ratio(mask),
            "latent/sparsity_rate": self.sparsity_rate(mask),
            "latent/dead_dims":     self.dead_dimensions(z_sparse),
        }
