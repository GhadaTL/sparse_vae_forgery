"""
models/projection_head.py
=========================
Étape ② — Projection Head
Compresse les patch tokens DINOv2 (B, 256, 768) vers μ et log σ² (B, 64).
Seule partie entraînable côté encodeur.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Encodeur VAE : patch tokens DINOv2 → (μ, log σ²).

    Architecture :
        Mean pool → FC(768,512)+LN+GELU → Dropout →
        FC(512,256)+LN+GELU → Dropout →
        Tête μ : FC(256,64)
        Tête log σ² : FC(256,64) + Clamp(-4,4)
    """

    def __init__(self, input_dim: int = 768, hidden_dims: list = None,
                 latent_dim: int = 64, dropout: float = 0.1,
                 logvar_clamp: tuple = (-4.0, 4.0)):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.logvar_clamp = logvar_clamp

        # Trunk partagé
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Têtes de sortie
        self.mu_head = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens : (B, 256, 768) — sortie DINOv2 sans CLS token

        Returns:
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)  — log σ², clampé dans [-4, 4]
        """
        # Mean pooling spatial : (B, 256, 768) → (B, 768)
        x = patch_tokens.mean(dim=1)

        # Trunk
        h = self.trunk(x)

        # Têtes
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(*self.logvar_clamp)

        return mu, logvar