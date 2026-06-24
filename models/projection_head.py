"""
models/projection_head.py
Projection Head : tokens DINOv2 (B, 256, 768) → μ, log σ² (B, 64).

C'est la seule partie entraînable côté encodeur.
"""
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Comprime les 256 patch tokens DINOv2 vers les paramètres (μ, log σ²)
    de la distribution latente VAE.

    Architecture :
        Mean pooling → FC(768→512)+LN+GELU → Dropout → 
        FC(512→256)+LN+GELU → Dropout → Têtes μ et log σ²

    Args:
        input_dim   : dimension des tokens DINOv2 (768 pour ViT-B)
        hidden_dim1 : première couche cachée (512)
        hidden_dim2 : deuxième couche cachée (256)
        latent_dim  : dimension de l'espace latent (64)
        dropout     : taux de dropout (0.1)
        logvar_clamp: bornes pour le clamping de log σ² ([-4, 4])
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim1: int = 512,
                 hidden_dim2: int = 256,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 logvar_clamp: tuple = (-4.0, 4.0)):
        super().__init__()

        self.logvar_min, self.logvar_max = logvar_clamp

        # --- Backbone de projection ---
        self.encoder = nn.Sequential(
            # Couche 1 : 768 → 512
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # Couche 2 : 512 → 256
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Têtes séparées pour μ et log σ²
        self.fc_mu     = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour une convergence stable."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens : (B, 256, 768) — patch tokens DINOv2

        Returns:
            mu     : (B, 64) — moyenne de q(z|x)
            logvar : (B, 64) — log-variance de q(z|x)
        """
        # Agrégation spatiale : mean pooling sur les 256 patches → (B, 768)
        x = patch_tokens.mean(dim=1)

        # Projection vers l'espace intermédiaire
        h = self.encoder(x)   # (B, 256)

        # Paramètres de la distribution
        mu     = self.fc_mu(h)      # (B, 64) — ∈ ℝ, pas d'activation
        logvar = self.fc_logvar(h)  # (B, 64) — clampé pour stabilité numérique
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        return mu, logvar
