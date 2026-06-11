"""
models/multiscale_decoder.py
============================
Étape ④ — Multi-Scale Decoder (FPN-style)
Reconstruit l'image à 3 résolutions : 16×16, 32×32, 64×64 px.
Aligné sur les tailles de forgeries copy-move de FMIDV-2022.
"""

import torch
import torch.nn as nn


class MultiScaleDecoder(nn.Module):
    """
    Décodeur pyramidal FPN.

    z_sparse (B,64) → x̂_coarse (B,3,16,16)
                    → x̂_medium (B,3,32,32)
                    → x̂_fine   (B,3,64,64)

    Architecture :
        Projection : Linear(64,512) → view(B,8,8,8)
        Niveau 1   : ConvTranspose(8→128, 4×4, s=2)  → (B,128,16,16) + BN + GELU
        Tête C     : Conv(128→3, 1×1) + Sigmoid       → x̂_coarse
        Niveau 2   : ConvTranspose(128→64, 4×4, s=2) → (B,64,32,32) + BN + GELU
        Tête M     : Conv(64→3, 1×1) + Sigmoid        → x̂_medium
        Niveau 3   : ConvTranspose(64→32, 4×4, s=2)  → (B,32,64,64) + BN + GELU
        Tête F     : Conv(32→3, 1×1) + Sigmoid        → x̂_fine
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Projection initiale : vecteur → feature map spatiale
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
        )
        # 512 = 8 channels × 8 × 8 spatial

        # ── Niveau 1 → 16×16 ──────────────────────────────────────────
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(8, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.head_coarse = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── Niveau 2 → 32×32 ──────────────────────────────────────────
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.head_medium = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── Niveau 3 → 64×64 ──────────────────────────────────────────
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.head_fine = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z_sparse: torch.Tensor):
        """
        Args:
            z_sparse : (B, latent_dim)

        Returns:
            x_hat_coarse : (B, 3, 16, 16)
            x_hat_medium : (B, 3, 32, 32)
            x_hat_fine   : (B, 3, 64, 64)
        """
        B = z_sparse.size(0)

        # Projection + reshape → (B, 8, 8, 8)
        h = self.proj(z_sparse)              # (B, 512)
        h = h.view(B, 8, 8, 8)              # (B, 8, 8, 8)

        # Niveau 1 : → (B, 128, 16, 16)
        f1 = self.up1(h)
        x_hat_coarse = self.head_coarse(f1)  # (B, 3, 16, 16)

        # Niveau 2 : skip connection depuis f1
        f2 = self.up2(f1)                    # (B, 64, 32, 32)
        x_hat_medium = self.head_medium(f2)  # (B, 3, 32, 32)

        # Niveau 3 : skip connection depuis f2
        f3 = self.up3(f2)                    # (B, 32, 64, 64)
        x_hat_fine = self.head_fine(f3)      # (B, 3, 64, 64)

        return x_hat_coarse, x_hat_medium, x_hat_fine


def prepare_multiscale_targets(image: torch.Tensor):
    """
    Redimensionne l'image originale (B,3,224,224) aux 3 résolutions cibles.

    Returns:
        x_c : (B, 3, 16, 16)
        x_m : (B, 3, 32, 32)
        x_f : (B, 3, 64, 64)
    """
    import torch.nn.functional as F
    x_c = F.interpolate(image, size=(16, 16), mode='bilinear', align_corners=False)
    x_m = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
    x_f = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
    return x_c, x_m, x_f