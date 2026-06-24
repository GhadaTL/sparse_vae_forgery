"""
models/multiscale_decoder.py
Décodeur multi-échelle de style FPN (Feature Pyramid Network).

Produit 3 reconstructions :
  x_hat_coarse : (B, 3, 16, 16)
  x_hat_medium : (B, 3, 32, 32)
  x_hat_fine   : (B, 3, 64, 64)

Architecture :
  z_sparse (B,64) → Linear(64,512)+GELU → view(B,8,8,8)
    → ConvTranspose2d 8×8 → 16×16  (128ch) + skip → tête coarse
    → ConvTranspose2d 16×16 → 32×32 (64ch)  + skip → tête medium
    → ConvTranspose2d 32×32 → 64×64 (32ch)  + skip → tête fine
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc Conv2d + BatchNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Bloc d'upsampling : ConvTranspose2d + BatchNorm + GELU.
    Double la résolution spatiale.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up(x)


class OutputHead(nn.Module):
    """Tête de sortie : Conv1×1 + Sigmoid → (B, 3, H, W)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1),
            nn.Sigmoid(),   # Sortie ∈ [0,1]
        )

    def forward(self, x):
        return self.head(x)


class MultiscaleDecoder(nn.Module):
    """
    Décodeur pyramidal FPN-style.

    Args:
        latent_dim      : dimension de z_sparse (64)
        proj_dim        : dimension de la projection initiale (512)
        coarse_channels : canaux au niveau 1 (128)
        medium_channels : canaux au niveau 2 (64)
        fine_channels   : canaux au niveau 3 (32)
    """

    def __init__(self,
                 latent_dim: int = 64,
                 proj_dim: int = 512,
                 coarse_channels: int = 128,
                 medium_channels: int = 64,
                 fine_channels: int = 32):
        super().__init__()

        # --- Projection initiale : z → feature map 8×8×8 ---
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 8 * 8 * 8),  # 8 feature maps 8×8
            nn.GELU(),
        )

        # --- Niveau 1 : 8×8 → 16×16 (coarse) ---
        self.up1 = UpBlock(8, coarse_channels)            # (B, 128, 16, 16)
        self.conv1 = ConvBlock(coarse_channels, coarse_channels)
        self.skip1 = nn.Conv2d(coarse_channels, coarse_channels, 1)  # Skip connection
        self.head_coarse = OutputHead(coarse_channels)

        # --- Niveau 2 : 16×16 → 32×32 (medium) ---
        self.up2 = UpBlock(coarse_channels, medium_channels)  # (B, 64, 32, 32)
        self.conv2 = ConvBlock(medium_channels, medium_channels)
        self.skip2 = nn.Conv2d(medium_channels, medium_channels, 1)
        self.head_medium = OutputHead(medium_channels)

        # --- Niveau 3 : 32×32 → 64×64 (fine) ---
        self.up3 = UpBlock(medium_channels, fine_channels)  # (B, 32, 64, 64)
        self.conv3 = ConvBlock(fine_channels, fine_channels)
        self.head_fine = OutputHead(fine_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_sparse: torch.Tensor):
        """
        Args:
            z_sparse : (B, 64) — vecteur latent sparse

        Returns:
            x_hat_coarse : (B, 3, 16, 16)
            x_hat_medium : (B, 3, 32, 32)
            x_hat_fine   : (B, 3, 64, 64)
        """
        B = z_sparse.shape[0]

        # Projection et reshape : (B, 64) → (B, 8, 8, 8)
        h = self.proj(z_sparse)        # (B, 512)
        h = h.view(B, 8, 8, 8)        # (B, 8, 8, 8) — attend proj_out = 8*8*8=512
        # Correction : proj(latent_dim → 512 → 8*8*8 = 512) OK

        # --- Niveau 1 : coarse 16×16 ---
        h1 = self.up1(h)               # (B, 128, 16, 16)
        h1 = self.conv1(h1)
        skip_1 = self.skip1(h1)        # Skip pour niveau suivant
        x_hat_coarse = self.head_coarse(h1)  # (B, 3, 16, 16)

        # --- Niveau 2 : medium 32×32 ---
        h2 = self.up2(h1)              # (B, 64, 32, 32)
        h2 = self.conv2(h2)
        skip_2 = self.skip2(h2)
        x_hat_medium = self.head_medium(h2)  # (B, 3, 32, 32)

        # --- Niveau 3 : fine 64×64 ---
        h3 = self.up3(h2)              # (B, 32, 64, 64)
        h3 = self.conv3(h3)
        x_hat_fine = self.head_fine(h3)      # (B, 3, 64, 64)

        return x_hat_coarse, x_hat_medium, x_hat_fine
