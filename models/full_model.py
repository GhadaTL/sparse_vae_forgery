"""
models/full_model.py
Modèle complet : DINOv2 → ProjectionHead → SparseLatentLayer → MultiscaleDecoder.
"""
import torch
import torch.nn as nn

from models.dinov2_extractor import DINOv2Extractor
from models.projection_head import ProjectionHead
from models.sparse_latent import SparseLatentLayer, AdaptiveKController
from models.multiscale_decoder import MultiscaleDecoder


class SparseVAE(nn.Module):
    """
    Pipeline complet de détection de falsification par détection d'anomalies.

    Flux :
        Image (B,3,224,224)
        → DINOv2 [frozen] → patch_tokens (B,256,768)
        → ProjectionHead → μ, log σ² (B,64)
        → SparseLatentLayer → z_sparse (B,64) + KL
        → MultiscaleDecoder → x̂_coarse, x̂_medium, x̂_fine
    """

    def __init__(self, cfg):
        super().__init__()

        # --- Composants ---
        self.dinov2 = DINOv2Extractor(
            model_name=cfg.dinov2.model
        )
        self.projection = ProjectionHead(
            input_dim=cfg.dinov2.feature_dim,
            hidden_dim1=cfg.projection.hidden_dim_1,
            hidden_dim2=cfg.projection.hidden_dim_2,
            latent_dim=cfg.projection.latent_dim,
            dropout=cfg.projection.dropout,
            logvar_clamp=(cfg.projection.logvar_clamp_min,
                          cfg.projection.logvar_clamp_max),
        )
        self.sparse_latent = SparseLatentLayer(
            latent_dim=cfg.sparse.latent_dim,
            K_init=cfg.sparse.K_init,
        )
        self.decoder = MultiscaleDecoder(
            latent_dim=cfg.sparse.latent_dim,
            proj_dim=cfg.decoder.proj_dim,
            coarse_channels=cfg.decoder.coarse_channels,
            medium_channels=cfg.decoder.medium_channels,
            fine_channels=cfg.decoder.fine_channels,
        )

        # Contrôleur adaptatif K (non-module PyTorch, pas de paramètres)
        self.k_controller = AdaptiveKController(
            K_init=cfg.sparse.K_init,
            K_min=cfg.sparse.K_min,
            K_max=cfg.sparse.K_max,
            tau_fraction=cfg.sparse.tau_fraction,
            tau_min=cfg.sparse.tau_min,
            recon_high=cfg.sparse.recon_loss_high,
            recon_low=cfg.sparse.recon_loss_low,
            recon_step=cfg.sparse.recon_step,
            ema_alpha=cfg.sparse.ema_alpha,
        )

        # Résumé des paramètres entraînables
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SparseVAE] Paramètres entraînables : {trainable:,}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, 224, 224)

        Returns:
            dict avec toutes les sorties nécessaires à la loss et aux diagnostics
        """
        # 1. Features DINOv2 (frozen)
        patch_tokens, cls_token = self.dinov2(x)

        # 2. Projection → μ, log σ²
        mu, logvar = self.projection(patch_tokens)

        # 3. Sparse latent : z, KL, masque
        z_sparse, kl_loss, kl_per_dim, mask = self.sparse_latent(mu, logvar)

        # 4. Reconstruction multi-échelle
        x_hat_coarse, x_hat_medium, x_hat_fine = self.decoder(z_sparse)

        return {
            "x_hat_coarse": x_hat_coarse,
            "x_hat_medium": x_hat_medium,
            "x_hat_fine":   x_hat_fine,
            "mu":           mu,
            "logvar":       logvar,
            "z_sparse":     z_sparse,
            "mask":         mask,
            "kl_loss":      kl_loss,
            "kl_per_dim":   kl_per_dim,
        }

    def update_k(self, kl_per_dim: torch.Tensor, recon_loss: float) -> int:
        """Délègue la mise à jour de K au contrôleur adaptatif."""
        new_k = self.k_controller.update(kl_per_dim, recon_loss)
        self.sparse_latent.k = new_k
        return new_k

    @property
    def current_k(self) -> int:
        return self.sparse_latent.k

    def get_trainable_params(self):
        """Retourne uniquement les paramètres entraînables (pas DINOv2)."""
        return [p for p in self.parameters() if p.requires_grad]
