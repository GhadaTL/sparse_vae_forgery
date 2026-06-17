import torch
import torch.nn as nn

from models.projection_head    import ProjectionHead
from models.sparse_latent      import SparseLatent
from models.multiscale_decoder import MultiScaleDecoder, prepare_multiscale_targets


def load_dinov2(model_name: str = "dinov2_vitb14") -> nn.Module:
    """Charge DINOv2 depuis torch.hub et le gèle complètement."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    return dinov2


def extract_dinov2_features(dinov2: nn.Module, image: torch.Tensor):
    """
    Extrait patch tokens depuis DINOv2.

    Args:
        image : (B, 3, 224, 224)

    Returns:
        patch_tokens : (B, 256, 768)
        cls_token    : (B, 768)
    """
    with torch.no_grad():
        out = dinov2.forward_features(image)
    patch_tokens = out["x_norm_patchtokens"]   # (B, 256, 768)
    cls_token    = out["x_norm_clstoken"]      # (B, 768)
    return patch_tokens, cls_token


class FullModel(nn.Module):
    """
    Pipeline complet :
        Image (B,3,224,224)
          → DINOv2 [frozen]       → patch_tokens (B, 256, 768)
          → ProjectionHead        → (mu, logvar)  (B, 64)
          → SparseLatent          → (z_sparse, kl, mask)
          → MultiScaleDecoder     → (x̂_c, x̂_m, x̂_f)

    Utilisé dans train.py avec K dynamique fourni par KController.
    """

    def __init__(self, config: dict):
        super().__init__()

        latent_dim   = config.get("latent_dim", 64)
        k_init       = config.get("k", 16)
        dropout      = config.get("dropout", 0.1)
        logvar_clamp = tuple(config.get("logvar_clamp", [-4.0, 4.0]))
        hidden_dims  = config.get("hidden_dims", [512, 256])

        # DINOv2 frozen
        self.dinov2 = load_dinov2()

        # Encodeur entraînable
        self.projection_head = ProjectionHead(
            input_dim    = 768,
            hidden_dims  = hidden_dims,
            latent_dim   = latent_dim,
            dropout      = dropout,
            logvar_clamp = logvar_clamp,
        )

        # Espace latent sparse (k dynamique passé en forward)
        self.sparse_latent = SparseLatent(latent_dim=latent_dim, k=k_init)

        # Décodeur entraînable
        self.decoder = MultiScaleDecoder(latent_dim=latent_dim)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, k: int = None):
        """
        Args:
            image : (B, 3, 224, 224)
            k     : Top-K actif (None → utilise k_default du SparseLatent)

        Returns:
            x_hat_c  : (B, 3, 16, 16)
            x_hat_m  : (B, 3, 32, 32)
            x_hat_f  : (B, 3, 64, 64)
            kl_loss  : scalaire
            z_sparse : (B, latent_dim)
            mask     : (B, latent_dim)
        """
        # 1. DINOv2 features (frozen)
        patch_tokens, _ = extract_dinov2_features(self.dinov2, image)

        # 2. Projection → (mu, logvar)
        mu, logvar = self.projection_head(patch_tokens)

        # 3. Reparameterization + Top-K + KL
        z_sparse, kl_loss, mask = self.sparse_latent(mu, logvar, k=k)

        # 4. Décodage multi-échelle
        x_hat_c, x_hat_m, x_hat_f = self.decoder(z_sparse)

        return x_hat_c, x_hat_m, x_hat_f, kl_loss, z_sparse, mask

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Retourne uniquement les paramètres entraînables (hors DINOv2)."""
        return (
            list(self.projection_head.parameters()) +
            list(self.sparse_latent.parameters())   +
            list(self.decoder.parameters())
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
