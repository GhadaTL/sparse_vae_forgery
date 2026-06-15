import torch
import torch.nn as nn

from models.projection_head import ProjectionHead
from models.sparse_latent import SparseLatentLayer
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
    Extrait patch tokens et CLS token depuis DINOv2.

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


class SparseVAE(nn.Module):
    """
    Pipeline complet :
        Image → DINOv2 → ProjectionHead → SparseLatentLayer → MultiScaleDecoder

    DINOv2 est frozen (non entraîné). Bien qu'inclus dans le module, l'optimiseur
    n'optimise que ProjectionHead et Decoder (voir train.py).
    """

    def __init__(self, latent_dim: int = 64, k: int = 16,
                 hidden_dims: list = None, dropout: float = 0.1,
                 logvar_clamp: tuple = (-4.0, 4.0)):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # DINOv2 frozen
        self.dinov2 = load_dinov2()
        
        self.projection_head = ProjectionHead(
            input_dim=768,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
        )
        self.sparse_latent = SparseLatentLayer(latent_dim=latent_dim, k=k)
        self.decoder = MultiScaleDecoder(latent_dim=latent_dim)

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens : (B, 256, 768) — sortie DINOv2

        Returns:
            x_hat_c  : (B, 3, 16, 16)
            x_hat_m  : (B, 3, 32, 32)
            x_hat_f  : (B, 3, 64, 64)
            kl_loss  : scalaire
            mu       : (B, latent_dim)
            logvar   : (B, latent_dim)
            z_sparse : (B, latent_dim)
            mask     : (B, latent_dim)
        """
        # Encodeur
        mu, logvar = self.projection_head(patch_tokens)
        z_sparse, kl_loss, mask = self.sparse_latent(mu, logvar)

        # Décodeur
        x_hat_c, x_hat_m, x_hat_f = self.decoder(z_sparse)

        return x_hat_c, x_hat_m, x_hat_f, kl_loss, mu, logvar, z_sparse, mask

    def encode(self, patch_tokens: torch.Tensor):
        """Encodage seul — utile pour l'inférence / visualisation."""
        mu, logvar = self.projection_head(patch_tokens)
        z_sparse, kl_loss, mask = self.sparse_latent(mu, logvar)
        return mu, logvar, z_sparse, mask

    def decode(self, z_sparse: torch.Tensor):
        """Décodage seul — utile pour générer des images depuis z."""
        return self.decoder(z_sparse)

    # ------------------------------------------------------------------ #
    #  Utilitaires                                                         #
    # ------------------------------------------------------------------ #

    def trainable_parameters(self):
        """Retourne uniquement les paramètres entraînables (hors DINOv2)."""
        return list(self.projection_head.parameters()) + \
               list(self.sparse_latent.parameters()) + \
               list(self.decoder.parameters())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)