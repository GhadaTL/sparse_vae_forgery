"""
models/full_model.py
====================
Full pipeline assembly:
    image → DINOv2 → ProjectionHead → SparseLatentLayer → MultiScaleDecoder

DINOv2 is frozen (no gradients). Only ProjectionHead, SparseLatentLayer,
and MultiScaleDecoder are trained.
"""

import torch
import torch.nn as nn

from models.projection_head   import ProjectionHead
from models.sparse_latent     import SparseLatentLayer          # fixed import
from models.multiscale_decoder import MultiScaleDecoder, prepare_multiscale_targets


# ──────────────────────────────────────────────────────────────
# DINOv2 loader
# ──────────────────────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitb14") -> nn.Module:
    """Load DINOv2 from torch.hub and freeze all parameters."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    return dinov2


@torch.no_grad()
def extract_dinov2_features(dinov2: nn.Module, image: torch.Tensor):
    """
    Extract patch tokens and CLS token from a frozen DINOv2.

    Args
        image : (B, 3, 224, 224) — ImageNet-normalized

    Returns
        patch_tokens : (B, 256, 768)
        cls_token    : (B, 768)
    """
    out          = dinov2.forward_features(image)
    patch_tokens = out["x_norm_patchtokens"]    # (B, 256, 768)
    cls_token    = out["x_norm_clstoken"]       # (B, 768)
    return patch_tokens, cls_token


# ──────────────────────────────────────────────────────────────
# SparseVAE  (was called "FullModel" in broken train.py)
# ──────────────────────────────────────────────────────────────

class SparseVAE(nn.Module):
    """
    Sparse VAE for identity document forgery detection (CDC §architecture).

    Forward input  : raw image (B, 3, 224, 224)
    Forward output : dict with all tensors needed for loss + evaluation

    Only ProjectionHead, SparseLatentLayer, and MultiScaleDecoder
    receive gradients. DINOv2 is always in eval() mode.
    """

    def __init__(
        self,
        latent_dim:   int   = 64,
        k:            int   = 16,
        hidden_dims:  list  = None,
        dropout:      float = 0.1,
        logvar_clamp: tuple = (-4.0, 4.0),
        dinov2_model: str   = "dinov2_vitb14",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim
        self.k          = k

        # ── frozen backbone ────────────────────────────────────
        self.dinov2 = load_dinov2(dinov2_model)

        # ── trainable encoder ──────────────────────────────────
        self.projection_head = ProjectionHead(
            input_dim    = 768,
            hidden_dims  = hidden_dims,
            latent_dim   = latent_dim,
            dropout      = dropout,
            logvar_clamp = logvar_clamp,
        )
        self.sparse_latent = SparseLatentLayer(
            latent_dim = latent_dim,
            k          = k,
        )

        # ── trainable decoder ──────────────────────────────────
        self.decoder = MultiScaleDecoder(latent_dim=latent_dim)

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(self, image: torch.Tensor) -> dict:
        """
        Full forward pass from raw image to reconstructions.

        Args
            image : (B, 3, 224, 224)

        Returns dict with keys:
            mu, logvar, z_sparse, mask, kl
            x_hat_coarse, x_hat_medium, x_hat_fine
            x_coarse, x_medium, x_fine   (downsampled targets for loss)
            cls_token                     (for monitoring / t-SNE)
        """
        # ① DINOv2 feature extraction (no grad)
        patch_tokens, cls_token = extract_dinov2_features(self.dinov2, image)

        # ② Projection Head → mu, logvar
        mu, logvar = self.projection_head(patch_tokens)

        # ③ Sparse Latent → z_sparse, kl, mask
        z_sparse, kl, mask = self.sparse_latent(mu, logvar)

        # ④ Multi-Scale Decoder → 3 reconstructions
        x_hat_c, x_hat_m, x_hat_f = self.decoder(z_sparse)

        # Multi-scale targets (downsampled from input)
        x_c, x_m, x_f = prepare_multiscale_targets(image)

        return {
            # Latent
            "mu":        mu,
            "logvar":    logvar,
            "z_sparse":  z_sparse,
            "mask":      mask,
            "kl":        kl,
            # Reconstructions
            "x_hat_coarse": x_hat_c,
            "x_hat_medium": x_hat_m,
            "x_hat_fine":   x_hat_f,
            # Targets (for loss computation)
            "x_coarse": x_c,
            "x_medium": x_m,
            "x_fine":   x_f,
            # Global feature (monitoring)
            "cls_token": cls_token,
        }

    # ──────────────────────────────────────────────────────────
    # Convenience: encode-only / decode-only
    # ──────────────────────────────────────────────────────────

    def encode(self, image: torch.Tensor):
        """
        Encode an image to sparse latent representation.
        Useful for inference and visualisation.

        Returns
            mu, logvar, z_sparse, mask
        """
        patch_tokens, _ = extract_dinov2_features(self.dinov2, image)
        mu, logvar      = self.projection_head(patch_tokens)
        z_sparse, kl, mask = self.sparse_latent(mu, logvar)
        return mu, logvar, z_sparse, mask

    def decode(self, z_sparse: torch.Tensor):
        """
        Decode a sparse latent vector to multi-scale reconstructions.

        Returns
            x_hat_coarse, x_hat_medium, x_hat_fine
        """
        return self.decoder(z_sparse)

    # ──────────────────────────────────────────────────────────
    # Train / eval mode helpers
    # ──────────────────────────────────────────────────────────

    def set_train_mode(self):
        """Train mode: DINOv2 stays in eval (frozen BN stats)."""
        self.train()
        self.dinov2.eval()

    def set_eval_mode(self):
        self.eval()

    # ──────────────────────────────────────────────────────────
    # Parameter utilities
    # ──────────────────────────────────────────────────────────

    def trainable_parameters(self):
        """Only ProjectionHead + SparseLatentLayer + Decoder — not DINOv2."""
        return (
            list(self.projection_head.parameters()) +
            list(self.sparse_latent.parameters()) +
            list(self.decoder.parameters())
        )

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


# Alias so any code that still references "FullModel" does not crash
FullModel = SparseVAE
