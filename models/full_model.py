import torch
import torch.nn as nn
import torchvision

class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # =========================
        # 1. DINOv2 backbone (frozen)
        # =========================
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',#official repository for DINOv2
            'dinov2_vitb14'#the specefic name of the model
        )

        for param in self.backbone.parameters():
            param.requires_grad = False  # IMPORTANT : freeze the backbone weights

        # =========================
        # 2. Modules du VAE
        # =========================
        self.projection = ProjectionHead(config)
        self.encoder = SparseLatent(config)
        self.decoder = MultiScaleDecoder(config)

    def forward(self, x):
        # 1. Features DINOv2
        features = self.backbone(x)  # (B, tokens, dim)

        # 2. Projection
        proj = self.projection(features)

        # 3. Latent sparse
        z, mu, logvar = self.encoder(proj)

        # 4. Reconstruction
        recon = self.decoder(z)

        return recon, mu, logvar