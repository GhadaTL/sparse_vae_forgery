"""
models/dinov2_extractor.py
Extracteur de features DINOv2 ViT-B/14 — entièrement gelé.
Aucun fine-tuning autorisé.
"""
import torch
import torch.nn as nn


class DINOv2Extractor(nn.Module):
    """
    Wrapper autour de DINOv2 ViT-B/14.

    Sorties :
        patch_tokens : (B, 256, 768) — un vecteur 768-dim par patch 14×14px
        cls_token    : (B, 768)      — représentation globale de l'image

    Le modèle est chargé une seule fois et tous ses paramètres sont gelés.
    """

    def __init__(self, model_name: str = "dinov2_vitb14"):
        super().__init__()

        print(f"[DINOv2] Chargement de {model_name} via torch.hub...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
        )

        # Geler TOUS les paramètres — aucun gradient
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # Informations d'architecture
        self.feature_dim = 768   # ViT-B
        self.patch_size  = 14
        self.num_patches = 256   # (224/14)^2

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[DINOv2] Paramètres: {total:,} total, {trainable} entraînables (frozen)")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, 224, 224) — batch d'images normalisées

        Returns:
            patch_tokens : (B, 256, 768)
            cls_token    : (B, 768)
        """
        with torch.no_grad():
            # get_intermediate_layers retourne les tokens de la dernière couche
            # return_class_token=True pour récupérer le token [CLS]
            outputs = self.backbone.get_intermediate_layers(
                x,
                n=1,
                return_class_token=True,
            )
            # outputs est une liste de tuples : (patch_tokens, cls_token)
            patch_tokens, cls_token = outputs[0]

        # patch_tokens : (B, 256, 768) | cls_token : (B, 768)
        return patch_tokens, cls_token

    def train(self, mode: bool = True):
        """Override : DINOv2 reste toujours en mode eval."""
        super().train(mode)
        self.backbone.eval()  # Force eval
        return self
