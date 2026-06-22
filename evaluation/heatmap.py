# evaluation/heatmap.py
# CDC §6 — Multi-Scale Anomaly Heatmap
# Localisation pixel-level des régions forgées sans supervision

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def compute_multiscale_heatmap(model, image, dinov2, projection_head, sparse_latent,
                                decoder, train_stats=None, device='cuda'):
    """
    Génère la heatmap d'anomalie pour une image.

    Args:
        model           : SparseVAE complet
        image           : (B, 3, 224, 224) ou (3, 224, 224)
        dinov2          : DINOv2 encoder (frozen)
        projection_head : Projection Head
        sparse_latent   : Sparse Latent Layer
        decoder         : Multi-Scale Decoder
        train_stats     : dict avec 'mean' et 'std' des heatmaps brutes sur val authentiques
        device          : 'cuda' ou 'cpu'

    Returns:
        H_smooth : numpy array (64, 64) — heatmap lissée normalisée
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    model.eval()

    with torch.no_grad():
        patch_tokens, cls_token = extract_dinov2_features(dinov2, image)
        mu, logvar = projection_head(patch_tokens)

        # ── CORRECTION : sparse_latent retourne 4 valeurs ────
        z_sparse, kl, kl_per_dim, mask = sparse_latent(mu, logvar)

        x_hat_c, x_hat_m, x_hat_f = decoder(z_sparse)

        x_c, x_m, x_f = prepare_multiscale_targets(image)

        E_c = ((x_c - x_hat_c)**2).mean(dim=1, keepdim=True)  # (B, 1, 16, 16)
        E_m = ((x_m - x_hat_m)**2).mean(dim=1, keepdim=True)  # (B, 1, 32, 32)
        E_f = ((x_f - x_hat_f)**2).mean(dim=1, keepdim=True)  # (B, 1, 64, 64)

        E_c_up = F.interpolate(E_c, size=(64, 64), mode='bilinear', align_corners=False)
        E_m_up = F.interpolate(E_m, size=(64, 64), mode='bilinear', align_corners=False)

        H = 0.2 * E_c_up + 0.3 * E_m_up + 0.5 * E_f  # (B, 1, 64, 64)

        if train_stats is not None:
            H_norm = (H - train_stats['mean']) / (train_stats['std'] + 1e-8)
        else:
            H_norm = H

        H_np = H_norm.squeeze().cpu().numpy()
        if H_np.ndim == 3:
            H_np = H_np[0]

        H_smooth = gaussian_filter(H_np, sigma=1.5)

    return H_smooth  # (64, 64)


def extract_dinov2_features(dinov2, image):
    with torch.no_grad():
        out = dinov2.forward_features(image)
    patch_tokens = out["x_norm_patchtokens"]
    cls_token    = out["x_norm_clstoken"]
    return patch_tokens, cls_token


def prepare_multiscale_targets(x: torch.Tensor):
    x_c = F.interpolate(x, (16, 16), mode="bilinear", align_corners=False)
    x_m = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
    x_f = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
    return x_c, x_m, x_f


def visualize_heatmap(original_image, heatmap, threshold_percentile=95):
    hm          = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    threshold   = np.percentile(hm, threshold_percentile)
    binary_mask = (hm > threshold).astype(np.uint8)
    hm_resized   = cv2.resize(hm,          (224, 224), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(binary_mask, (224, 224))
    return hm_resized, mask_resized


def plot_heatmap(original_image, heatmap, title="Anomaly Heatmap", save_path=None):
    hm_resized, _ = visualize_heatmap(original_image, heatmap)

    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))

    img_min, img_max = original_image.min(), original_image.max()
    img_norm   = (original_image - img_min) / (img_max - img_min + 1e-8)
    hm_colored = plt.cm.hot(hm_resized)[:, :, :3]
    overlay    = 0.7 * img_norm + 0.3 * hm_colored

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(img_norm);             axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(hm_resized, cmap='hot'); axes[1].set_title("Heatmap");      axes[1].axis('off')
    axes[2].imshow(overlay);              axes[2].set_title(title);            axes[2].axis('off')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_heatmap_stats(model, dinov2, projection_head, sparse_latent, decoder,
                           val_loader_authentic, device='cuda'):
    """
    Calcule mean et std des heatmaps brutes sur le val set authentique (CDC §6.4).
    """
    all_heatmaps = []
    model.eval()

    with torch.no_grad():
        for batch in val_loader_authentic:
            image = batch['image'] if isinstance(batch, dict) else batch[0]
            H = compute_multiscale_heatmap(
                model, image, dinov2, projection_head, sparse_latent, decoder,
                train_stats=None, device=device,
            )
            all_heatmaps.append(H)

    hm_stack = np.stack(all_heatmaps)   # (N, 64, 64)
    return {'mean': float(hm_stack.mean()), 'std': float(hm_stack.std())}


def compute_localization_metrics(heatmap_pred, mask_gt, threshold_percentile=95):
    """
    Calcule les métriques de localisation : IoU, Pixel-AUC, AP (CDC §6.5).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    hm        = (heatmap_pred - heatmap_pred.min()) / (heatmap_pred.max() - heatmap_pred.min() + 1e-8)
    threshold = np.percentile(hm, threshold_percentile)
    mask_pred = (hm > threshold).astype(np.uint8)

    hm_flat        = hm.flatten()
    mask_gt_flat   = mask_gt.flatten()
    mask_pred_flat = mask_pred.flatten()

    intersection = (mask_pred_flat & mask_gt_flat).sum()
    union        = (mask_pred_flat | mask_gt_flat).sum()
    iou          = float(intersection / (union + 1e-8))

    try:
        pixel_auc = float(roc_auc_score(mask_gt_flat, hm_flat))
    except Exception:
        pixel_auc = 0.0

    try:
        ap = float(average_precision_score(mask_gt_flat, hm_flat))
    except Exception:
        ap = 0.0

    return {'iou': iou, 'pixel_auc': pixel_auc, 'ap': ap}