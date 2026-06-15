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
        model : SparseVAE complet (alternatif : passer les composants séparés)
        image : (B, 3, 224, 224) ou (3, 224, 224)
        dinov2 : DINOv2 encoder (frozen)
        projection_head : Projection Head
        sparse_latent : Sparse Latent Layer
        decoder : Multi-Scale Decoder
        train_stats : dict avec 'mean' et 'std' des heatmaps brutes sur val authentiques
        device : 'cuda' ou 'cpu'
    
    Returns:
        H_smooth : numpy array (64, 64) — heatmap lissée normalisée
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, 3, 224, 224)
    
    image = image.to(device)
    model.eval()
    
    with torch.no_grad():
        # ── Forward pass complet ──
        patch_tokens, cls_token = extract_dinov2_features(dinov2, image)
        mu, logvar = projection_head(patch_tokens)
        z_sparse, kl, mask = sparse_latent(mu, logvar)
        x_hat_c, x_hat_m, x_hat_f = decoder(z_sparse)
        
        # ── Cibles multi-échelle ──
        x_c, x_m, x_f = prepare_multiscale_targets(image)
        
        # ── Erreurs pixelwise : mean sur canaux RGB → (B, 1, H, W) ──
        E_c = ((x_c - x_hat_c)**2).mean(dim=1, keepdim=True)  # (B, 1, 16, 16)
        E_m = ((x_m - x_hat_m)**2).mean(dim=1, keepdim=True)  # (B, 1, 32, 32)
        E_f = ((x_f - x_hat_f)**2).mean(dim=1, keepdim=True)  # (B, 1, 64, 64)
        
        # ── Upscaling des erreurs grossières vers 64×64 ──
        E_c_up = F.interpolate(E_c, size=(64, 64), mode='bilinear', align_corners=False)
        E_m_up = F.interpolate(E_m, size=(64, 64), mode='bilinear', align_corners=False)
        
        # ── Fusion pondérée (CDC §6.2) ──
        H = 0.2 * E_c_up + 0.3 * E_m_up + 0.5 * E_f  # (B, 1, 64, 64)
        
        # ── Normalisation par les stats du val authentique ──
        if train_stats is not None:
            H_norm = (H - train_stats['mean']) / (train_stats['std'] + 1e-8)
        else:
            H_norm = H
        
        # ── Lissage gaussien pour réduire le bruit ──
        H_np = H_norm.squeeze().cpu().numpy()  # (64, 64) ou (B, 64, 64)
        
        if H_np.ndim == 3:  # Batch
            H_np = H_np[0]
        
        H_smooth = gaussian_filter(H_np, sigma=1.5)
    
    return H_smooth  # numpy array (64, 64)


def extract_dinov2_features(dinov2, image):
    """
    Extrait patch tokens et CLS token depuis DINOv2.
    
    Args:
        image : (B, 3, 224, 224)
    
    Returns:
        patch_tokens : (B, 256, 768)
        cls_token : (B, 768)
    """
    with torch.no_grad():
        out = dinov2.forward_features(image)
    patch_tokens = out["x_norm_patchtokens"]   # (B, 256, 768)
    cls_token = out["x_norm_clstoken"]         # (B, 768)
    return patch_tokens, cls_token


def prepare_multiscale_targets(x: torch.Tensor):
    """
    Redimensionne x aux 3 résolutions cibles (CDC §4.4).
    
    Args:
        x : (B, 3, 224, 224)
    
    Returns:
        x_c, x_m, x_f : (B, 3, 16, 16), (B, 3, 32, 32), (B, 3, 64, 64)
    """
    x_c = F.interpolate(x, (16, 16), mode="bilinear", align_corners=False)
    x_m = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
    x_f = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
    return x_c, x_m, x_f


def visualize_heatmap(original_image, heatmap, threshold_percentile=95):
    """
    Superpose la heatmap sur l'image originale.
    
    Args:
        original_image : (3, 224, 224) ou (224, 224, 3) numpy/tensor
        heatmap : (64, 64) numpy array
        threshold_percentile : percentile pour seuillage binaire
    
    Returns:
        hm_resized : (224, 224) heatmap redimensionnée
        mask_resized : (224, 224) masque binaire
    """
    # Normaliser heatmap en [0,1] pour visualisation
    hm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Seuillage binaire
    threshold = np.percentile(hm, threshold_percentile)
    binary_mask = (hm > threshold).astype(np.uint8)
    
    # Redimensionner vers résolution originale (224×224)
    hm_resized = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(binary_mask, (224, 224))
    
    return hm_resized, mask_resized


def plot_heatmap(original_image, heatmap, title="Anomaly Heatmap", save_path=None):
    """
    Visualise l'image originale avec la heatmap superposée.
    
    Args:
        original_image : (3, 224, 224) ou (224, 224, 3)
        heatmap : (64, 64) ou (224, 224)
        title : titre du graphique
        save_path : chemin de sauvegarde optionnel
    """
    hm_resized, mask_resized = visualize_heatmap(original_image, heatmap)
    
    # Convertir image si nécessaire
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    
    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))
    
    # Normaliser image
    img_min, img_max = original_image.min(), original_image.max()
    img_norm = (original_image - img_min) / (img_max - img_min + 1e-8)
    
    # Colormap pour heatmap
    hm_colored = plt.cm.hot(hm_resized)[:, :, :3]
    
    # Superposition
    overlay = 0.7 * img_norm + 0.3 * hm_colored
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(img_norm)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(hm_resized, cmap='hot')
    axes[1].set_title("Heatmap")
    axes[1].colorbar()
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compute_heatmap_stats(model, dinov2, projection_head, sparse_latent, decoder,
                          val_loader_authentic, device='cuda'):
    """
    Calcule mean et std des heatmaps brutes sur le val set authentique (CDC §6.4).
    Ces stats servent à normaliser les heatmaps lors de l'inférence.
    
    Args:
        model : SparseVAE complet
        dinov2, projection_head, sparse_latent, decoder : composants
        val_loader_authentic : DataLoader sur documents authentiques
        device : 'cuda' ou 'cpu'
    
    Returns:
        dict avec 'mean' et 'std'
    """
    all_heatmaps = []
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader_authentic:
            # batch peut être un dict {'image': ...} ou un tuple (image, label)
            if isinstance(batch, dict):
                image = batch['image']
            else:
                image = batch[0]
            
            H = compute_multiscale_heatmap(
                model, image, dinov2, projection_head, sparse_latent, decoder,
                train_stats=None, device=device
            )
            all_heatmaps.append(H)
    
    hm_stack = np.stack(all_heatmaps)  # (N, 64, 64)
    
    return {
        'mean': hm_stack.mean(),
        'std': hm_stack.std()
    }


def compute_localization_metrics(heatmap_pred, mask_gt, threshold_percentile=95):
    """
    Calcule les métriques de localisation : IoU, Pixel-AUC, AP (CDC §6.5).
    
    Args:
        heatmap_pred : (64, 64) heatmap normalisée
        mask_gt : (64, 64) masque ground-truth binaire (1=forgé, 0=authentique)
        threshold_percentile : percentile pour seuillage
    
    Returns:
        dict avec métriques : iou, pixel_auc, ap
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    
    # Normaliser heatmap
    hm = (heatmap_pred - heatmap_pred.min()) / (heatmap_pred.max() - heatmap_pred.min() + 1e-8)
    
    # Seuillage binaire
    threshold = np.percentile(hm, threshold_percentile)
    mask_pred = (hm > threshold).astype(np.uint8)
    
    # Aplatir
    hm_flat = hm.flatten()
    mask_gt_flat = mask_gt.flatten()
    mask_pred_flat = mask_pred.flatten()
    
    # IoU
    intersection = (mask_pred_flat & mask_gt_flat).sum()
    union = (mask_pred_flat | mask_gt_flat).sum()
    iou = intersection / (union + 1e-8)
    
    # Pixel-level AUC-ROC
    try:
        pixel_auc = roc_auc_score(mask_gt_flat, hm_flat)
    except:
        pixel_auc = 0.0
    
    # Average Precision
    try:
        ap = average_precision_score(mask_gt_flat, hm_flat)
    except:
        ap = 0.0
    
    return {
        'iou': iou,
        'pixel_auc': pixel_auc,
        'ap': ap
    }
