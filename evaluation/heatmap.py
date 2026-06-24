"""
evaluation/heatmap.py
Génération et normalisation des heatmaps d'anomalie multi-échelle.

H(i,j) = 0.2×E_c + 0.3×E_m + 0.5×E_f  (toutes upscalées à 64×64)
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# =============================================================================
# Calcul de la heatmap brute
# =============================================================================

def compute_raw_heatmap(x_hat_coarse: torch.Tensor,
                        x_hat_medium: torch.Tensor,
                        x_hat_fine:   torch.Tensor,
                        x_coarse:     torch.Tensor,
                        x_medium:     torch.Tensor,
                        x_fine:       torch.Tensor,
                        heatmap_size: int = 64) -> torch.Tensor:
    """
    Calcule la heatmap d'erreur de reconstruction à 64×64.

    Args:
        x_hat_* : reconstructions du décodeur
        x_*     : cibles correspondantes

    Returns:
        H : (B, 1, 64, 64) — heatmap brute (avant normalisation)
    """
    # Erreur pixelwise par échelle : moyenne sur les 3 canaux RGB → (B,1,H,W)
    E_c = ((x_coarse - x_hat_coarse) ** 2).mean(dim=1, keepdim=True)  # (B,1,16,16)
    E_m = ((x_medium - x_hat_medium) ** 2).mean(dim=1, keepdim=True)  # (B,1,32,32)
    E_f = ((x_fine   - x_hat_fine)   ** 2).mean(dim=1, keepdim=True)  # (B,1,64,64)

    # Upscaling des erreurs grossières vers 64×64
    E_c_up = F.interpolate(E_c, size=(heatmap_size, heatmap_size),
                           mode='bilinear', align_corners=False)
    E_m_up = F.interpolate(E_m, size=(heatmap_size, heatmap_size),
                           mode='bilinear', align_corners=False)

    # Fusion pondérée : fine est dominant
    H = 0.2 * E_c_up + 0.3 * E_m_up + 0.5 * E_f  # (B, 1, 64, 64)
    return H


# =============================================================================
# Statistiques de normalisation (calibrées sur le val authentique)
# =============================================================================

def compute_heatmap_stats(model,
                          val_loader,
                          device: torch.device,
                          heatmap_size: int = 64) -> dict:
    """
    Calcule mean et std des heatmaps brutes sur le val set authentique.
    Ces stats servent à normaliser (z-score) les heatmaps à l'évaluation.

    Args:
        model      : SparseVAE en mode eval
        val_loader : DataLoader MIDV-2020 val (authentiques uniquement)
        device     : torch.device

    Returns:
        dict {'mean': float, 'std': float}
    """
    from utils.helpers import prepare_multiscale_targets

    model.eval()
    all_vals = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            x_c, x_m, x_f = prepare_multiscale_targets(images, device)
            H = compute_raw_heatmap(
                outputs["x_hat_coarse"], outputs["x_hat_medium"], outputs["x_hat_fine"],
                x_c, x_m, x_f, heatmap_size
            )
            all_vals.append(H.cpu().numpy().ravel())

    all_vals = np.concatenate(all_vals)
    stats = {"mean": float(all_vals.mean()), "std": float(all_vals.std() + 1e-8)}
    print(f"[Heatmap Stats] mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    return stats


# =============================================================================
# Heatmap normalisée + lissée (pour une image)
# =============================================================================

def compute_multiscale_heatmap(model,
                               image: torch.Tensor,
                               train_stats: dict,
                               device: torch.device,
                               sigma: float = 1.5,
                               heatmap_size: int = 64) -> np.ndarray:
    """
    Génère la heatmap normalisée et lissée pour une image.

    Args:
        model       : SparseVAE
        image       : (1, 3, 224, 224) ou (B, 3, 224, 224)
        train_stats : {'mean': float, 'std': float} du val authentique
        sigma       : sigma du lissage gaussien

    Returns:
        H_smooth : numpy array (64, 64) ou (B, 64, 64)
    """
    from utils.helpers import prepare_multiscale_targets

    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        x_c, x_m, x_f = prepare_multiscale_targets(image, device)

        H = compute_raw_heatmap(
            outputs["x_hat_coarse"], outputs["x_hat_medium"], outputs["x_hat_fine"],
            x_c, x_m, x_f, heatmap_size
        )  # (B, 1, 64, 64)

    # Normalisation z-score par les stats du val authentique
    if train_stats is not None:
        H_norm = (H - train_stats["mean"]) / (train_stats["std"] + 1e-8)
    else:
        H_norm = H

    H_np = H_norm.squeeze(1).cpu().numpy()  # (B, 64, 64) ou (64, 64)

    # Lissage gaussien sur chaque image du batch
    if H_np.ndim == 3:
        H_smooth = np.stack([gaussian_filter(h, sigma=sigma) for h in H_np])
    else:
        H_smooth = gaussian_filter(H_np, sigma=sigma)

    return H_smooth


# =============================================================================
# Visualisation (optionnelle)
# =============================================================================

def visualize_heatmap(original_image: np.ndarray,
                      heatmap: np.ndarray,
                      threshold_percentile: int = 95,
                      save_path: str = None):
    """
    Superpose la heatmap sur l'image originale avec colormap.

    Args:
        original_image      : (H, W, 3) uint8
        heatmap             : (64, 64) float — heatmap normalisée
        threshold_percentile: seuil binaire (percentile)
        save_path           : chemin de sauvegarde (ou None pour afficher)
    """
    import cv2
    import matplotlib.pyplot as plt

    # Normaliser heatmap en [0, 1]
    hm = heatmap.copy().astype(np.float32)
    hm_min, hm_max = hm.min(), hm.max()
    hm = (hm - hm_min) / (hm_max - hm_min + 1e-8)

    # Seuillage binaire
    threshold = np.percentile(hm, threshold_percentile)
    binary_mask = (hm > threshold).astype(np.uint8)

    # Upscale heatmap vers résolution originale
    H, W = original_image.shape[:2]
    hm_resized   = cv2.resize(hm,          (W, H), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Colormap JET pour la heatmap
    hm_colored = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay    = cv2.addWeighted(original_image, 0.6, hm_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_image);           axes[0].set_title("Original")
    axes[1].imshow(hm_resized, cmap="jet");   axes[1].set_title("Heatmap")
    axes[2].imshow(mask_resized, cmap="gray"); axes[2].set_title("Masque binaire")
    axes[3].imshow(overlay[:, :, ::-1]);       axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return hm_resized, mask_resized
