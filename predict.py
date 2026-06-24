"""
predict.py
Test sur une seule image : décision AUTHENTIQUE / FORGÉ + heatmap de localisation.

Usage Colab :
    %cd /content/drive/MyDrive/sparse_vae_forgery
    !python predict.py --image /chemin/vers/image.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

# ── Résolution des chemins (compatible Colab + local) ──────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# ───────────────────────────────────────────────────────────────────────────

from utils.helpers import set_seed, load_config, load_checkpoint, prepare_multiscale_targets
from data.datasets import get_eval_transform, get_midv2020_loaders
from models.full_model import SparseVAE
from evaluation.anomaly_scorer import AnomalyScorer
from evaluation.heatmap import compute_heatmap_stats


# =============================================================================
# Chargement et préparation d'une image
# =============================================================================

def load_image(image_path: str, image_size: int = 224):
    """Charge une image et retourne le tensor (1, 3, H, W) + l'image PIL originale."""
    transform = get_eval_transform(image_size)
    pil_img   = Image.open(image_path).convert("RGB")
    tensor    = transform(pil_img).unsqueeze(0)   # (1, 3, 224, 224)
    return tensor, pil_img


# =============================================================================
# Calcul de la heatmap pour une image
# =============================================================================

def compute_single_heatmap(model, image_tensor, device,
                            heatmap_stats, sigma=1.5, heatmap_size=64):
    """
    Génère la heatmap d'anomalie normalisée et lissée pour une image.

    Returns:
        H_smooth   : (64, 64) numpy — heatmap lissée
        recon_fine : (3, 64, 64) numpy — reconstruction fine pour visualisation
        outputs    : dict des sorties du modèle
    """
    model.eval()
    with torch.no_grad():
        img = image_tensor.to(device)
        outputs = model(img)
        _, _, x_f = prepare_multiscale_targets(img, device)
        x_c = F.interpolate(img, size=(16, 16), mode='bilinear', align_corners=False)
        x_m = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)

        # Erreurs par échelle → (1, 1, H, W)
        E_c = ((x_c - outputs["x_hat_coarse"]) ** 2).mean(dim=1, keepdim=True)
        E_m = ((x_m - outputs["x_hat_medium"]) ** 2).mean(dim=1, keepdim=True)
        E_f = ((x_f - outputs["x_hat_fine"])   ** 2).mean(dim=1, keepdim=True)

        E_c_up = F.interpolate(E_c, size=(heatmap_size, heatmap_size),
                               mode='bilinear', align_corners=False)
        E_m_up = F.interpolate(E_m, size=(heatmap_size, heatmap_size),
                               mode='bilinear', align_corners=False)

        H = 0.2 * E_c_up + 0.3 * E_m_up + 0.5 * E_f   # (1, 1, 64, 64)

        # Normalisation z-score
        if heatmap_stats is not None:
            H = (H - heatmap_stats["mean"]) / (heatmap_stats["std"] + 1e-8)

        H_np = H.squeeze().cpu().numpy()                 # (64, 64)
        H_smooth = gaussian_filter(H_np, sigma=sigma)

        # Reconstruction fine pour visualisation
        recon_np = outputs["x_hat_fine"].squeeze().cpu().numpy()  # (3, 64, 64)
        recon_np = np.transpose(recon_np, (1, 2, 0))              # (64, 64, 3)
        recon_np = np.clip(recon_np, 0, 1)

    return H_smooth, recon_np, outputs


# =============================================================================
# Score d'anomalie pour une image
# =============================================================================

def compute_single_score(model, image_tensor, device, scorer):
    """Calcule le score d'anomalie composite pour une image."""
    scores, labels = scorer.predict_batch(model, image_tensor, device)
    return float(scores[0]), int(labels[0])


# =============================================================================
# Visualisation complète
# =============================================================================

def visualize_prediction(pil_image, H_smooth, recon_np,
                          score, label, threshold,
                          kl_mean, active_k,
                          save_path=None):
    """
    Génère la figure de visualisation avec 5 panneaux :
      1. Image originale
      2. Reconstruction fine (64×64 upscalée)
      3. Heatmap d'anomalie (colormap JET)
      4. Masque binaire des zones suspectes
      5. Overlay heatmap sur image originale
    """
    # Préparer l'image originale en numpy
    orig_np = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0

    # Normaliser heatmap en [0, 1]
    hm = H_smooth.copy().astype(np.float32)
    hm_min, hm_max = hm.min(), hm.max()
    hm_norm = (hm - hm_min) / (hm_max - hm_min + 1e-8)

    # Masque binaire : zones au-dessus du percentile 90
    thresh_bin = np.percentile(hm_norm, 90)
    binary_mask = (hm_norm > thresh_bin).astype(np.uint8)

    # Upscale heatmap et masque à 224×224
    hm_224    = cv2.resize(hm_norm,    (224, 224), interpolation=cv2.INTER_LINEAR)
    mask_224  = cv2.resize(binary_mask,(224, 224), interpolation=cv2.INTER_NEAREST)
    recon_224 = cv2.resize(recon_np,   (224, 224), interpolation=cv2.INTER_LINEAR)

    # Overlay coloré
    hm_colored = cv2.applyColorMap((hm_224 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_colored = hm_colored[:, :, ::-1].astype(np.float32) / 255.0
    overlay    = np.clip(orig_np * 0.55 + hm_colored * 0.45, 0, 1)

    # ── Décision ──────────────────────────────────────────────────────────
    decision  = "FORGÉ 🔴"    if label == 1 else "AUTHENTIQUE 🟢"
    color_dec = "#e74c3c"     if label == 1 else "#2ecc71"
    score_pct = min(100, max(0, (score / (threshold * 2)) * 100))

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # Titre principal
    fig.suptitle(
        f"DÉCISION : {decision}     |     Score d'anomalie : {score:.4f}  "
        f"(seuil θ = {threshold:.4f})",
        fontsize=16, fontweight="bold", color=color_dec,
        y=0.97
    )

    axes = []
    titles = ["Image Originale", "Reconstruction Fine\n(Décodeur 64×64)",
              "Heatmap d'Anomalie", "Zones Suspectes\n(Masque Binaire)",
              "Overlay Heatmap"]
    images_to_show = [orig_np, recon_224, hm_224, mask_224, overlay]
    cmaps = [None, None, "jet", "gray", None]

    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1)
        axes.append(ax)
        if cmaps[i]:
            ax.imshow(images_to_show[i], cmap=cmaps[i])
        else:
            ax.imshow(np.clip(images_to_show[i], 0, 1))
        ax.set_title(titles[i], color="white", fontsize=10, pad=8)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    # Barre de score
    ax_bar = fig.add_axes([0.15, 0.04, 0.70, 0.035])
    ax_bar.set_facecolor("#2c2c54")
    ax_bar.barh(0, score_pct, height=0.8,
                color=color_dec, alpha=0.85)
    ax_bar.barh(0, 100, height=0.8,
                color="#444", alpha=0.3)
    # Ligne seuil
    thresh_pct = min(100, (threshold / (threshold * 2)) * 100)
    ax_bar.axvline(x=50, color="yellow", linewidth=2, linestyle="--")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Score d'anomalie (normalisé)", color="white", fontsize=9)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#555")

    # Métadonnées
    meta = (f"K actif = {active_k}/64  |  KL mean = {kl_mean:.3f}  |  "
            f"Modèle : DINOv2-ViT-B/14 + Sparse VAE")
    fig.text(0.5, 0.01, meta, ha="center", fontsize=9,
             color="#aaaaaa", style="italic")

    plt.tight_layout(rect=[0, 0.10, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualisation] Sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close()
    return fig


# =============================================================================
# Pipeline principal
# =============================================================================

def predict(image_path:  str,
            cfg_path:    str = "configs/default.yaml",
            checkpoint:  str = None,
            output_dir:  str = "results/predictions",
            show_plot:   bool = True):

    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Predict] Device : {device}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Charger le modèle ─────────────────────────────────────────────────
    model     = SparseVAE(cfg).to(device)
    ckpt_path = checkpoint or cfg.training.best_model_path
    state     = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state["model"])
    model.k_controller.load_state_dict(state["k_ctrl"])
    model.eval()
    print(f"[Predict] Modèle chargé — epoch={state['epoch']}  K={model.current_k}")

    # ── Calibration du scorer sur val authentiques ────────────────────────
    print("[Predict] Calibration du scorer...")
    _, val_loader = get_midv2020_loaders(cfg)

    scorer = AnomalyScorer(
        weights=(
            cfg.scoring.weights.recon,
            cfg.scoring.weights.kl,
            cfg.scoring.weights.heatmap,
            cfg.scoring.weights.latent,
        ),
        threshold_percentile=cfg.scoring.threshold_percentile,
        sigma=cfg.scoring.heatmap_sigma,
        heatmap_size=cfg.scoring.heatmap_size,
    )
    scorer.fit(model, val_loader, device)

    # Stats heatmap pour normalisation
    heatmap_stats = scorer.heatmap_stats

    # ── Charger l'image ───────────────────────────────────────────────────
    print(f"[Predict] Image : {image_path}")
    image_tensor, pil_image = load_image(image_path, cfg.data.image_size)

    # ── Score d'anomalie ──────────────────────────────────────────────────
    score, label = compute_single_score(model, image_tensor, device, scorer)

    # ── Heatmap ───────────────────────────────────────────────────────────
    H_smooth, recon_np, outputs = compute_single_heatmap(
        model, image_tensor, device, heatmap_stats,
        sigma=cfg.scoring.heatmap_sigma,
        heatmap_size=cfg.scoring.heatmap_size,
    )

    # KL mean pour affichage
    kl_mean = outputs["kl_per_dim"].mean().item()

    # ── Affichage console ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    decision = "FORGÉ 🔴" if label == 1 else "AUTHENTIQUE 🟢"
    print(f"  DÉCISION    : {decision}")
    print(f"  Score       : {score:.4f}  (seuil θ = {scorer.threshold:.4f})")
    print(f"  K actif     : {model.current_k}/64")
    print(f"  KL mean     : {kl_mean:.4f}")
    print("=" * 55 + "\n")

    # ── Visualisation ─────────────────────────────────────────────────────
    img_name  = Path(image_path).stem
    save_path = str(Path(output_dir) / f"{img_name}_prediction.png")

    visualize_prediction(
        pil_image    = pil_image,
        H_smooth     = H_smooth,
        recon_np     = recon_np,
        score        = score,
        label        = label,
        threshold    = scorer.threshold,
        kl_mean      = kl_mean,
        active_k     = model.current_k,
        save_path    = save_path,
    )

    # ── Affichage inline Colab ─────────────────────────────────────────────
    if show_plot:
        try:
            from IPython.display import display, Image as IPImage
            display(IPImage(filename=save_path))
        except ImportError:
            pass

    return {
        "decision":  "FORGÉ" if label == 1 else "AUTHENTIQUE",
        "score":     score,
        "threshold": scorer.threshold,
        "label":     label,
        "heatmap":   H_smooth,
        "save_path": save_path,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction sur une seule image — Sparse VAE Forgery Detection"
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Chemin vers l'image à analyser (.jpg / .png)")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output",     type=str, default="results/predictions")
    args = parser.parse_args()

    predict(
        image_path = args.image,
        cfg_path   = args.config,
        checkpoint = args.checkpoint,
        output_dir = args.output,
    )