"""
evaluation/anomaly_scorer.py
Score d'anomalie composite :
  S_final = 0.4×S_recon + 0.2×S_kl + 0.3×S_heatmap + 0.1×S_latent

Calibré sur le val set authentique (seuil θ = percentile 95).
"""
import numpy as np
import torch
from tqdm import tqdm

from evaluation.heatmap import compute_multiscale_heatmap, compute_heatmap_stats
from utils.helpers import prepare_multiscale_targets
import torch.nn.functional as F


class AnomalyScorer:
    """
    Calcule, normalise et agrège les 4 signaux d'anomalie.

    Composantes :
        S_recon   : MSE(x_fine, x̂_fine) — erreur de reconstruction fine
        S_kl      : KL(q(z|x) || N(0,I)) — déviation de la prior
        S_heatmap : percentile 95 de H_smooth — pic d'anomalie localisé
        S_latent  : ||z_sparse||₂ / K — intensité des activations actives

    Args:
        weights              : (w_recon, w_kl, w_hm, w_lat) — poids de fusion
        threshold_percentile : percentile pour le seuil θ (défaut 95)
        sigma                : sigma du lissage gaussien pour la heatmap
        heatmap_size         : résolution de la heatmap (64)
    """

    def __init__(self,
                 weights: tuple = (0.4, 0.2, 0.3, 0.1),
                 threshold_percentile: int = 95,
                 sigma: float = 1.5,
                 heatmap_size: int = 64):
        self.w_recon, self.w_kl, self.w_hm, self.w_lat = weights
        self.threshold_percentile = threshold_percentile
        self.sigma = sigma
        self.heatmap_size = heatmap_size

        # Stats de normalisation (calibrées sur val authentique)
        self.stats = {}
        self.heatmap_stats = None
        self.threshold = None

    # ------------------------------------------------------------------
    # Calcul des scores bruts pour un batch
    # ------------------------------------------------------------------

    def _compute_raw_scores(self,
                            model,
                            images: torch.Tensor,
                            device: torch.device) -> dict:
        """
        Retourne les 4 scores bruts (non normalisés) pour un batch.

        Returns:
            dict avec listes : 'recon', 'kl', 'heatmap', 'latent'
        """
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            x_c, x_m, x_f = prepare_multiscale_targets(images, device)

            # S_recon : MSE fine
            s_recon = F.mse_loss(outputs["x_hat_fine"], x_f,
                                 reduction="none").mean(dim=[1, 2, 3])  # (B,)

            # S_kl : KL totale par image
            kl_bd = -0.5 * (1 + outputs["logvar"]
                            - outputs["mu"].pow(2)
                            - outputs["logvar"].exp())
            s_kl = kl_bd.sum(dim=1)  # (B,)

            # S_latent : ||z_sparse||₂ / K
            k = max(1, outputs["mask"].sum(dim=1).float().mean().item())
            s_latent = outputs["z_sparse"].norm(dim=1) / k  # (B,)

        # S_heatmap : percentile 95 de la heatmap lissée — calculé image par image
        s_heatmap_list = []
        for i in range(images.shape[0]):
            H = compute_multiscale_heatmap(
                model,
                images[i:i+1],
                train_stats=self.heatmap_stats,
                device=device,
                sigma=self.sigma,
                heatmap_size=self.heatmap_size,
            )
            s_heatmap_list.append(float(np.percentile(H, 95)))

        return {
            "recon":   s_recon.cpu().numpy().tolist(),
            "kl":      s_kl.cpu().numpy().tolist(),
            "heatmap": s_heatmap_list,
            "latent":  s_latent.cpu().numpy().tolist(),
        }

    # ------------------------------------------------------------------
    # Calibration sur le val authentique
    # ------------------------------------------------------------------

    def fit(self, model, val_loader, device: torch.device):
        """
        Calibre le scorer sur le val set authentique :
          - Calcule les stats de normalisation (μ, σ) pour chaque composante
          - Calcule les heatmap stats
          - Fixe le seuil θ = percentile 95 du score composite

        Args:
            model      : SparseVAE
            val_loader : DataLoader val authentiques (MIDV-2020)
            device     : torch.device
        """
        print("[AnomalyScorer] Calibration sur le val authentique...")

        # 1. Stats heatmap
        self.heatmap_stats = compute_heatmap_stats(model, val_loader, device,
                                                   self.heatmap_size)

        # 2. Collecter tous les scores bruts
        raw = {"recon": [], "kl": [], "heatmap": [], "latent": []}
        for batch in tqdm(val_loader, desc="  Calibration"):
            images = batch["image"]
            batch_scores = self._compute_raw_scores(model, images, device)
            for k in raw:
                raw[k].extend(batch_scores[k])

        # 3. Stats de normalisation
        for k in raw:
            vals = np.array(raw[k])
            self.stats[k] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std() + 1e-8),
            }

        # 4. Score composite sur les authentiques pour fixer θ
        composite_scores = self._compute_composite_from_raw(raw)
        self.threshold = float(np.percentile(composite_scores, self.threshold_percentile))
        print(f"[AnomalyScorer] Seuil θ calibré = {self.threshold:.4f}")

    def _normalize_score(self, value: float, key: str) -> float:
        """Z-score d'un score brut selon les stats du val authentique."""
        return (value - self.stats[key]["mean"]) / self.stats[key]["std"]

    def _compute_composite_from_raw(self, raw: dict) -> np.ndarray:
        """Calcule les scores composites pour un ensemble de scores bruts."""
        n = len(raw["recon"])
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = (
                self.w_recon  * self._normalize_score(raw["recon"][i],   "recon")
                + self.w_kl   * self._normalize_score(raw["kl"][i],      "kl")
                + self.w_hm   * self._normalize_score(raw["heatmap"][i], "heatmap")
                + self.w_lat  * self._normalize_score(raw["latent"][i],  "latent")
            )
        return scores

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict_batch(self, model, images: torch.Tensor,
                      device: torch.device) -> tuple:
        """
        Calcule le score d'anomalie composite pour un batch.

        Returns:
            scores : (B,) numpy — scores d'anomalie
            labels : (B,) numpy — 1=FORGÉ, 0=AUTHENTIQUE
        """
        raw = self._compute_raw_scores(model, images, device)
        composite = self._compute_composite_from_raw(raw)
        labels = (composite > self.threshold).astype(int)
        return composite, labels

    def predict_dataset(self, model, test_loader, device: torch.device) -> dict:
        """
        Évalue sur un dataset complet (FMIDV-2022).

        Returns:
            dict avec 'scores', 'preds', 'labels', 'paths'
        """
        all_scores, all_preds, all_labels, all_paths = [], [], [], []

        for batch in tqdm(test_loader, desc="[Évaluation]"):
            images = batch["image"]
            true_labels = batch["label"].numpy()
            paths = batch["path"]

            scores, preds = self.predict_batch(model, images, device)
            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(true_labels.tolist())
            all_paths.extend(paths)

        return {
            "scores": np.array(all_scores),
            "preds":  np.array(all_preds),
            "labels": np.array(all_labels),
            "paths":  all_paths,
        }
