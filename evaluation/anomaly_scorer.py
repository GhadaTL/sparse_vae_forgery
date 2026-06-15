# evaluation/anomaly_scorer.py
# CDC §7 — Score d'Anomalie Composite
# Agrégation multi-signal → classification finale authentique / forgé

import torch
import numpy as np
from pathlib import Path
import json


class AnomalyScorer:
    """
    Calcule un score d'anomalie composite combinant 4 signaux :
    - S_recon : erreur de reconstruction
    - S_kl : KL divergence
    - S_heatmap : pic de la heatmap d'anomalie
    - S_latent : norme des activations latentes
    
    CDC §7 — implémentation complète.
    """
    
    def __init__(self, weights=(0.4, 0.2, 0.3, 0.1)):
        """
        Args:
            weights : tuple (w_recon, w_kl, w_heatmap, w_latent)
        """
        self.w_recon, self.w_kl, self.w_heatmap, self.w_latent = weights
        self.stats = {}  # Statistiques de normalisation calibrées sur val authentiques
        self.threshold = None  # Seuil θ calibré
    
    def fit(self, model, dinov2, projection_head, sparse_latent, decoder,
            val_loader_authentic, heatmap_stats, device='cuda'):
        """
        Calibre le scorer sur le val set authentique.
        Calcule les statistiques de normalisation et le seuil θ (CDC §7.3).
        
        Args:
            model : SparseVAE complet
            dinov2, projection_head, sparse_latent, decoder : composants
            val_loader_authentic : DataLoader sur documents authentiques
            heatmap_stats : dict avec 'mean' et 'std' de la heatmap (de compute_heatmap_stats)
            device : 'cuda' ou 'cpu'
        """
        print("Calibrating AnomalyScorer on validation set...")
        
        scores = {'recon': [], 'kl': [], 'hm': [], 'lat': []}
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader_authentic):
                # Extraire l'image du batch
                if isinstance(batch, dict):
                    image = batch['image']
                else:
                    image = batch[0]
                
                image = image.to(device)
                
                # Calcul des scores bruts
                s = self._compute_raw_scores(
                    model, image, dinov2, projection_head, sparse_latent, decoder,
                    heatmap_stats, device
                )
                
                for k in scores:
                    scores[k].append(s[k])
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{batch_idx + 1}] Processed")
        
        # Calcul des stats pour normalisation
        for k in scores:
            vals = np.array(scores[k])
            self.stats[k] = {
                'mean': vals.mean(),
                'std': vals.std() + 1e-8
            }
        
        # Seuil θ = percentile 95 du score composite sur les authentiques
        composite_scores = []
        for s_recon, s_kl, s_hm, s_lat in zip(
            scores['recon'], scores['kl'], scores['hm'], scores['lat']
        ):
            s_dict = {'recon': s_recon, 'kl': s_kl, 'hm': s_hm, 'lat': s_lat}
            s_norm = self._normalize(s_dict)
            s_comp = self._composite(s_norm)
            composite_scores.append(s_comp)
        
        self.threshold = np.percentile(composite_scores, 95)
        
        print(f"✅ Calibration complete")
        print(f"   Threshold θ = {self.threshold:.4f}")
        print(f"   Stats: {self.stats}")
    
    def predict(self, model, dinov2, projection_head, sparse_latent, decoder,
                image, heatmap_stats, device='cuda'):
        """
        Prédit le label (AUTHENTIQUE / FORGÉ) et retourne le score.
        
        Args:
            model : SparseVAE complet
            dinov2, projection_head, sparse_latent, decoder : composants
            image : (3, 224, 224) ou (B, 3, 224, 224)
            heatmap_stats : dict avec 'mean' et 'std' de la heatmap
            device : 'cuda' ou 'cpu'
        
        Returns:
            score : score composite scalaire
            label : 'AUTHENTIQUE' ou 'FORGÉ'
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        with torch.no_grad():
            raw = self._compute_raw_scores(
                model, image, dinov2, projection_head, sparse_latent, decoder,
                heatmap_stats, device
            )
            norm = self._normalize(raw)
            score = self._composite(norm)
        
        label = 'FORGÉ' if score > self.threshold else 'AUTHENTIQUE'
        
        return score, label
    
    def _compute_raw_scores(self, model, image, dinov2, projection_head, sparse_latent,
                            decoder, heatmap_stats, device='cuda'):
        """
        Calcule les 4 composantes de score brutes (CDC §7.2).
        
        Returns:
            dict avec 'recon', 'kl', 'hm', 'lat'
        """
        model.eval()
        
        with torch.no_grad():
            # ── Forward pass complet ──
            from evaluation.heatmap import extract_dinov2_features
            patch_tokens, cls_token = extract_dinov2_features(dinov2, image)
            mu, logvar = projection_head(patch_tokens)
            z_sparse, kl, mask = sparse_latent(mu, logvar)
            x_hat_c, x_hat_m, x_hat_f = decoder(z_sparse)
            
            # ── Cibles multi-échelle ──
            import torch.nn.functional as F
            x_c = F.interpolate(image, (16, 16), mode="bilinear", align_corners=False)
            x_m = F.interpolate(image, (32, 32), mode="bilinear", align_corners=False)
            x_f = F.interpolate(image, (64, 64), mode="bilinear", align_corners=False)
            
            # ── S_recon : MSE sur la résolution fine (16, 16, 64, 64) ──
            mse_f = F.mse_loss(x_f, x_hat_f, reduction='mean').item()
            s_recon = mse_f
            
            # ── S_kl : KL divergence ──
            s_kl = kl.mean().item()
            
            # ── S_heatmap : Percentile 95 de la heatmap lissée ──
            from evaluation.heatmap import compute_multiscale_heatmap
            hm = compute_multiscale_heatmap(
                model, image, dinov2, projection_head, sparse_latent, decoder,
                train_stats=heatmap_stats, device=device
            )
            s_heatmap = np.percentile(hm, 95)
            
            # ── S_latent : norme moyenne des dims actives ──
            # ||z_sparse||_2 / K
            z_norm = torch.norm(z_sparse, p=2, dim=1).mean().item()
            K = z_sparse.shape[1]
            s_latent = z_norm / K
        
        return {
            'recon': s_recon,
            'kl': s_kl,
            'hm': s_heatmap,
            'lat': s_latent
        }
    
    def _normalize(self, raw_scores):
        """
        Normalise chaque composante par z-score (CDC §7.2).
        z-score = (x - μ) / σ
        """
        norm = {}
        for key in ['recon', 'kl', 'hm', 'lat']:
            if key in raw_scores:
                mean = self.stats[key]['mean']
                std = self.stats[key]['std']
                norm[key] = (raw_scores[key] - mean) / std
            else:
                norm[key] = 0.0
        return norm
    
    def _composite(self, norm_scores):
        """
        Calcule le score composite pondéré (CDC §7.2).
        S_final = 0.4×S_norm_recon + 0.2×S_norm_kl + 0.3×S_norm_heatmap + 0.1×S_norm_latent
        """
        return (
            self.w_recon * norm_scores['recon'] +
            self.w_kl * norm_scores['kl'] +
            self.w_heatmap * norm_scores['hm'] +
            self.w_latent * norm_scores['lat']
        )
    
    def save(self, path):
        """Sauvegarde le scorer calibré (threshold et stats)."""
        config = {
            'weights': [self.w_recon, self.w_kl, self.w_heatmap, self.w_latent],
            'threshold': float(self.threshold),
            'stats': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in self.stats.items()}
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Scorer saved to {path}")
    
    def load(self, path):
        """Charge un scorer calibré."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.w_recon, self.w_kl, self.w_heatmap, self.w_latent = config['weights']
        self.threshold = config['threshold']
        self.stats = config['stats']
        print(f"✅ Scorer loaded from {path}")
    
    def batch_predict(self, model, dinov2, projection_head, sparse_latent, decoder,
                      data_loader, heatmap_stats, device='cuda'):
        """
        Évalue un batch complet (val/test set).
        
        Returns:
            dict avec 'scores', 'labels', 'predictions'
        """
        all_scores = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    images = batch['image']
                    labels = batch.get('label', None)
                else:
                    images = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                
                images = images.to(device)
                
                for i in range(images.shape[0]):
                    img = images[i:i+1]
                    score, pred_label = self.predict(
                        model, dinov2, projection_head, sparse_latent, decoder,
                        img, heatmap_stats, device
                    )
                    all_scores.append(score)
                    all_labels.append(pred_label)
        
        return {
            'scores': np.array(all_scores),
            'predictions': np.array(all_labels),
        }


def evaluate_anomaly_detector(scores, true_labels, threshold=None):
    """
    Évalue les performances du détecteur d'anomalies.
    
    Args:
        scores : array de scores
        true_labels : array de labels (0=authentique, 1=forgé)
        threshold : seuil à utiliser (si None, utilise percentile 95)
    
    Returns:
        dict avec métriques : tp, fp, tn, fn, precision, recall, f1, auc_roc
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
    
    if threshold is None:
        threshold = np.percentile(scores, 95)
    
    predictions = (scores > threshold).astype(int)
    
    # Matrice de confusion
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()
    
    # Métriques
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    try:
        auc_roc = roc_auc_score(true_labels, scores)
    except:
        auc_roc = 0.0
    
    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'threshold': float(threshold)
    }
