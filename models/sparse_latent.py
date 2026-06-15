import torch
import torch.nn as nn


class TopKStraightThrough(torch.autograd.Function):
    """CDC §3.2 — straight-through estimator."""
    @staticmethod
    def forward(ctx, z: torch.Tensor, k: int) -> torch.Tensor:
        z_abs     = z.abs()
        threshold, _ = z_abs.topk(k, dim=1)
        threshold = threshold[:, -1:]          # (B, 1) — seuil de coupure
        mask      = (z_abs >= threshold).float()
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None               # STE : gradient passe intact


def top_k_sparsity(z: torch.Tensor, k: int):
    """Retourne (z_sparse, mask) — CDC §3.2."""
    z_abs        = z.abs()
    threshold, _ = z_abs.topk(k, dim=1)
    threshold    = threshold[:, -1:]
    mask         = (z_abs >= threshold).float()
    z_sparse     = TopKStraightThrough.apply(z, k)
    return z_sparse, mask


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                   training: bool = True) -> torch.Tensor:
    """
    CDC §2.4 — reparametrization trick.
    Training : z = µ + σ·ε
    Inférence : moyenne de L=10 échantillons pour stabilité
    """
    std = torch.exp(0.5 * logvar)
    if training:
        eps = torch.randn_like(std)
        return mu + std * eps
    else:
        # CDC §2.4 : L=10 échantillons à l'inférence
        samples = [mu + std * torch.randn_like(std) for _ in range(10)]
        return torch.stack(samples).mean(dim=0)


class SparseLatentLayer(nn.Module):
    """
    CDC §3.5 — implémentation complète de la couche latente.
    Reparametrization → Top-K (adaptatif selon les données) → KL (calculée sur z original)
    """
    def __init__(self, latent_dim: int = 64, k: int = 16, 
                 adaptive_k: bool = True, k_selection_method: str = "variance"):
        super().__init__()
        self.k_fixed       = k
        self.latent_dim    = latent_dim
        self.adaptive_k    = adaptive_k
        self.k_selection_method = k_selection_method  # "variance", "threshold", ou "entropy"
        self.k             = k  # k actuel (sera mis à jour dynamiquement)

    def select_k_by_variance(self, z: torch.Tensor, variance_threshold: float = 0.9) -> int:
        """
        Sélectionne k pour capturer variance_threshold % de la variance totale.
        Exemple : 90% de la variance = garder dimensions importantes
        """
        z_abs = z.abs()  # (B, latent_dim)
        
        # Variance par dimension
        variance = torch.var(z_abs, dim=0)  # (latent_dim,)
        
        # Trier par variance décroissante
        sorted_var, _ = torch.sort(variance, descending=True)
        
        # Variance cumulative normalisée
        cum_var = torch.cumsum(sorted_var, dim=0) / sorted_var.sum()
        
        # Nombre de dimensions pour atteindre threshold
        k = (cum_var < variance_threshold).sum().item() + 1
        k = max(1, min(k, self.latent_dim))  # Limiter entre 1 et latent_dim
        
        return k

    def select_k_by_threshold(self, z: torch.Tensor, alpha: float = 1.5) -> int:
        """
        Sélectionne k basé sur un seuil statistique : mean + alpha * std
        Dimensions > threshold restent actives.
        Exemple : alpha=1.5 = 1.5 écarts-types au-dessus de la moyenne
        """
        z_abs = z.abs()  # (B, latent_dim)
        
        # Statistiques par dimension
        mean_val = z_abs.mean(dim=0)   # (latent_dim,)
        std_val = z_abs.std(dim=0)     # (latent_dim,)
        
        threshold = mean_val + alpha * std_val  # (latent_dim,)
        
        # Nombre de dimensions significatives
        active = (z_abs.mean(dim=0) > threshold).sum().item()
        k = max(1, active)
        k = min(k, self.latent_dim)
        
        return k

    def select_k_by_entropy(self, z: torch.Tensor, target_entropy: float = 0.7) -> int:
        """
        Sélectionne k pour minimiser l'entropie de la représentation.
        Dimensions avec forte magnitude = entropie basse = importantes
        """
        z_abs = z.abs()  # (B, latent_dim)
        
        # Magnitude moyenne par dimension, triée
        magnitude = z_abs.mean(dim=0)  # (latent_dim,)
        sorted_mag, _ = torch.sort(magnitude, descending=True)
        
        # Normaliser entre 0 et 1
        norm_mag = sorted_mag / (sorted_mag.max() + 1e-8)
        
        # Entropie cumulative
        entropy = -(norm_mag * torch.log(norm_mag + 1e-8)).sum()
        
        # Trouver k où entropie = target_entropy * max_entropy
        for k in range(1, self.latent_dim + 1):
            partial_entropy = -(norm_mag[:k] * torch.log(norm_mag[:k] + 1e-8)).sum()
            if partial_entropy / entropy > target_entropy:
                return k
        
        return self.latent_dim

    def select_adaptive_k(self, z: torch.Tensor) -> int:
        """
        Sélectionne automatiquement k selon la méthode configurée.
        """
        if self.k_selection_method == "variance":
            return self.select_k_by_variance(z, variance_threshold=0.90)
        elif self.k_selection_method == "threshold":
            return self.select_k_by_threshold(z, alpha=1.5)
        elif self.k_selection_method == "entropy":
            return self.select_k_by_entropy(z, target_entropy=0.7)
        else:
            return self.k_fixed

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        # 1. Reparametrization
        z = reparameterize(mu, logvar, self.training)   # (B, latent_dim)

        # 2. Sélectionner k automatiquement si activé
        if self.adaptive_k:
            self.k = self.select_adaptive_k(z)
        else:
            self.k = self.k_fixed

        # 3. Top-K sparsity avec k adaptatif
        z_sparse, mask = top_k_sparsity(z, self.k)      # (B, latent_dim)

        # 4. KL calculée sur z original (pas z_sparse) — CDC §3.5
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        return z_sparse, kl_loss, mask

    def active_ratio(self, mask: torch.Tensor) -> float:
        """CDC §3.4 — fraction de dimensions actives (cible 0.3–0.7)."""
        return mask.float().mean().item()

    def sparsity_rate(self, z_sparse: torch.Tensor) -> float:
        """CDC §3.4 — % de zéros dans z_sparse (cible ~75% pour K=16)."""
        return (z_sparse == 0).float().mean().item()

    def get_current_k(self) -> int:
        """Retourne la valeur de k actuellement utilisée."""
        return self.k