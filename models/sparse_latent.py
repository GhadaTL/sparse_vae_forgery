"""
models/sparse_latent.py
Couche Latente Sparse + Contrôleur Adaptatif K.

Contribution principale :
  - Reparametrization trick (entraînement stochastique / inférence stable)
  - Top-K hard sparsity avec straight-through estimator
  - AdaptiveKController : apprend automatiquement K à partir de la KL par dim
    et de la reconstruction loss — sans jamais fixer K manuellement.
"""
import torch
import torch.nn as nn


# =============================================================================
# Reparametrization Trick
# =============================================================================

def reparameterize(mu: torch.Tensor,
                   logvar: torch.Tensor,
                   training: bool = True,
                   n_samples: int = 10) -> torch.Tensor:
    """
    Reparametrization trick : z = μ + σ ⊙ ε, ε ~ N(0, I).
    Différentiable par rapport à μ et logvar.

    En inférence : moyenne de n_samples échantillons pour la stabilité.

    Args:
        mu       : (B, D)
        logvar   : (B, D)
        training : bool
        n_samples: nombre d'échantillons en inférence

    Returns:
        z : (B, D)
    """
    std = torch.exp(0.5 * logvar)   # σ = exp(0.5 × log σ²)

    if training:
        eps = torch.randn_like(std)  # ε ~ N(0, I)
        return mu + std * eps
    else:
        # Moyenne de n_samples pour réduire la variance en inférence
        samples = [mu + std * torch.randn_like(std) for _ in range(n_samples)]
        return torch.stack(samples, dim=0).mean(dim=0)


# =============================================================================
# Top-K Hard Sparsity
# =============================================================================

def top_k_sparsity(z: torch.Tensor, k: int):
    """
    Applique une sparsité Top-K sur z.
    Seules les K dimensions de plus forte magnitude sont conservées.

    Gradient : straight-through estimator (le gradient passe intégralement).

    Args:
        z : (B, D) — vecteur latent dense
        k : nombre de dimensions à conserver

    Returns:
        z_sparse : (B, D) — z avec D-K dimensions mises à zéro
        mask     : (B, D) — masque binaire (1 = actif, 0 = zéro)
    """
    k = max(1, min(k, z.shape[1]))  # Sécurité : k ∈ [1, D]

    z_abs = z.abs()   # (B, D)

    # Trouver la K-ième plus grande valeur par ligne → seuil
    topk_vals, _ = z_abs.topk(k, dim=1)        # (B, K)
    threshold = topk_vals[:, -1:].detach()      # (B, 1) — la plus petite des K

    # Masque binaire : 1 pour les K dims actives
    mask = (z_abs >= threshold).float()          # (B, D)

    # Application du masque — straight-through : gradient non masqué en backward
    z_sparse = z * mask

    return z_sparse, mask


# =============================================================================
# KL Divergence — par dimension et totale
# =============================================================================

def kl_divergence(mu: torch.Tensor,
                  logvar: torch.Tensor):
    """
    KL(q(z|x) || N(0,I)) par dimension et totale.

    KL_i = -0.5 × (1 + log σ_i² - μ_i² - σ_i²)

    Args:
        mu     : (B, D)
        logvar : (B, D)

    Returns:
        kl_loss    : scalaire — moyenne sur batch et dimensions
        kl_per_dim : (D,)    — KL par dimension (moyenne sur le batch)
    """
    # KL par (batch, dim) : (B, D)
    kl_bd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Par dimension : moyenne sur le batch → (D,)
    kl_per_dim = kl_bd.mean(dim=0)

    # Scalaire : somme sur D puis moyenne sur B
    kl_loss = kl_bd.sum(dim=1).mean()

    return kl_loss, kl_per_dim


# =============================================================================
# Contrôleur Adaptatif K
# =============================================================================

class AdaptiveKController:
    """
    Apprend automatiquement la meilleure valeur de K à chaque époque.

    Logique :
      1. Calculer τ = max(tau_fraction × KL_mean, tau_min)
      2. Compter les dimensions informatives : KL_i > τ → K_target
      3. Ajuster selon la reconstruction : si recon élevée → augmenter K
      4. Contraindre K ∈ [K_min, K_max]
      5. Appliquer via EMA pour éviter les oscillations

    Args:
        K_init      : valeur initiale de K (8)
        K_min       : valeur minimale (2)
        K_max       : valeur maximale (50)
        tau_fraction: fraction de KL_mean pour le seuil (0.1)
        tau_min     : seuil minimum (1e-4)
        recon_high  : seuil reconstruction pour augmenter K (0.05)
        recon_low   : seuil reconstruction pour réduire K (0.01)
        recon_step  : pas d'adaptation en K (2)
        ema_alpha   : coefficient EMA (0.9) — proche de 1 = évolution lente
    """

    def __init__(self,
                 K_init: int = 8,
                 K_min: int = 2,
                 K_max: int = 50,
                 tau_fraction: float = 0.1,
                 tau_min: float = 1e-4,
                 recon_high: float = 0.05,
                 recon_low: float = 0.01,
                 recon_step: int = 2,
                 ema_alpha: float = 0.9):
        self.K = float(K_init)   # Float pour l'EMA
        self.K_min = K_min
        self.K_max = K_max
        self.tau_fraction = tau_fraction
        self.tau_min = tau_min
        self.recon_high = recon_high
        self.recon_low  = recon_low
        self.recon_step = recon_step
        self.ema_alpha  = ema_alpha

        self._K_ema = float(K_init)    # EMA interne

    @property
    def current_K(self) -> int:
        """K courant entier (utilisé dans top_k_sparsity)."""
        return int(round(self._K_ema))

    def update(self,
               kl_per_dim: torch.Tensor,
               recon_loss: float) -> int:
        """
        Met à jour K à partir des diagnostics de l'époque.

        Args:
            kl_per_dim : (D,) — KL par dimension (moyenne sur le batch/époque)
            recon_loss : scalaire float — reconstruction loss de validation

        Returns:
            K courant (entier)
        """
        kl_np = kl_per_dim.detach().cpu().float()

        # --- Étape 1 : Calcul de τ ---
        kl_mean = kl_np.mean().item()
        tau = max(self.tau_fraction * kl_mean, self.tau_min)

        # --- Étape 2 : Compter les dimensions informatives ---
        n_informative = int((kl_np > tau).sum().item())
        K_target = float(n_informative)

        # --- Étape 3 : Ajuster selon la reconstruction ---
        if recon_loss > self.recon_high:
            K_target += self.recon_step   # Reconstruction mauvaise → plus de dims
        elif recon_loss < self.recon_low:
            K_target -= self.recon_step   # Reconstruction trop facile → moins de dims

        # --- Étape 4 : Contraindre dans [K_min, K_max] ---
        K_target = float(max(self.K_min, min(self.K_max, K_target)))

        # --- Étape 5 : EMA pour évolution progressive ---
        # K_new = α × K_old + (1-α) × K_target
        self._K_ema = self.ema_alpha * self._K_ema + (1.0 - self.ema_alpha) * K_target

        # Contraindre à nouveau après EMA
        self._K_ema = max(float(self.K_min), min(float(self.K_max), self._K_ema))

        return self.current_K

    def state_dict(self) -> dict:
        return {"K_ema": self._K_ema}

    def load_state_dict(self, state: dict):
        self._K_ema = state["K_ema"]


# =============================================================================
# Couche Latente Sparse
# =============================================================================

class SparseLatentLayer(nn.Module):
    """
    Couche latente combinant :
      - Reparametrization trick
      - Top-K hard sparsity (K adaptatif)
      - Calcul de la KL divergence par dimension

    Args:
        latent_dim : dimension de l'espace latent (64)
        K_init     : valeur initiale de K (8)
    """

    def __init__(self,
                 latent_dim: int = 64,
                 K_init: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self._k = K_init

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int):
        self._k = max(1, min(value, self.latent_dim))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Args:
            mu     : (B, D) — moyenne de q(z|x)
            logvar : (B, D) — log-variance de q(z|x)

        Returns:
            z_sparse   : (B, D) — vecteur latent sparse
            kl_loss    : scalaire — KL totale (pour la loss)
            kl_per_dim : (D,)   — KL par dimension (pour AdaptiveKController)
            mask       : (B, D) — masque Top-K
        """
        # 1. Reparametrization trick
        z = reparameterize(mu, logvar, training=self.training)

        # 2. Top-K sparsity
        z_sparse, mask = top_k_sparsity(z, k=self._k)

        # 3. KL divergence (calculée sur z original, pas z_sparse)
        kl_loss, kl_per_dim = kl_divergence(mu, logvar)

        return z_sparse, kl_loss, kl_per_dim, mask
