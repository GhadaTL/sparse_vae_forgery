"""
utils/helpers.py
Fonctions utilitaires : seed, config, diagnostics latents.
"""
import random
import numpy as np
import torch
import yaml
from pathlib import Path


# =============================================================================
# Reproductibilité
# =============================================================================

def set_seed(seed: int = 42):
    """Fixe toutes les sources d'aléatoire pour la reproductibilité."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Toutes les graines fixées à {seed}")


# =============================================================================
# Chargement de la configuration
# =============================================================================

class DotDict(dict):
    """Dict accessible via attributs : cfg.training.lr"""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Clé '{key}' introuvable dans la config")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str = "configs/default.yaml") -> DotDict:
    """Charge le fichier YAML de configuration."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return DotDict(cfg)


# =============================================================================
# β Scheduler — 3 phases fixes
# =============================================================================

def get_beta(epoch: int, cfg) -> float:
    """
    Calcule β selon le protocole en 3 phases :
      Phase 1 (1..phase1_end)  : β = 0
      Phase 2 (phase1_end+1..phase2_end) : warm-up linéaire 0 → beta_final
      Phase 3 (phase2_end+1..) : β = beta_final
    epoch est 1-indexé.
    """
    p1 = cfg.beta.phase1_end
    p2 = cfg.beta.phase2_end
    bf = cfg.beta.beta_final

    if epoch <= p1:
        return 0.0
    elif epoch <= p2:
        progress = (epoch - p1) / (p2 - p1)
        return float(bf * progress)
    else:
        return float(bf)


def get_phase(epoch: int, cfg) -> int:
    """Retourne la phase courante (1, 2 ou 3)."""
    if epoch <= cfg.beta.phase1_end:
        return 1
    elif epoch <= cfg.beta.phase2_end:
        return 2
    else:
        return 3


# =============================================================================
# Diagnostics de l'espace latent
# =============================================================================

def compute_latent_diagnostics(mu: torch.Tensor,
                                logvar: torch.Tensor,
                                mask: torch.Tensor,
                                kl_per_dim: torch.Tensor) -> dict:
    """
    Calcule les métriques LEA (Latent Efficiency Analysis).

    Args:
        mu       : (B, D) — moyennes latentes
        logvar   : (B, D) — log-variances latentes
        mask     : (B, D) — masque binaire Top-K (0 ou 1)
        kl_per_dim : (D,)  — KL par dimension

    Returns:
        dict avec active_ratio, sparsity_rate, kl_mean, posterior_collapse,
        kl_explosion, variance_per_dim
    """
    with torch.no_grad():
        # Active ratio : fraction de dims avec |z_i| > 0.01 en moyenne batch
        active_ratio = (mask.float().mean(dim=0) > 0.1).float().mean().item()

        # Sparsity rate : % de dims actives moyenné sur le batch
        sparsity_rate = mask.float().mean().item()

        # KL moyenne
        kl_mean = kl_per_dim.mean().item()

        # Diagnostics
        posterior_collapse = kl_mean < 0.5
        kl_explosion = kl_mean > 10.0

        # Variance par dimension (sur le batch)
        var_per_dim = mu.var(dim=0).cpu().numpy()

    return {
        "active_ratio": active_ratio,
        "sparsity_rate": sparsity_rate,
        "kl_mean": kl_mean,
        "kl_per_dim": kl_per_dim.cpu().numpy(),
        "posterior_collapse": posterior_collapse,
        "kl_explosion": kl_explosion,
        "var_per_dim": var_per_dim,
        "dead_dims": int((var_per_dim < 0.001).sum()),
    }


# =============================================================================
# Utilitaires d'images
# =============================================================================

def prepare_multiscale_targets(images: torch.Tensor, device: torch.device):
    """
    Redimensionne les images originales (B, 3, 224, 224) vers les 3 résolutions
    cibles du décodeur : 16×16, 32×32, 64×64.
    """
    import torch.nn.functional as F
    x_coarse = F.interpolate(images, size=(16, 16), mode='bilinear', align_corners=False)
    x_medium = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
    x_fine   = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
    return x_coarse.to(device), x_medium.to(device), x_fine.to(device)


# =============================================================================
# Sauvegarde / Chargement de checkpoints
# =============================================================================

def save_checkpoint(state: dict, path: str):
    """Sauvegarde un checkpoint PyTorch."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Charge un checkpoint PyTorch."""
    return torch.load(path, map_location=device)


# =============================================================================
# Logger simple (console + optionnel WandB)
# =============================================================================

class Logger:
    """Logger léger avec support optionnel WandB."""

    def __init__(self, use_wandb: bool = False, project: str = "sparse-vae"):
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=project)

    def log(self, metrics: dict, step: int = None):
        msg = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                         for k, v in metrics.items())
        print(f"  [{step}] {msg}" if step is not None else f"  {msg}")
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_image(self, name: str, img_array, step: int = None):
        if self.use_wandb:
            import wandb
            wandb.log({name: wandb.Image(img_array)}, step=step)
