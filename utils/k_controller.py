import numpy as np

class KController:
    """
    Adaptive K Controller basé sur trois signaux :
      1. KL per dimension  → dimensions informatives
      2. Active ratio      → contrainte de densité
      3. Reconstruction loss → contrainte de qualité
    Tous les hyperparamètres sont injectés depuis configs/default.yaml.
    """
    def __init__(
        self,
        latent_dim,
        initial_k,
        min_k,
        max_k,
        tau_fraction,
        tau_min,
        ema_alpha,
        ratio_min,
        ratio_max,
        recon_loss_low,   # seuil bas  : recon trop basse → réduire K
        recon_loss_high,  # seuil haut : recon trop haute → augmenter K
        recon_step,       # incrément/décrement K quand recon hors bornes
    ):
        self.latent_dim     = latent_dim
        self.k_current      = initial_k
        self.k_initial      = initial_k
        self.k_min          = min_k
        self.k_max          = max_k
        self.tau_fraction   = tau_fraction
        self.tau_min        = tau_min
        self.ema_alpha      = ema_alpha
        self.ratio_min      = ratio_min
        self.ratio_max      = ratio_max
        self.recon_loss_low  = recon_loss_low
        self.recon_loss_high = recon_loss_high
        self.recon_step      = recon_step
        self.k_history       = []

    def update(self, kl_per_dim, recon_loss):
        """
        Args:
            kl_per_dim : array shape (latent_dim,) — KL par dimension
            recon_loss : float — reconstruction loss moyenne de l'epoch
        Returns:
            K final contrôlé
        """
        kl_per_dim = np.clip(np.array(kl_per_dim, dtype=np.float32), 0.0, None)

        # ── 1. KL → k_target de base ──────────────────────────
        kl_mean = kl_per_dim.mean()
        tau     = max(self.tau_fraction * kl_mean, self.tau_min)

        active_mask = kl_per_dim > tau
        k_target    = int(np.sum(active_mask))
        ratio_raw   = k_target / self.latent_dim

        # ── 2. Contrainte active_ratio ────────────────────────
        recon_reason = "ok"
        if ratio_raw < self.ratio_min:
            k_target = int(self.ratio_min * self.latent_dim)
        elif ratio_raw > self.ratio_max:
            k_target = int(self.ratio_max * self.latent_dim)

        # ── 3. Contrainte reconstruction loss ─────────────────
        if recon_loss > self.recon_loss_high:
            # recon trop élevée → K trop petit, modèle n'a pas assez de capacité
            k_target    = k_target + self.recon_step
            recon_reason = f"↑ recon={recon_loss:.4f} > {self.recon_loss_high}"
        elif recon_loss < self.recon_loss_low:
            # recon trop basse → K trop grand, risque de mémorisation
            k_target    = k_target - self.recon_step
            recon_reason = f"↓ recon={recon_loss:.4f} < {self.recon_loss_low}"

        # ── 4. EMA ────────────────────────────────────────────
        k_smooth = self.ema_alpha * self.k_current + (1 - self.ema_alpha) * k_target
        k_new    = max(self.k_min, min(self.k_max, int(round(k_smooth))))

        # ── 5. Update state ───────────────────────────────────
        active_ratio_final = k_new / self.latent_dim
        k_old          = self.k_current
        self.k_current = k_new
        self.k_history.append(k_new)

        # ── 6. Log ────────────────────────────────────────────
        print(
            f"K-Control | K {k_old} → {k_new} | "
            f"k_target={k_target} (raw_ratio={ratio_raw:.3f}) | "
            f"active_ratio={active_ratio_final:.3f} | "
            f"tau={tau:.5f} | kl_mean={kl_mean:.5f} | "
            f"recon={recon_reason}"
        )
        return self.k_current

    def get_k(self):
        return self.k_current

    def reset(self):
        self.k_current = self.k_initial
        self.k_history = []

    def get_history(self):
        return {"k_history": self.k_history}