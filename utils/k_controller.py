# utils/k_controller.py
# Adaptation dynamique de K selon qualité reconstruction

import numpy as np


class KController:
    """
    Adapte K(t) dynamiquement selon qualité de reconstruction.
    
    Logique:
    - Si recon_loss augmente → augmenter K (plus de dimensions actives)
    - Si recon_loss stable/bon → garder ou réduire K progressivement
    - K ∈ [min_k, max_k]
    
    CDC §3.4 — Critères K
    """
    
    def __init__(self, initial_k=16, min_k=1, max_k=64, 
                 step=1, window=5, threshold_pct=5.0):
        """
        Args:
            initial_k : K au démarrage
            min_k : K minimum
            max_k : K maximum
            step : incrément/décrement par adaptation
            window : fenêtre de moyennes mobiles (nombre epochs)
            threshold_pct : seuil de variation (%) pour adapter K
        """
        self.k_current = initial_k
        self.k_initial = initial_k
        self.k_min = min_k
        self.k_max = max_k
        self.k_history = [initial_k]
        
        self.recon_losses = []
        self.window = window
        self.step = step
        self.threshold_pct = threshold_pct / 100.0  # Convert to ratio
        
    def update(self, recon_loss):
        """
        Adapte K selon trend de recon_loss.
        
        Args:
            recon_loss : scalaire — perte reconstruction moyenne de l'epoch
        
        Returns:
            k_new : nouvelle valeur de K pour prochain epoch
        """
        self.recon_losses.append(recon_loss)
        
        # Pas assez de données pour décider
        if len(self.recon_losses) < self.window:
            return self.k_current
        
        # Moyenne mobile récente vs ancienne
        avg_recent = np.mean(self.recon_losses[-self.window:])
        avg_old = np.mean(self.recon_losses[-(2*self.window):-self.window])
        
        # Calcul variation relative
        if avg_old > 0:
            relative_change = (avg_recent - avg_old) / avg_old
        else:
            relative_change = 0.0
        
        # Adaptation K
        k_old = self.k_current
        
        if relative_change > self.threshold_pct:
            # Loss augmente: augmenter K
            self.k_current = min(self.k_current + self.step, self.k_max)
            direction = "↑"
        elif relative_change < -self.threshold_pct:
            # Loss diminue: réduire K progressivement
            self.k_current = max(self.k_current - self.step, self.k_min)
            direction = "↓"
        else:
            # Stable: maintenir K
            direction = "→"
        
        self.k_history.append(self.k_current)
        
        if k_old != self.k_current:
            print(f"  K-Control: {direction} K {k_old} → {self.k_current} "
                  f"(recon_loss: {avg_recent:.5f}, Δ={relative_change*100:+.1f}%)")
        
        return self.k_current
    
    def get_k(self):
        """Retourne K actuel."""
        return self.k_current
    
    def reset(self):
        """Reset contrôleur."""
        self.k_current = self.k_initial
        self.k_history = [self.k_initial]
        self.recon_losses = []
    
    def get_history(self):
        """Retourne historique K(t) et recon_loss(t)."""
        return {
            "k_history": self.k_history,
            "recon_losses": self.recon_losses
        }
