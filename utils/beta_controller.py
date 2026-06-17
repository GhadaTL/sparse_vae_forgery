# utils/beta_controller.py
# Adaptation dynamique de β selon niveau de sparsité obtenu

import numpy as np


class BetaController:
    """
    Adapte β(t) dynamiquement selon SPARSITÉ RÉELLE obtenue.
    
    Logique:
    - Phase 1 (epoch < 20): β = 0 (reconstruction pure)
    - Phase 2+ :
      - Si sparsity < target → augmenter β (pour pousser sparsité)
      - Si sparsity > target → réduire β
      - β ∈ [0, beta_max]
    
    CDC §5.4 — Protocole 3 phases adaptatif
    """
    
    def __init__(self, beta_max=4.0, target_sparsity=0.85,
                 step=0.1, warmup_epoch=20):
        """
        Args:
            beta_max : β maximum (phase 3)
            target_sparsity : pourcentage de sparsité visé (ex: 0.85 = 85%)
            step : incrément/décrement de β
            warmup_epoch : fin de phase 1 (pas d'adaptation avant)
        """
        self.beta = 0.0
        self.beta_max = beta_max
        self.beta_min = 0.0
        self.target_sparsity = target_sparsity
        self.step = step
        self.warmup_epoch = warmup_epoch
        
        self.beta_history = [0.0]
        self.sparsity_history = []
        
    def update(self, sparsity_rate, epoch):
        """
        Adapte β selon sparsité réelle.
        
        Args:
            sparsity_rate : % de zéros dans z_sparse (float 0-1)
            epoch : epoch courant
        
        Returns:
            beta_new : nouvelle valeur de β
        """
        self.sparsity_history.append(sparsity_rate)
        
        # Phase 1: pas d'adaptation, β = 0
        if epoch < self.warmup_epoch:
            self.beta = 0.0
        
        # Phase 2-3: adapter β selon sparsité
        else:
            beta_old = self.beta
            tolerance = 0.02  # ±2% tolérance autour target
            
            if sparsity_rate < (self.target_sparsity - tolerance):
                # Sparsité insuffisante: augmenter β
                self.beta = min(self.beta + self.step, self.beta_max)
                direction = "↑"
            elif sparsity_rate > (self.target_sparsity + tolerance):
                # Sparsité trop forte: réduire β
                self.beta = max(self.beta - self.step, self.beta_min)
                direction = "↓"
            else:
                # Sparsité optimale: maintenir β
                direction = "→"
            
            self.beta_history.append(self.beta)
            
            if abs(beta_old - self.beta) > 1e-6:
                print(f"  β-Control: {direction} β {beta_old:.2f} → {self.beta:.2f} "
                      f"(sparsity: {sparsity_rate:.2f}, target: {self.target_sparsity:.2f})")
        
        return self.beta
    
    def get_beta(self):
        """Retourne β actuel."""
        return self.beta
    
    def reset(self):
        """Reset contrôleur."""
        self.beta = 0.0
        self.beta_history = [0.0]
        self.sparsity_history = []
    
    def get_history(self):
        """Retourne historique β(t) et sparsity(t)."""
        return {
            "beta_history": self.beta_history,
            "sparsity_history": self.sparsity_history
        }
