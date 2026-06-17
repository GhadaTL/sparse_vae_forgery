# 🔴 ANALYSE: CE QUI MANQUE POUR LA MÉTHODE ADAPTATIVE

## COMPARAISON: Actuel vs Adaptatif

### ❌ CODE ACTUEL (Statique)

```python
# train.py
model = SparseVAE(latent_dim=64, k=16)  # K FIXÉ = 16 pour toute l'époque

for epoch in range(100):
    for batch in train_loader:
        z = encoder(x)
        z_sparse = top_k(z, k=16)  # ← TOUJOURS 16, jamais adapté
        x_hat = decoder(z_sparse)
        
        recon_loss = compute_recon_loss(x, x_hat)
        sparsity = compute_sparsity(z_sparse)  # Calculé mais NON UTILISÉ
        
        beta = beta_annealing(epoch)  # β adapté selon PHASE (epoch)
        kl_loss = compute_kl(z)
        
        total_loss = recon_loss + beta * kl_loss
        total_loss.backward()

# ✅ Ce qui existe
├─ K = 16 (fixé)
├─ β adapté selon phase (3 phases: 0 → 0-4 → 4)
├─ Top-K sparsité implémenté
├─ KL divergence calculé
└─ Reconstruction multi-échelle OK

# ❌ CE QUI MANQUE
├─ Contrôleur K(t) adaptatif
├─ Contrôleur β(t) adaptatif selon sparsité
├─ Feedback loop: recon_loss → K
├─ Feedback loop: sparsity → β
└─ Historique des changements K et β
```

---

## CE QUI DOIT ÊTRE AJOUTÉ

### 1️⃣ **CONTRÔLEUR K ADAPTATIF** (MANQUANT)

```python
class KController:
    """
    Adapte K(t) selon qualité reconstruction
    
    Logique:
    - Si recon_loss augmente → augmenter K (plus de dimensions)
    - Si recon_loss stabil/bon → garder ou réduire K
    - Min: 1, Max: latent_dim
    """
    
    def __init__(self, initial_k=16, min_k=1, max_k=64, step=1, window=5):
        self.k_current = initial_k
        self.k_history = [initial_k]
        self.recon_losses = []
        self.window = window  # Moyenne mobile sur N epochs
        self.step = step      # Incrément/décrement par adaptation
        
    def update(self, recon_loss):
        """Adapte K selon trend de recon_loss"""
        self.recon_losses.append(recon_loss)
        
        if len(self.recon_losses) < self.window:
            return self.k_current  # Pas assez de données
        
        # Moyenne mobile
        avg_recent = np.mean(self.recon_losses[-self.window:])
        avg_old = np.mean(self.recon_losses[-(2*self.window):-self.window])
        
        # Logique adaptation
        if avg_recent > avg_old * 1.05:  # Loss augmente (+5%)
            self.k_current = min(self.k_current + self.step, self.max_k)
            print(f"↑ K augmente: {self.k_current} (recon loss ↑)")
        elif avg_recent < avg_old * 0.95:  # Loss diminue (-5%)
            self.k_current = max(self.k_current - self.step, self.min_k)
            print(f"↓ K diminue: {self.k_current} (recon loss ↓)")
        
        self.k_history.append(self.k_current)
        return self.k_current
```

### 2️⃣ **CONTRÔLEUR β ADAPTATIF** (MANQUANT)

```python
class BetaController:
    """
    Adapte β(t) selon NIVEAU DE SPARSITÉ obtenu
    
    Logique:
    - Si sparsity_rate < target (ex: 85%) → augmenter β pour pousser sparsité
    - Si sparsity_rate > target → réduire β
    - β toujours dans [0, beta_max]
    """
    
    def __init__(self, beta_max=4.0, target_sparsity=0.85, step=0.1):
        self.beta = 0.0
        self.beta_max = beta_max
        self.target_sparsity = target_sparsity
        self.step = step
        self.beta_history = [0.0]
        
    def update(self, sparsity_rate, epoch):
        """Adapte β selon sparsité réelle"""
        
        # Phase 1: pas de sparsity constraint
        if epoch < 20:
            self.beta = 0.0
        
        # Phase 2-3: adapter β selon sparsité
        else:
            if sparsity_rate < self.target_sparsity:
                self.beta = min(self.beta + self.step, self.beta_max)
                print(f"↑ β augmente: {self.beta:.2f} (sparsity {sparsity_rate:.2f} < target {self.target_sparsity:.2f})")
            elif sparsity_rate > self.target_sparsity + 0.05:
                self.beta = max(self.beta - self.step, 0.0)
                print(f"↓ β diminue: {self.beta:.2f} (sparsity {sparsity_rate:.2f} > target)")
            else:
                print(f"→ β stable: {self.beta:.2f} (sparsity optimal {sparsity_rate:.2f})")
        
        self.beta_history.append(self.beta)
        return self.beta
```

### 3️⃣ **CALCUL DES MÉTRIQUES DE SPARSITÉ** (MANQUANT)

```python
def compute_sparsity_metrics(z_sparse, latent_dim=64):
    """
    Calcule métriques de sparsité pour feedback contrôleur
    
    Returns:
        dict avec:
        - sparsity_rate: % de zéros dans z_sparse
        - active_ratio: % dimensions actives (non-zéro)
        - n_collapsed: nombre dimensions totalement inactives sur batch
        - mean_active_per_sample: moyenne dims actives par sample
    """
    batch_size = z_sparse.shape[0]
    
    # Zéros dans le batch entier
    total_zeros = (z_sparse == 0).float().sum().item()
    total_elements = z_sparse.numel()
    sparsity_rate = total_zeros / total_elements
    active_ratio = 1.0 - sparsity_rate
    
    # Dimensions totalement inactives (zéro pour TOUS les samples)
    dims_active = (z_sparse.abs() > 0).any(dim=0)  # (latent_dim,)
    n_collapsed = (dims_active == 0).sum().item()
    
    # Moyenne dims actives par sample
    active_per_sample = (z_sparse != 0).sum(dim=1).float().mean().item()
    
    return {
        "sparsity_rate": sparsity_rate,           # % zéros
        "active_ratio": active_ratio,             # % non-zéros
        "n_collapsed": n_collapsed,               # dims morts
        "mean_active_per_sample": active_per_sample
    }
```

### 4️⃣ **MODIFICATION TRAIN.PY** (MANQUANTE)

```python
# Avant (statique)
model = SparseVAE(latent_dim=64, k=16)

for epoch in range(100):
    for batch in train_loader:
        z_sparse = model.encode(x)  # k=16 toujours
        ...

# Après (adaptatif)
model = SparseVAE(latent_dim=64, k=16)  # k initial

k_controller = KController(initial_k=16, max_k=64)
beta_controller = BetaController(beta_max=4.0, target_sparsity=0.85)

for epoch in range(100):
    epoch_metrics = {"recon_losses": [], "sparsities": []}
    
    for batch in train_loader:
        # 1. Forward avec K(t) ADAPTATIF
        z_sparse, kl = model.encode(x, k=k_controller.k_current)
        x_hat = model.decode(z_sparse)
        
        # 2. Perte reconstruction
        recon_loss = compute_recon_loss(x, x_hat)
        epoch_metrics["recon_losses"].append(recon_loss.item())
        
        # 3. Calculer sparsité RÉELLE
        sparsity_metrics = compute_sparsity_metrics(z_sparse)
        epoch_metrics["sparsities"].append(sparsity_metrics["sparsity_rate"])
        
        # 4. β adaptatif selon sparsité réelle
        beta_t = beta_controller.update(
            sparsity_metrics["sparsity_rate"],
            epoch
        )
        
        # 5. Perte totale avec β adaptatif
        total_loss = recon_loss + beta_t * kl
        total_loss.backward()
        optimizer.step()
    
    # Fin d'epoch: adapter K selon trend recon_loss
    avg_recon = np.mean(epoch_metrics["recon_losses"])
    k_controller.update(avg_recon)
    
    print(f"Epoch {epoch}: "
          f"K(t)={k_controller.k_current}, "
          f"β(t)={beta_controller.beta:.2f}, "
          f"sparsity={np.mean(epoch_metrics['sparsities']):.2f}")
```

---

## RÉSUMÉ: CE QUI MANQUE

| Composant | Statut | Fichier | Action |
|-----------|--------|--------|--------|
| **KController** | ❌ MANQUANT | utils/k_controller.py | À créer |
| **BetaController** | ❌ MANQUANT | utils/beta_controller.py | À créer |
| **compute_sparsity_metrics()** | ❌ MANQUANT | losses/total_loss.py | À ajouter |
| **train.py adaptatif** | ❌ INCOMPLET | train.py | À modifier |
| **SparseVAE.encode(k)** | ⚠️ PARTIEL | models/full_model.py | À adapter pour k variable |

---

## STRUCTURE À CRÉER

```
utils/
├─ k_controller.py         ← KController classe
├─ beta_controller.py      ← BetaController classe
└─ __init__.py

losses/total_loss.py       ← Ajouter compute_sparsity_metrics()

train.py                   ← Modifier pour utiliser contrôleurs

models/full_model.py       ← Adapter encode() pour k variable
```

---

## POINTS CRITIQUES

1. **K et β ne sont PAS contrôlés actuellement** — juste phase-based β
2. **Pas de feedback loop** — sparsity calculée mais non utilisée
3. **Pas de scheduler adaptatif** — seulement β_annealing fixe
4. **SparseVAE.encode() reçoit z mais pas k variable** — doit être modifié
5. **train.py n'utilise pas les métriques de sparsité** — juste calcule et ignore

---

## ÉTAPES DE CORRECTION

1. ✅ Créer `KController` avec logique adaptation
2. ✅ Créer `BetaController` avec logique adaptation
3. ✅ Ajouter `compute_sparsity_metrics()` 
4. ✅ Modifier `train.py` pour utiliser contrôleurs
5. ✅ Adapter `models/full_model.py` pour k variable
6. ✅ Tracker et visualizer l'évolution K(t) et β(t)
