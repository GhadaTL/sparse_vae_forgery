# INTÉGRATION DES CONTRÔLEURS ADAPTATIFS — Guide Complet

## QUOI MODIFIER DANS train.py

### Avant (statique)
```python
# train.py — ACTUEL (ligne ~130)

model = SparseVAE(
    latent_dim = cfg["latent_dim"],  # 64
    k          = k,                   # FIXÉ! (ex: 16)
    dropout    = cfg["dropout"],
).to(device)

# ... plus tard dans la boucle d'entraînement (ligne ~160)

for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    
    for x, _ in train_loader:
        x = x.to(device)
        x_c, x_m, x_f = prepare_multiscale_targets(x)
        
        tokens = model.dinov2(x)
        mu, log_var = model.projection_head(tokens)
        z_sparse, kl, mask = model.sparse_latent(mu, log_var)  # K toujours 16!
        x_hat_c, x_hat_m, x_hat_f = model.decoder(z_sparse)
        
        metrics = total_loss(
            x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f,
            kl_loss  = kl,
            epoch    = epoch,
            # ❌ Pas de β adaptatif selon sparsité
        )
```

### Après (adaptatif)
```python
# train.py — MODIFIÉ

from utils import KController, BetaController
from losses.total_loss import compute_sparsity_metrics

model = SparseVAE(
    latent_dim = cfg["latent_dim"],  # 64
    k          = k,                   # K INITIAL
    dropout    = cfg["dropout"],
).to(device)

# ✅ AJOUTER: Contrôleurs adaptatifs
k_controller = KController(
    initial_k = k,
    min_k = 4,
    max_k = 60,
    step = 2,
    window = 5
)

beta_controller = BetaController(
    beta_max = cfg.get("beta_max", 4.0),
    target_sparsity = cfg.get("target_sparsity", 0.85),
    step = 0.1,
    warmup_epoch = 20
)

# ... dans la boucle d'entraînement

for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    
    # ✅ Métriques par epoch
    epoch_recon_losses = []
    epoch_sparsities = []
    
    for x, _ in train_loader:
        x = x.to(device)
        x_c, x_m, x_f = prepare_multiscale_targets(x)
        
        tokens = model.dinov2(x)
        mu, log_var = model.projection_head(tokens)
        
        # ✅ CHANGEMENT: Utiliser K(t) adaptatif
        k_t = k_controller.get_k()  # K courant adapté
        z_sparse, kl, mask = model.sparse_latent(mu, log_var, k=k_t)
        
        x_hat_c, x_hat_m, x_hat_f = model.decoder(z_sparse)
        
        metrics = total_loss(
            x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f,
            kl_loss  = kl,
            epoch    = epoch,
        )
        
        # ✅ AJOUTER: Calculer sparsité réelle
        sparsity_metrics = compute_sparsity_metrics(z_sparse, cfg["latent_dim"])
        
        epoch_recon_losses.append(metrics["l_recon"].item())
        epoch_sparsities.append(sparsity_metrics["sparsity_rate"])
        
        metrics["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # ✅ FIN D'EPOCH: Adapter K et β
    avg_recon_loss = sum(epoch_recon_losses) / len(epoch_recon_losses)
    avg_sparsity = sum(epoch_sparsities) / len(epoch_sparsities)
    
    # Adapter K selon trend recon_loss
    k_controller.update(avg_recon_loss)
    
    # Adapter β selon sparsité réelle
    beta_t = beta_controller.update(avg_sparsity, epoch)
    
    print(f"Epoch {epoch:03d} | "
          f"val_loss={val_loss:.5f} | "
          f"K(t)={k_controller.get_k()} | "
          f"β(t)={beta_t:.2f} | "
          f"sparsity={avg_sparsity:.2f}")
```

---

## CHANGEMENTS DÉTAILLÉS

### 1️⃣ Imports (début de train.py)

```python
# AJOUTER après autres imports

from utils import KController, BetaController
from losses.total_loss import compute_sparsity_metrics
```

### 2️⃣ Initialisation (après création du modèle)

```python
# Après: model = SparseVAE(...).to(device)

# AJOUTER:
k_controller = KController(
    initial_k=k,                              # K initial (ex: 16)
    min_k=4,                                  # K minimum
    max_k=60,                                 # K maximum
    step=2,                                   # Incrément/décrement
    window=5                                  # Fenêtre moyenne mobile
)

beta_controller = BetaController(
    beta_max=cfg.get("beta_max", 4.0),       # β max de config
    target_sparsity=cfg.get("target_sparsity", 0.85),  # 85% zéros
    step=0.1,                                 # Incrément β
    warmup_epoch=20                           # Phase 1 sans adaptation
)

# Logs
print(f"✅ Contrôleurs activés:")
print(f"   K: {k_controller.k_current} (range [{k_controller.k_min}, {k_controller.k_max}])")
print(f"   β: {beta_controller.beta} (max {beta_controller.beta_max}, target sparsity {beta_controller.target_sparsity:.0%})")
```

### 3️⃣ Boucle d'entraînement (dans for epoch)

```python
# AJOUTER après: model.train()

epoch_recon_losses = []
epoch_sparsities = []
```

### 4️⃣ Boucle batch (dans for x, _ in train_loader)

```python
# MODIFIER: Ligne où on appelle sparse_latent

# ❌ AVANT:
# z_sparse, kl, mask = model.sparse_latent(mu, log_var)

# ✅ APRÈS:
k_t = k_controller.get_k()
z_sparse, kl, mask = model.sparse_latent(mu, log_var, k=k_t)

# ... après total_loss(...)

# ✅ AJOUTER:
sparsity_metrics = compute_sparsity_metrics(z_sparse, cfg["latent_dim"])

epoch_recon_losses.append(metrics["l_recon"].item())
epoch_sparsities.append(sparsity_metrics["sparsity_rate"])
```

### 5️⃣ Fin d'epoch (après validation loss)

```python
# APRÈS: Calcul de val_loss mais AVANT le print/early_stopping

# ✅ AJOUTER:
avg_recon_loss = sum(epoch_recon_losses) / len(epoch_recon_losses)
avg_sparsity = sum(epoch_sparsities) / len(epoch_sparsities)

# Adapter K selon trend recon_loss (moyenne mobile)
k_controller.update(avg_recon_loss)

# Adapter β selon sparsité réelle obtenue
beta_t = beta_controller.update(avg_sparsity, epoch)

# Tracker historique
metrics_log = {
    "epoch": epoch,
    "k_current": k_controller.get_k(),
    "beta_current": beta_t,
    "sparsity": avg_sparsity,
    "recon_loss": avg_recon_loss,
}
```

### 6️⃣ Print d'epoch (modifier le print existant)

```python
# ❌ AVANT:
# print(f"  Epoch {epoch:03d} | val_loss={val_loss:.5f} | β={beta_cur:.2f}")

# ✅ APRÈS:
print(f"  Epoch {epoch:03d} | "
      f"val_loss={val_loss:.5f} | "
      f"K={k_controller.get_k():<3d} | "
      f"β={beta_controller.get_beta():.2f} | "
      f"sparsity={avg_sparsity:.2%}")
```

---

## MODIFICATION MODÈLE (sparse_latent.py)

Le `SparseLatentLayer` doit accepter **k variable** :

```python
# AVANT (sparse_latent.py ligne ~50):
class SparseLatentLayer(nn.Module):
    def __init__(self, latent_dim=64, k=16):
        self.k = k  # ❌ K fixé!
    
    def forward(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        z_sparse = self.topk_fn.apply(z, self.k)  # ❌ K toujours self.k
        ...

# APRÈS (sparse_latent.py ligne ~50):
class SparseLatentLayer(nn.Module):
    def __init__(self, latent_dim=64, k=16):
        self.k_default = k  # K par défaut
    
    def forward(self, mu, logvar, k=None):  # ✅ k optionnel
        if k is None:
            k = self.k_default  # Fallback
        z = self.reparameterize(mu, logvar)
        z_sparse = self.topk_fn.apply(z, k)  # ✅ Utiliser k passé
        ...
```

---

## RÉSUMÉ DES FICHIERS À MODIFIER

| Fichier | Changement | Priorité |
|---------|-----------|----------|
| `train.py` | Utiliser KController, BetaController, compute_sparsity_metrics | 🔴 CRITIQUE |
| `models/sparse_latent.py` | Accepter k variable en forward | 🔴 CRITIQUE |
| `losses/total_loss.py` | ✅ DÉJÀ FAIT (compute_sparsity_metrics ajouté) | ✅ |
| `utils/k_controller.py` | ✅ DÉJÀ CRÉÉ | ✅ |
| `utils/beta_controller.py` | ✅ DÉJÀ CRÉÉ | ✅ |
| `utils/__init__.py` | ✅ DÉJÀ CRÉÉ | ✅ |

---

## RÉSULTATS ATTENDUS

Après intégration complète, vous verrez dans les logs:

```
Epoch   1 | val_loss=0.52341 | K=16  | β=0.00 | sparsity=12%
Epoch   2 | val_loss=0.48932 | K=16  | β=0.00 | sparsity=15%
...
Epoch  20 | val_loss=0.31245 | K=16  | β=0.00 | sparsity=82%
Epoch  21 | val_loss=0.30567 | K=16  | β=0.05 | sparsity=83%  ← β commence à augmenter
Epoch  22 | val_loss=0.29876 | K=17  | β=0.10 | sparsity=84%  ← K augmente
Epoch  23 | val_loss=0.29234 | K=18  | β=0.15 | sparsity=85%  ← K continue, β augmente
...
Epoch  50 | val_loss=0.15623 | K=22  | β=2.30 | sparsity=85%  ← Approche β_max
Epoch  51 | val_loss=0.14532 | K=22  | β=4.00 | sparsity=85%  ← β fixé à max
Epoch  52 | val_loss=0.14231 | K=21  | β=4.00 | sparsity=85%  ← Optimisation fine
```

---

## POINTS IMPORTANTS

1. **K augmente** quand reconstruction loss se dégrado (plus de dimensions pour mieux reconstruire)
2. **K diminue** quand reconstruction loss s'améliore (compression pour tester sparsité extrême)
3. **β augmente** quand sparsité < target (pour pousser KL et forcer sparsité)
4. **β diminue** quand sparsité > target (trop creux, besoin de plus de dims)
5. **Phase 1 (20 epochs)**: β=0 toujours, K peut adapté selon recon
6. **Phase 2-3**: Adaptation complète K et β

---

## TEST RAPIDE

Après modification, lancez:

```bash
python train.py --k 16 --beta_max 4.0 --config configs/default.yaml 2>&1 | head -50
```

Vous devriez voir:
```
✅ Contrôleurs activés:
   K: 16 (range [4, 60])
   β: 0 (max 4.0, target sparsity 85%)
...
Epoch 001 | K-Control: ...
Epoch 021 | β-Control: ...
```
