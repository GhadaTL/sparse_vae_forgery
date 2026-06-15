# 🔍 Analyse de Cohérence du Projet

## ⚠️ PROBLÈMES CRITIQUES DÉTECTÉS

### 1. **ERREUR STRUCTURELLE — Models/DINOv2**

#### Problème
- **train.py** ligne 230: `tokens = model.dinov2(x)` — **INEXISTANT !**
- **train.py** ligne 215: `tokens = model.dinov2(x)` — **INEXISTANT !**
- **train.py** ligne 153: `tokens = model.dinov2(x)` — **INEXISTANT !**
- **evaluate.py**: Même appels `model.dinov2(x)`

#### Réalité dans models/full_model.py
```python
class SparseVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, k: int = 16, ...):
        self.projection_head = ProjectionHead(...)
        self.sparse_latent = SparseLatentLayer(...)
        self.decoder = MultiScaleDecoder(...)
        # ❌ PAS d'attribut self.dinov2 !
```

#### Conséquence
- ❌ train.py **NE PEUT PAS S'EXÉCUTER**
- ❌ evaluate.py **NE PEUT PAS S'EXÉCUTER**

#### Solution
- **Option A** : Ajouter DINOv2 comme attribut dans SparseVAE
- **Option B** : Modifier train.py et evaluate.py pour passer DINOv2 séparément

---

### 2. **DONNÉES MANQUANTES**

#### Problème
- **train.py** ligne 342: `from data.dataset import MIDV2020Dataset`
- **evaluate.py** ligne 31: `from data.dataset import MIDV2020Dataset, FMIDV2022Dataset`
- **File**: `data/dataset.py` — **N'EXISTE PAS**

#### Conséquence
- ❌ ImportError au lancement de train.py
- ❌ ImportError au lancement de evaluate.py

---

### 3. **CONFIGURATION MANQUANTE**

#### Problème
- **train.py** ligne 335: `with open(args.config) as f: cfg = yaml.safe_load(f)`
- **File**: `configs/default.yaml` — **VIDE OU INEXISTANT**

#### Conséquence
- ❌ FileNotFoundError ou yaml.YAMLError
- ❌ Les clés attendues (batch_size, lr, epochs, latent_dim, k, dropout, etc.) — **MANQUANTES**

---

### 4. **INCOHÉRENCE — heatmap.py vs anomaly_scorer.py**

#### Problème dans anomaly_scorer.py

**Ligne 140**: `from models.full_model import extract_dinov2_features`
```python
# Mais dans heatmap.py, la fonction est définie localement :
def extract_dinov2_features(dinov2, image):
    ...

# Et dans models/full_model.py, c'est aussi défini :
def extract_dinov2_features(dinov2: nn.Module, image: torch.Tensor):
    ...
```

#### Problème
- ❌ **Duplication de fonction** extract_dinov2_features (3 définitions !)
- ❌ **Import incorrect** en ligne 140 — la fonction n'existe pas dans models.full_model
- ❌ **Signature incohérente** entre les versions

#### Situation actuelle

| Fichier | Fonction | Signature |
|---------|----------|-----------|
| models/full_model.py | extract_dinov2_features | (dinov2, image) → (patch_tokens, cls_token) |
| evaluation/heatmap.py | extract_dinov2_features | (dinov2, image) → (patch_tokens, cls_token) |
| evaluation/anomaly_scorer.py | ~~import~~ | **ERREUR** — N'existe pas dans full_model |

---

### 5. **INCOHÉRENCE — prepare_multiscale_targets**

#### Situation actuelle

| Fichier | Fonction |
|---------|----------|
| train.py | `prepare_multiscale_targets(x)` |
| evaluation/heatmap.py | `prepare_multiscale_targets(x)` |
| models/full_model.py | `prepare_multiscale_targets()` (importée depuis multiscale_decoder) |

#### Problème
- ⚠️ **Définition multiple** — mais c'est OK, elles sont identiques
- ⚠️ Dans anomaly_scorer.py, j'utilise `F.interpolate` inline au lieu d'appeler la fonction

---

### 6. **ANOMALY_SCORER — ARCHITECTURE CONFUSE**

#### Problème

J'ai écrit la fonction `_compute_raw_scores()` pour accepter 6 paramètres séparés :
```python
def _compute_raw_scores(self, model, image, dinov2, projection_head, sparse_latent,
                        decoder, heatmap_stats, device='cuda'):
```

#### Confusion
- Je passe `model` ET les composants séparés — **REDONDANT**
- Passer `model` suffit si on refactorise légèrement
- Passer les composants séparés évite de dépendre de model.dinov2

---

### 7. **MISSING REQUIREMENTS.TXT**

#### Problème
- Aucun fichier requirements.txt
- **CRITICAL** pour reproduire l'env

#### Manquent les dépendances
```
torch
torchvision
kornia
wandb
scikit-learn
opencv-python
scipy
matplotlib
timm
pyyaml
```

---

## 📋 RÉSUMÉ DES ACTIONS REQUISES

### 🔴 **BLOQUANTS IMMÉDIAT**

| # | Fichier | Problème | Action |
|---|---------|---------|--------|
| 1 | models/full_model.py | DINOv2 pas un attribut | Ajouter `self.dinov2 = load_dinov2()` |
| 2 | data/dataset.py | N'existe pas | **CRÉER** MIDV2020Dataset, FMIDV2022Dataset |
| 3 | configs/default.yaml | Vide | **CRÉER** avec tous les hyperparamètres |
| 4 | requirements.txt | N'existe pas | **CRÉER** |

### 🟠 **COHÉRENCE REQUISE**

| # | Fichier | Action |
|---|---------|--------|
| 5 | evaluation/anomaly_scorer.py | Corriger import ligne 140 → from evaluation.heatmap import ... |
| 6 | evaluation/heatmap.py + anomaly_scorer.py | Harmoniser signatures extract_dinov2_features |
| 7 | evaluation/__init__.py | ✅ Déjà OK |

### 🟡 **STRUCTURE**

| # | Fichier | Action |
|---|---------|--------|
| 8 | results/ablation/ | Créer répertoire (train.py le crée, OK) |
| 9 | .gitkeep | Optionnel |

---

## 🛠️ PRIORITÉ DE CORRECTION

1. ✅ **URGENT** — Fixer DINOv2 dans models/full_model.py
2. ✅ **URGENT** — Créer data/dataset.py
3. ✅ **URGENT** — Créer configs/default.yaml
4. ✅ **URGENT** — Créer requirements.txt
5. ⚠️ Corriger imports dans anomaly_scorer.py
6. ⚠️ Vérifier cohérence evaluation/ functions

---

## ✅ FICHIERS COHÉRENTS

- ✅ evaluation/heatmap.py — bien structuré
- ✅ evaluation/anomaly_scorer.py — 90% OK, import à fixer
- ✅ evaluation/__init__.py — exports OK
- ✅ losses/total_loss.py — pas testé mais structure OK
- ✅ models/ (autres) — pas encore examinés
