# 📋 DESCRIPTION COMPLÈTE DU PROJET - Sparse VAE Forgery Detection

## 1. CONTEXTE ET OBJECTIF

**Projet**: Détection de documents forgés (falsifiés) en utilisant un **Sparse VAE** avec des features **DINOv2** comme détecteur d'anomalies.

**Objectif Principal**: 
- Entraîner un modèle VAE sparse sur des documents authentiques (MIDV-2020)
- Localiser pixel-level les régions forgées (heatmap)
- Classifier images: AUTHENTIQUE vs FORGÉ avec score de confiance
- Évaluer sur documents forgés (FMIDV-2022)

**Référence Technique**: Cahier des Charges (CDC) 22 pages — sections §2 à §7

---

## 2. ARCHITECTURE GÉNÉRALE

### Pipeline 7-étapes (CDC §2-7)

```
IMAGE (224×224)
    ↓
[ÉTAPE 1] DINOv2 (frozen)
    └─ Extrait 257 tokens (256 patches + 1 CLS) de dimension 768
    └─ Modèle congelé (non entraîné) depuis Meta/Facebook
    
    ↓
[ÉTAPE 2] Projection Head (Encoder)
    └─ Réduit 256 patches → μ et logvar de dimension 64
    └─ MLP avec 2 couches cachées [512, 256], dropout
    
    ↓
[ÉTAPE 3] Sparse Latent Layer (Goulot d'étranglement)
    └─ Applique sparsité: garder TOP-K dimensions sur 64 (K=16 par défaut)
    └─ Calcule KL divergence (reconstruction du σ² initial)
    └─ Utilise Straight-Through Estimator (STE) pour backprop
    
    ↓
[ÉTAPE 4] Multi-Scale Decoder
    └─ Génère reconstructions à 3 résolutions:
       ├─ Coarse (16×16)
       ├─ Medium (32×32)
       └─ Fine (64×64)
    
    ↓
[ÉTAPE 5] Perte Multi-Échelle + β-annealing
    └─ L_total = L_recon + β·L_KL
    └─ Protocole 3-phases:
       ├─ Phase 1 (epoch 1-20): β = 0 (reconstruction pure)
       ├─ Phase 2 (epoch 21-50): β linéaire 0 → β_max
       └─ Phase 3 (epoch 51-100): β = β_max (fixé)
    
    ↓
[ÉTAPE 6] Heatmap Anomalie (Localisation)
    └─ Fusionne erreurs reconstruction 3 échelles:
       └─ H = 0.2×E_coarse + 0.3×E_medium + 0.5×E_fine
    └─ Normalisée, lissée (Gaussian σ=1.5)
    └─ Résolution finale: 64×64 → upscale 224×224
    
    ↓
[ÉTAPE 7] Score Anomalie Composite
    └─ Agrège 4 signaux:
       ├─ S_recon: MSE reconstruction fine
       ├─ S_kl: KL divergence
       ├─ S_heatmap: percentile 95 de la heatmap
       └─ S_latent: norme activations latentes
    └─ Normalisation z-score + poids composites
    └─ Classification: AUTHENTIQUE (score < θ) vs FORGÉ (score ≥ θ)
```

---

## 3. TECHNOLOGIE ET STACK

### Framework et Librairies

| Composant | Technologie |
|-----------|-------------|
| **Modèle DINOv2** | `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` |
| **Framework ML** | PyTorch 2.0+ |
| **Perte SSIM** | `kornia.losses.SSIMLoss` |
| **Métriques** | `scikit-learn` (AUC-ROC, F1, Precision, Recall) |
| **Traitement image** | OpenCV, scipy.ndimage |
| **Config** | PyYAML |
| **Experiment tracking** | wandb (optionnel) |

### Dimensionalité des Tensors

```
Image input:           (B, 3, 224, 224)
DINOv2 patches:        (B, 256, 768)
Projection Head μ:     (B, 64)
Sparse z:              (B, 64) avec K=16 dims actif
Decoder output (fine): (B, 3, 64, 64)
Heatmap final:         (64, 64) [normalisée 0-1]
Anomaly score:         scalar
```

---

## 4. FICHIERS IMPLÉMENTÉS ET RÔLE

### 4.1 Architecture (models/)

#### `models/full_model.py` ✅ COMPLET
- **Classe**: `SparseVAE`
- **Responsabilités**:
  - Combine tous les composants (DINOv2, ProjectionHead, SparseLatent, Decoder)
  - Charge DINOv2 frozen via `load_dinov2()`
  - Méthodes: `forward()`, `encode()`, `decode()`
- **Fonction**: `extract_dinov2_features(dinov2, image)` → (patch_tokens, cls_token)
- **Note**: DINOv2 est attribut mais **jamais optimisé** (pas dans optimizer)

#### `models/projection_head.py` ✅ COMPLET
- **Classe**: `ProjectionHead`
- **Responsabilités**: Encodeur MLP
  - Input: 256 patches (768 dims chacun) → flat concatenation
  - Hidden: [512, 256] avec dropout
  - Output: μ (B, 64), logvar (B, 64)

#### `models/sparse_latent.py` ✅ COMPLET
- **Classe**: `SparseLatentLayer`
- **Responsabilités**: 
  - Reparameterization trick (μ, σ) → z
  - Top-K sparsité (K=16, reste = 0)
  - KL divergence loss
  - STE (Straight-Through Estimator) pour gradients

#### `models/multiscale_decoder.py` ✅ COMPLET
- **Classe**: `MultiScaleDecoder`
- **Responsabilités**:
  - Décode z_sparse → 3 reconstructions:
    - Coarse: 16×16
    - Medium: 32×32
    - Fine: 64×64
  - Architectures séparées pour chaque échelle

### 4.2 Perte (losses/)

#### `losses/total_loss.py` ✅ COMPLET
- **Fonction**: `total_loss(x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f, kl_loss, epoch, beta_max)`
- **Calcul**:
  - `L_coarse = 0.1 × (0.8×MSE + 0.2×(1-SSIM))` @ 16×16
  - `L_medium = 0.3 × (0.8×MSE + 0.2×(1-SSIM))` @ 32×32
  - `L_fine = 0.6 × (0.8×MSE + 0.2×(1-SSIM))` @ 64×64
  - `L_recon = L_coarse + L_medium + L_fine`
  - `L_total = L_recon + β(epoch)·KL`
- **Fonction**: `beta_annealing(epoch, beta_max)` → Phase 3 protocol

### 4.3 Entraînement (root scripts)

#### `train.py` ✅ COMPLET
- **Fonction**: Entraîne UN run (K, β)
- **Commande**: `python train.py --k 16 --beta_max 4.0 --config configs/default.yaml`
- **Responsabilités**:
  - Charge config YAML
  - Crée DataLoaders (MIDV2020)
  - Initialise SparseVAE
  - Entraîne 100 epochs avec early stopping (patience=15)
  - Validation projection head epoch 20 (CDC §2.6)
  - Sauvegarde checkpoint + JSON
- **Sorties**:
  - `checkpoints/k{K}_beta{β}_best.pt`
  - `results/ablation/k{K}_beta{β}.json`

#### `ablation.py` ✅ COMPLET
- **Fonction**: Lance MULTIPLE runs (grid search)
- **Commande**: `python ablation.py`
- **Responsabilités**:
  - Lit k_values et beta_max_values de config
  - Boucle: pour chaque (K, β), lance `train.py`
  - Filtre runs "qualified" selon critères
- **Sorties**: 12 JSON (ou N selon grid)

### 4.4 Évaluation (evaluation/)

#### `evaluation/heatmap.py` ✅ IMPLÉMENTÉ (320 lignes)
- **Fonction**: Localisation pixel-level des forgeries
- **Fonctions principales**:
  - `compute_multiscale_heatmap(model, image, dinov2, projection_head, sparse_latent, decoder, train_stats, device)` 
    - Fusionne 3 erreurs reconstruction pondérées
    - Retourne heatmap (64, 64) lissée normalisée
  - `visualize_heatmap(original_image, heatmap)` 
    - Superpose heatmap sur image originale (224×224)
    - Seuillage binaire percentile 95
  - `plot_heatmap(original_image, heatmap, title, save_path)`
    - Matplotlib visualization
  - `compute_heatmap_stats(model, val_loader, device)`
    - Calibre sur val authentiques (μ, σ pour normalisation)
  - `compute_localization_metrics(heatmap, ground_truth)`
    - IoU, Pixel-AUC, Average Precision
- **Intégration**: Étape 6 pipeline, utilisée par anomaly_scorer

#### `evaluation/anomaly_scorer.py` ✅ IMPLÉMENTÉ (270 lignes)
- **Classe**: `AnomalyScorer`
- **Responsabilités**:
  - Calibration et inférence du score d'anomalie composite
  
- **Méthodes principales**:
  - `fit(model, dinov2, projection_head, sparse_latent, decoder, val_loader_authentic, heatmap_stats, device)`
    - Calibre sur val authentiques
    - Calcule μ, σ pour chaque composante
    - Définit seuil θ = percentile 95
  
  - `predict(model, dinov2, ..., image, heatmap_stats, device)`
    - Retourne (score, label) pour image unique
    - Label: "AUTHENTIQUE" ou "FORGÉ"
  
  - `_compute_raw_scores(model, image, ...)` [Interne]
    - Calcule 4 composantes:
      - S_recon: MSE reconstruction fine
      - S_kl: KL divergence
      - S_heatmap: percentile 95 heatmap
      - S_latent: ||z||_2 / K
  
  - `_normalize(raw_scores)` [Interne]
    - Z-score norm: (x - μ) / σ
  
  - `_composite(norm_scores)` [Interne]
    - Pondération: 0.4×S_recon + 0.2×S_kl + 0.3×S_heatmap + 0.1×S_latent
  
  - `save(path)` / `load(path)`
    - Persiste calibration en JSON
  
  - `batch_predict(model, ..., data_loader, ...)` 
    - Évalue dataset entier

- **Fonction**: `evaluate_anomaly_detector(scores, true_labels, threshold)`
  - Calcule TP, FP, TN, FN, Precision, Recall, F1, AUC-ROC

#### `evaluation/metrics.py` ✅ COMPLET
- **Fonction**: `validate_projection_head(model, val_loader, device)`
  - Valide après Phase 1 (epoch 20)
  - Vérifie dimensionalité et qualité features

#### `evaluate.py` ✅ COMPLET
- **Fonction**: Évaluation finale sur test set
- **Commande**: `python evaluate.py`
- **Responsabilités**:
  - Charge tous runs "qualified" de ablation.py
  - Pour chaque run:
    - Charge checkpoint
    - Calcule erreurs sur MIDV2020 val (authentiques)
    - Calcule erreurs sur FMIDV2022 (forgeries)
    - Dérive AUC-ROC
    - Met à jour JSON → auc_roc: ...
- **Sorties**: Updated JSON avec AUC-ROC

### 4.5 Configuration

#### `configs/default.yaml` ✅ COMPLET
```yaml
# Architecture
latent_dim: 64              # Dimension z
k: 16                       # Top-K sparsité

# Optimisation
batch_size: 32
lr: 1.0e-4
wd: 1.0e-4
epochs: 100
patience: 15

# Loss
alpha: 0.1                  # Poids L_coarse
lambda_m: 0.3               # Poids L_medium
lambda_f: 0.6               # Poids L_fine
epsilon: 0.2                # Poids SSIM

# Ablation
k_values: [8, 16, 32]
beta_max_values: [0.5, 1.0, 2.0, 4.0]
```

### 4.6 Données

#### `data/dataset.py` ⚠️ STUB (À COMPLÉTER)
- **Classes**: 
  - `MIDV2020Dataset(split='train'/'val'/'test')`
    - Doit charger images authentiques 224×224 normalisées
    - label=0 (authentique)
  - `FMIDV2022Dataset()`
    - Doit charger images forgées 224×224
    - label=1 (forgé)
  - `FantasyIDDataset()`, `SIDTDDataset()` (optionnels cross-dataset)

---

## 5. WORKFLOWS IMPLÉMENTÉS

### 5.1 Entraînement (CDC §5.4)

```
train.py --k K --beta_max β
├─ Charge data MIDV2020
├─ Initialise SparseVAE + optimizer
├─ 100 epochs:
│  ├─ Forward pass: image → heatmap
│  ├─ Backward: L_total.backward()
│  ├─ Validation: calc val_loss
│  ├─ Si epoch==20: validate_projection_head()
│  ├─ Early stopping: patience=15
│  └─ Save best checkpoint
├─ Compute latent criteria
└─ Save JSON results
```

### 5.2 Évaluation Anomaly Score

```
evaluate.py
├─ Pour chaque run qualifié:
│  ├─ Charge checkpoint
│  ├─ Crée AnomalyScorer
│  ├─ fit() sur val authentiques
│  ├─ predict() sur FMIDV2022
│  ├─ Calcule AUC-ROC
│  └─ Update JSON
└─ Rank runs par AUC-ROC
```

### 5.3 Inférence (Runtime)

```
test_image_dino.py --image doc.jpg
├─ Charge best checkpoint
├─ Charge AnomalyScorer
├─ Pour image:
│  ├─ Forward SparseVAE
│  ├─ Compute heatmap
│  ├─ Compute anomaly score
│  ├─ Predict label
│  └─ Visualize
└─ Output: label + score + heatmap image
```

---

## 6. FICHIERS CRÉÉS ET ÉTAT

### ✅ COMPLET ET COHÉRENT

```
models/
├─ full_model.py           ✅ SparseVAE complet
├─ projection_head.py      ✅ Encoder MLP
├─ sparse_latent.py        ✅ Top-K + KL
├─ multiscale_decoder.py   ✅ 3-scale decoder
└─ __init__.py             ✅ Exports

losses/
├─ total_loss.py           ✅ Multi-scale loss + β-annealing
└─ __init__.py             ✅ Exports

evaluation/
├─ heatmap.py              ✅ Localisation (320 lignes)
├─ anomaly_scorer.py       ✅ Scoring + classification (270 lignes)
├─ metrics.py              ✅ Validation
└─ __init__.py             ✅ Exports

configs/
└─ default.yaml            ✅ Hyperparamètres

train.py                   ✅ UN run entraînement
ablation.py                ✅ Grid search
evaluate.py                ✅ AUC-ROC final
test_image_dino.py         ✅ Inférence
requirements.txt           ✅ Dépendances
```

### ⚠️ À COMPLÉTER

```
data/
├─ dataset.py              🔴 STUB (à remplir)
│  ├─ MIDV2020Dataset      🔴 À implémenter
│  └─ FMIDV2022Dataset     🔴 À implémenter
└─ __init__.py             ✅ Exports
```

---

## 7. COMMENT UTILISER

### Installation

```bash
pip install -r requirements.txt
```

### Étape 1: Préparer Données

```python
# À faire dans data/dataset.py
class MIDV2020Dataset(Dataset):
    def __init__(self, split='train', root_dir='data/midv2020'):
        # Charger images authentiques 224×224 normalisées [0,1]
        # label: 0 (authentique)
        ...
    
    def __getitem__(self, idx):
        return image, label

class FMIDV2022Dataset(Dataset):
    def __init__(self, root_dir='data/fmidv2022'):
        # Charger images forgées 224×224
        # label: 1 (forgé)
        ...
```

### Étape 2: UN RUN TEST (30-60 min)

```bash
python train.py --k 16 --beta_max 4.0 --config configs/default.yaml

# OUTPUT:
# ✅ checkpoints/k16_beta4_best.pt
# ✅ results/ablation/k16_beta4.json
```

### Étape 3: ABLATION (optionnel, 4-6h)

```bash
python ablation.py

# Teste 12 combinaisons (K=[8,16,32] × β=[0.5,1,2,4])
# OUTPUT: 12 checkpoints + 12 JSON
```

### Étape 4: ÉVALUATION (30 min)

```bash
python evaluate.py

# Calcule AUC-ROC de chaque run
# OUTPUT: Updated JSON avec auc_roc
```

### Étape 5: TEST INFÉRENCE (<1 sec/image)

```bash
python test_image_dino.py --image path/to/document.jpg

# OUTPUT:
# Score: 0.234
# Label: AUTHENTIQUE
# Heatmap: visualization.png
```

---

## 8. POINTS CLÉS À COMPRENDRE

1. **DINOv2 Frozen**: Jamais entraîné, extrait seulement features
2. **Sparse Latent**: Seulement K=16/64 dims actif (réduction 75%)
3. **Multi-Scale Loss**: Combine 3 résolutions avec poids [0.1, 0.3, 0.6]
4. **β-Annealing**: 3 phases pour équilibrer reconstruction vs regularization
5. **Heatmap**: Fusion 3-échelles → localisation forgeries
6. **Anomaly Score**: 4 signaux normalisés → prédiction binaire
7. **Modular Design**: Chaque script indépendant, chacun peut s'exécuter seul
8. **Early Stopping**: patience=15 pour éviter overfitting

---

## 9. RÉSUMÉ TECHNIQUE

| Aspect | Valeur |
|--------|--------|
| **Encoder** | DINOv2 ViT-B/14 (frozen) |
| **Bottleneck** | Sparse latent 64→16 dims |
| **Decoder** | 3 branches (16×16, 32×32, 64×64) |
| **Loss** | Multi-scale MSE+SSIM + β·KL |
| **Training** | 100 epochs, batch=32, early stopping |
| **Evaluation** | AUC-ROC sur forgeries |
| **Anomaly Detection** | 4-signal composite + threshold |
| **Heatmap** | 64×64 multi-scale fusion |
| **Framework** | PyTorch |
| **GPU Required** | Fortement recommandé (2GB+) |

---

## 10. PROCHAINES ÉTAPES

1. **Remplir `data/dataset.py`** avec chargement MIDV2020 et FMIDV2022
2. **Tester UN run**: `python train.py --k 16 --beta_max 4.0`
3. **Vérifier output**: checkpoints/ et results/ablation/
4. **Lancer ablation**: `python ablation.py` (overnight)
5. **Évaluer**: `python evaluate.py`
6. **Tester inférence**: `python test_image_dino.py --image ...`

---

**Tous les codes sont prêts, cohérents et documentés. Le projet est modular et peut s'exécuter étape par étape.**
