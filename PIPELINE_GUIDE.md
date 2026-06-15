# 📊 PIPELINE COMPLET - Sparse VAE Forgery Detection

## 1️⃣ ARCHITECTURE GÉNÉRALE

```
╔════════════════════════════════════════════════════════════════════════╗
║                     7-ÉTAPE PIPELINE (CDC §)                          ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  IMAGE (224×224)                                                      ║
║     │                                                                  ║
║     ▼                                                                  ║
║  [1] DINOV2 (frozen) ──────────► 257 tokens (256 patches + 1 CLS)   ║
║     │ models/full_model.py                                           ║
║     ▼                                                                  ║
║  [2] PROJECTION HEAD ──────────► μ, logvar (B, 64)                   ║
║     │ models/projection_head.py                                      ║
║     ▼                                                                  ║
║  [3] SPARSE LATENT ────────────► z_sparse (B, 64, K=16 actif)        ║
║     │ models/sparse_latent.py                                        ║
║     ▼                                                                  ║
║  [4] MULTI-SCALE DECODER ─────► x_hat @ 3 résolutions              ║
║     │ models/multiscale_decoder.py                                   ║
║     ▼                                                                  ║
║  [5] RECONSTRUCTION LOSS ─────► L_recon + β·L_KL                    ║
║     │ losses/total_loss.py                                           ║
║     ├─► Entraînement (train.py)                                     ║
║     │                                                                  ║
║     ▼                                                                  ║
║  [6] HEATMAP (Étape 7) ───────► Localisation pixel-level            ║
║     │ evaluation/heatmap.py                                          ║
║     ▼                                                                  ║
║  [7] ANOMALY SCORE (Étape 7) ─► Score composite + AUTHENTIQUE/FORGÉ ║
║     └─► evaluation/anomaly_scorer.py                                 ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 2️⃣ FLUX D'EXÉCUTION COMPLET

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: ENTRAÎNEMENT                             │
└─────────────────────────────────────────────────────────────────────────┘

ÉTAPE 1 → train.py --k 16 --beta_max 4.0 --config configs/default.yaml
├─ Charge: config.yaml (hyperparamètres)
├─ Charge: data/dataset.py → MIDV2020Dataset('train')
├─ Crée: SparseVAE (models/full_model.py)
│  └─ Charge DINOv2 (frozen)
│  └─ Crée ProjectionHead, SparseLatentLayer, MultiScaleDecoder
├─ Entraîne: 3 phases (β=0 → β annealing → β=4)
│  ├─ Phase 1 (1-20): β = 0
│  ├─ Phase 2 (21-50): β linéaire 0 → 4
│  └─ Phase 3 (51-100): β = 4
├─ Valide: Projection Head à epoch 20 (evaluation/metrics.py)
├─ Early stopping: patience=15
├─ Sauvegarde:
│  ├─ checkpoints/k{K}_beta{β}_best.pt ← poids du modèle
│  └─ results/ablation/k{K}_beta{β}.json ← métriques (sans AUC-ROC)
│
└─ OUTPUT: Best checkpoint + JSON

┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: ABLATION (optionnel)                       │
└─────────────────────────────────────────────────────────────────────────┘

ÉTAPE 2 → ablation.py (grid search)
├─ Lit: configs/default.yaml → k_values, beta_max_values
├─ Pour chaque (K, β):
│  ├─ Lance: train.py --k K --beta_max β
│  ├─ Collecte: JSON résultat
│  └─ Évalue: Critères K et β
├─ Filtre: "qualified": true/false
│  └─ Critères: sparsity_rate, n_collapsed, active_ratio, final_kl
└─ OUTPUT: results/ablation/*.json (filtrés)

┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: ÉVALUATION (TEST AUC)                        │
└─────────────────────────────────────────────────────────────────────────┘

ÉTAPE 3 → evaluate.py
├─ Lit: results/ablation/*.json (runs "qualified": true)
├─ Pour chaque run qualifié:
│  ├─ Charge: checkpoint_best.pt
│  ├─ Charge: data/dataset.py
│  │  ├─ MIDV2020Dataset('val') ← authentiques pour calibration
│  │  └─ FMIDV2022Dataset() ← forgeries pour test
│  ├─ Calcule: Erreurs reconstruction sur les 2 datasets
│  ├─ Dérive: AUC-ROC (sklearn)
│  └─ Met à jour: results/ablation/k{K}_beta{β}.json → auc_roc: ...
│
├─ Crée: AnomalyScorer (evaluation/anomaly_scorer.py)
│  ├─ fit() sur val authentiques
│  └─ predict() sur test set
│
└─ OUTPUT: Updated JSON + scores

┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 4: TEST INFÉRENCE (RUNTIME)                      │
└─────────────────────────────────────────────────────────────────────────┘

ÉTAPE 4 → test_image_dino.py ou script personnalisé
├─ Charge: Best checkpoint
├─ Charge: AnomalyScorer (calibré sur val authentiques)
├─ Pour chaque image test:
│  ├─ Forward SparseVAE → heatmap
│  ├─ Anomaly score (4 composantes)
│  ├─ Prédiction: AUTHENTIQUE ou FORGÉ
│  └─ Visualisation: heatmap superposée
│
└─ OUTPUT: Prédictions + visualisations
```

---

## 3️⃣ DÉPENDANCES ENTRE FICHIERS

```
                   ┌──────────────────────────┐
                   │  configs/default.yaml    │
                   │  (hyperparamètres)       │
                   └────────┬─────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
    ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
    │  train.py   │   │ ablation.py  │   │evaluate.py  │
    │             │   │              │   │             │
    └─────┬───────┘   └──────┬───────┘   └──────┬──────┘
          │                  │                   │
          └──────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ models/         │
                    │ ├─ full_model   │
                    │ ├─ proj_head    │
                    │ ├─ sparse_lat   │
                    │ └─ decoder      │
                    └────────┬────────┘
                             │
                    ┌────────▼─────────────┐
                    │ data/               │
                    │ ├─ dataset.py       │
                    │ │  ├─ MIDV2020      │
                    │ │  └─ FMIDV2022     │
                    │ └─ data/            │
                    │    ├─ midv2020/     │
                    │    └─ fmidv2022/    │
                    └────────┬────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌──────────────┐      ┌────────────────┐     ┌───────────────┐
│ losses/      │      │ evaluation/    │     │ checkpoints/  │
│ total_loss   │      │ ├─ heatmap.py  │     │ k*_beta*.pt   │
│              │      │ ├─ metrics.py  │     │               │
│              │      │ ├─ anomaly_    │     │ results/      │
│              │      │ │  scorer.py   │     │ ablation/     │
│              │      │ └─ __init__.py │     │ *.json        │
│              │      └────────────────┘     └───────────────┘
└──────────────┘
```

---

## 4️⃣ FICHIERS CLÉS ET RESPONSABILITÉS

### 🎯 **Fichiers d'Entrée (Données)**
```
data/dataset.py ─────► MIDV2020Dataset('train'/'val') ──► Images authentiques
                       FMIDV2022Dataset() ──────────────► Images forgées
```

### 🧠 **Fichiers de Modèle (Architecture)**
```
models/full_model.py ──────────► SparseVAE (complète)
                                 ├─ load_dinov2() [frozen]
                                 ├─ extract_dinov2_features()
                                 
models/projection_head.py ─────► ProjectionHead (μ, logvar)

models/sparse_latent.py ───────► SparseLatentLayer (sparsité Top-K)

models/multiscale_decoder.py ──► MultiScaleDecoder (3 résolutions)
```

### 💡 **Fichiers d'Entraînement**
```
losses/total_loss.py ──────────► total_loss() + beta_annealing()
                                 └─ MSE + SSIM + KL

train.py ──────────────────────► train_one_run() (1 run)
                                 ├─ 3 phases
                                 ├─ Early stopping
                                 └─ Checkpoint save

ablation.py ───────────────────► Grid search (K, β)
```

### 🔍 **Fichiers d'Évaluation**
```
evaluation/metrics.py ─────────► validate_projection_head()

evaluation/heatmap.py ─────────► compute_multiscale_heatmap()
                                 └─ Localisation forgeries

evaluation/anomaly_scorer.py ──► AnomalyScorer class
                                 ├─ fit() — calibration
                                 └─ predict() — inférence

evaluate.py ───────────────────► Calcule AUC-ROC final
```

### 🏃 **Fichier de Test**
```
test_image_dino.py ────────────► Inférence sur une image
```

---

## 5️⃣ COMMENT LANCER LE TRAVAIL

### **Option A: UN SEUL RUN (Démarrage Rapide)**

```bash
# 1. Installer dépendances
pip install -r requirements.txt

# 2. Remplir data/dataset.py (charger MIDV2020 et FMIDV2022)
# ... À faire par l'utilisateur

# 3. Lancer un seul run (K=16, β=4)
python train.py --k 16 --beta_max 4.0 --config configs/default.yaml

# OUTPUT:
# ✅ checkpoints/k16_beta4_best.pt
# ✅ results/ablation/k16_beta4.json (sans AUC-ROC)
```

### **Option B: ABLATION COMPLÈTE (Grid Search)**

```bash
# 1. Installer dépendances
pip install -r requirements.txt

# 2. Remplir data/dataset.py

# 3. Lancer ablation (teste toutes combinaisons)
python ablation.py

# OUTPUT:
# ✅ results/ablation/k8_beta0.5.json
# ✅ results/ablation/k8_beta1.0.json
# ✅ results/ablation/k16_beta4.json
# ... (12 runs si k_values=[8,16,32] × beta_values=[0.5,1,2,4])
```

### **Option C: ÉVALUATION FINALE (AUC-ROC)**

```bash
# 1. Après ablation.py

# 2. Évaluer les runs qualifiés
python evaluate.py

# OUTPUT:
# ✅ results/ablation/k*_beta*.json (WITH auc_roc: 0.98)
```

### **Option D: TEST INFÉRENCE (1 image)**

```bash
# 1. Après evaluation.py

# 2. Tester une image
python test_image_dino.py \
    --image path/to/test.jpg \
    --checkpoint checkpoints/k16_beta4_best.pt \
    --anomaly_scorer checkpoints/k16_beta4_scorer.json

# OUTPUT:
# ✅ Score: 0.234
# ✅ Label: AUTHENTIQUE
# ✅ Heatmap visualization
```

---

## 6️⃣ NON, PAS UN SEUL "MAIN"

Le projet est **modulaire** — chaque étape est un script indépendant:

| Étape | Script | Commande | Prérequis |
|-------|--------|----------|-----------|
| 1 | train.py | `python train.py --k 16 --beta_max 4` | config.yaml, dataset.py |
| 2 | ablation.py | `python ablation.py` | config.yaml, dataset.py |
| 3 | evaluate.py | `python evaluate.py` | Runs from ablation.py |
| 4 | test_image_dino.py | `python test_image_dino.py ...` | Best checkpoint |

**Raison**: Chaque étape prend du temps (entraînement = heures), donc on lance chacune séparément.

---

## 7️⃣ WORKFLOW RECOMMANDÉ

```bash
# ═══════════════════════════════════════════════════════════════

# STEP 1: Setup
pip install -r requirements.txt

# STEP 2: Préparer données (vous devez faire ça)
# 🔴 Remplir data/dataset.py avec MIDV2020Dataset et FMIDV2022Dataset

# ═══════════════════════════════════════════════════════════════

# STEP 3: Test une configuration (quick test)
python train.py --k 16 --beta_max 4.0 --config configs/default.yaml
# ⏱️ ~30-60 min sur GPU

# STEP 4: Vérifier résultat
cat results/ablation/k16_beta4.json
# ✅ {"val_loss": ..., "sparsity_rate": ..., "auc_roc": null}

# ═══════════════════════════════════════════════════════════════

# STEP 5: Ablation complète (optionnel)
python ablation.py
# ⏱️ ~4-6 heures (12 runs × 30 min)

# STEP 6: Évaluation finale
python evaluate.py
# ⏱️ ~30 min
# ✅ results/ablation/k*_beta*.json (WITH auc_roc)

# ═══════════════════════════════════════════════════════════════

# STEP 7: Test inférence
python test_image_dino.py --image path/to/doc.jpg
```

---

## 8️⃣ VÉRIFICATION COHÉRENCE (Avant de lancer)

```bash
# Vérifie que tout import bien
python -c "from models import SparseVAE; print('✅ Models OK')"
python -c "from evaluation import AnomalyScorer; print('✅ Evaluation OK')"
python -c "import yaml; yaml.safe_load(open('configs/default.yaml')); print('✅ Config OK')"

# Vérifie datasets
python -c "from data import MIDV2020Dataset; print('✅ Dataset OK (needs impl)')"
```

---

## 📌 RÉSUMÉ

| Aspect | Réponse |
|--------|---------|
| **Besoin UN main?** | ❌ NON — chaque script est autonome |
| **Ordre d'exécution** | train.py → ablation.py → evaluate.py → test_image_dino.py |
| **Peut-on sauter des étapes?** | ✅ OUI — chaque étape indépendante |
| **Combien de temps?** | ~30-60 min per run, ~6h grid search |
| **Fichiers config requis** | configs/default.yaml ✅, data/dataset.py 🔴 (À FAIRE) |
| **Sorties principales** | checkpoints/*.pt + results/ablation/*.json |

