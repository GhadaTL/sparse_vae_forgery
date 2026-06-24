# Adaptive Sparse VAE — Identity Document Forgery Detection

**"Adaptive Sparse Variational Autoencoder with DINOv2 Features for Document Forgery Detection"**

Wissal KARABAKA & Ghada TLILI — ENET'COM Sfax, CRNS 2026  
Encadrantes : Mme Sonda AMMAR, Mme Mariem Regaieg

---

## Architecture

```
Image RGB 224×224
    → DINOv2 ViT-B/14 [frozen]   (B, 256, 768)
    → ProjectionHead              (B, 64) μ, log σ²
    → SparseLatentLayer           (B, 64) z_sparse  +  KL
    → MultiscaleDecoder           x̂_coarse (16²), x̂_medium (32²), x̂_fine (64²)
```

**Contribution principale** : Contrôleur Adaptatif K — apprend automatiquement le nombre de dimensions latentes actives à partir de la KL par dimension et de la reconstruction loss.

---

## Structure du projet

```
sparse_vae_forgery/
├── configs/
│   └── default.yaml         ← TOUS les hyperparamètres ici
├── models/
│   ├── dinov2_extractor.py  ← DINOv2 frozen
│   ├── projection_head.py   ← tokens → μ, log σ²
│   ├── sparse_latent.py     ← Top-K + AdaptiveKController
│   ├── multiscale_decoder.py← décodeur FPN 3 échelles
│   └── full_model.py        ← pipeline complet
├── losses/
│   └── total_loss.py        ← MSE + SSIM + β×KL
├── utils/
│   └── helpers.py           ← seed, config, β schedule, diagnostics
├── data/
│   └── datasets.py          ← MIDV-2020 (train/val) + FMIDV-2022 (test)
├── evaluation/
│   ├── heatmap.py           ← heatmap multi-échelle
│   ├── anomaly_scorer.py    ← score composite calibré
│   └── metrics.py           ← AUC-ROC, F1, confusion matrix
├── train.py                 ← entraînement principal
├── evaluate.py              ← évaluation finale + scores_anomalie.csv
├── ablation.py              ← étude d'ablation
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Données

### Structure attendue

```
data/
├── midv2020/            ← Documents authentiques MIDV-2020
│   ├── alb_id/
│   │   └── <doc_id>/images/frame0000.jpg ...
│   └── ...
└── fmidv2022/           ← FMIDV-2022 (test uniquement)
    ├── authentic/
    │   └── <images>.jpg
    └── forged/
        └── <images>.jpg
```

> ⚠️ Les forgeries FMIDV-2022 ne doivent **jamais** apparaître pendant l'entraînement.

Les images MIDV-2020 (64×64 originales) sont automatiquement redimensionnées à 224×224.

---

## Utilisation

### 1. Entraînement

```bash
python train.py --config configs/default.yaml
```

Le protocole en 3 phases est automatique :
- **Phase 1** (epochs 1-30) : β=0, reconstruction pure
- **Phase 2** (epochs 31-130) : warm-up linéaire β 0→4
- **Phase 3** (epochs 131-300) : β=4.0, K adaptatif actif

Sorties : `checkpoints/best_model.pth` et `checkpoints/last_model.pth`

### 2. Évaluation finale

```bash
python evaluate.py --config configs/default.yaml --output results/
```

Produit :
- `results/scores_anomalie.csv` — score d'anomalie de chaque image
- `results/metrics.json` — AUC-ROC, Precision, Recall, F1, Matrice de confusion

### 3. Étude d'ablation

```bash
python ablation.py --config configs/default.yaml --output results/ablation/
```

---

## Protocole β (3 phases fixes)

| Phase | Epochs    | β         | Objectif                  | Sélection best model     |
|-------|-----------|-----------|---------------------------|--------------------------|
| 1     | 1–30      | 0         | Reconstruction pure       | val reconstruction loss  |
| 2     | 31–130    | 0 → 4     | Structurer l'espace latent| —                        |
| 3     | 131–300   | 4.0 fixe  | Sparse VAE complet        | val total loss           |

---

## Adaptive K Controller

K évolue progressivement via EMA :

```
τ = max(0.1 × KL_mean, 1e-4)
K_target = #{dim i : KL_i > τ}
Si recon_loss > 0.05 → K_target += 2
Si recon_loss < 0.01 → K_target -= 2
K_new = 0.9 × K_old + 0.1 × K_target
K ∈ [2, 50]
```

---

## Score d'Anomalie Composite

```
S_final = 0.4 × S_recon + 0.2 × S_kl + 0.3 × S_heatmap + 0.1 × S_latent
```

Chaque composante est normalisée par z-score (μ, σ calculés sur le val authentique).  
Seuil θ = percentile 95 du score composite sur les authentiques.

---

## Diagnostics

| Métrique         | Seuil alarme          | Interprétation               |
|------------------|-----------------------|------------------------------|
| KL_mean < 0.5    | Posterior collapse    | Augmenter lr ou diminuer β   |
| KL_mean > 10     | KL explosion          | Diminuer β ou lr             |
| active_ratio < 0.3 | Sparsité excessive  | Augmenter K_init             |
| dead_dims > 10   | Dimensions mortes     | Vérifier projection head     |

---

## Reproductibilité

```python
# Automatique dans train.py et evaluate.py
set_seed(42)
```

---

## Référence

Basé sur :
- DINOv2 (Oquab et al., 2023) — Meta AI
- β-VAE (Higgins et al., 2017)
- MIDV-2020 / FMIDV-2022 datasets
