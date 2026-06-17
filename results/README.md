# results/

Répertoire pour résultats et artefacts.

## results/ablation/

Résultats des runs d'ablation générés par:
- `train.py` → JSON avec métriques d'entraînement (val_loss, sparsity_rate, etc.)
- `evaluate.py` → JSON mis à jour avec AUC-ROC final

Fichiers: `k{K}_beta{β}.json`

Exemple:
```json
{
  "k": 16,
  "beta": 4.0,
  "val_loss": 0.0523,
  "sparsity_rate": 0.89,
  "active_ratio": 0.16,
  "n_collapsed": 4,
  "final_kl": 2.134,
  "auc_roc": 0.9823,
  "checkpoint": "checkpoints/k16_beta4_best.pt",
  "valid": true,
  "stopped_early": false
}
```
