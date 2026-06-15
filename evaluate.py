# evaluate.py
# CDC §5.4 — Étape 4 : Test AUC-ROC sur FMIDV-2022
#
# Uniquement sur les runs marqués "qualified": true par ablation.py.
# Pour chaque run qualifié :
#   1. Charge le checkpoint (SparseVAE identique à train.py)
#   2. Calcule les erreurs de reconstruction sur MIDV-2020 val (authentiques)
#   3. Calcule les erreurs de reconstruction sur FMIDV-2022 (forgeries)
#   4. Dérive l'AUC-ROC  (label 0=auth, 1=forgery ; score = erreur recon)
#   5. Met à jour le JSON → auc_roc = <valeur>
#
# Le forward reprend exactement la chaîne de train.py :
#   dinov2 → projection_head → sparse_latent → decoder (multi-échelle)
#   La perte de reconstruction est la moyenne des MSE sur les 3 échelles.

import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models.full_model import SparseVAE
from data.dataset      import MIDV2020Dataset, FMIDV2022Dataset

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/ablation")
CONFIG_PATH = Path("configs/default.yaml")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Utilitaires JSON ──────────────────────────────────────────────────────────

def load_qualified_results() -> list[tuple[Path, dict]]:
    """Retourne uniquement les runs marqués qualified=True par ablation.py."""
    items = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("qualified", False):
            items.append((path, r))
        else:
            label = path.stem
            reason = (
                "filtre K" if not r.get("filter_k_passed", True)
                else "filtre β" if not r.get("filter_beta_passed", True)
                else "non valide"
            )
            print(f"  [skip {reason}] {label}")
    return items


def save_result(path: Path, result: dict) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Cibles multi-échelle (identique à train.py) ───────────────────────────────

def prepare_multiscale_targets(x: torch.Tensor):
    """
    CDC §4.4 — x : (B, 3, 224, 224)
    Retourne les 3 cibles aux résolutions 16×16, 32×32, 64×64.
    """
    x_c = F.interpolate(x, (16, 16), mode="bilinear", align_corners=False)
    x_m = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
    x_f = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
    return x_c, x_m, x_f


# ── Scorer : erreur de reconstruction multi-échelle ───────────────────────────

@torch.no_grad()
def reconstruction_errors(model: SparseVAE,
                           loader: DataLoader) -> np.ndarray:
    """
    Calcule l'erreur de reconstruction par image.
    Score = moyenne des MSE sur les 3 échelles.
    Score élevé → image suspecte (forgery probable).
    """
    model.eval()
    errors = []

    for x, _ in loader:
        x = x.to(DEVICE)
        x_c, x_m, x_f = prepare_multiscale_targets(x)

        # Forward identique à train.py
        tokens                     = model.dinov2(x)
        mu, log_var                = model.projection_head(tokens)
        z_sparse, _, _             = model.sparse_latent(mu, log_var)
        x_hat_c, x_hat_m, x_hat_f = model.decoder(z_sparse)

        # Erreur par image : moyenne des 3 MSE pixel-wise
        err_c = F.mse_loss(x_hat_c, x_c, reduction="none").view(x.size(0), -1).mean(1)
        err_m = F.mse_loss(x_hat_m, x_m, reduction="none").view(x.size(0), -1).mean(1)
        err_f = F.mse_loss(x_hat_f, x_f, reduction="none").view(x.size(0), -1).mean(1)

        score = (err_c + err_m + err_f) / 3.0   # (B,)
        errors.append(score.cpu().numpy())

    return np.concatenate(errors)


# ── Évaluation d'un run qualifié ──────────────────────────────────────────────

def evaluate_run(result: dict,
                 val_loader: DataLoader,
                 forg_loader: DataLoader,
                 cfg: dict) -> float:
    """
    Charge le checkpoint du run, calcule l'AUC-ROC sur FMIDV-2022.
    Retourne l'AUC.
    """
    ckpt_path = Path(result["checkpoint"])
    k         = result["k"]

    model = SparseVAE(
        latent_dim = cfg["latent_dim"],
        k          = k,
        dropout    = cfg["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE)
    )

    # Erreurs de reconstruction
    auth_errors = reconstruction_errors(model, val_loader)    # label 0
    forg_errors = reconstruction_errors(model, forg_loader)   # label 1

    scores = np.concatenate([auth_errors, forg_errors])
    labels = np.concatenate([
        np.zeros(len(auth_errors), dtype=int),
        np.ones(len(forg_errors),  dtype=int),
    ])

    auc = float(roc_auc_score(labels, scores))
    return auc


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("ÉTAPE 4 — Test AUC sur runs qualifiés (FMIDV-2022)")
    print("=" * 55)

    cfg              = load_config()
    qualified_runs   = load_qualified_results()

    if not qualified_runs:
        print("\n⚠️  Aucun run qualifié. Lancez ablation.py --step filter d'abord.")
        return

    # ── DataLoaders ───────────────────────────────────────────
    val_dataset  = MIDV2020Dataset(split="val")
    forg_dataset = FMIDV2022Dataset()

    val_loader  = DataLoader(
        val_dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = 4,
    )
    forg_loader = DataLoader(
        forg_dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = 4,
    )

    # ── Évaluation ────────────────────────────────────────────
    for json_path, result in qualified_runs:

        ckpt = result.get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            print(f"  ⚠️  Checkpoint manquant : {ckpt} — run ignoré")
            continue

        run_label = Path(ckpt).stem
        auc       = evaluate_run(result, val_loader, forg_loader, cfg)

        print(f"  Évaluation : {run_label:<22} → AUC={auc:.4f}")

        result["auc_roc"] = round(auc, 4)
        save_result(json_path, result)

    print(f"\n✅ {len(qualified_runs)} runs évalués.")
    print(f"   AUC-ROC écrits dans {RESULTS_DIR}/")
    print(f"   → Lancez maintenant : python ablation.py --step decide")


if __name__ == "__main__":
    main()