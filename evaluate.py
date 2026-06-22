# evaluate.py
# CDC §5.4 — Évaluation complète sur FMIDV-2022
#
# - 500 images forgées  (FMIDV-2022)  tirées avec seed fixe
# - 500 images authentiques (MIDV-2020 val) tirées avec seed fixe
# - Calibration heatmap stats sur les authentiques
# - Métriques : AUC-ROC, Precision, Recall, F1, TP/FP/TN/FN, threshold
# - Heatmaps (.npy) sauvegardées pour chaque image forgée
#
# Usage :
#   python evaluate.py
#   python evaluate.py --config configs/default.yaml

import argparse
import json
import warnings
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics  import roc_auc_score

warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

from models.full_model     import SparseVAE
from data.dataset          import MIDV2020Dataset, FMIDV2022Dataset
from evaluation.heatmap    import compute_multiscale_heatmap

N_SAMPLES = 500
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# 2. Sampling reproductible
# ─────────────────────────────────────────────────────────────

def sample_subset(dataset, n: int, seed: int) -> Subset:
    """Tire n indices reproductibles sans remise."""
    rng     = np.random.default_rng(seed)
    n       = min(n, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


# ─────────────────────────────────────────────────────────────
# 3. Chargement modèle
# ─────────────────────────────────────────────────────────────

def load_model(cfg: dict) -> tuple:
    """
    Recrée SparseVAE identique à train.py et charge best_model.pth.
    K est figé dans model_state — β inutile à l'évaluation.
    """
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {ckpt_path}\n"
            "Lancez train.py d'abord."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = SparseVAE(
        latent_dim   = cfg["model"]["latent_dim"],
        k            = cfg["model"]["k"],
        dropout      = cfg["model"]["dropout"],
        logvar_clamp = tuple(cfg["model"]["logvar_clamp"]),
        dinov2_model = cfg["model"]["dinov2_name"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.set_eval_mode()
    return model, ckpt


# ─────────────────────────────────────────────────────────────
# 4. Calibration heatmap stats
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_heatmap_stats(model: SparseVAE,
                              loader: DataLoader) -> dict:
    """
    Calcule mean/std des heatmaps brutes sur le val authentique (CDC §6.4).
    Sert à normaliser les heatmaps lors de l'évaluation et de l'inférence.
    """
    all_hm = []
    for batch in loader:
        images = batch["image"].to(DEVICE)
        for i in range(images.size(0)):
            hm = compute_multiscale_heatmap(
                model, images[i:i+1],
                model.dinov2, model.projection_head,
                model.sparse_latent, model.decoder,
                train_stats=None, device=DEVICE,
            )
            all_hm.append(hm)

    stack = np.stack(all_hm)   # (N, 64, 64)
    return {"mean": float(stack.mean()), "std": float(stack.std())}


# ─────────────────────────────────────────────────────────────
# 5. Calcul scores composites
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(model: SparseVAE,
                   loader: DataLoader,
                   heatmap_stats: dict,
                   save_heatmaps_dir: Path = None) -> dict:
    """
    Calcule 4 signaux bruts par image (CDC §7.2) :
        S_recon   : MSE fine-scale (64×64) par image
        S_kl      : KL divergence scalaire
        S_heatmap : percentile 95 de la heatmap normalisée
        S_latent  : ||z_sparse||_2 / latent_dim

    Retourne un dict de 4 arrays bruts — la pondération
    et le z-score global sont faits dans main().
    """
    raw = {"recon": [], "kl": [], "hm": [], "lat": []}

    for batch in loader:
        images = batch["image"].to(DEVICE)
        paths  = batch["path"]

        # Forward complet — cohérent avec train.py
        outputs = model(images)

        # S_recon : MSE fine scale par image (B,)
        err_f = F.mse_loss(
            outputs["x_hat_fine"], outputs["x_fine"],
            reduction="none",
        ).view(images.size(0), -1).mean(dim=1)
        raw["recon"].extend(err_f.cpu().tolist())

        # S_kl : somme de kl_per_dim (même valeur pour tout le batch
        # car kl_per_dim est moyenné sur le batch dans sparse_latent)
        s_kl = outputs["kl_per_dim"].sum().item()
        raw["kl"].extend([s_kl] * images.size(0))

        # S_latent : ||z_sparse||_2 / latent_dim par image (B,)
        z_norm = torch.norm(outputs["z_sparse"], p=2, dim=1) / model.latent_dim
        raw["lat"].extend(z_norm.cpu().tolist())

        # S_heatmap : image par image
        for i in range(images.size(0)):
            hm = compute_multiscale_heatmap(
                model, images[i:i+1],
                model.dinov2, model.projection_head,
                model.sparse_latent, model.decoder,
                train_stats=heatmap_stats, device=DEVICE,
            )
            raw["hm"].append(float(np.percentile(hm, 95)))

            # Sauvegarde heatmap .npy pour les forgées
            if save_heatmaps_dir is not None:
                save_heatmaps_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(paths[i]).stem
                np.save(save_heatmaps_dir / f"{stem}_heatmap.npy", hm)

    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────
# 6. Métriques complètes
# ─────────────────────────────────────────────────────────────

def compute_metrics(auth_composite: np.ndarray,
                    forg_composite: np.ndarray,
                    threshold_percentile: int) -> dict:
    """
    AUC-ROC, Precision, Recall, F1, TP/FP/TN/FN.
    Seuil θ = percentile(auth_composite, threshold_percentile).
    """
    scores = np.concatenate([auth_composite, forg_composite])
    labels = np.concatenate([
        np.zeros(len(auth_composite), dtype=int),
        np.ones(len(forg_composite),  dtype=int),
    ])

    theta = float(np.percentile(auth_composite, threshold_percentile))
    preds = (scores > theta).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = 0.0

    return {
        "auc_roc":               round(auc,             4),
        "precision":             round(float(precision), 4),
        "recall":                round(float(recall),    4),
        "f1":                    round(float(f1),        4),
        "threshold":             round(theta,            6),
        "threshold_percentile":  threshold_percentile,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_auth": len(auth_composite),
        "n_forg": len(forg_composite),
    }


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    w   = cfg["evaluation"]["anomaly_weights"]   # [w_recon, w_kl, w_hm, w_lat]

    print("=" * 55)
    print("ÉVALUATION — MIDV-2020 vs FMIDV-2022")
    print("=" * 55)

    # ── Modèle ────────────────────────────────────────────────
    model, ckpt = load_model(cfg)
    print(f"\n  Checkpoint epoch : {ckpt.get('epoch', '?')}")
    print(f"  val_loss         : {ckpt.get('val_loss', 0):.4f}")
    print(f"  K (sparse_latent): {model.sparse_latent.k}")
    print(f"  device           : {DEVICE}")

    # ── Datasets + sampling 500/500 ───────────────────────────
    auth_ds_full = MIDV2020Dataset(
        root_dir   = cfg["data"]["midv2020_path"],
        split      = "val",
        image_size = cfg["data"]["image_size"],
    )
    forg_ds_full = FMIDV2022Dataset(
        root_dir   = cfg["data"]["fmidv2022_path"],
        image_size = cfg["data"]["image_size"],
    )

    auth_ds = sample_subset(auth_ds_full, N_SAMPLES, seed=SEED)
    forg_ds = sample_subset(forg_ds_full, N_SAMPLES, seed=SEED)

    print(f"\n  Authentiques : {len(auth_ds)}/{len(auth_ds_full)}")
    print(f"  Forgées      : {len(forg_ds)}/{len(forg_ds_full)}")

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    auth_loader = DataLoader(auth_ds, batch_size=bs, shuffle=False, num_workers=nw)
    forg_loader = DataLoader(forg_ds, batch_size=bs, shuffle=False, num_workers=nw)

    # ── Calibration heatmap stats ─────────────────────────────
    print("\n  [1/4] Calibration heatmap stats sur authentiques...")
    heatmap_stats = calibrate_heatmap_stats(model, auth_loader)
    print(f"        mean={heatmap_stats['mean']:.5f}  std={heatmap_stats['std']:.5f}")

    # ── Scores authentiques ───────────────────────────────────
    print("\n  [2/4] Calcul scores authentiques...")
    auth_raw = compute_scores(model, auth_loader, heatmap_stats)

    # ── Scores forgées + sauvegarde heatmaps ─────────────────
    print("\n  [3/4] Calcul scores forgées + sauvegarde heatmaps...")
    heatmap_dir = Path(cfg["output"]["results_dir"]) / "heatmaps"
    forg_raw    = compute_scores(model, forg_loader, heatmap_stats,
                                 save_heatmaps_dir=heatmap_dir)

    # ── Z-score global (auth + forg ensemble) ─────────────────
    print("\n  [4/4] Calcul métriques...")

    def global_zscore(a, b):
        combined  = np.concatenate([a, b])
        mu, sigma = combined.mean(), combined.std() + 1e-8
        return (a - mu) / sigma, (b - mu) / sigma

    auth_z, forg_z = {}, {}
    for key in ("recon", "kl", "hm", "lat"):
        auth_z[key], forg_z[key] = global_zscore(auth_raw[key], forg_raw[key])

    def composite(z):
        return w[0]*z["recon"] + w[1]*z["kl"] + w[2]*z["hm"] + w[3]*z["lat"]

    auth_composite = composite(auth_z)
    forg_composite = composite(forg_z)

    # ── Métriques ─────────────────────────────────────────────
    metrics = compute_metrics(
        auth_composite, forg_composite,
        threshold_percentile=cfg["evaluation"]["threshold_percentile"],
    )

    # ── Affichage ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS")
    print("=" * 55)
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.6f}  "
          f"(p{metrics['threshold_percentile']} authentiques)")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    print(f"  N auth={metrics['n_auth']}  N forg={metrics['n_forg']}")

    # ── Sauvegarde JSON + heatmap_stats ───────────────────────
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Métriques
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Heatmap stats → utilisées par inference.py
    with open(out_dir / "heatmap_stats.json", "w") as f:
        json.dump(heatmap_stats, f, indent=2)

    print(f"\n  ✅ Métriques     → {out_dir}/evaluation_results.json")
    print(f"  ✅ Heatmap stats → {out_dir}/heatmap_stats.json")
    print(f"  ✅ Heatmaps .npy → {heatmap_dir}/")


if __name__ == "__main__":
    main()