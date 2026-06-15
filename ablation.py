# ablation.py
# CDC §5.4 — pipeline d'ablation K × β
# Étape 2 : filtre K  (sparsity_rate, active_ratio, n_collapsed)
# Étape 3 : filtre β  (final_kl)
# Étape 5 : décision finale (max AUC-ROC → K*, β*, best_model.pt)
#
# Dépend de :
#   results/ablation/<run>.json  ← produits par train.py
#   results/ablation/<run>.json  ← auc_roc complété par evaluate.py
#
# Usage :
#   python ablation.py --step filter      # Étapes 2 + 3
#   python ablation.py --step decide      # Étape 5
#   python ablation.py                    # Tout (filtres → décision)

import argparse
import json
import shutil
import yaml
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path("results/ablation")
CKPT_DIR     = Path("checkpoints")
BEST_MODEL   = Path("best_model.pt")
DEFAULT_YAML = Path("configs/default.yaml")

# ── Critères filtre K  (CDC §3.4) ─────────────────────────────────────────────
LATENT_DIM        = 64
SPARSITY_TOL      = 0.05    # |sparsity_rate - (1 - k/64)| ≤ tolérance
ACTIVE_RATIO_MIN  = 0.30
ACTIVE_RATIO_MAX  = 0.70
MAX_COLLAPSED     = 0       # n_collapsed doit être == 0

# ── Critère filtre β  (CDC §2.6) ──────────────────────────────────────────────
KL_MIN = 2.0   # nats
KL_MAX = 10.0  # nats


# ── Utilitaires JSON ──────────────────────────────────────────────────────────

def load_all_results() -> list[tuple[Path, dict]]:
    """Charge tous les JSON de results/ablation/ triés par run_id."""
    items = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            items.append((path, json.load(f)))
    return items


def save_result(path: Path, result: dict) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Étape 2 — Filtre K ────────────────────────────────────────────────────────

def filter_k(results: list[tuple[Path, dict]]) -> int:
    """
    Critères (CDC §3.4) :
      1. sparsity_rate ≈ 1 - k/64  (± SPARSITY_TOL)
      2. active_ratio  ∈ [0.3, 0.7]
      3. n_collapsed   == 0

    Ajoute le flag "filter_k_passed" dans chaque JSON.
    Retourne le nombre de runs éliminés.
    """
    print("=" * 55)
    print("ÉTAPE 2 — Filtre K")
    print("=" * 55)

    n_eliminated = 0

    for path, r in results:

        # Runs déjà invalidés par train.py (phase 1 échouée)
        if not r.get("valid", True):
            r["filter_k_passed"] = False
            save_result(path, r)
            print(f"  ❌ {r.get('run_id', path.stem):<18} → run invalide (train.py)")
            n_eliminated += 1
            continue

        k            = r["k"]
        sparsity     = r["sparsity_rate"]
        active_ratio = r["active_ratio"]
        n_collapsed  = r["n_collapsed"]
        expected_sp  = 1.0 - k / LATENT_DIM

        ok_sparsity  = abs(sparsity - expected_sp) <= SPARSITY_TOL
        ok_ratio     = ACTIVE_RATIO_MIN <= active_ratio <= ACTIVE_RATIO_MAX
        ok_collapsed = n_collapsed <= MAX_COLLAPSED

        passed = ok_sparsity and ok_ratio and ok_collapsed
        r["filter_k_passed"] = passed
        save_result(path, r)

        run_label = path.stem

        if passed:
            print(f"  ✅ {run_label:<18}  "
                  f"sparsité={sparsity:.2f}  "
                  f"ratio={active_ratio:.2f}  "
                  f"dims_mortes={n_collapsed}")
        else:
            reasons = []
            if not ok_sparsity:
                reasons.append(
                    f"sparsité={sparsity:.2f} ≠ {expected_sp:.2f}±{SPARSITY_TOL}"
                )
            if not ok_ratio:
                reasons.append(
                    f"active_ratio={active_ratio:.2f} "
                    f"∉ [{ACTIVE_RATIO_MIN},{ACTIVE_RATIO_MAX}]"
                )
            if not ok_collapsed:
                reasons.append(f"n_collapsed={n_collapsed} > 0")
            print(f"  ❌ {run_label:<18} → {', '.join(reasons)}")
            n_eliminated += 1

    n_passed = len(results) - n_eliminated
    print(f"  → {n_passed} runs passent\n")
    return n_eliminated


# ── Étape 3 — Filtre β ────────────────────────────────────────────────────────

def filter_beta(results: list[tuple[Path, dict]]) -> int:
    """
    Critère (CDC §2.6) :
      4. final_kl ∈ [2.0, 10.0] nats

    Ajoute les flags "filter_beta_passed" et "qualified" dans chaque JSON.
    Ne teste que les runs ayant passé filter_k.
    Retourne le nombre de runs éliminés par ce filtre.
    """
    print("=" * 55)
    print("ÉTAPE 3 — Filtre β")
    print("=" * 55)

    n_eliminated = 0

    for path, r in results:

        if not r.get("filter_k_passed", False):
            # Déjà éliminé — on propage les flags sans recompter
            r.setdefault("filter_beta_passed", False)
            r.setdefault("qualified", False)
            save_result(path, r)
            continue

        kl     = r["final_kl"]
        passed = KL_MIN <= kl <= KL_MAX
        r["filter_beta_passed"] = passed
        r["qualified"]          = passed   # qualifié = passe les deux filtres
        save_result(path, r)

        run_label = path.stem

        if passed:
            print(f"  ✅ {run_label:<18}  KL={kl:.3f}")
        else:
            print(f"  ❌ {run_label:<18} → "
                  f"KL={kl:.3f} ∉ [{KL_MIN},{KL_MAX}]")
            n_eliminated += 1

    n_qualified = sum(1 for _, r in results if r.get("qualified", False))
    print(f"  → {n_qualified} runs qualifiés\n")
    return n_eliminated


# ── Étape 5 — Décision finale ─────────────────────────────────────────────────

def decide(results: list[tuple[Path, dict]]) -> None:
    """
    Parmi les runs qualifiés avec auc_roc renseigné :
      → sélectionne le max AUC-ROC
      → copie le checkpoint → best_model.pt
      → écrit configs/default.yaml avec K* et β*
    """
    print("=" * 55)
    print("ÉTAPE 5 — Décision finale : max AUC")
    print("=" * 55)

    qualified = [
        (p, r) for p, r in results
        if r.get("qualified", False) and r.get("auc_roc") is not None
    ]

    if not qualified:
        print("⚠️  Aucun run qualifié avec auc_roc disponible.")
        print("    Lancez evaluate.py d'abord.")
        return

    # Tri décroissant par AUC-ROC
    qualified.sort(key=lambda x: x[1]["auc_roc"], reverse=True)

    # ── Tableau récapitulatif ─────────────────────────────────
    header = f"  {'Run':<20} {'k':>4} {'β':>4} {'AUC':>8} {'KL':>8} {'ratio':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, (_, r) in enumerate(qualified):
        star      = " ★" if i == 0 else ""
        run_label = Path(r["checkpoint"]).stem if r.get("checkpoint") else "?"
        print(f"  {run_label:<20} {r['k']:>4} {int(r['beta']):>4} "
              f"{r['auc_roc']:>8.4f} {r['final_kl']:>8.3f} "
              f"{r['active_ratio']:>7.2f}{star}")

    _, best = qualified[0]
    best_ckpt = Path(best["checkpoint"]) if best.get("checkpoint") else None

    # ── Copie checkpoint ─────────────────────────────────────
    if best_ckpt and best_ckpt.exists():
        shutil.copy2(best_ckpt, BEST_MODEL)
        print(f"\n  best_model.pt ← {best_ckpt}")
    else:
        print(f"\n  ⚠️  Checkpoint introuvable : {best_ckpt}")

    # ── Écriture default.yaml ─────────────────────────────────
    DEFAULT_YAML.parent.mkdir(parents=True, exist_ok=True)

    # Charger le yaml existant si disponible pour ne pas écraser les autres clés
    existing_cfg = {}
    if DEFAULT_YAML.exists():
        with open(DEFAULT_YAML) as f:
            existing_cfg = yaml.safe_load(f) or {}

    existing_cfg.update({
        "k":    best["k"],
        "beta": best["beta"],
        # Métriques de référence (lecture seule, pour traçabilité)
        "_ablation": {
            "best_run":      Path(best["checkpoint"]).stem if best.get("checkpoint") else None,
            "auc_roc":       best["auc_roc"],
            "final_kl":      round(best["final_kl"], 4),
            "active_ratio":  round(best["active_ratio"], 4),
            "sparsity_rate": round(best["sparsity_rate"], 4),
        },
    })

    with open(DEFAULT_YAML, "w") as f:
        yaml.dump(existing_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  default.yaml  ← k*={best['k']}, β*={best['beta']}, "
          f"AUC={best['auc_roc']:.4f}\n")


# ── Résumé global ─────────────────────────────────────────────────────────────

def print_summary(results: list[tuple[Path, dict]],
                  n_elim_k: int,
                  n_elim_beta: int) -> None:

    total      = len(results)
    n_valid    = sum(1 for _, r in results if r.get("valid", True))
    n_qualified = sum(1 for _, r in results if r.get("qualified", False))
    n_tested   = sum(1 for _, r in results
                     if r.get("qualified") and r.get("auc_roc") is not None)

    print("=" * 55)
    print("RÉSUMÉ ABLATION")
    print("=" * 55)
    print(f"  Runs totaux           : {total}")
    print(f"  Runs valides (train)  : {n_valid}")
    print(f"  Éliminés filtre K     : {n_elim_k}")
    print(f"  Éliminés filtre β     : {n_elim_beta}  "
          f"(sur les {total - n_elim_k} restants)")
    print(f"  Runs testés sur FMIDV : {n_tested}")

    qualified_with_auc = [
        (p, r) for p, r in results
        if r.get("qualified") and r.get("auc_roc") is not None
    ]
    if qualified_with_auc:
        _, best = max(qualified_with_auc, key=lambda x: x[1]["auc_roc"])
        print()
        print("  ┌──────────────────────────────────┐")
        print(f"  │  k*  = {best['k']:<27}│")
        print(f"  │  β*  = {best['beta']:<27}│")
        print(f"  │  AUC = {best['auc_roc']:<27}│")
        print("  └──────────────────────────────────┘")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline d'ablation K/β — CDC §5.4"
    )
    parser.add_argument(
        "--step",
        choices=["filter", "decide", "all"],
        default="all",
        help="Étape à exécuter (default: all)",
    )
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"⚠️  Dossier {RESULTS_DIR} introuvable — lancez train.py d'abord.")
        return

    n_elim_k    = 0
    n_elim_beta = 0

    if args.step in ("filter", "all"):
        results     = load_all_results()
        n_elim_k    = filter_k(results)
        results     = load_all_results()    # recharger après écriture
        n_elim_beta = filter_beta(results)

    if args.step in ("decide", "all"):
        results = load_all_results()
        decide(results)

    if args.step == "all":
        results = load_all_results()
        print_summary(results, n_elim_k, n_elim_beta)


if __name__ == "__main__":
    main()