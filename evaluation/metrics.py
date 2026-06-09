# evaluation/metrics.py
import torch


@torch.no_grad()
def validate_projection_head(model, val_loader, device):
    """
    CDC §2.6 — validation de la projection head.
    Forward pass arrêté après ProjectionHead.
    Décodeur et SparseLatent ne sont pas appelés.
    """
    all_mu, all_logvar = [], []

    model.eval()  # désactive Dropout dans ProjectionHead

    for x, _ in val_loader:
        x      = x.to(device)

        # ── Forward partiel : s'arrête après ProjectionHead ──
        tokens      = model.dinov2(x)                # étape 1
        mu, log_var = model.projection_head(tokens)  # étape 2
        # ── On n'appelle pas plus loin ────────────────────────

        all_mu.append(mu.cpu())
        all_logvar.append(log_var.cpu())

    mu_all     = torch.cat(all_mu,     dim=0)  # (N, 64)
    logvar_all = torch.cat(all_logvar, dim=0)  # (N, 64)

    # ── Critère 1 : µ_moyen ≈ 0, σ²_moyen ≈ 1 ───────────────
    mu_moyen  = mu_all.mean().item()
    var_moyen = logvar_all.exp().mean().item()

    # ── Critère 2 : KL ∈ [2, 10] nats ────────────────────────
    kl = -0.5 * torch.sum(
        1 + logvar_all - mu_all.pow(2) - logvar_all.exp(),
        dim=1
    ).mean().item()

    # ── Critère 3 : Var(µ_i) > 0.01 pour tout i ──────────────
    var_dims    = mu_all.var(dim=0)           # (64,)
    n_collapsed = (var_dims < 0.01).sum().item()

    # ── Affichage ─────────────────────────────────────────────
    print(f"  µ_moyen     = {mu_moyen:.4f}   (cible ≈ 0)")
    print(f"  σ²_moyen    = {var_moyen:.4f}   (cible ≈ 1)")
    print(f"  KL moyenne  = {kl:.3f} nats  (cible 2–10)")
    print(f"  Dims mortes = {n_collapsed}/64  (cible = 0)")

    # ── Décision ──────────────────────────────────────────────
    ok_phase1 = (
        abs(mu_moyen)        < 0.5 and
        abs(var_moyen - 1.0) < 0.5 and
        n_collapsed          == 0
    )
    ok_phase3 = ok_phase1 and (2.0 <= kl <= 10.0)

    return {
        "mu_moyen":    mu_moyen,
        "var_moyen":   var_moyen,
        "kl_moyen":    kl,
        "n_collapsed": n_collapsed,
        "ok_phase1":   ok_phase1,   # sans critère KL
        "ok_phase3":   ok_phase3,   # avec critère KL
    }