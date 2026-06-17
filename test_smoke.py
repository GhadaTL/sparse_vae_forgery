"""
test_smoke.py
=============
Smoke test — vérifie que toute la pipeline fonctionne
avec des tenseurs aléatoires (sans données réelles ni DINOv2).

Usage :
    python test_smoke.py
"""

import torch
import torch.nn as nn
from models.projection_head    import ProjectionHead
from models.sparse_latent      import SparseLatent
from models.multiscale_decoder import MultiScaleDecoder, prepare_multiscale_targets
from losses.total_loss         import reconstruction_loss, compute_sparsity_metrics
from utils.k_controller        import KController
from utils.beta_controller     import BetaController
from adaptive.logger           import TrainingLogger

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BATCH_SIZE  = 4
LATENT_DIM  = 64
K_INIT      = 16
EPOCHS      = 30
DEVICE      = torch.device("cpu")

print("=" * 55)
print("SMOKE TEST — Sparse VAE Forgery Detection")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# MODULES  (sans DINOv2 — patch_tokens simulés directement)
# ─────────────────────────────────────────────────────────
projection_head = ProjectionHead(input_dim=768, latent_dim=LATENT_DIM, dropout=0.1)
sparse_latent   = SparseLatent(latent_dim=LATENT_DIM, k=K_INIT)
decoder         = MultiScaleDecoder(latent_dim=LATENT_DIM)

optimizer = torch.optim.Adam(
    list(projection_head.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)

# ─────────────────────────────────────────────────────────
# CONTROLLERS
# ─────────────────────────────────────────────────────────
k_controller    = KController(initial_k=K_INIT, min_k=1, max_k=64, step=1, window=5)
beta_controller = BetaController(beta_max=4.0, target_sparsity=0.85, step=0.1, warmup_epoch=20)
logger          = TrainingLogger()

current_k    = k_controller.get_k()
current_beta = beta_controller.get_beta()

# ─────────────────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT SIMULÉE
# ─────────────────────────────────────────────────────────
for epoch in range(EPOCHS):

    projection_head.train()
    decoder.train()

    # Simule patch_tokens DINOv2 directement : (B, 256, 768)
    patch_tokens = torch.randn(BATCH_SIZE, 256, 768)

    # Image originale simulée pour les cibles
    x = torch.rand(BATCH_SIZE, 3, 224, 224)

    # ProjectionHead → (mu, logvar)
    mu, logvar = projection_head(patch_tokens)

    # SparseLatent → (z_sparse, kl, mask)
    z_sparse, kl_loss, mask = sparse_latent(mu, logvar, k=current_k)

    # Cibles multi-échelle
    x_c, x_m, x_f = prepare_multiscale_targets(x)

    # Décodeur
    x_hat_c, x_hat_m, x_hat_f = decoder(z_sparse)

    # Loss
    recon = reconstruction_loss(x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f)
    loss  = recon["l_recon"] + current_beta * kl_loss

    # Sparsité
    sp       = compute_sparsity_metrics(z_sparse, LATENT_DIM)
    sparsity = sp["sparsity_rate"]

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Adaptation K et β
    current_k    = k_controller.update(loss.item())
    current_beta = beta_controller.update(sparsity, epoch)

    # Log
    logger.log(epoch=epoch, K=current_k, beta=current_beta,
               loss=loss.item(), sparsity=sparsity)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"loss={loss.item():.4f} | "
          f"K={current_k:2d} | "
          f"beta={current_beta:.3f} | "
          f"sparsity={sparsity:.2f}")

# ─────────────────────────────────────────────────────────
# VÉRIFICATIONS FINALES
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("VÉRIFICATIONS")
print("=" * 55)

assert x_hat_c.shape == (BATCH_SIZE, 3, 16, 16),  f"Erreur shape coarse : {x_hat_c.shape}"
assert x_hat_m.shape == (BATCH_SIZE, 3, 32, 32),  f"Erreur shape medium : {x_hat_m.shape}"
assert x_hat_f.shape == (BATCH_SIZE, 3, 64, 64),  f"Erreur shape fine   : {x_hat_f.shape}"
assert z_sparse.shape == (BATCH_SIZE, LATENT_DIM), f"Erreur shape z      : {z_sparse.shape}"
assert kl_loss.item() >= 0,                        "KL loss négative !"
assert 0.0 <= sparsity <= 1.0,                     "Sparsity hors [0,1] !"

k_history = k_controller.get_history()["k_history"]
b_history = beta_controller.get_history()["beta_history"]

print(f"✅ Shapes outputs     : OK")
print(f"✅ KL loss            : {kl_loss.item():.4f}")
print(f"✅ Sparsity finale    : {sparsity:.2f}")
print(f"✅ K historique       : {k_history}")
print(f"✅ Beta historique    : {[round(b,2) for b in b_history]}")
print(f"\n✅ SMOKE TEST PASSÉ — pipeline fonctionnelle")
