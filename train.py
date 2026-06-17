import torch
import torch.optim as optim
from tqdm import tqdm

from models.full_model         import FullModel, prepare_multiscale_targets
from utils.k_controller        import KController
from utils.beta_controller     import BetaController
from adaptive.logger           import TrainingLogger
from losses.total_loss         import reconstruction_loss, compute_sparsity_metrics


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train(config, dataloader, val_loader=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # MODEL
    # ----------------------------
    model     = FullModel(config).to(device)
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr  = config["lr"],
        weight_decay = config.get("wd", 1e-4)
    )

    # ----------------------------
    # CONTROLLERS (utils — conformes CDC)
    # ----------------------------
    k_controller = KController(
        initial_k     = config["k"],        # CDC §3.4 — K initial = 16
        min_k         = 1,
        max_k         = 64,
        step          = 1,
        window        = 5,
        threshold_pct = 5.0
    )

    beta_controller = BetaController(
        beta_max         = config["beta_max"],  # CDC §5.4 — beta_max = 4.0
        target_sparsity  = 0.85,                # CDC §5.4 — cible 85%
        step             = 0.1,
        warmup_epoch     = 20                   # CDC §5.4 — Phase 1 : β = 0
    )

    logger     = TrainingLogger()
    num_epochs = config["epochs"]

    # K et beta courants (mis à jour à chaque fin d'epoch)
    current_k    = k_controller.get_k()
    current_beta = beta_controller.get_beta()

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(num_epochs):

        model.train()

        epoch_loss     = 0.0
        epoch_sparsity = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for x, _ in loop:

            x = x.to(device)

            # ----------------------------
            # CIBLES MULTI-ÉCHELLE
            # ----------------------------
            x_c, x_m, x_f = prepare_multiscale_targets(x)

            # ----------------------------
            # FORWARD  (k dynamique)
            # ----------------------------
            x_hat_c, x_hat_m, x_hat_f, kl_loss, z_sparse, mask = model(
                x, k=current_k
            )

            # ----------------------------
            # LOSS
            # ----------------------------
            recon = reconstruction_loss(
                x_c, x_hat_c,
                x_m, x_hat_m,
                x_f, x_hat_f,
            )
            loss = recon["l_recon"] + current_beta * kl_loss

            # ----------------------------
            # SPARSITÉ RÉELLE
            # ----------------------------
            sparsity_metrics = compute_sparsity_metrics(z_sparse, config.get("latent_dim", 64))
            sparsity         = sparsity_metrics["sparsity_rate"]

            # ----------------------------
            # BACKPROP
            # ----------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss     += loss.item()
            epoch_sparsity += sparsity

            loop.set_postfix({
                "loss"    : f"{loss.item():.4f}",
                "K"       : current_k,
                "beta"    : f"{current_beta:.3f}",
                "sparse"  : f"{sparsity:.2f}",
            })

        # =========================================================
        # MÉTRIQUES EPOCH
        # =========================================================
        avg_loss     = epoch_loss     / len(dataloader)
        avg_sparsity = epoch_sparsity / len(dataloader)

        # =========================================================
        # ADAPTATION K et β
        # =========================================================
        current_k    = k_controller.update(avg_loss)
        current_beta = beta_controller.update(avg_sparsity, epoch)

        # =========================================================
        # LOGGING
        # =========================================================
        logger.log(
            epoch    = epoch,
            K        = current_k,
            beta     = current_beta,
            loss     = avg_loss,
            sparsity = avg_sparsity,
        )

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Loss     : {avg_loss:.4f}")
        print(f"  Sparsity : {avg_sparsity:.4f}")
        print(f"  K        : {current_k}  |  Beta : {current_beta:.4f}")

        # =========================================================
        # VALIDATION
        # =========================================================
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    x_c, x_m, x_f = prepare_multiscale_targets(x)

                    x_hat_c, x_hat_m, x_hat_f, kl_loss, z_sparse, _ = model(
                        x, k=current_k
                    )

                    recon    = reconstruction_loss(x_c, x_hat_c, x_m, x_hat_m, x_f, x_hat_f)
                    loss_val = recon["l_recon"] + current_beta * kl_loss
                    val_loss += loss_val.item()

            val_loss /= len(val_loader)
            print(f"  Val Loss : {val_loss:.4f}")

    # =========================================================
    # SAUVEGARDE
    # =========================================================
    torch.save(model.state_dict(), config["save_path"])
    logger.save(config["log_path"])

    print("\nEntraînement terminé.")
    print(f"Modèle sauvegardé : {config['save_path']}")
    print(f"Logs sauvegardés  : {config['log_path']}")
