import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ----------------------------
# MODELS
# ----------------------------
from models.full_model import FullModel

# ----------------------------
# CONTROLLERS (utils)
# ----------------------------
from utils.k_controller import KController
from utils.beta_controller import BetaController

# ----------------------------
# ADAPTIVE MODULES
# ----------------------------
from adaptive.logger import TrainingLogger

# ----------------------------
# LOSS
# ----------------------------
from losses.total_loss import total_loss_fn


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train(config, dataloader, val_loader=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # MODEL
    # ----------------------------
    model = FullModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # ----------------------------
    # CONTROLLERS (utils — conformes CDC)
    # ----------------------------
    k_controller = KController(
        initial_k=config["k"],           # CDC §3.4 — K initial = 16
        min_k=1,
        max_k=64,
        step=1,
        window=5,
        threshold_pct=5.0
    )

    beta_controller = BetaController(
        beta_max=config["beta_max"],      # CDC §5.4 — beta_max = 4.0
        target_sparsity=0.85,             # CDC §5.4 — cible sparsité 85%
        step=0.1,
        warmup_epoch=20                   # CDC §5.4 — Phase 1 : epoch < 20 → β = 0
    )

    logger = TrainingLogger()

    num_epochs = config["epochs"]

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Récupérer K et beta courants
        current_k    = k_controller.get_k()
        current_beta = beta_controller.get_beta()

        for x, _ in loop:

            x = x.to(device)

            # ----------------------------
            # FORWARD
            # ----------------------------
            z = model.encoder(x)

            z_sparse = model.sparse_latent(
                z,
                k=current_k
            )

            x_hat = model.decoder(z_sparse)

            # ----------------------------
            # LOSS
            # ----------------------------
            loss, sparsity = total_loss_fn(
                x,
                x_hat,
                beta=current_beta,
                z_sparse=z_sparse
            )

            # ----------------------------
            # BACKPROP
            # ----------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss      += loss.item()
            epoch_sparsity  += sparsity

            loop.set_postfix({
                "loss": loss.item(),
                "K": current_k,
                "beta": round(current_beta, 4)
            })

        # =========================================================
        # METRICS (epoch level)
        # =========================================================

        avg_loss     = epoch_loss     / len(dataloader)
        avg_sparsity = epoch_sparsity / len(dataloader)

        # =========================================================
        # ADAPTATION (KController + BetaController)
        # =========================================================

        current_k    = k_controller.update(avg_loss)
        current_beta = beta_controller.update(avg_sparsity, epoch)

        # =========================================================
        # LOGGING
        # =========================================================

        logger.log(
            epoch=epoch,
            K=current_k,
            beta=current_beta,
            loss=avg_loss,
            sparsity=avg_sparsity
        )

        print(f"\nEpoch {epoch}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Sparsity: {avg_sparsity:.4f}")
        print(f"K: {current_k} | Beta: {current_beta:.4f}")

        # =========================================================
        # OPTIONAL VALIDATION
        # =========================================================

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)

                    z        = model.encoder(x)
                    z_sparse = model.sparse_latent(z, k=current_k)
                    x_hat    = model.decoder(z_sparse)

                    loss, _ = total_loss_fn(
                        x,
                        x_hat,
                        beta=current_beta,
                        z_sparse=z_sparse
                    )

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

    # =========================================================
    # SAVE MODEL + LOGS
    # =========================================================

    torch.save(model.state_dict(), config["save_path"])
    logger.save(config["log_path"])

    print("\nTraining finished.")
    print(f"Model saved to: {config['save_path']}")
    print(f"Logs saved to: {config['log_path']}")
