"""
train.py — AURA Model Training Pipeline
=========================================
Trains the FlowAutoencoder on CICIDS2017 BENIGN data.
Optionally also trains the STGNN with synthetic graph snapshots.

Usage:
  python train.py                    # Train both models
  python train.py --ae-only          # Train autoencoder only (faster)
  python train.py --epochs 10        # Override epoch count
  python train.py --quick            # Quick 5-epoch sanity check
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import config as cfg
from aura.data_loader import CICIDSDataLoader, CSV_FILES
from aura.models import FlowAutoencoder, AuraSTGNN, AURAModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder Training
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    ae:         FlowAutoencoder,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
    epochs:     int = cfg.AE_EPOCHS,
    device:     str = f"cuda" if torch.cuda.is_available() else "cpu",
) -> FlowAutoencoder:
    """
    Train the unsupervised autoencoder on benign flow features.

    Strategy
    --------
    Phase 1 (first 2/3 of epochs): Pure MSE reconstruction on benign data.
    Phase 2 (last 1/3 of epochs):  MSE + Contrastive loss using synthetic
                                    negative samples (simulated attack features).

    This two-phase approach first establishes a stable normal manifold, then
    hardens the boundary by explicitly pushing attack-like latents away from it.
    """
    ae = ae.to(device)
    optimizer  = torch.optim.Adam(ae.parameters(), lr=cfg.AE_LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(train_data.to(device)),
        batch_size=cfg.AE_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data.to(device)),
        batch_size=cfg.AE_BATCH_SIZE
    )

    contrastive_start = int(epochs * 0.65)
    best_val_loss = float("inf")

    print(f"\n{'='*58}")
    print(f"  Training FlowAutoencoder: {epochs} epochs  device={device}")
    print(f"  Contrastive phase starts at epoch {contrastive_start}")
    print(f"{'='*58}")

    for epoch in range(1, epochs + 1):
        ae.train()
        epoch_loss = 0.0
        use_contrastive = epoch > contrastive_start

        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_hat, z = ae(batch)

            z_neg = None
            if use_contrastive:
                # Synthetic negative: perturb normal features to simulate attack
                z_neg = ae.encode(
                    (batch + torch.randn_like(batch) * 0.3).clamp(0, 1)
                ).detach()

            loss = ae.reconstruction_loss(batch, x_hat, z, z_neg=z_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (vbatch,) in val_loader:
                xh, _ = ae(vbatch)
                val_loss += nn.functional.mse_loss(xh, vbatch).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ae.state_dict(), cfg.MODELS_DIR / "autoencoder_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            phase = "MSE+Contrast" if use_contrastive else "MSE only"
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={epoch_loss/len(train_loader):.5f}  "
                  f"val={val_loss:.5f}  [{phase}]  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Load best model
    ae.load_state_dict(torch.load(cfg.MODELS_DIR / "autoencoder_best.pth"))
    print(f"\n✓ Autoencoder training complete.  Best val loss: {best_val_loss:.5f}")
    return ae


def train_stgnn(
    gnn:     AuraSTGNN,
    graphs:  list,   # List of (graph_dict, labels) from stream_graphs()
    epochs:  int = cfg.GNN_EPOCHS,
    device:  str = "cpu",
) -> AuraSTGNN:
    """
    Train the STGNN in a semi-supervised manner using available labels.

    Loss function: Binary Cross-Entropy on per-node anomaly scores.
    Since node labels are approximated from edge labels (any node with an
    incident labelled-attack edge is a 'suspicious' node), this is a
    weakly supervised approach — appropriate for the hackathon timeline.
    """
    gnn = gnn.to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.GNN_LEARNING_RATE)
    bce_loss  = nn.BCELoss()

    print(f"\n{'='*58}")
    print(f"  Training AuraSTGNN: {epochs} epochs  device={device}")
    print(f"  Graphs available: {len(graphs)}")
    print(f"{'='*58}")

    for epoch in range(1, epochs + 1):
        gnn.train()
        epoch_loss = 0.0

        for graph, edge_labels in graphs:
            x          = graph["x"].to(device)
            edge_index = graph["edge_index"].to(device)

            # Approximate node labels: node is suspicious if any incident edge is attack
            N = x.shape[0]
            node_labels = torch.zeros(N, device=device)
            if edge_labels.sum() > 0:
                attack_edges = edge_labels.bool()
                src = edge_index[0][attack_edges]
                dst = edge_index[1][attack_edges]
                node_labels[src] = 1.0
                node_labels[dst] = 1.0

            optimizer.zero_grad()
            scores, _ = gnn(x, edge_index)
            loss = bce_loss(scores, node_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/max(len(graphs), 1):.5f}")

    torch.save(gnn.state_dict(), cfg.MODELS_DIR / "stgnn_trained.pth")
    print(f"✓ STGNN training complete.")
    return gnn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AURA Model Training")
    parser.add_argument("--ae-only",  action="store_true", help="Train autoencoder only")
    parser.add_argument("--epochs",   type=int, default=None, help="Override epoch count")
    parser.add_argument("--quick",    action="store_true",   help="5-epoch quick test")
    args = parser.parse_args()

    epochs = 5 if args.quick else (args.epochs or cfg.AE_EPOCHS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # ── Step 1: Load and preprocess data ────────────────────────────────────
    print("\n[Phase 1] Loading CICIDS2017 data and fitting scaler …")
    loader = CICIDSDataLoader(load_fraction=0.15 if args.quick else cfg.DATA_LOAD_FRACTION)
    scaler = loader.fit_scaler()

    # ── Step 2: Collect ALL benign flows into a flat tensor  ─────────────────
    print("[Phase 2] Collecting benign training data …")
    benign_flows = []
    attack_graphs_for_gnn = []

    # Only use Monday (pure benign) for AE training; attack CSVs for GNN
    for graph, labels in loader.stream_graphs(scaler, csv_files=[CSV_FILES[0]]):
        # All edges in Monday should be benign — use for AE baseline
        benign_flows.append(graph["edge_attr"])

        if len(attack_graphs_for_gnn) < 100:
            attack_graphs_for_gnn.append((graph, labels))

        if len(benign_flows) >= 50:  # Cap for speed during hackathon
            break

    if not benign_flows:
        logger.error("No benign data collected.  Check CSV paths in config.py.")
        sys.exit(1)

    all_benign = torch.cat(benign_flows, dim=0)   # [N_total, F]
    logger.info(f"Total benign flows collected: {all_benign.shape[0]}")

    # Train/val split (80/20)
    n_val      = int(len(all_benign) * 0.20)
    n_train    = len(all_benign) - n_val
    train_data, val_data = random_split(
        all_benign, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_tensor = all_benign[list(train_data.indices)]
    val_tensor   = all_benign[list(val_data.indices)]

    # ── Step 3: Train Autoencoder ────────────────────────────────────────────
    print("[Phase 3] Training FlowAutoencoder …")
    ae = FlowAutoencoder()
    ae = train_autoencoder(ae, train_tensor, val_tensor, epochs=epochs, device=device)

    # ── Step 4: Train STGNN (optional) ───────────────────────────────────────
    if not args.ae_only and attack_graphs_for_gnn:
        # Also stream attack data for GNN training
        print("[Phase 4] Collecting attack graph windows for STGNN …")
        for csv_file in CSV_FILES[1:4]:   # Attack-containing CSVs
            for graph, labels in loader.stream_graphs(scaler, csv_files=[csv_file]):
                attack_graphs_for_gnn.append((graph, labels))
                if len(attack_graphs_for_gnn) >= 200:
                    break

        print(f"[Phase 4] Training AuraSTGNN on {len(attack_graphs_for_gnn)} graphs …")
        gnn = AuraSTGNN()
        gnn = train_stgnn(gnn, attack_graphs_for_gnn, epochs=min(epochs, cfg.GNN_EPOCHS), device=device)
    else:
        logger.info("Skipping STGNN training (--ae-only or no attack graphs).")
        gnn = AuraSTGNN()

    # ── Step 5: Save full bundle ──────────────────────────────────────────────
    bundle = AURAModelBundle()
    bundle.autoencoder.load_state_dict(ae.state_dict())
    if not args.ae_only:
        bundle.stgnn.load_state_dict(gnn.state_dict())

    bundle_path = cfg.MODELS_DIR / "aura_bundle.pth"
    torch.save(bundle.state_dict(), bundle_path)
    print(f"\n✓ Full AURA bundle saved: {bundle_path}")
    print(f"  Total parameters: {bundle.total_params():,}")

    # ── Step 6: Save scaler for inference ────────────────────────────────────
    import joblib
    scaler_path = cfg.MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")

    print(f"\n{'='*58}")
    print("  AURA Training Complete — Ready for inference and demo!")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
