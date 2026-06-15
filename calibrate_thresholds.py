"""
calibrate_thresholds.py — AURA Threshold Calibration & Feature Audit
======================================================================

Run this ONCE before any demo. It does two things:

  1. THRESHOLD CALIBRATION
     Loads the pre-trained autoencoder and runs it over clean Monday CSV
     (benign normal traffic only). Prints percentile statistics of the
     resulting MSE distribution and recommends concrete values for:
       config.MSE_THRESHOLD_HIGH   (→ 99th-percentile of normal MSE)
       config.MSE_THRESHOLD_MEDIUM (→ 90th-percentile of normal MSE)

  2. FEATURE INDEX AUDIT
     Reads the actual column ordering from Monday CSV (after stripping
     whitespace, exactly as data_loader does) and compares every entry
     in config.FEATURE_INDEX_MAP against the real column position.
     Prints PASS / MISMATCH / MISSING for each key so you can fix any
     off-by-one errors before the injection pipeline silently corrupts
     wrong features.

Usage
-----
    python calibrate_thresholds.py              # uses saved AE checkpoint
    python calibrate_thresholds.py --train-quick # trains a fresh AE quickly then calibrates

If no checkpoint is found and --train-quick is not passed, the script
falls back to a randomly-initialised AE and notes that numbers
are meaningless --- you MUST train first.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import config as cfg
from aura.models import FlowAutoencoder
from aura.data_loader import CICIDSDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate")

# ── Constants ─────────────────────────────────────────────────────────────────
AE_CHECKPOINT = cfg.MODELS_DIR / "autoencoder_best.pth"
BENIGN_CSV    = "Monday-WorkingHours.pcap_ISCX.csv"

# How many graph windows to sample for calibration (more = better estimate)
MAX_CALIBRATION_WINDOWS = 200


# =============================================================================
# 1. LOAD OR TRAIN AUTOENCODER
# =============================================================================

def load_or_train_ae(train_quick: bool) -> tuple[FlowAutoencoder, bool]:
    """
    Returns (ae_model, is_trained).
    is_trained=False means the model is random and numbers are garbage.
    """
    ae = FlowAutoencoder()

    if AE_CHECKPOINT.exists():
        log.info(f"Loading checkpoint: {AE_CHECKPOINT}")
        state = torch.load(AE_CHECKPOINT, map_location="cpu", weights_only=True)
        ae.load_state_dict(state)
        ae.eval()
        log.info("Checkpoint loaded successfully.")
        return ae, True

    if train_quick:
        log.warning("No checkpoint found — running quick training (5 epochs on Monday benign data).")
        _quick_train(ae)
        ae.eval()
        return ae, True

    log.error(
        "No checkpoint found at %s and --train-quick not passed.\n"
        "MSE numbers will be meaningless (random weights).\n"
        "Run:  python train.py --ae-only --quick   first, then re-run this script.",
        AE_CHECKPOINT,
    )
    ae.eval()
    return ae, False


def _quick_train(ae: FlowAutoencoder, epochs: int = 5):
    """Very fast AE training pass — only for immediate demo needs."""
    import torch.optim as optim

    loader = CICIDSDataLoader(load_fraction=0.1)
    scaler = loader.fit_scaler()

    optimiser = optim.Adam(ae.parameters(), lr=cfg.AE_LEARNING_RATE)
    ae.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = []
        for graph, _ in loader.stream_graphs(scaler, csv_files=[BENIGN_CSV]):
            edge_attr = graph["edge_attr"]   # [E, 78]
            if edge_attr.shape[0] == 0:
                continue
            optimiser.zero_grad()
            x_hat, z = ae(edge_attr)
            loss = torch.nn.functional.mse_loss(x_hat, edge_attr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss.append(loss.item())
        log.info(f"  Quick-train epoch {epoch}/{epochs}  mean_loss={np.mean(epoch_loss):.6f}")

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), AE_CHECKPOINT)
    log.info(f"Quick-trained AE saved to {AE_CHECKPOINT}")


# =============================================================================
# 2. COLLECT NORMAL MSE DISTRIBUTION
# =============================================================================

def collect_normal_mse(ae: FlowAutoencoder) -> np.ndarray:
    """
    Stream the Monday (benign) CSV through the AE and collect per-flow MSE.
    Returns a flat numpy array of MSE values.
    """
    loader = CICIDSDataLoader(load_fraction=cfg.DATA_LOAD_FRACTION)
    log.info("Fitting scaler on benign data …")
    scaler = loader.fit_scaler()

    all_mse = []
    windows_processed = 0

    log.info(f"Streaming benign windows (max {MAX_CALIBRATION_WINDOWS}) …")
    for graph, labels in loader.stream_graphs(scaler, csv_files=[BENIGN_CSV]):
        # Only use edges labelled benign (label=0) — pure normal traffic
        edge_attr = graph["edge_attr"]    # [E, 78]
        benign_mask = (labels == 0)

        if benign_mask.sum() == 0:
            continue

        x_benign = edge_attr[benign_mask]   # [E', 78]
        mse = ae.anomaly_score(x_benign)    # [E']  — no_grad inside
        all_mse.extend(mse.cpu().numpy().tolist())

        windows_processed += 1
        if windows_processed >= MAX_CALIBRATION_WINDOWS:
            break

    all_mse = np.array(all_mse, dtype=np.float32)
    log.info(f"Collected {len(all_mse):,} MSE samples from {windows_processed} windows.")
    return all_mse


# =============================================================================
# 3. PRINT STATISTICS & RECOMMENDATIONS
# =============================================================================

def print_mse_report(mse_values: np.ndarray, is_trained: bool):
    """Print the MSE distribution report with threshold recommendations."""

    print("\n" + "=" * 70)
    print("  AURA — AUTOENCODER MSE CALIBRATION REPORT")
    print("=" * 70)

    if not is_trained:
        print("\n  ⚠️  WARNING: Model is UNTRAINED (random weights).")
        print("     Run 'python train.py --ae-only --quick' first.")
        print("     These numbers are MEANINGLESS.\n")

    if len(mse_values) == 0:
        print("  ❌ No MSE samples collected. Check CSV path.")
        return

    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    pct_values  = {p: float(np.percentile(mse_values, p)) for p in percentiles}

    print(f"\n  Samples collected : {len(mse_values):,}")
    print(f"  Min MSE           : {mse_values.min():.6f}")
    print(f"  Mean MSE          : {mse_values.mean():.6f}")
    print(f"  Max MSE           : {mse_values.max():.6f}")
    print(f"  Std MSE           : {mse_values.std():.6f}")
    print()
    print("  Percentile breakdown:")
    for p, v in pct_values.items():
        bar = "█" * int(v * 200)
        print(f"    P{str(p).ljust(5)} : {v:.6f}  {bar}")

    p90  = pct_values[90]
    p99  = pct_values[99]
    p995 = pct_values[99.5]

    print("\n" + "-" * 70)
    print("  RECOMMENDATIONS FOR config.py")
    print("-" * 70)
    print(f"\n  MSE_THRESHOLD_MEDIUM = {p90:.4f}   # 90th-percentile normal MSE")
    print(f"  MSE_THRESHOLD_HIGH   = {p99:.4f}   # 99th-percentile normal MSE")
    print()
    print(f"  (Conservative: use P99.5 = {p995:.4f} for HIGH if false-positive rate is high)")
    print()

    # Sanity check current config values
    current_high   = cfg.MSE_THRESHOLD_HIGH
    current_medium = cfg.MSE_THRESHOLD_MEDIUM

    print("  Current config values:")
    print(f"    MSE_THRESHOLD_HIGH   = {current_high}")
    print(f"    MSE_THRESHOLD_MEDIUM = {current_medium}")
    print()

    if current_high < p99:
        print(f"  ⚠️  ALERT: current HIGH threshold ({current_high}) is BELOW the 99th percentile")
        print(f"     of normal traffic ({p99:.4f}). Normal traffic will fire as HIGH → 3-tier")
        print(f"     response loses meaning. Increase to at least {p99:.4f}.")
    elif current_high > p995 * 3:
        print(f"  ⚠️  ALERT: current HIGH threshold ({current_high}) is very conservative.")
        print(f"     Real attacks may only reach {p995:.4f}–{pct_values[99.9]:.4f}. Consider lowering.")
    else:
        print(f"  ✓  HIGH threshold looks reasonable.")

    if current_medium < p90:
        print(f"  ⚠️  ALERT: current MEDIUM threshold ({current_medium}) is below P90 of normal MSE.")
        print(f"     Abundant false MEDIUM alerts likely. Recommended: {p90:.4f}")
    else:
        print(f"  ✓  MEDIUM threshold looks reasonable.")

    print()
    print("  To apply recommendations, edit config.py lines 217–218:")
    print(f"    MSE_THRESHOLD_HIGH   = {p99:.4f}")
    print(f"    MSE_THRESHOLD_MEDIUM = {p90:.4f}")
    print("=" * 70)


# =============================================================================
# 4. FEATURE INDEX AUDIT
# =============================================================================

def audit_feature_index_map():
    """
    Compare FEATURE_INDEX_MAP against the real column ordering from the CSV.
    Prints PASS / MISMATCH / MISSING for every entry.
    """
    print("\n" + "=" * 70)
    print("  AURA — FEATURE_INDEX_MAP AUDIT")
    print("=" * 70)

    benign_path = cfg.CSV_DIR / BENIGN_CSV
    if not benign_path.exists():
        print(f"\n  ❌ CSV not found: {benign_path}")
        print("     Cannot audit feature ordering without the source CSV.")
        print("=" * 70)
        return

    import pandas as pd

    log.info(f"Reading CSV header from {benign_path} …")
    # Read only the header row — fastest possible
    header_df = pd.read_csv(benign_path, nrows=0, low_memory=False)
    # Strip whitespace exactly as data_loader does
    columns = [c.strip() for c in header_df.columns]

    # Remove label column to get feature columns only (matches data_loader logic)
    label_col_clean = cfg.LABEL_COL.strip()
    feature_cols = [c for c in columns if c != label_col_clean]

    # Build name→index lookup from actual CSV
    actual_index: dict[str, int] = {col: idx for idx, col in enumerate(feature_cols)}

    print(f"\n  Total feature columns in CSV : {len(feature_cols)}")
    print(f"  config.FEATURE_DIM           : {cfg.FEATURE_DIM}")
    if len(feature_cols) != cfg.FEATURE_DIM:
        print(f"  ❌ MISMATCH! CSV has {len(feature_cols)} features but FEATURE_DIM={cfg.FEATURE_DIM}")
    else:
        print(f"  ✓  Feature count matches FEATURE_DIM.")
    print()

    # Build a reverse map: NF-UNSW-NB15-v3 column name variants → config key
    # The CSV uses proper names like "Destination Port", "Flow Duration" etc.
    # We need to map config keys (snake_case) to likely CSV column names.
    # We do a case-insensitive substring match as a heuristic.

    # Explicit canonical mapping: config_key → expected CSV column substring
    CANONICAL_MAP = {
        "dest_port":          "Destination Port",
        "flow_duration":      "Flow Duration",
        "fwd_packets":        "Total Fwd Packets",
        "bwd_packets":        "Total Backward Packets",
        "fwd_bytes":          "Total Length of Fwd Packets",
        "bwd_bytes":          "Total Length of Bwd Packets",
        "fwd_pkt_len_max":    "Fwd Packet Length Max",
        "fwd_pkt_len_min":    "Fwd Packet Length Min",
        "fwd_pkt_len_mean":   "Fwd Packet Length Mean",
        "fwd_pkt_len_std":    "Fwd Packet Length Std",
        "bwd_pkt_len_max":    "Bwd Packet Length Max",
        "bwd_pkt_len_min":    "Bwd Packet Length Min",
        "bwd_pkt_len_mean":   "Bwd Packet Length Mean",
        "bwd_pkt_len_std":    "Bwd Packet Length Std",
        "flow_bytes_s":       "Flow Bytes/s",
        "flow_pkts_s":        "Flow Packets/s",
        "flow_iat_mean":      "Flow IAT Mean",
        "flow_iat_std":       "Flow IAT Std",
        "flow_iat_max":       "Flow IAT Max",
        "flow_iat_min":       "Flow IAT Min",
        "fwd_iat_total":      "Fwd IAT Total",
        "fwd_psh_flags":      "Fwd PSH Flags",
        "pkt_len_std":        "Packet Length Std",
        "pkt_len_var":        "Packet Length Variance",
        "syn_flag_count":     "SYN Flag Count",
        "rst_flag_count":     "RST Flag Count",
        "psh_flag_count":     "PSH Flag Count",
        "ack_flag_count":     "ACK Flag Count",
        "subflow_fwd_bytes":  "Subflow Fwd Bytes",
        "subflow_bwd_bytes":  "Subflow Bwd Bytes",
        "idle_mean":          "Idle Mean",
        "idle_std":           "Idle Std",
    }

    passed   = 0
    mismatched = 0
    missing  = 0

    print(f"  {'Config Key':<22} {'Config Idx':>10}  {'CSV Column Found':<35} {'CSV Real Idx':>12}  Status")
    print(f"  {'-'*22} {'-'*10}  {'-'*35} {'-'*12}  ------")

    for config_key, config_idx in sorted(cfg.FEATURE_INDEX_MAP.items(), key=lambda x: x[1]):
        expected_col_name = CANONICAL_MAP.get(config_key, None)

        if expected_col_name is None:
            # Try case-insensitive fuzzy match
            matches = [c for c in feature_cols if config_key.replace("_", " ").lower() in c.lower()]
            expected_col_name = matches[0] if matches else None

        if expected_col_name is None or expected_col_name not in actual_index:
            # Try case-insensitive fallback
            ci_matches = [c for c in feature_cols
                          if (expected_col_name or "").lower() in c.lower() or
                             c.lower() in (expected_col_name or "").lower()]
            if ci_matches:
                expected_col_name = ci_matches[0]

        if expected_col_name and expected_col_name in actual_index:
            real_idx = actual_index[expected_col_name]
            if real_idx == config_idx:
                status = "✓ PASS"
                passed += 1
            else:
                status = f"❌ MISMATCH (real={real_idx})"
                mismatched += 1
        else:
            real_idx = "?"
            status = "⚠️  NOT FOUND IN CSV"
            missing += 1

        col_display = (expected_col_name or "?")[:35]
        print(f"  {config_key:<22} {config_idx:>10}  {col_display:<35} {str(real_idx):>12}  {status}")

    print()
    print(f"  Results: {passed} PASS  |  {mismatched} MISMATCH  |  {missing} NOT FOUND")

    if mismatched > 0 or missing > 0:
        print()
        print("  ❌ ACTION REQUIRED: FEATURE_INDEX_MAP has incorrect indices.")
        print("     The injection profiles are corrupting wrong features silently.")
        print("     Update FEATURE_INDEX_MAP in config.py with the 'real' column indices shown above.")
    else:
        print()
        print("  ✓  All FEATURE_INDEX_MAP entries match the actual CSV column ordering.")

    # Print the full feature column list for manual verification
    print()
    print("  Full feature column list (index → name) from CSV:")
    for i, col in enumerate(feature_cols):
        marker = " ←── in FEATURE_INDEX_MAP" if i in cfg.FEATURE_INDEX_MAP.values() else ""
        print(f"    [{i:3d}] {col}{marker}")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AURA threshold calibration + feature audit")
    parser.add_argument(
        "--train-quick", action="store_true",
        help="If no checkpoint exists, run 5-epoch quick AE training before calibrating."
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Skip MSE calibration, only run the feature index audit."
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Skip feature index audit, only run MSE calibration."
    )
    args = parser.parse_args()

    # ── Feature Index Audit ───────────────────────────────────────────────────
    if not args.calibrate_only:
        audit_feature_index_map()

    # ── MSE Calibration ───────────────────────────────────────────────────────
    if not args.audit_only:
        ae, is_trained = load_or_train_ae(args.train_quick)
        mse_values = collect_normal_mse(ae)
        print_mse_report(mse_values, is_trained)

        if is_trained and len(mse_values) > 0:
            p90 = float(np.percentile(mse_values, 90))
            p99 = float(np.percentile(mse_values, 99))

            # Write recommended values to a JSON file for easy reference
            import json
            results = {
                "n_samples":              int(len(mse_values)),
                "mse_min":                float(mse_values.min()),
                "mse_mean":               float(mse_values.mean()),
                "mse_max":                float(mse_values.max()),
                "mse_std":                float(mse_values.std()),
                "p50":                    float(np.percentile(mse_values, 50)),
                "p75":                    float(np.percentile(mse_values, 75)),
                "p90":                    float(np.percentile(mse_values, 90)),
                "p95":                    float(np.percentile(mse_values, 95)),
                "p99":                    float(np.percentile(mse_values, 99)),
                "p99_5":                  float(np.percentile(mse_values, 99.5)),
                "p99_9":                  float(np.percentile(mse_values, 99.9)),
                "recommended_MSE_THRESHOLD_MEDIUM": round(p90, 4),
                "recommended_MSE_THRESHOLD_HIGH":   round(p99, 4),
                "current_MSE_THRESHOLD_MEDIUM":     cfg.MSE_THRESHOLD_MEDIUM,
                "current_MSE_THRESHOLD_HIGH":       cfg.MSE_THRESHOLD_HIGH,
            }
            out_path = cfg.LOGS_DIR / "calibration_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
