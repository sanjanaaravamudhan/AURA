"""
aura/data_loader.py — Phase 1: Data Ingestion & Topological Mapping
=====================================================================

Pipeline Design
---------------
The CICIDS2017 'MachineLearningCSV' variant ships 78 NetFlow statistical
features per row but *strips* the Source/Destination IP columns for privacy.
To still construct a meaningful graph topology we apply a deterministic
synthetic node-mapping heuristic (see `_assign_synthetic_nodes`).

Processing chain (in order):
  1. Raw CSV  →  strip column whitespace  →  drop Inf/NaN
  2. Label column extracted; rows split into BENIGN and ATTACK splits
  3. Benign split sanitised with IsolationForest (Poisoned Baseline Defence)
  4. MinMaxScaler fitted on sanitised benign data; applied to all splits
  5. Rolling WINDOW_SIZE-row snapshots  →  PyTorch tensors
  6. Synthetic edges built with TTL counter; expired edges pruned each window
  7. Node features = per-node mean aggregation of incident edge features

Returns
-------
  A Python generator that yields (graph_dict, label_vector) tuples.
  graph_dict = {
      "x"          : Tensor[N_nodes, F]   — node feature matrix
      "edge_index" : Tensor[2, E]         — COO sparse adjacency
      "edge_attr"  : Tensor[E, F]         — edge (flow) features
      "ttl"        : dict[(src,dst) → int]— remaining TTL per edge
  }
  label_vector : Tensor[E] — 0=benign, 1=attack  (one per edge/flow)
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Project-level config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from inspection of the dataset
# ─────────────────────────────────────────────────────────────────────────────

# All CSV files in the MachineLearningCVE folder, ordered by weekday.
CSV_FILES: List[str] = [
    "C:\Users\Student2\AURA\dataset\Monday-WorkingHours.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Tuesday-WorkingHours.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Wednesday-workingHours.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "C:\Users\Student2\AURA\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017 CSVs use inconsistent leading/trailing spaces in column names
    (e.g. ' Label', ' Flow Duration').  This function strips all of them so
    downstream code can reference columns by clean names.
    """
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean_infinities_and_nans(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Replace np.inf / -np.inf with NaN, then forward-fill (temporal continuity)
    and finally back-fill remaining leading NaNs.  Any residual rows are
    dropped as a last resort.

    This is a known data-quality issue in CICIDS2017 — especially in the
    'Flow Bytes/s' and 'Flow Packets/s' columns which occasionally see
    divide-by-zero artifacts during pcap replay.
    """
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Forward-fill preserves temporal continuity within NetFlow streams
    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].bfill()
    # Hard drop any remaining NaN rows (should be minimal after ff/bf)
    before = len(df)
    df = df.dropna(subset=feature_cols)
    after = len(df)
    if before - after > 0:
        logger.warning(f"Dropped {before - after} rows with residual NaN/Inf after fill.")
    return df


def _isolationforest_sanitise(
    X: np.ndarray,
    contamination: float = cfg.IF_CONTAMINATION
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poisoned-Baseline Defence
    --------------------------
    Before fitting the autoencoder's baseline, we run IsolationForest on the
    BENIGN-only split.  This scrubs extreme statistical outliers — which could
    be mislabelled attack flows, sensor glitches, or adversarially injected
    'poison' rows designed to skew the normal distribution.

    IsolationForest works by randomly partitioning the feature space;
    anomalous points (those requiring fewer splits to isolate) receive a
    negative anomaly score.  Setting contamination=0.02 flags the 2% most
    extreme rows.

    Parameters
    ----------
    X             : ndarray[N, F] — normalised benign feature matrix
    contamination : expected fraction of outliers in the benign split

    Returns
    -------
    X_clean       : ndarray[N', F] — sanitised benign feature matrix (N' ≤ N)
    mask          : boolean ndarray[N] — True = row kept
    """
    logger.info(f"Running IsolationForest baseline sanitisation "
                f"(contamination={contamination}) on {len(X)} benign rows …")
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,  # Use all available cores
    )
    preds = iso.fit_predict(X)           # +1 = inlier, -1 = outlier
    mask  = preds == 1                   # Keep only inliers
    X_clean = X[mask]
    removed = int((~mask).sum())
    logger.info(f"IsolationForest removed {removed} suspicious rows "
                f"({100*removed/len(X):.2f}% of benign split). "
                f"Clean baseline size: {len(X_clean)} rows.")
    return X_clean, mask


def _assign_synthetic_nodes(
    df: pd.DataFrame,
    num_nodes: int = cfg.NUM_SYNTHETIC_NODES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Topology Mapping
    ---------------------------
    Since the MachineLearningCSV variant omits Source/Destination IP columns,
    we reconstruct a plausible flow topology using a deterministic heuristic:

      src_id = row_index % num_nodes
      dst_id = (row_index + Destination_Port_bucket) % num_nodes

    'Destination_Port_bucket' groups ports into well-known service buckets
    (e.g., 80/443 → bucket 0, 22 → bucket 1, etc.), which mimics a real
    network where different service types cluster on specific server nodes.

    This preserves the relational structure needed for GNN message-passing
    while being honest in demos ("simulated topology from flow statistics").

    Returns
    -------
    src_nodes : ndarray[E] — source node IDs (int, 0 … num_nodes-1)
    dst_nodes : ndarray[E] — destination node IDs (int, 0 … num_nodes-1)
    """
    port_col = "Destination Port" if "Destination Port" in df.columns else df.columns[0]
    ports = df[port_col].values.astype(int)

    # Map ports to 5 service buckets: HTTP(80,8080), HTTPS(443), SSH(22),
    # DNS(53), and everything else — this adds semantic structure to the topology.
    def _port_bucket(p: int) -> int:
        if p in (80, 8080, 8000):  return 1
        if p in (443, 8443):       return 2
        if p == 22:                return 3
        if p == 53:                return 4
        return 0  # Generic / ephemeral

    port_buckets  = np.array([_port_bucket(int(p)) for p in ports])
    row_indices   = np.arange(len(df))

    src_nodes = row_indices % num_nodes
    dst_nodes = (row_indices + port_buckets + num_nodes // 2) % num_nodes

    # Self-loops add no topological information — remove them
    self_loop_mask = src_nodes == dst_nodes
    dst_nodes[self_loop_mask] = (dst_nodes[self_loop_mask] + 1) % num_nodes

    return src_nodes.astype(np.int64), dst_nodes.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# TTL Edge Decay Tracker
# ─────────────────────────────────────────────────────────────────────────────

class TTLEdgeTracker:
    """
    Lightweight Time-To-Live (TTL) tracker for graph edges.

    In real NetFlow analysis, an "edge" (connection between two IPs) should
    only persist in the graph as long as the hosts are actively communicating.
    If no traffic is seen on an edge for EDGE_TTL_WINDOWS consecutive windows,
    the edge is pruned — keeping the graph sparse and the GNN computationally
    light.

    This also has a security benefit: it resets the topological baseline for
    hosts that may have been idle (preventing stale 'trusted' edges from
    masking lateral movement that reactivates an old connection).
    """

    def __init__(self, ttl: int = cfg.EDGE_TTL_WINDOWS):
        self.ttl      = ttl
        # Maps (src, dst) → remaining TTL ticks
        self._counters: Dict[Tuple[int, int], int] = defaultdict(lambda: ttl)

    def update(self, active_edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Call once per window tick.
        1. All edges in `active_edges` have their TTL reset to maximum.
        2. All other tracked edges have their TTL decremented by 1.
        3. Edges whose TTL reaches 0 are pruned.

        Returns the list of all currently *live* edges (active + not-yet-expired).
        """
        active_set = set(active_edges)

        # Reset TTL for currently active edges
        for e in active_set:
            self._counters[e] = self.ttl

        # Decrement dormant edges
        dormant = set(self._counters.keys()) - active_set
        for e in dormant:
            self._counters[e] -= 1

        # Prune expired edges
        expired = [e for e, ttl in self._counters.items() if ttl <= 0]
        for e in expired:
            del self._counters[e]

        return list(self._counters.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Node Feature Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_features(
    edge_features: np.ndarray,    # shape [E, F]
    src_nodes:     np.ndarray,    # shape [E]
    dst_nodes:     np.ndarray,    # shape [E]
    num_nodes:     int,
    feature_dim:   int,
) -> np.ndarray:
    """
    Aggregate per-edge (flow) features into per-node feature vectors.

    Each node's feature vector is computed as the mean of all edge features
    for edges incident to that node (both inbound and outbound).  This
    captures the cumulative statistical behaviour of a host — its typical
    byte ratios, packet rates, and IAT distributions — and forms the input
    tensor X fed to the STGNN.

    Shape:  X ∈ ℝ^{N × F}  where N = num_nodes, F = feature_dim.
    """
    # Initialise with zeros — nodes with no incident edges stay as zeros
    X = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)

    # Accumulate edge features into both endpoint nodes
    for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
        X[s] += edge_features[i]
        X[d] += edge_features[i]
        counts[s] += 1
        counts[d] += 1

    # Normalise by degree (avoid NaN for isolated nodes via max)
    counts = np.maximum(counts, 1.0)
    X = X / counts[:, np.newaxis]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Core Public API
# ─────────────────────────────────────────────────────────────────────────────

class CICIDSDataLoader:
    """
    End-to-end data pipeline for CICIDS2017 MachineLearningCSV → PyTorch graphs.

    Usage
    -----
    >>> loader = CICIDSDataLoader()
    >>> scaler = loader.fit_scaler()              # Step 1: fit on benign data
    >>> for graph, labels in loader.stream_graphs(scaler):  # Step 2: stream
    ...     # graph["x"], graph["edge_index"], graph["edge_attr"]
    ...     pass
    """

    def __init__(
        self,
        csv_dir:        Path = cfg.CSV_DIR,
        benign_csv:     str  = "Monday-WorkingHours.pcap_ISCX.csv",
        load_fraction:  float = cfg.DATA_LOAD_FRACTION,
        window_size:    int   = cfg.WINDOW_SIZE,
        num_nodes:      int   = cfg.NUM_SYNTHETIC_NODES,
    ):
        self.csv_dir       = csv_dir
        self.benign_csv    = benign_csv
        self.load_fraction = load_fraction
        self.window_size   = window_size
        self.num_nodes     = num_nodes
        self._ttl_tracker  = TTLEdgeTracker()
        self._scaler: Optional[MinMaxScaler] = None

        # Discovered at scan time
        self._feature_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a single CICIDS CSV file, returning a clean DataFrame."""
        path = self.csv_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        logger.info(f"Loading {filename} …")
        # Estimate rows for fractional loading (avoids full parse)
        # Using skiprows with a probability approximation
        total_rows = sum(1 for _ in open(path)) - 1  # minus header
        n_rows     = max(100, int(total_rows * self.load_fraction))
        df = pd.read_csv(path, nrows=n_rows, low_memory=False)
        df = _strip_column_whitespace(df)

        # Identify feature columns (everything except Label)
        if self._feature_cols is None:
            all_cols = list(df.columns)
            label_stripped = cfg.LABEL_COL.strip()
            self._feature_cols = [c for c in all_cols if c != label_stripped]
            logger.info(f"Discovered {len(self._feature_cols)} feature columns.")

        df = _clean_infinities_and_nans(df, self._feature_cols)
        return df

    def _label_to_binary(self, series: pd.Series) -> np.ndarray:
        """Convert string labels to binary:  0 = BENIGN,  1 = ATTACK."""
        return (series.str.strip() != cfg.BENIGN_LABEL).astype(np.int64).values

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def fit_scaler(self) -> MinMaxScaler:
        """
        Fit a MinMaxScaler on the sanitised BENIGN-only training split.

        Why benign-only?  If we fit the scaler on mixed data containing
        attacks, extreme attack values (e.g., DDoS byte floods) will
        compress the normal-traffic range and reduce the autoencoder's
        sensitivity to subtle anomalies.

        Returns the fitted scaler so it can be persisted between phases.
        """
        df = self._load_csv(self.benign_csv)
        label_col_clean = cfg.LABEL_COL.strip()

        benign_df = df[df[label_col_clean].str.strip() == cfg.BENIGN_LABEL]
        logger.info(f"Benign training rows before sanitisation: {len(benign_df)}")

        X_benign = benign_df[self._feature_cols].values.astype(np.float32)

        # ── Step 3: Poisoned Baseline Defence ────────────────────────────────
        X_clean, _ = _isolationforest_sanitise(X_benign)

        # ── Step 4: Fit scaler on clean benign distribution ──────────────────
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_clean)
        self._scaler = scaler
        logger.info("MinMaxScaler fitted on sanitised benign baseline.")
        return scaler

    def stream_graphs(
        self,
        scaler: MinMaxScaler,
        csv_files: Optional[List[str]] = None,
    ) -> Generator[Tuple[Dict, torch.Tensor], None, None]:
        """
        Generator that yields (graph_dict, label_tensor) snapshots from
        CICIDS2017 CSVs using a rolling WINDOW_SIZE-row window.

        Each yield represents one "tick" of the 1-second network sensor.

        graph_dict keys
        ---------------
        x          : FloatTensor[N, F]  — node feature matrix
        edge_index : LongTensor[2, E]   — COO sparse adjacency (after TTL pruning)
        edge_attr  : FloatTensor[E, F]  — edge (flow) feature matrix
        ttl_state  : dict               — TTL counters for live edges (for UI)
        """
        if csv_files is None:
            csv_files = CSV_FILES

        for csv_file in csv_files:
            try:
                df = self._load_csv(csv_file)
            except FileNotFoundError:
                logger.warning(f"Skipping missing file: {csv_file}")
                continue

            label_col_clean = cfg.LABEL_COL.strip()
            labels_all = self._label_to_binary(df[label_col_clean])

            # Apply the pre-fitted scaler (clamp=True silently clips unseen extremes)
            X_scaled = scaler.transform(
                df[self._feature_cols].values.astype(np.float32)
            ).clip(0, 1)

            # Synthetic node assignment (row-level)
            src_all, dst_all = _assign_synthetic_nodes(df, self.num_nodes)

            n_windows = len(df) // self.window_size
            logger.info(f"Streaming {n_windows} windows from {csv_file} …")

            for w in range(n_windows):
                s = w * self.window_size
                e = s + self.window_size

                # ── Slice this window's data ──────────────────────────────
                X_window      = X_scaled[s:e]       # [W, F]
                src_window    = src_all[s:e]         # [W]
                dst_window    = dst_all[s:e]         # [W]
                labels_window = labels_all[s:e]      # [W]

                # ── TTL Edge Decay ────────────────────────────────────────
                active_edges  = list(zip(src_window.tolist(), dst_window.tolist()))
                live_edges    = self._ttl_tracker.update(active_edges)

                # Build index mapping from live edges back to window rows
                live_edge_set = set(live_edges)
                keep_mask     = np.array([
                    (int(src_window[i]), int(dst_window[i])) in live_edge_set
                    for i in range(len(src_window))
                ])

                # Apply mask — only include flows on live edges
                if keep_mask.sum() == 0:
                    continue  # Empty window after TTL pruning; skip

                X_edge    = X_window[keep_mask]
                src_edge  = src_window[keep_mask]
                dst_edge  = dst_window[keep_mask]
                labels_w  = labels_window[keep_mask]

                # ── Node Feature Aggregation (X matrix for GNN) ───────────
                X_node = _build_node_features(
                    X_edge, src_edge, dst_edge,
                    self.num_nodes, len(self._feature_cols)
                )

                # ── Build PyTorch tensors ─────────────────────────────────
                edge_index = torch.tensor(
                    np.stack([src_edge, dst_edge], axis=0),
                    dtype=torch.long
                )                             # shape: [2, E]

                x = torch.tensor(X_node, dtype=torch.float32)
                # shape: [N, F]

                edge_attr = torch.tensor(X_edge, dtype=torch.float32)
                # shape: [E, F]

                graph_dict = {
                    "x":          x,
                    "edge_index": edge_index,
                    "edge_attr":  edge_attr,
                    "ttl_state":  dict(self._ttl_tracker._counters),
                    "window_id":  f"{csv_file}:w{w}",
                }

                label_tensor = torch.tensor(labels_w, dtype=torch.long)

                yield graph_dict, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AURA Data Loader — Sanity Check ===")
    loader = CICIDSDataLoader(load_fraction=0.05)

    print("Fitting scaler on benign baseline …")
    scaler = loader.fit_scaler()

    print("Streaming first 3 graph windows …")
    for i, (graph, labels) in enumerate(loader.stream_graphs(scaler, csv_files=[CSV_FILES[0]])):
        print(f"\n[Window {i}]  id={graph['window_id']}")
        print(f"  x.shape        = {graph['x'].shape}        (Nodes × Features)")
        print(f"  edge_index.shape= {graph['edge_index'].shape}  (2 × Edges)")
        print(f"  edge_attr.shape = {graph['edge_attr'].shape}  (Edges × Features)")
        print(f"  labels.shape    = {labels.shape}   | attack ratio={labels.float().mean():.3f}")
        print(f"  live edges (TTL)= {len(graph['ttl_state'])}")
        if i >= 2:
            break

    print("\n✓ Data loader test passed.")
