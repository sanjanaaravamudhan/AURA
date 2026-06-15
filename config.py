"""
config.py — AURA Global Configuration
======================================
Single source of truth for all hyperparameters, paths, and system constants.
Centralising config prevents magic numbers from scattering across modules and
makes hackathon tuning fast (one file to change).
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.resolve()
CSV_DIR    = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR   = BASE_DIR / "logs"
CONTRACTS_DIR = BASE_DIR / "contracts"

# Ensure output dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (NF-UNSW-NB15-v3)
# ─────────────────────────────────────────────────────────────────────────────

# NF-UNSW-NB15-v3 uses real source/destination IPs for genuine topology.
NUM_SYNTHETIC_NODES = 20

# Column name for the target label
LABEL_COL = "Label"

# The label value that represents benign (normal) traffic in NF-UNSW-NB15-v3
# Label column is binary: 0 = Benign, 1 = Attack
BENIGN_LABEL = 0

# Fraction of data to load per CSV (1.0 = all rows; reduce for speed during dev)
DATA_LOAD_FRACTION = 0.3   # 30 % is enough to demo; use 1.0 for full training

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH / TTL EDGE DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Rolling time-window size in simulated "ticks" (1 tick ≈ 1 second of NetFlow)
WINDOW_SIZE = 60          # number of flow rows per graph snapshot

# Time-To-Live: an edge is pruned after this many windows without traffic
EDGE_TTL_WINDOWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER (Layer 1 — Statistical Tripwire)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM     = 47    # Number of normalised NetFlow statistical features (NF-UNSW-NB15-v3)
ENCODER_DIMS    = [32, 24]   # Progressive compression (smaller network for 47 features)
LATENT_DIM      = 16    # Bottleneck: the latent fingerprint space
DECODER_DIMS    = [24, 32]   # Mirror of encoder (symmetric reconstruction)

AE_LEARNING_RATE = 1e-3
AE_EPOCHS        = 30        # Enough for convergence on NF-UNSW-NB15-v3 subset
AE_BATCH_SIZE    = 256

# Contrastive negative-sampling margin (pushes attack embeddings away from
# the normal manifold during simulated baseline hardening)
CONTRASTIVE_MARGIN = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# STGNN (Layer 2 — Contextual Validator)
# ─────────────────────────────────────────────────────────────────────────────

# Node feature dimensionality fed to the GNN
# Each node's feature vector = mean of its incident edge (flow) features
GNN_INPUT_DIM  = FEATURE_DIM
GNN_HIDDEN_DIM = 64
GNN_OUTPUT_DIM = 32          # Latent node embedding dimension
GNN_LEARNING_RATE = 5e-4
GNN_EPOCHS     = 20

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLDING (Exponential Moving Average over batch MSE)
# ─────────────────────────────────────────────────────────────────────────────

# EMA smoothing factor (α). Higher = reacts faster but is noisier.
# Lower = more stable but slower to adapt.
EMA_ALPHA = 0.05

# An alert is raised when:  loss > EMA_mean + (EMA_SIGMA_MULTIPLIER × EMA_std)
EMA_SIGMA_MULTIPLIER = 3.0

# Warm-up batches before thresholds are active (avoids cold-start false alarms)
EMA_WARMUP_BATCHES = 50

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY ENGINE — Temporal Accumulator + EMA Trajectory
# ─────────────────────────────────────────────────────────────────────────────

# Rolling window for per-node event accumulation (seconds).
# Events older than this are purged before escalation rules are evaluated.
TEMPORAL_WINDOW_SECONDS = 300   # 5 minutes — configurable

# EMA trajectory persistence threshold.
# K consecutive readings above 2.0σ → MEDIUM floor.
# K consecutive readings above 2.5σ → HIGH floor.
K_CONSECUTIVE_READINGS  = 5     # configurable


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING (Flower + Krum Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

FL_SERVER_ADDRESS   = "localhost:8080"
FL_NUM_ROUNDS       = 3          # 3 rounds for 3 clients — 1 hash per round on ledger
FL_MIN_CLIENTS      = 3          # Minimum clients needed to start a round
FL_MIN_AVAILABLE    = 3          # All 3 orgs must be present before round 1

# Krum: number of clients to select per round (must be ≤ total clients - 2)
# Krum drops the m clients whose weight updates are most distant from the median.
KRUM_NUM_TO_SELECT  = 2          # Select 2 from 3 mock clients (drops 1 straggler)

# Straggler policy: if a client doesn't respond within this many seconds, drop it
FL_ROUND_TIMEOUT_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST AGGREGATION (replaces Krum — Upgrade 6)
# ─────────────────────────────────────────────────────────────────────────────

# Number of synthetic benign samples the server holds as its trusted root dataset.
# These are used to train the server model by one step each round so it computes
# a reference gradient direction for cosine trust scoring of client updates.
# Range: 100–500 recommended; lower = faster, higher = more robust server gradient.
FLTRUST_ROOT_SAMPLES   = 200

# Learning rate used for the server's single-step root-dataset gradient update.
# Kept separate from AE_LEARNING_RATE so it can be tuned independently.
FLTRUST_SERVER_LR      = 1e-3

# Trust score at or below this value causes the client to be flagged as Byzantine
# in the detection log (fed into Upgrade 3).  ReLU already zeroes negatives;
# this threshold lets you also zero out near-zero trust scores from noisy clients.
FLTRUST_MIN_TRUST_SCORE = 0.0   # 0.0 = ReLU only (strict); raise to e.g. 0.05 to be stricter

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE ENGINE — Critical Infrastructure Allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Tier-1 "never hard-isolate" nodes (simulated by synthetic node IDs)
# In production these would be real IP CIDRs or hostnames.
CRITICAL_ALLOWLIST = {
    "node_0":  "Domain Controller (AD)",
    "node_1":  "Core HR Database",
    "node_2":  "Payment Gateway",
    "node_3":  "SCADA / ICS Controller",
}

# Confidence thresholds for the 3-tier response policy
CONFIDENCE_LOW_THRESHOLD  = 0.40   # Below this: log only
CONFIDENCE_MED_THRESHOLD  = 0.70   # Below this: throttle + HITL
# Above MED_THRESHOLD → full isolation for non-critical nodes

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN / GANACHE (Immutable Audit Log)
# ─────────────────────────────────────────────────────────────────────────────

GANACHE_URL              = "http://127.0.0.1:7545"
CONTRACT_ADDRESS_FILE    = str(MODELS_DIR / "contract_address.txt")
CONTRACT_ABI_FILE        = str(CONTRACTS_DIR / "ModelRegistry.abi")

# If Ganache is not running, AURA falls back to local SHA-256 file logging
BLOCKCHAIN_FALLBACK_LOG  = str(LOGS_DIR / "blockchain_fallback.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST (Baseline Sanitisation)
# ─────────────────────────────────────────────────────────────────────────────

# Contamination: expected fraction of mislabelled / poisoned rows in the
# "normal" training split.  NF-UNSW-NB15-v3 is ~94.6% benign but we
# apply a small contamination rate defensively.
IF_CONTAMINATION = 0.02   # 2 % — removes extreme statistical outliers

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_REFRESH_INTERVAL_MS = 1500   # Streamlit auto-refresh period
ALERT_LOG_FILE = str(LOGS_DIR / "aura_alerts.jsonl")
EVENT_LOG_FILE = str(LOGS_DIR / "aura_events.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM INJECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# NetFlow feature index lookup — used by _run_inject_inference() in api_server.py
# to resolve feature names to their column positions without hardcoding integers.
# Derived from NF-UNSW-NB15-v3 schema (IPs, ports, timestamps, Label, Attack stripped).
#
# Feature column order (47 features, 0-indexed):
#   0:  PROTOCOL                    1:  L7_PROTO
#   2:  IN_BYTES                    3:  IN_PKTS
#   4:  OUT_BYTES                   5:  OUT_PKTS
#   6:  TCP_FLAGS                   7:  CLIENT_TCP_FLAGS
#   8:  SERVER_TCP_FLAGS            9:  FLOW_DURATION_MILLISECONDS
#  10:  DURATION_IN                11:  DURATION_OUT
#  12:  MIN_TTL                    13:  MAX_TTL
#  14:  LONGEST_FLOW_PKT           15:  SHORTEST_FLOW_PKT
#  16:  MIN_IP_PKT_LEN             17:  MAX_IP_PKT_LEN
#  18:  SRC_TO_DST_SECOND_BYTES    19:  DST_TO_SRC_SECOND_BYTES
#  20:  RETRANSMITTED_IN_BYTES     21:  RETRANSMITTED_IN_PKTS
#  22:  RETRANSMITTED_OUT_BYTES    23:  RETRANSMITTED_OUT_PKTS
#  24:  SRC_TO_DST_AVG_THROUGHPUT  25:  DST_TO_SRC_AVG_THROUGHPUT
#  26:  NUM_PKTS_UP_TO_128_BYTES   27:  NUM_PKTS_128_TO_256_BYTES
#  28:  NUM_PKTS_256_TO_512_BYTES  29:  NUM_PKTS_512_TO_1024_BYTES
#  30:  NUM_PKTS_1024_TO_1514_BYTES
#  31:  TCP_WIN_MAX_IN             32:  TCP_WIN_MAX_OUT
#  33:  ICMP_TYPE                  34:  ICMP_IPV4_TYPE
#  35:  DNS_QUERY_ID               36:  DNS_QUERY_TYPE
#  37:  DNS_TTL_ANSWER             38:  FTP_COMMAND_RET_CODE
#  39:  SRC_TO_DST_IAT_MIN         40:  SRC_TO_DST_IAT_MAX
#  41:  SRC_TO_DST_IAT_AVG         42:  SRC_TO_DST_IAT_STDDEV
#  43:  DST_TO_SRC_IAT_MIN         44:  DST_TO_SRC_IAT_MAX
#  45:  DST_TO_SRC_IAT_AVG         46:  DST_TO_SRC_IAT_STDDEV
FEATURE_INDEX_MAP: dict = {
    "protocol":                 0,
    "l7_proto":                 1,
    "in_bytes":                 2,
    "in_pkts":                  3,
    "out_bytes":                4,
    "out_pkts":                 5,
    "tcp_flags":                6,
    "client_tcp_flags":         7,
    "server_tcp_flags":         8,
    "flow_duration":            9,
    "duration_in":              10,
    "duration_out":             11,
    "min_ttl":                  12,
    "max_ttl":                  13,
    "longest_flow_pkt":         14,
    "shortest_flow_pkt":        15,
    "min_ip_pkt_len":           16,
    "max_ip_pkt_len":           17,
    "src_to_dst_second_bytes":  18,
    "dst_to_src_second_bytes":  19,
    "retransmitted_in_bytes":   20,
    "retransmitted_in_pkts":    21,
    "retransmitted_out_bytes":  22,
    "retransmitted_out_pkts":   23,
    "src_to_dst_avg_throughput": 24,
    "dst_to_src_avg_throughput": 25,
    "num_pkts_up_to_128_bytes": 26,
    "num_pkts_128_to_256_bytes": 27,
    "num_pkts_256_to_512_bytes": 28,
    "num_pkts_512_to_1024_bytes": 29,
    "num_pkts_1024_to_1514_bytes": 30,
    "tcp_win_max_in":           31,
    "tcp_win_max_out":          32,
    "icmp_type":                33,
    "icmp_ipv4_type":           34,
    "dns_query_id":             35,
    "dns_query_type":           36,
    "dns_ttl_answer":           37,
    "ftp_command_ret_code":     38,
    "src_to_dst_iat_min":       39,
    "src_to_dst_iat_max":       40,
    "src_to_dst_iat_avg":       41,
    "src_to_dst_iat_stddev":    42,
    "dst_to_src_iat_min":       43,
    "dst_to_src_iat_max":       44,
    "dst_to_src_iat_avg":       45,
    "dst_to_src_iat_stddev":    46,
}

# MSE severity thresholds for custom injection events.
# These values are calibrated to the current AE's reconstruction error scale.
# Raise MSE_THRESHOLD_HIGH to require stronger anomaly evidence for HIGH tier.
MSE_THRESHOLD_HIGH   = 0.7   # MSE above this → AlertSeverity.HIGH
MSE_THRESHOLD_MEDIUM = 0.4   # MSE above this → AlertSeverity.MEDIUM  (else LOW)

# Corruption profiles for each simulated attack type.
# Each profile maps feature-group names to their corruption ranges:
#   {feature_key_from_FEATURE_INDEX_MAP: (lo, hi)}
# _run_inject_inference() applies each group in order and skips absent keys.
ATTACK_CORRUPTION_PROFILES: dict = {
    "ddos": {
        "in_pkts":                  (0.90, 0.99),
        "out_pkts":                 (0.85, 0.99),
        "src_to_dst_second_bytes":  (0.88, 0.99),
        "src_to_dst_iat_avg":       (0.00, 0.03),   # near-zero = flood
        "src_to_dst_iat_stddev":    (0.00, 0.02),   # robotic regularity
        "tcp_flags":                (0.80, 0.99),    # mass SYN without ACK
        "flow_duration":            (0.00, 0.05),    # very short flows
    },
    "lateral": {
        "flow_duration":            (0.50, 0.75),
        "in_pkts":                  (0.50, 0.65),
        "src_to_dst_iat_stddev":    (0.75, 0.95),   # high jitter = evasion
        "dst_to_src_iat_stddev":    (0.75, 0.95),
        "duration_in":              (0.80, 0.98),    # beacon-like pauses
        "duration_out":             (0.03, 0.10),    # robotic regularity
        "client_tcp_flags":         (0.60, 0.80),
    },
    "exfil": {
        "in_bytes":                 (0.88, 0.99),    # large outbound
        "out_bytes":                (0.00, 0.06),    # nothing coming back
        "src_to_dst_second_bytes":  (0.85, 0.99),
        "dst_to_src_second_bytes":  (0.00, 0.04),
        "src_to_dst_iat_avg":       (0.40, 0.55),   # regulated pacing
        "src_to_dst_iat_stddev":    (0.00, 0.03),   # robotic timing
        "flow_duration":            (0.80, 0.98),
    },
    "port_scan": {
        "flow_duration":            (0.00, 0.03),
        "in_bytes":                 (0.00, 0.04),
        "out_bytes":                (0.00, 0.03),
        "tcp_flags":                (0.80, 0.99),    # RST/SYN mix
        "client_tcp_flags":         (0.70, 0.90),
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "web": {
        "in_bytes":                 (0.80, 0.95),
        "out_bytes":                (0.20, 0.30),
        "client_tcp_flags":         (0.85, 0.98),    # PSH flags
        "server_tcp_flags":         (0.85, 0.98),
        "flow_duration":            (0.05, 0.12),
        "src_to_dst_iat_avg":       (0.02, 0.08),
    },
    "exploits": {
        "in_bytes":                 (0.70, 0.95),
        "longest_flow_pkt":         (0.85, 0.99),    # oversized payloads
        "tcp_flags":                (0.60, 0.85),
        "flow_duration":            (0.10, 0.30),
        "retransmitted_in_bytes":   (0.40, 0.70),    # retransmits from instability
        "src_to_dst_iat_stddev":    (0.50, 0.80),
    },
    "fuzzers": {
        "in_bytes":                 (0.60, 0.90),
        "in_pkts":                  (0.70, 0.95),
        "longest_flow_pkt":         (0.70, 0.99),
        "shortest_flow_pkt":        (0.00, 0.05),    # mixed sizes = fuzzing
        "src_to_dst_iat_stddev":    (0.80, 0.99),    # chaotic timing
        "flow_duration":            (0.10, 0.40),
        "protocol":                 (0.80, 0.99),    # unusual protocols
    },
    "reconnaissance": {
        "flow_duration":            (0.00, 0.05),    # very short probe flows
        "in_bytes":                 (0.00, 0.08),    # minimal data
        "out_bytes":                (0.00, 0.06),
        "tcp_flags":                (0.70, 0.95),    # SYN probes
        "num_pkts_up_to_128_bytes": (0.80, 0.99),    # small packets
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "backdoor": {
        "flow_duration":            (0.60, 0.90),
        "in_bytes":                 (0.40, 0.65),
        "out_bytes":                (0.40, 0.65),    # symmetric C2 traffic
        "src_to_dst_iat_avg":       (0.50, 0.70),    # periodic beaconing
        "src_to_dst_iat_stddev":    (0.00, 0.05),    # robotic regularity
        "dst_to_src_iat_avg":       (0.50, 0.70),
        "dst_to_src_iat_stddev":    (0.00, 0.05),
    },
    "custom": {
        # Generic high-variance anomaly: packet size, chaotic IAT, unusual protocols
        "longest_flow_pkt":         (0.85, 0.99),
        "shortest_flow_pkt":        (0.85, 0.99),
        "max_ip_pkt_len":           (0.85, 0.99),
        "min_ip_pkt_len":           (0.85, 0.99),
        "src_to_dst_iat_avg":       (0.02, 0.05),   # near-zero IAT
        "src_to_dst_iat_stddev":    (0.92, 0.99),   # chaotic
        "protocol":                 (0.90, 0.99),
    },
}
