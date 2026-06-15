"""
aura/ae_explainer.py — AE Feature Attribution & Attack Classification
=======================================================================

Given a per-feature reconstruction residual vector |x - x_hat| ∈ ℝ^47,
this module:

  1. Names the top contributing features in human-readable terms
  2. Matches the residual pattern against known attack signatures
  3. Produces a plain-English explanation panel for the SOC operator

Design
------
We use a lightweight dot-product similarity between the (normalized) residual
vector and pre-defined attack signature vectors.  Each signature encodes which
features SHOULD be anomalous for that attack type, weighted by expected severity.
No additional model is needed — it's a lookup/scoring pass on top of the AE.

This is interpretable-by-design: the operator can see exactly WHICH features
drove the alert AND why the system inferred a specific attack category.
"""

import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Feature Index → Human-Readable Name
# All 47 NF-UNSW-NB15-v3 features (IPs, ports, timestamps, Label, Attack stripped)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: Dict[int, str] = {
    0:  "Protocol",
    1:  "L7 Protocol",
    2:  "Inbound Bytes",
    3:  "Inbound Packets",
    4:  "Outbound Bytes",
    5:  "Outbound Packets",
    6:  "TCP Flags",
    7:  "Client TCP Flags",
    8:  "Server TCP Flags",
    9:  "Flow Duration (ms)",
    10: "Duration In",
    11: "Duration Out",
    12: "Min TTL",
    13: "Max TTL",
    14: "Longest Flow Pkt",
    15: "Shortest Flow Pkt",
    16: "Min IP Pkt Len",
    17: "Max IP Pkt Len",
    18: "Src→Dst Bytes/s",
    19: "Dst→Src Bytes/s",
    20: "Retransmitted In Bytes",
    21: "Retransmitted In Pkts",
    22: "Retransmitted Out Bytes",
    23: "Retransmitted Out Pkts",
    24: "Src→Dst Avg Throughput",
    25: "Dst→Src Avg Throughput",
    26: "Pkts ≤128 Bytes",
    27: "Pkts 128–256 Bytes",
    28: "Pkts 256–512 Bytes",
    29: "Pkts 512–1024 Bytes",
    30: "Pkts 1024–1514 Bytes",
    31: "TCP Win Max In",
    32: "TCP Win Max Out",
    33: "ICMP Type",
    34: "ICMP IPv4 Type",
    35: "DNS Query ID",
    36: "DNS Query Type",
    37: "DNS TTL Answer",
    38: "FTP Command Ret Code",
    39: "Src→Dst IAT Min",
    40: "Src→Dst IAT Max",
    41: "Src→Dst IAT Avg",
    42: "Src→Dst IAT Stddev",
    43: "Dst→Src IAT Min",
    44: "Dst→Src IAT Max",
    45: "Dst→Src IAT Avg",
    46: "Dst→Src IAT Stddev",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature Groups (for grouped explanation display)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[int]] = {
    "Volume / Bytes":     [2, 3, 4, 5, 20, 21, 22, 23],
    "Bandwidth / Rates":  [18, 19, 24, 25],
    "Timing / IAT":       [9, 10, 11, 39, 40, 41, 42, 43, 44, 45, 46],
    "TCP / Protocol":     [0, 6, 7, 8],
    "Packet Size":        [14, 15, 16, 17, 26, 27, 28, 29, 30],
    "Application Layer":  [1, 33, 34, 35, 36, 37, 38],
    "Window / TTL":       [12, 13, 31, 32],
}


# ─────────────────────────────────────────────────────────────────────────────
# Attack Signature Vectors
# ─────────────────────────────────────────────────────────────────────────────
# Each signature is a sparse dict: {feature_index: expected_high_residual_weight}
# Values are relative importances — they get L2-normalised at match time.
# Based on the NF-UNSW-NB15-v3 attack taxonomy + attack_injector.py profiles.

ATTACK_SIGNATURES: Dict[str, Dict[int, float]] = {
    "DoS": {
        3:  1.0,   # Inbound Packets (flood)
        5:  0.9,   # Outbound Packets
        18: 0.9,   # Src→Dst Bytes/s
        41: 0.8,   # Src→Dst IAT Avg (near-zero = flood)
        42: 0.7,   # Src→Dst IAT Stddev
        6:  0.8,   # TCP Flags (SYN flood)
        9:  0.6,   # Flow Duration (short)
    },
    "Reconnaissance": {
        9:  1.0,   # Flow Duration (very short probes)
        2:  0.8,   # Inbound Bytes (minimal)
        4:  0.8,   # Outbound Bytes (minimal)
        6:  0.9,   # TCP Flags (SYN probes)
        26: 0.7,   # Pkts ≤128 Bytes (small packets)
        18: 0.5,   # Src→Dst Bytes/s (low)
    },
    "Exploits": {
        2:  0.9,   # Inbound Bytes
        14: 1.0,   # Longest Flow Pkt (oversized payloads)
        6:  0.7,   # TCP Flags
        9:  0.6,   # Flow Duration
        20: 0.8,   # Retransmitted In Bytes (instability)
        42: 0.6,   # Src→Dst IAT Stddev
    },
    "Fuzzers": {
        2:  0.8,   # Inbound Bytes
        3:  0.9,   # Inbound Packets
        14: 0.9,   # Longest Flow Pkt
        15: 0.7,   # Shortest Flow Pkt (mixed sizes)
        42: 1.0,   # Src→Dst IAT Stddev (chaotic)
        0:  0.6,   # Protocol (unusual)
    },
    "Generic": {
        2:  0.8,   # Inbound Bytes
        4:  0.8,   # Outbound Bytes
        6:  0.7,   # TCP Flags
        9:  0.6,   # Flow Duration
        18: 0.7,   # Src→Dst Bytes/s
        42: 0.5,   # Src→Dst IAT Stddev
    },
    "Backdoor": {
        9:  0.8,   # Flow Duration (long sessions)
        2:  0.7,   # Inbound Bytes
        4:  0.7,   # Outbound Bytes (symmetric C2)
        41: 1.0,   # Src→Dst IAT Avg (periodic beaconing)
        42: 0.9,   # Src→Dst IAT Stddev (robotic regularity)
        45: 0.9,   # Dst→Src IAT Avg
        46: 0.9,   # Dst→Src IAT Stddev
    },
    "Shellcode": {
        2:  1.0,   # Inbound Bytes (payload delivery)
        14: 0.9,   # Longest Flow Pkt
        6:  0.7,   # TCP Flags
        9:  0.6,   # Flow Duration (short burst)
        7:  0.8,   # Client TCP Flags (PSH)
    },
    "Worms": {
        3:  0.9,   # Inbound Packets (propagation)
        5:  0.9,   # Outbound Packets (propagation)
        18: 1.0,   # Src→Dst Bytes/s
        19: 0.8,   # Dst→Src Bytes/s
        9:  0.7,   # Flow Duration
        6:  0.6,   # TCP Flags
    },
    "Analysis": {
        9:  1.0,   # Flow Duration (extended probing)
        2:  0.6,   # Inbound Bytes
        4:  0.6,   # Outbound Bytes
        41: 0.8,   # Src→Dst IAT Avg
        42: 0.7,   # Src→Dst IAT Stddev
        26: 0.5,   # Pkts ≤128 Bytes
    },
    "Data Exfiltration": {
        2:  1.0,   # Inbound Bytes (large outbound)
        18: 0.9,   # Src→Dst Bytes/s
        9:  0.8,   # Flow Duration (sustained)
        41: 0.7,   # Src→Dst IAT Avg (regulated pacing)
        42: 0.6,   # Src→Dst IAT Stddev (robotic)
        4:  0.5,   # Outbound Bytes (near-zero return)
        19: 0.5,   # Dst→Src Bytes/s
    },
    "Lateral Movement": {
        42: 1.0,   # Src→Dst IAT Stddev (high jitter)
        10: 0.9,   # Duration In (beacon-like)
        9:  0.8,   # Flow Duration
        3:  0.7,   # Inbound Packets
        46: 0.6,   # Dst→Src IAT Stddev
        7:  0.5,   # Client TCP Flags
    },
    "Port Scan": {
        6:  1.0,   # TCP Flags (RST/SYN)
        7:  0.9,   # Client TCP Flags
        9:  0.8,   # Flow Duration (very short)
        2:  0.7,   # Inbound Bytes (minimal)
        4:  0.7,   # Outbound Bytes (minimal)
        18: 0.5,   # Src→Dst Bytes/s
    },
    "Web Attack": {
        2:  0.9,   # Inbound Bytes
        4:  0.8,   # Outbound Bytes
        7:  1.0,   # Client TCP Flags (PSH)
        8:  0.9,   # Server TCP Flags
        9:  0.7,   # Flow Duration (short bursts)
        41: 0.6,   # Src→Dst IAT Avg
    },
}


# Human-readable explanations per attack type
ATTACK_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "DoS": {
        "icon":    "🌊",
        "summary": "Denial-of-Service flood attack detected",
        "detail":  (
            "Packet rate and bandwidth are abnormally high with near-zero "
            "inter-arrival time — consistent with a UDP/SYN/ICMP flood. "
            "Incomplete TCP handshakes (high SYN, low ACK) confirm the source "
            "is NOT establishing legitimate connections. "
            "Action: rate-limit the source subnet and engage upstream scrubbing."
        ),
        "why_high": "Inbound Packets and TCP Flags are the primary drivers — "
                    "the model has never seen legitimate traffic at this rate.",
    },
    "Reconnaissance": {
        "icon":    "🔍",
        "summary": "Network reconnaissance / port scan detected",
        "detail":  (
            "Multiple extremely short flows with minimal byte transfer and "
            "high RST + SYN flag counts — the attacker is probing which services "
            "are open without completing any connection. "
            "Action: block the scanning IP, alert vulnerability management team."
        ),
        "why_high": "TCP Flags and very short Flow Duration are the primary "
                    "drivers — legitimate flows do not terminate this abruptly en masse.",
    },
    "Exploits": {
        "icon":    "💣",
        "summary": "Exploit attempt detected (buffer overflow / code execution)",
        "detail":  (
            "Unusually large packet sizes combined with TCP flag anomalies and "
            "retransmission bursts suggest payload delivery for a known exploit. "
            "The oversized packets and timing jitter indicate automated tooling. "
            "Action: isolate the target host, check for compromise indicators, "
            "verify patch levels."
        ),
        "why_high": "Longest Flow Pkt and Retransmitted In Bytes are the primary "
                    "drivers — exploit payloads create abnormal packet size distributions.",
    },
    "Fuzzers": {
        "icon":    "🔀",
        "summary": "Fuzzing attack detected (input mutation / protocol abuse)",
        "detail":  (
            "Chaotic timing variance with mixed packet sizes (very large and "
            "very small alternating) on unusual protocols — consistent with "
            "automated fuzzing tools probing for vulnerabilities. "
            "Action: rate-limit the source, review application error logs for crashes."
        ),
        "why_high": "Src→Dst IAT Stddev and packet size variance are the primary "
                    "drivers — fuzzing creates chaotic, non-human traffic patterns.",
    },
    "Generic": {
        "icon":    "⚡",
        "summary": "Generic network attack pattern detected",
        "detail":  (
            "Broad anomalies across volume, timing, and protocol features suggest "
            "a multi-vector or generic network attack. The pattern does not closely "
            "match a single specific attack type but deviates significantly from "
            "normal traffic across multiple feature groups. "
            "Action: escalate to Tier-2 analysis, capture full PCAP for forensics."
        ),
        "why_high": "Spread anomalies across volume and timing features — "
                    "indicates a broad-spectrum attack or novel variant.",
    },
    "Backdoor": {
        "icon":    "🚪",
        "summary": "Backdoor / C2 communication detected",
        "detail":  (
            "Symmetric bidirectional traffic with periodic beaconing intervals "
            "(very consistent IAT with near-zero jitter) over sustained connections. "
            "This is the hallmark of command-and-control communication from an "
            "implanted backdoor. "
            "Action: isolate the host immediately, initiate EDR investigation, "
            "check for lateral movement."
        ),
        "why_high": "Src→Dst IAT Avg/Stddev and symmetric byte ratios are the "
                    "primary drivers — robotic periodic beaconing is never legitimate.",
    },
    "Shellcode": {
        "icon":    "🐚",
        "summary": "Shellcode payload delivery detected",
        "detail":  (
            "Large inbound payload with oversized packets and aggressive TCP push "
            "flags on short-duration flows — consistent with shellcode injection. "
            "The payload size and delivery pattern match known exploit kit behaviour. "
            "Action: quarantine the target, scan for injected code, review memory dumps."
        ),
        "why_high": "Inbound Bytes and Longest Flow Pkt are the primary "
                    "drivers — shellcode payloads create distinctive size signatures.",
    },
    "Worms": {
        "icon":    "🐛",
        "summary": "Network worm propagation detected",
        "detail":  (
            "High bidirectional packet rates with elevated throughput — the pattern "
            "suggests automated self-replication across the network. Both inbound "
            "and outbound traffic spikes indicate the host is both receiving worm "
            "payloads and actively scanning/infecting other hosts. "
            "Action: network-wide containment, identify patient zero, apply patches."
        ),
        "why_high": "Src→Dst Bytes/s and bidirectional packet counts are the "
                    "primary drivers — worm propagation creates symmetric high-rate flows.",
    },
    "Analysis": {
        "icon":    "🔬",
        "summary": "Deep analysis / probing activity detected",
        "detail":  (
            "Extended-duration flows with methodical timing (consistent IAT) and "
            "small packet sizes — consistent with automated service enumeration "
            "or vulnerability scanning tools performing deep analysis. "
            "Action: review scan targets, assess exposure, block the source IP."
        ),
        "why_high": "Flow Duration and Src→Dst IAT Avg are the primary "
                    "drivers — analysis probes are characteristically slow and methodical.",
    },
    "Data Exfiltration": {
        "icon":    "📤",
        "summary": "Data exfiltration (low & slow) detected",
        "detail":  (
            "Extreme asymmetry: large forward (outbound) byte count vs near-zero "
            "backward (inbound) bytes over a sustained, long connection. "
            "Robotic inter-arrival timing (low Stddev) indicates machine-scripted "
            "exfiltration rather than human-driven traffic. "
            "Action: terminate the connection, inspect endpoint for malware, "
            "check DLP logs for data classification hits."
        ),
        "why_high": "Inbound Bytes and Src→Dst Bytes/s ratios are the primary "
                    "drivers — upload-only sustained flows are outside the normal manifold.",
    },
    "Lateral Movement": {
        "icon":    "↔️",
        "summary": "Internal lateral movement / east-west threat detected",
        "detail":  (
            "High timing jitter (Src→Dst IAT Stddev) combined with long idle periods "
            "between bursts is the hallmark of a compromised host performing "
            "internal reconnaissance. The GNN (Layer 2) should confirm abnormal "
            "device-to-device connectivity not seen during training. "
            "Action: isolate the source host, initiate EDR investigation."
        ),
        "why_high": "Src→Dst IAT Stddev and Duration In are the primary drivers — "
                    "the beacon-like sleep-burst pattern is not present in normal flows.",
    },
    "Port Scan": {
        "icon":    "🔍",
        "summary": "Port scanning activity detected",
        "detail":  (
            "Multiple extremely short flows with minimal byte transfer and "
            "high RST + SYN TCP flag counts — the attacker is probing services. "
            "Action: block the scanning IP, alert vulnerability management team."
        ),
        "why_high": "TCP Flags and very short Flow Duration are the primary "
                    "drivers — legitimate flows do not terminate this abruptly.",
    },
    "Web Attack": {
        "icon":    "💉",
        "summary": "Web application attack detected (SQLi / XSS)",
        "detail":  (
            "Elevated PSH flags and large forward payload sizes on short-duration "
            "flows suggest HTTP request manipulation — consistent with SQL injection "
            "or XSS payloads being submitted. "
            "Action: review WAF logs, block the offending IP, audit database "
            "query logs for injection attempts."
        ),
        "why_high": "Client TCP Flags and Server TCP Flags are primary drivers — "
                    "legitimate HTTP traffic does not push this many payloads per flow.",
    },
    "Unknown Anomaly": {
        "icon":    "❓",
        "summary": "Anomalous pattern — no close attack signature match",
        "detail":  (
            "The reconstruction error is elevated but the feature residual pattern "
            "does not closely match any known attack signature. This may indicate "
            "a novel attack variant, misconfigured device, or legitimate but unusual "
            "traffic pattern. "
            "Action: review the top contributing features manually and escalate "
            "to Tier-2 analysis."
        ),
        "why_high": "Spread residuals across multiple unrelated feature groups — "
                    "no single attack taxonomy matches well.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Explanation Function
# ─────────────────────────────────────────────────────────────────────────────

def explain_ae(
    residuals:  np.ndarray,   # [47] mean absolute per-feature residual
    top_k:      int = 5,
    min_score:  float = 0.05, # minimum similarity to claim a match
) -> Dict:
    """
    Given a per-feature reconstruction residual vector, return a structured
    explanation dict for the dashboard.

    Parameters
    ----------
    residuals  : np.ndarray [47] — mean |x - x_hat| per feature
    top_k      : how many top features to surface
    min_score  : cosine similarity threshold below which we say "Unknown"

    Returns
    -------
    dict with keys:
      top_features    : list of (feature_name, residual_value, feature_index)
      group_residuals : dict {group_name: mean_residual}
      inferred_attack : str — matched attack label (or "Unknown Anomaly")
      match_score     : float ∈ [0,1]
      explanation     : dict (icon, summary, detail, why_high)
    """
    residuals = np.array(residuals, dtype=np.float32)
    n_feats   = len(residuals)

    # ── Top-K contributing features ───────────────────────────────────────
    top_indices = np.argsort(residuals)[::-1][:top_k]
    top_features = [
        (FEATURE_NAMES.get(int(i), f"Feature_{i}"), float(residuals[i]), int(i))
        for i in top_indices
    ]

    # ── Group-level residuals ─────────────────────────────────────────────
    group_residuals: Dict[str, float] = {}
    for group_name, indices in FEATURE_GROUPS.items():
        valid = [residuals[i] for i in indices if i < n_feats]
        group_residuals[group_name] = float(np.mean(valid)) if valid else 0.0

    # ── Attack signature matching (cosine similarity) ─────────────────────
    # Build a dense residual vector from the sparse signature
    best_attack = "Unknown Anomaly"
    best_score  = 0.0

    r_norm = np.linalg.norm(residuals)
    if r_norm > 1e-8:
        r_unit = residuals / r_norm

        for attack_name, sig_dict in ATTACK_SIGNATURES.items():
            # Build dense signature vector
            sig_vec = np.zeros(n_feats, dtype=np.float32)
            for feat_idx, weight in sig_dict.items():
                if feat_idx < n_feats:
                    sig_vec[feat_idx] = weight

            sig_norm = np.linalg.norm(sig_vec)
            if sig_norm < 1e-8:
                continue

            sig_unit = sig_vec / sig_norm
            score    = float(np.dot(r_unit, sig_unit))   # cosine similarity

            if score > best_score:
                best_score  = score
                best_attack = attack_name

    if best_score < min_score:
        best_attack = "Unknown Anomaly"

    explanation = ATTACK_EXPLANATIONS.get(best_attack, ATTACK_EXPLANATIONS["Unknown Anomaly"])

    return {
        "top_features":    top_features,
        "group_residuals": group_residuals,
        "inferred_attack": best_attack,
        "match_score":     round(best_score, 3),
        "explanation":     explanation,
    }
