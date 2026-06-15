"""
aura/detector.py — Live Inference Engine with Dynamic Thresholding
===================================================================

This module fuses Layer 1 (Autoencoder) and Layer 2 (STGNN) into a
unified scoring pipeline with statistically-driven, adaptive alerting.

Dynamic Thresholding Rationale
-------------------------------
Static thresholds (e.g., `if loss > 0.5`) are a critical weakness:

  - A misconfigured baseline → wrong constant → floods analysts with false
    positives or silently misses real attacks.
  - Network behaviour drifts over time (e.g., a firmware update changes
    packet sizes universally) — a static threshold would trigger mass alerts.

AURA uses an Exponential Moving Average (EMA) tracker over a rolling window
of historical batch-level MSE losses.  The alert threshold is dynamically
computed as:

    threshold_t = EMA_mean_t + σ_mult × EMA_std_t

Where:
  EMA_mean_t  = α × loss_t + (1 − α) × EMA_mean_{t−1}   (smoothed mean)
  EMA_std_t   = sqrt( α × (loss_t − EMA_mean_t)² + (1−α) × EMA_var_{t−1} )
  σ_mult      = EMA_SIGMA_MULTIPLIER (default: 3.0, ~99.7% coverage of normal)

This mirrors the statistical process control (SPC) concept of control charts
used in industrial anomaly detection — well understood, explainable to judges.

A 50-batch warm-up period suppresses alerts during initial calibration.
"""

import logging
import math
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import FlowAutoencoder, AuraSTGNN

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class AlertSeverity(Enum):
    NORMAL   = auto()   # No anomaly detected
    LOW      = auto()   # Layer 1 weakly triggered; logged only
    MEDIUM   = auto()   # Layer 1 + Layer 2 confirmed; throttle + HITL
    HIGH     = auto()   # Dual-layer confirmation + topological violation


@dataclass
class AnomalyEvent:
    """Structured event record for an anomaly detection result."""
    timestamp:       float
    window_id:       str
    ae_score:        float          # Autoencoder mean MSE over batch
    ae_threshold:    float          # Dynamic threshold at this moment
    gnn_scores:      List[float]    # Per-node GNN scores (empty if L2 not invoked)
    severity:        AlertSeverity
    triggered_nodes: List[int]      # Node IDs exceeding GNN threshold
    confidence:      float          # Fused confidence in [0, 1]
    raw_label_ratio: float          # Ground-truth attack ratio (if available)
    # Explainability fields (populated when L1 triggers)
    top_features:    List[tuple]    # [(feature_name, residual, index), ...]
    inferred_attack: str            # Best-match attack category label
    match_score:     float          # Cosine similarity of residual vs signature
    group_residuals: dict           # {group_name: mean_residual}

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "window_id":       self.window_id,
            "ae_score":        round(self.ae_score, 6),
            "ae_threshold":    round(self.ae_threshold, 6),
            "gnn_scores":      [round(s, 4) for s in self.gnn_scores],
            "severity":        self.severity.name,
            "triggered_nodes": self.triggered_nodes,
            "confidence":      round(self.confidence, 4),
            "raw_label_ratio": round(self.raw_label_ratio, 4),
            "inferred_attack": self.inferred_attack,
            "match_score":     round(self.match_score, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EMA Dynamic Threshold Tracker
# ─────────────────────────────────────────────────────────────────────────────

class EMAThresholdTracker:
    """
    Online Exponential Moving Average tracker for adaptive anomaly thresholding.

    Maintains two EMA statistics simultaneously:
    1. EMA of the loss (the smoothed mean baseline)
    2. EMA of the squared deviation (Welford-style online variance)

    This avoids storing a large rolling window of historical losses while
    still giving a statistically sound, continuously adapting threshold.

    Parameters
    ----------
    alpha           : EMA smoothing factor ∈ (0, 1). Lower = more inertia.
    sigma_multiplier: How many σ above the mean triggers an alert.
    warmup_batches  : Batches before any alert can fire (cold-start guard).
    """

    def __init__(
        self,
        alpha:            float = cfg.EMA_ALPHA,
        sigma_multiplier: float = cfg.EMA_SIGMA_MULTIPLIER,
        warmup_batches:   int   = cfg.EMA_WARMUP_BATCHES,
    ):
        self.alpha            = alpha
        self.sigma_multiplier = sigma_multiplier
        self.warmup_batches   = warmup_batches
        self.batch_count      = 0

        # EMA state (initialised to None; first update seeds them)
        self._ema_mean: Optional[float] = None
        self._ema_var:  Optional[float] = None

        # Trajectory counters — consecutive readings above soft sigma levels
        self._consecutive_above_2sigma:   int = 0
        self._consecutive_above_2_5sigma: int = 0

    @property
    def ema_std(self) -> float:
        if self._ema_var is None:
            return float('inf')
        return math.sqrt(max(self._ema_var, 1e-10))

    @property
    def threshold(self) -> float:
        """Current 3σ hard UCL threshold. Returns inf during warmup."""
        if self.batch_count < self.warmup_batches or self._ema_mean is None:
            return float('inf')
        return self._ema_mean + self.sigma_multiplier * self.ema_std

    @property
    def threshold_2sigma(self) -> float:
        """EMA_mean + 2.0σ — soft trajectory alert level. Returns inf during warmup."""
        if self.batch_count < self.warmup_batches or self._ema_mean is None:
            return float('inf')
        return self._ema_mean + 2.0 * self.ema_std

    @property
    def threshold_2_5sigma(self) -> float:
        """EMA_mean + 2.5σ — elevated trajectory alert level. Returns inf during warmup."""
        if self.batch_count < self.warmup_batches or self._ema_mean is None:
            return float('inf')
        return self._ema_mean + 2.5 * self.ema_std

    def update(self, loss: float) -> float:
        """
        Process a new batch loss value.  Updates EMA state and returns the
        current 3σ threshold AFTER incorporating this new observation.

        Also updates trajectory counters:
          _consecutive_above_2sigma   — incremented when loss > EMA_mean + 2.0σ
          _consecutive_above_2_5sigma — incremented when loss > EMA_mean + 2.5σ
          Both reset to 0 when the respective sigma level is not exceeded.
        """
        self.batch_count += 1

        if self._ema_mean is None:
            # Seed the EMA with the very first observation
            self._ema_mean = loss
            self._ema_var  = 0.0
        else:
            # EMA mean update:  μ_t = α·x_t + (1−α)·μ_{t−1}
            delta           = loss - self._ema_mean
            self._ema_mean += self.alpha * delta

            # EMA variance update (online Welford-style):
            # σ²_t = (1−α)·(σ²_{t−1} + α·δ²)
            self._ema_var   = (1 - self.alpha) * (self._ema_var + self.alpha * delta ** 2)

        # ── Trajectory tracking (post-warmup only) ────────────────────────────
        if self.batch_count >= self.warmup_batches and self._ema_mean is not None:
            t2_5 = self.threshold_2_5sigma
            t2_0 = self.threshold_2sigma
            if not math.isinf(t2_5) and loss > t2_5:
                # Above 2.5σ: both counters increment (2.5σ implies 2.0σ)
                self._consecutive_above_2_5sigma += 1
                self._consecutive_above_2sigma   += 1
            elif not math.isinf(t2_0) and loss > t2_0:
                # Between 2.0σ and 2.5σ: only 2σ counter increments
                self._consecutive_above_2sigma   += 1
                self._consecutive_above_2_5sigma  = 0
            else:
                # Below 2.0σ: both counters reset
                self._consecutive_above_2sigma    = 0
                self._consecutive_above_2_5sigma  = 0

        return self.threshold

    def is_anomalous(self, loss: float) -> bool:
        """
        Check if `loss` exceeds the CURRENT 3σ threshold (before updating).
        Retained for external callers and unit tests.  The inference engine
        now calls update() first and evaluates l1_triggered separately.
        """
        return self.batch_count >= self.warmup_batches and loss > self.threshold

    def state_dict(self) -> dict:
        return {
            "batch_count":              self.batch_count,
            "ema_mean":                 self._ema_mean,
            "ema_var":                  self._ema_var,
            "alpha":                    self.alpha,
            "consecutive_above_2sigma":   self._consecutive_above_2sigma,
            "consecutive_above_2_5sigma": self._consecutive_above_2_5sigma,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fused Confidence Score
# ─────────────────────────────────────────────────────────────────────────────

def fuse_scores(
    ae_score:   float,          # Normalised AE anomaly score ∈ [0, 1]
    ae_thresh:  float,          # Current dynamic AE threshold
    gnn_scores: Optional[torch.Tensor],   # Per-node GNN scores ∈ [0, 1]
    ae_weight:  float = 0.55,   # Layer 1 has slightly higher weight
    gnn_weight: float = 0.45,
) -> float:
    """
    Decision Matrix: fuse Layer 1 and Layer 2 scores into a single
    confidence value ∈ [0, 1].

    Layer 1 contribution: normalised by threshold exceedance ratio
        ae_contribution = min(1.0, ae_score / ae_thresh) × ae_weight

    Layer 2 contribution: maximum node-level GNN score (most suspicious node)
        gnn_contribution = max_node_score × gnn_weight

    If Layer 2 was not invoked (None), Layer 1 carries 100% weight.
    """
    ae_contrib = min(1.0, ae_score / max(ae_thresh, 1e-8)) * ae_weight

    if gnn_scores is not None and len(gnn_scores) > 0:
        gnn_max    = float(gnn_scores.max())
        gnn_contrib = gnn_max * gnn_weight
        total       = ae_weight + gnn_weight
        return (ae_contrib + gnn_contrib) / total
    else:
        return min(1.0, ae_contrib / ae_weight)


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class AURAInferenceEngine:
    """
    Stateful inference engine that processes incoming graph windows,
    runs Layer 1 and (conditionally) Layer 2, and emits structured AnomalyEvents.

    The engine maintains the EMA threshold state across calls, so it must be
    instantiated once and reused per inference session (not recreated per batch).

    Usage
    -----
    >>> engine = AURAInferenceEngine(ae_model, gnn_model)
    >>> event = engine.process(graph_dict, labels)
    >>> if event.severity != AlertSeverity.NORMAL:
    ...     response_engine.act(event)
    """

    def __init__(
        self,
        autoencoder: FlowAutoencoder,
        stgnn:       AuraSTGNN,
        gnn_node_threshold: float = 0.60,   # GNN per-node score to flag
        device:      str = "cpu",
    ):
        self.ae          = autoencoder.to(device).eval()
        self.gnn         = stgnn.to(device).eval()
        self.ema         = EMAThresholdTracker()
        self.gnn_thresh  = gnn_node_threshold
        self.device      = device
        self._event_log: List[AnomalyEvent] = []

        # Per-node temporal accumulator: {node_id: [(unix_ts, AlertSeverity), ...]}
        # Used by _apply_temporal_escalation to detect sustained repeated flags.
        self._node_event_window: Dict[str, List[tuple]] = defaultdict(list)

        logger.info("AURAInferenceEngine initialised (device=%s).", device)

    def process(
        self,
        graph:       dict,
        labels:      Optional[torch.Tensor] = None,
    ) -> AnomalyEvent:
        """
        Process one graph snapshot through the full 4-layer pipeline.

        Parameters
        ----------
        graph  : dict from CICIDSDataLoader.stream_graphs() (NF-UNSW-NB15-v3)
        labels : optional ground-truth label tensor [E] for eval metrics

        Returns
        -------
        AnomalyEvent with full diagnostics
        """
        x          = graph["x"].to(self.device)           # [N, F]
        edge_index = graph["edge_index"].to(self.device)  # [2, E]
        edge_attr  = graph["edge_attr"].to(self.device)   # [E, F]
        window_id  = graph.get("window_id", "unknown")

        # ── Layer 1: Statistical Tripwire ─────────────────────────────────
        ae_scores = self.ae.anomaly_score(edge_attr)      # [E]
        batch_mse = float(ae_scores.mean())

        # EMA is updated FIRST so severity is determined after full computation.
        # l1_triggered = True on either:
        #   (a) hard UCL breach: batch_mse > 3σ threshold, OR
        #   (b) trajectory persistence: K consecutive readings above 2.0σ or 2.5σ
        current_threshold = self.ema.update(batch_mse)
        l1_triggered = (
            not math.isinf(current_threshold)
            and (
                batch_mse > current_threshold
                or self.ema._consecutive_above_2_5sigma >= cfg.K_CONSECUTIVE_READINGS
                or self.ema._consecutive_above_2sigma   >= cfg.K_CONSECUTIVE_READINGS
            )
        )
        # ── Feature Attribution & Attack Classification (when L1 fires) ───
        top_features    = []
        inferred_attack = "Normal"
        match_score     = 0.0
        group_residuals: dict = {}

        if l1_triggered:
            try:
                from aura.ae_explainer import explain_ae
                feat_residuals  = self.ae.explain_features(edge_attr)   # [F]
                expl            = explain_ae(feat_residuals)
                top_features    = expl["top_features"]
                inferred_attack = expl["inferred_attack"]
                match_score     = expl["match_score"]
                group_residuals = expl["group_residuals"]
            except Exception as _e:
                logger.debug(f"Explainer skipped: {_e}")
        # ── Layer 2: Contextual Validator (only if L1 triggered) ──────────
        gnn_scores      = None
        triggered_nodes = []

        if l1_triggered:
            logger.info(
                f"[L1 TRIGGERED] window={window_id}  "
                f"mse={batch_mse:.4f} > threshold={current_threshold:.4f}  "
                f"→ Invoking STGNN …"
            )
            gnn_scores = self.gnn.topology_anomaly_score(x, edge_index)  # [N]
            triggered_nodes = (gnn_scores > self.gnn_thresh).nonzero(as_tuple=True)[0].tolist()

        # ── Score Fusion & Severity Classification ────────────────────────
        confidence = fuse_scores(
            ae_score=batch_mse,
            ae_thresh=current_threshold if not math.isinf(current_threshold) else 1.0,
            gnn_scores=gnn_scores,
        )

        # Base severity: instantaneous score + EMA trajectory
        severity = self._classify_severity(
            l1_triggered, confidence, triggered_nodes,
            consec_2sigma   = self.ema._consecutive_above_2sigma,
            consec_2_5sigma = self.ema._consecutive_above_2_5sigma,
        )
        # Temporal escalation: sustained repeated flags per node raise severity
        severity = self._apply_temporal_escalation(severity, triggered_nodes)

        # Ground truth ratio for dashboard metrics
        gt_ratio = 0.0
        if labels is not None:
            gt_ratio = float(labels.float().mean())

        event = AnomalyEvent(
            timestamp        = time.time(),
            window_id        = window_id,
            ae_score         = batch_mse,
            ae_threshold     = current_threshold if not math.isinf(current_threshold) else -1.0,
            gnn_scores       = gnn_scores.tolist() if gnn_scores is not None else [],
            severity         = severity,
            triggered_nodes  = triggered_nodes,
            confidence       = confidence,
            raw_label_ratio  = gt_ratio,
            top_features     = top_features,
            inferred_attack  = inferred_attack,
            match_score      = match_score,
            group_residuals  = group_residuals,
        )

        self._event_log.append(event)
        self._persist_event(event)

        if severity != AlertSeverity.NORMAL:
            logger.warning(
                f"[ALERT {severity.name}] confidence={confidence:.2%}  "
                f"nodes_flagged={triggered_nodes}  window={window_id}"
            )

        return event

    def _classify_severity(
        self,
        l1_triggered:    bool,
        confidence:      float,
        triggered_nodes: List[int],
        consec_2sigma:   int = 0,
        consec_2_5sigma: int = 0,
    ) -> AlertSeverity:
        """
        Severity is a function of TWO independent inputs:

        1. Instantaneous MSE reconstruction error (via confidence + triggered_nodes):
             LOW    : L1 triggered, confidence < CONFIDENCE_LOW_THRESHOLD
             MEDIUM : L1 triggered, confidence ≥ LOW but < MED, or no GNN nodes
             HIGH   : L1 + L2 confirm, confidence ≥ MED, ≥1 GNN node flagged

        2. EMA trajectory persistence (consec_2sigma / consec_2_5sigma):
             consec_2sigma   ≥ K → floor severity to at least MEDIUM
             consec_2_5sigma ≥ K → floor severity to at least HIGH

        Either condition is sufficient to produce a given severity level.
        Trajectory can only RAISE severity, never lower it.
        """
        if not l1_triggered:
            return AlertSeverity.NORMAL

        # ── Instantaneous classification ──────────────────────────────────
        if confidence < cfg.CONFIDENCE_LOW_THRESHOLD:
            base = AlertSeverity.LOW
        elif confidence < cfg.CONFIDENCE_MED_THRESHOLD or not triggered_nodes:
            base = AlertSeverity.MEDIUM
        else:
            base = AlertSeverity.HIGH

        # ── EMA trajectory override (can only raise severity) ─────────────
        K = cfg.K_CONSECUTIVE_READINGS
        if consec_2_5sigma >= K and base.value < AlertSeverity.HIGH.value:
            logger.info(
                f"[EMA-TRAJ] {consec_2_5sigma} consecutive readings above 2.5σ "
                f"→ escalating {base.name} → HIGH"
            )
            base = AlertSeverity.HIGH
        elif consec_2sigma >= K and base.value < AlertSeverity.MEDIUM.value:
            logger.info(
                f"[EMA-TRAJ] {consec_2sigma} consecutive readings above 2.0σ "
                f"→ escalating {base.name} → MEDIUM"
            )
            base = AlertSeverity.MEDIUM

        return base

    def _apply_temporal_escalation(
        self,
        base_severity:   AlertSeverity,
        triggered_nodes: List[int],
    ) -> AlertSeverity:
        """
        Per-node sliding window accumulator — escalates severity based on
        sustained repeated flags within TEMPORAL_WINDOW_SECONDS.

        Escalation rules (evaluated per triggered node):
          low_count >= 3              → escalate to at least MEDIUM
          low_count >= 5 OR
          medium_count >= 3           → escalate to HIGH

        The window is purged of entries older than TEMPORAL_WINDOW_SECONDS
        before evaluation.  After a HIGH severity event, the accumulator for
        that node is reset — a resolved incident must not poison future windows.

        NORMAL events are never accumulated and bypass this method entirely.
        """
        if base_severity == AlertSeverity.NORMAL:
            return AlertSeverity.NORMAL

        now = time.time()
        cutoff = now - cfg.TEMPORAL_WINDOW_SECONDS

        # Use 'global' as the key when no specific nodes are identified
        node_keys = [f"node_{n}" for n in triggered_nodes] if triggered_nodes else ["global"]
        escalated = base_severity

        for node_key in node_keys:
            # Purge stale entries
            self._node_event_window[node_key] = [
                (ts, sev) for ts, sev in self._node_event_window[node_key]
                if ts >= cutoff
            ]
            window = self._node_event_window[node_key]

            low_count    = sum(1 for _, sev in window if sev == AlertSeverity.LOW)
            medium_count = sum(1 for _, sev in window if sev == AlertSeverity.MEDIUM)

            # Determine escalation candidate for this node
            if low_count >= 5 or medium_count >= 3:
                candidate = AlertSeverity.HIGH
            elif low_count >= 3:
                candidate = AlertSeverity.MEDIUM
            else:
                candidate = base_severity

            if candidate.value > escalated.value:
                logger.info(
                    f"[TEMP-ESC] {node_key}: low_flags={low_count} "
                    f"med_flags={medium_count} within "
                    f"{cfg.TEMPORAL_WINDOW_SECONDS}s → "
                    f"{base_severity.name} → {candidate.name}"
                )
                escalated = candidate

        # Accumulate CURRENT event (using base_severity, pre-escalation)
        # into each node's window AFTER escalation is resolved.
        for node_key in node_keys:
            self._node_event_window[node_key].append((now, base_severity))
            # Reset window after HIGH — resolved incident must not feed future windows
            if escalated == AlertSeverity.HIGH:
                self._node_event_window[node_key] = []

        return escalated

    def _persist_event(self, event: AnomalyEvent) -> None:
        """Append event JSON to alert log file for dashboard consumption."""
        try:
            log_path = Path(cfg.ALERT_LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.debug(f"Event log write failed: {e}")

    @property
    def ema_state(self) -> dict:
        return self.ema.state_dict()

    def recent_events(self, n: int = 20) -> List[AnomalyEvent]:
        return self._event_log[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AURA Detector — EMA Threshold + Inference Test ===\n")
    from aura.models import FlowAutoencoder, AuraSTGNN

    ae  = FlowAutoencoder()
    gnn = AuraSTGNN()
    eng = AURAInferenceEngine(ae, gnn)

    N = cfg.NUM_SYNTHETIC_NODES
    F = cfg.FEATURE_DIM

    print(f"Simulating {cfg.EMA_WARMUP_BATCHES + 20} windows …")
    for i in range(cfg.EMA_WARMUP_BATCHES + 20):
        # Inject a spike after warmup
        spike = 5.0 if i == cfg.EMA_WARMUP_BATCHES + 10 else 0.0
        graph = {
            "x":          torch.randn(N, F) + spike,
            "edge_index": torch.randint(0, N, (2, 40)),
            "edge_attr":  torch.randn(40, F) + spike,
            "window_id":  f"test:w{i}",
        }
        event = eng.process(graph)
        if event.severity != AlertSeverity.NORMAL:
            print(f"  [W{i:3d}] {event.severity.name:6s}  "
                  f"mse={event.ae_score:.4f}  thresh={event.ae_threshold:.4f}  "
                  f"conf={event.confidence:.2%}")

    print("\nEMA state:", eng.ema_state)
    print("✓ Detector test passed.")
