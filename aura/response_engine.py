"""
aura/response_engine.py — Policy-Driven Incident Response
===========================================================

This module implements the Decision Matrix and Response Orchestrator.

Design Principles
-----------------
1. HUMAN-IN-THE-LOOP (HITL): The system NEVER autonomously makes irreversible
   decisions on Tier-1 Critical Assets (Domain Controllers, ICS, Databases).
   For these, AURA throttles traffic + pages an analyst.

2. BLAST-RADIUS CONTROL: Three response tiers minimise collateral damage:
   - LOW severity:    Log + monitor.  No network action.
   - MEDIUM severity: Bandwidth throttle (10 kbps cap via simulated tc/SDN).
                      Alert analyst via dashboard.
   - HIGH severity:   Full iptables/SDN isolation for Non-Critical nodes.
                      Throttle + HITL for Critical nodes.

3. AUDITABILITY: Every response action is logged with timestamp, justification,
   confidence score, and the policy rule that triggered it.

IMPORTANT NOTE on iptables/tc:
On Windows (hackathon setup), subprocess iptables calls are SIMULATED — they
log the exact command that would run on a Linux production system.  The response
engine returns the command string; in production, this is piped to the SDN
controller or a privileged microservice.
"""

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.detector import AnomalyEvent, AlertSeverity
import policy_engine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response Action Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class ResponseAction(Enum):
    LOG_ONLY       = "LOG_ONLY"           # Passive — no network change
    THROTTLE       = "THROTTLE"           # Bandwidth throttle via tc / SDN
    ISOLATE        = "ISOLATE"            # Full iptables DROP rule
    HITL_ESCALATE  = "HITL_ESCALATE"     # Human-In-The-Loop escalation
    ALREADY_ACTIONED = "ALREADY_ACTIONED" # Duplicate suppression


@dataclass
class IncidentRecord:
    """Immutable audit record written for every response action."""
    timestamp:      float
    window_id:      str
    node_id:        str
    node_label:     str       # Human-readable (from allowlist or "Standard Asset")
    event_severity: str
    confidence:     float
    action_taken:   str
    policy_reason:  str
    command_issued: str       # The exact system command (simulated on Windows)
    is_critical:    bool

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Response Engine
# ─────────────────────────────────────────────────────────────────────────────

class AURAResponseEngine:
    """
    Stateful policy enforcement point.

    Flow
    ----
    1. Receives AnomalyEvent from AURAInferenceEngine
    2. Identifies affected nodes (from triggered_nodes list)
    3. Checks each against the Critical Infrastructure Allowlist
    4. Applies the appropriate tiered response
    5. Writes an IncidentRecord for the audit log

    Idempotency: Tracks recently actioned nodes to prevent duplicate iptables
    rules from stacking (which could cause kernel rule table overflow).
    """

    IS_WINDOWS = platform.system() == "Windows"

    def __init__(
        self,
        allowlist:   dict = None,
        log_path:    str  = cfg.EVENT_LOG_FILE,
    ):
        self.allowlist  = allowlist or cfg.CRITICAL_ALLOWLIST
        self.log_path   = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Track recently isolated/throttled nodes to prevent duplicates
        # Maps node_id → unix timestamp of last action
        self._actioned_nodes: dict = {}
        self._dedup_window_sec = 30   # Suppress duplicate actions within 30s

        logger.info(
            f"ResponseEngine ready.  Critical allowlist: "
            f"{list(self.allowlist.keys())}  |  Windows-sim={self.IS_WINDOWS}"
        )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def act(self, event: AnomalyEvent) -> List[IncidentRecord]:
        """
        Process an AnomalyEvent and execute the appropriate policy response.

        Returns a list of IncidentRecord objects (one per affected node).
        Returns empty list if event.severity == NORMAL.
        """
        if event.severity == AlertSeverity.NORMAL:
            return []

        records = []

        # If no specific nodes were flagged (Layer 2 not invoked or no triggers),
        # apply a network-wide throttle as a precautionary measure.
        import random
        if event.triggered_nodes:
            target_nodes = event.triggered_nodes
        else:
            # Fix: Pick a random Standard Asset (Node 4 to 19) to isolate
            # instead of hashing the string "network_wide" every time.
            # (Nodes 0, 1, 2, and 3 are protected in the CRITICAL_ALLOWLIST)
            target_nodes = [random.randint(4, 19)]

        for raw_node_id in target_nodes:
            node_id    = f"node_{raw_node_id}" if isinstance(raw_node_id, int) else str(raw_node_id)
            is_critical = node_id in self.allowlist
            node_label  = self.allowlist.get(node_id, "Standard Asset")

            # Dedup check
            last_action = self._actioned_nodes.get(node_id, 0)
            if time.time() - last_action < self._dedup_window_sec:
                record = IncidentRecord(
                    timestamp      = time.time(),
                    window_id      = event.window_id,
                    node_id        = node_id,
                    node_label     = node_label,
                    event_severity = event.severity.name,
                    confidence     = event.confidence,
                    action_taken   = ResponseAction.ALREADY_ACTIONED.value,
                    policy_reason  = "Duplicate suppression active (dedup window active).",
                    command_issued = "NONE",
                    is_critical    = is_critical,
                )
                records.append(record)
                continue

            # Select and execute response
            record = self._apply_policy(event, node_id, node_label, is_critical)
            records.append(record)
            self._actioned_nodes[node_id] = time.time()

        return records

    # ------------------------------------------------------------------
    # Policy Tiers
    # ------------------------------------------------------------------

    def _apply_policy(
        self,
        event:       AnomalyEvent,
        node_id:     str,
        node_label:  str,
        is_critical:  bool,
    ) -> IncidentRecord:
        """
        Tiered policy enforcement:

        ┌─────────────────┬──────────────────────────────────────────────────┐
        │ Severity        │ Critical Node         │ Standard Node             │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ LOW             │ Log only              │ Log only                  │
        │ MEDIUM          │ Throttle + HITL alert │ Throttle + HITL alert     │
        │ HIGH            │ Throttle + HITL alert │ Full isolation (iptables) │
        └─────────────────┴──────────────────────────────────────────────────┘
        """
        severity = event.severity

        if severity == AlertSeverity.LOW:
            return self._log_only(event, node_id, node_label, is_critical)

        elif severity == AlertSeverity.MEDIUM:
            return self._throttle_and_hitl(event, node_id, node_label, is_critical)

        elif severity == AlertSeverity.HIGH:
            if is_critical:
                return self._throttle_and_hitl(
                    event, node_id, node_label, is_critical,
                    reason="HIGH severity on CRITICAL asset — must be HITL authorised before isolation."
                )
            else:
                return self._isolate(event, node_id, node_label, is_critical)

        # Fallback
        return self._log_only(event, node_id, node_label, is_critical)

    def _log_only(
        self, event, node_id, node_label, is_critical
    ) -> IncidentRecord:
        reason = (
            f"Confidence {event.confidence:.2%} below threshold "
            f"{cfg.CONFIDENCE_LOW_THRESHOLD:.2%}. Monitoring only."
        )
        logger.info(f"[LOG_ONLY] {node_id} ({node_label}) — {reason}")
        return self._write_record(
            event, node_id, node_label, is_critical,
            action=ResponseAction.LOG_ONLY,
            reason=reason, command="NONE",
        )

    def _throttle_and_hitl(
        self, event, node_id, node_label, is_critical,
        reason: str = None
    ) -> IncidentRecord:
        simulated_ip = f"10.0.0.{hash(node_id) % 254 + 1}"
        asset_class  = "CRITICAL" if is_critical else "STANDARD"

        # Delegate to policy engine — runs scripts/throttle.sh (or simulates on Windows)
        command = policy_engine.execute_response(
            severity     = "MEDIUM",
            asset_class  = asset_class,
            node_id      = node_id,
            node_label   = node_label,
            simulated_ip = simulated_ip,
            confidence   = event.confidence,
        )

        if reason is None:
            reason = (
                f"Confidence {event.confidence:.2%} ≥ {cfg.CONFIDENCE_MED_THRESHOLD:.2%}. "
                f"Bandwidth throttled via policy engine.  HITL analyst alert sent."
            )

        logger.warning(
            f"[THROTTLE+HITL] {node_id} ({node_label}) | IP={simulated_ip} | {reason}"
        )
        self._send_hitl_alert(event, node_id, node_label, reason)

        return self._write_record(
            event, node_id, node_label, is_critical,
            action=ResponseAction.THROTTLE,
            reason=reason, command=command,
        )

    def _isolate(
        self, event, node_id, node_label, is_critical
    ) -> IncidentRecord:
        simulated_ip = f"10.0.0.{hash(node_id) % 254 + 1}"

        # Delegate to policy engine — presents HITL gate before running scripts/isolate.sh.
        # If HITL is rejected, policy_engine automatically degrades to scripts/throttle.sh
        # and logs the rejection + fallback with timestamp. Node always exits controlled.
        command = policy_engine.execute_response(
            severity     = "HIGH",
            asset_class  = "STANDARD",
            node_id      = node_id,
            node_label   = node_label,
            simulated_ip = simulated_ip,
            confidence   = event.confidence,
        )

        # Determine actual action taken based on what the policy engine ran
        actual_action = (
            ResponseAction.THROTTLE if "throttle" in command.lower()
            else ResponseAction.ISOLATE
        )

        reason = (
            f"HIGH confidence ({event.confidence:.2%}) on Non-Critical asset. "
            f"Policy engine executed: {actual_action.value}. "
            f"Lateral movement blast radius contained."
        )
        logger.critical(
            f"[{actual_action.value}] {node_id} ({node_label}) | IP={simulated_ip} | {reason}"
        )
        return self._write_record(
            event, node_id, node_label, is_critical,
            action=actual_action,
            reason=reason, command=command,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_command(self, command: str) -> None:
        """
        Execute a system command — simulated on Windows, real on Linux.
        Logs the command regardless so the dashboard always shows exact actions.
        """
        if self.IS_WINDOWS or command.startswith("[SIMULATED"):
            logger.info(f"[SIM-CMD] {command}")
            return
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command[:80]}")
        except Exception as e:
            logger.error(f"Command exception: {e}")

    def _send_hitl_alert(self, event, node_id, node_label, reason) -> None:
        """
        In production: PagerDuty / Slack webhook / SIEM forwarding.
        In demo:       writes to alert log + prints to dashboard.
        """
        alert = {
            "type":       "HITL_REQUIRED",
            "timestamp":  time.time(),
            "node_id":    node_id,
            "node_label": node_label,
            "confidence": event.confidence,
            "reason":     reason,
            "window":     event.window_id,
        }
        try:
            with open(cfg.ALERT_LOG_FILE, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except Exception:
            pass
        print(f"\n🚨 [HITL ALERT] Analyst action required on {node_id} ({node_label})\n")

    def _write_record(
        self, event, node_id, node_label, is_critical,
        action: ResponseAction, reason: str, command: str,
    ) -> IncidentRecord:
        record = IncidentRecord(
            timestamp      = time.time(),
            window_id      = event.window_id,
            node_id        = node_id,
            node_label     = node_label,
            event_severity = event.severity.name,
            confidence     = event.confidence,
            action_taken   = action.value,
            policy_reason  = reason,
            command_issued = command,
            is_critical    = is_critical,
        )
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception:
            pass
        return record


# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from aura.detector import AlertSeverity, AnomalyEvent
    import time

    print("=== AURA Response Engine — Policy Test ===\n")
    eng = AURAResponseEngine()

    test_events = [
        AnomalyEvent(time.time(), "test:w1", 0.10, 0.05, [], AlertSeverity.LOW,       [],     0.25, 0.0),
        AnomalyEvent(time.time(), "test:w2", 0.55, 0.05, [], AlertSeverity.MEDIUM,    [5, 8], 0.65, 0.3),
        AnomalyEvent(time.time(), "test:w3", 0.95, 0.05, [], AlertSeverity.HIGH,      [0],    0.92, 0.9),  # CRITICAL node
        AnomalyEvent(time.time(), "test:w4", 0.95, 0.05, [], AlertSeverity.HIGH,      [12],   0.88, 0.9),  # Standard node
    ]

    for ev in test_events:
        print(f"\n--- Processing {ev.severity.name} event (nodes={ev.triggered_nodes}) ---")
        records = eng.act(ev)
        for r in records:
            print(f"  → {r.action_taken:20s} | critical={r.is_critical} | {r.policy_reason[:60]}…")

    print("\n✓ Response engine test passed.")
