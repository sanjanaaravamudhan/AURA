"""
run_federation_networked.py — AURA Cross-Network FL Launcher
=============================================================

This script launches the AURA Federated Learning pipeline in TRUE networked
mode: the aggregation server and each organisation client run as SEPARATE
OS processes, communicating via Flower's gRPC transport layer.

Architecture
------------

  ┌─────────────────────────────────────────────────────────────────┐
  │  This machine                                                   │
  │                                                                 │
  │  [Process 0 — FL Server]   0.0.0.0:8080                        │
  │      KrumFedAURA strategy                                       │
  │      Byzantine-robust Krum aggregation                          │
  │      SHA-256 → Blockchain mint (final round only)               │
  │                    ▲  ▲  ▲                                      │
  │                    │  │  │  (gRPC over loopback / LAN)          │
  │  [Process 1]──────╯  │  ╰──────[Process 3]                     │
  │  org_hospital_1       │         org_university_3                │
  │  Simulated net:       │         Simulated net:                  │
  │  192.168.1.0/24       │         172.16.1.0/24                   │
  │                 [Process 2]                                     │
  │                 org_bank_2 (Byzantine / adversarial)            │
  │                 Simulated net: 10.0.1.0/24                      │
  └─────────────────────────────────────────────────────────────────┘

Flow
----
1. Server starts, waits for MIN_AVAILABLE_CLIENTS to connect
2. Each client dials the server via gRPC (separate process / could be
   a separate machine — just change --server to the remote IP)
3. Server runs Krum selection each round — org_bank_2 (Byzantine) is
   consistently dropped; hospital + university weights are aggregated
4. Final round: SHA-256(aggregated weights) minted on blockchain once
5. Each client verifies received weights against the on-chain hash

To run on REAL separate machines
---------------------------------
  Machine A (server):   python run_federation_networked.py --server-only
  Machine B (hospital): python -m aura.fl_client --client-id org_hospital_1 --server <A_IP>:8080
  Machine C (bank):     python -m aura.fl_client --client-id org_bank_2 --server <A_IP>:8080 --byzantine
  Machine D (univ):     python -m aura.fl_client --client-id org_university_3 --server <A_IP>:8080

Usage (single machine demo)
----------------------------
  python run_federation_networked.py
  python run_federation_networked.py --server-address 0.0.0.0:8080
  python run_federation_networked.py --server-only          # headless server
  python run_federation_networked.py --rounds 5
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
PYTHON    = sys.executable
SERVER_PY = ROOT / "aura" / "fl_server.py"
CLIENT_PY = ROOT / "aura" / "fl_client.py"

# ── Organisation definitions ──────────────────────────────────────────────────
ORGS = [
    {
        "client_id":    "org_hospital_1",
        "network_sim":  "192.168.1.0/24",
        "byzantine":    False,
        "samples":      500,
        "description":  "Hospital Network — normal traffic",
    },
    {
        "client_id":    "org_bank_2",
        "network_sim":  "10.0.1.0/24",
        "byzantine":    True,
        "samples":      500,
        "description":  "Bank Network — ADVERSARIAL (Byzantine poisoning attempt)",
    },
    {
        "client_id":    "org_university_3",
        "network_sim":  "172.16.1.0/24",
        "byzantine":    False,
        "samples":      500,
        "description":  "University Network — normal traffic",
    },
    {
        "client_id":    "org_isp_4",
        "network_sim":  "10.10.0.0/24",
        "byzantine":    False,
        "samples":      500,
        "description":  "ISP Network — normal traffic",
    },
    {
        "client_id":    "org_retail_5",
        "network_sim":  "172.31.0.0/24",
        "byzantine":    False,
        "samples":      500,
        "description":  "Retail Network — normal traffic",
    },
]


def print_banner(server_address: str) -> None:
    print("\n" + "=" * 62)
    print("  AURA — Cross-Network Federated Learning")
    print("  Mode: TRUE NETWORKED (separate OS processes, gRPC)")
    print("=" * 62)
    print(f"  Server:  {server_address}")
    for org in ORGS:
        tag = " [BYZANTINE]" if org["byzantine"] else ""
        print(f"  Client:  {org['client_id']:25s}  net={org['network_sim']}{tag}")
    print("=" * 62 + "\n")


def start_server_process(server_address: str, rounds: int) -> subprocess.Popen:
    """Launch the gRPC FL server in a subprocess."""
    cmd = [
        PYTHON, str(SERVER_PY),
        "--address", server_address,
        "--rounds",  str(rounds),
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    print(f"[LAUNCHER] Server process PID {proc.pid} started.")
    return proc


def start_client_process(org: dict, server_address: str) -> subprocess.Popen:
    """Launch a single org's FL client in a subprocess."""
    cmd = [
        PYTHON, str(CLIENT_PY),
        "--client-id",   org["client_id"],
        "--server",      server_address,
        "--samples",     str(org["samples"]),
        "--network-sim", org["network_sim"],
    ]
    if org["byzantine"]:
        cmd.append("--byzantine")

    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    print(f"[LAUNCHER] {org['client_id']} process PID {proc.pid} | {org['description']}")
    return proc


def stream_output(proc: subprocess.Popen, label: str) -> None:
    """Read and print all remaining stdout from a finished process."""
    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [{label}] {line}")


def run(server_address: str, rounds: int, server_only: bool) -> None:
    print_banner(server_address)

    # ── 1. Start server ──────────────────────────────────────────────────────
    server_proc = start_server_process(server_address, rounds)

    if server_only:
        print("[LAUNCHER] --server-only mode: waiting for remote clients …")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            server_proc.terminate()
        finally:
            stream_output(server_proc, "SERVER")
        return

    # ── 2. Wait for server to be ready ───────────────────────────────────────
    print("[LAUNCHER] Waiting 3 s for server gRPC socket to open …")
    time.sleep(3)

    # ── 3. Start all org clients (each is a separate process = separate network) ──
    client_procs = []
    for org in ORGS:
        proc = start_client_process(org, server_address)
        client_procs.append((proc, org["client_id"]))
        time.sleep(1.0)   # stagger start so all 3 connect before round 1 begins

    print(f"\n[LAUNCHER] {len(client_procs)} client processes running. "
          f"Waiting for federation to complete …\n")

    # ── 4. Wait for all clients to finish ────────────────────────────────────
    for proc, label in client_procs:
        proc.wait()
        stream_output(proc, label)

    # ── 5. Wait for server to finish and collect output ──────────────────────
    server_proc.wait()
    stream_output(server_proc, "SERVER")

    # ── 6. Chain verification ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Federation complete — running blockchain verification …")
    print("=" * 62)

    verify = subprocess.run(
        [PYTHON, str(ROOT / "verify_chain.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
        cwd=str(ROOT),
    )
    print(verify.stdout)
    if verify.returncode != 0:
        print("[VERIFY ERROR]", verify.stderr)


# ── Server __main__ support (fl_server.py needs CLI args) ────────────────────

def _patch_server_cli():
    """Add CLI argument parsing to fl_server.py's __main__ block."""
    # We launch fl_server.py directly; it needs to accept --address / --rounds.
    # This is handled inside fl_server.py's __main__ block below.
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA Networked FL Launcher")
    parser.add_argument(
        "--server-address", default="localhost:8080",
        help="gRPC server address (default: localhost:8080). "
             "For cross-machine: use the server's LAN IP, e.g. 192.168.0.10:8080"
    )
    parser.add_argument("--rounds",      type=int, default=3)
    parser.add_argument("--server-only", action="store_true",
                        help="Start server and wait for remote clients (no local client procs)")
    args = parser.parse_args()

    run(
        server_address = args.server_address,
        rounds         = args.rounds,
        server_only    = args.server_only,
    )

