"""
aura/models.py — Layer 1 (Autoencoder) + Layer 2 (STGNN)
===========================================================

Model Architecture Summary
---------------------------

Layer 1 — Statistical Tripwire (Unsupervised Autoencoder)
  Input:   x ∈ ℝ^{E × F}  (E edge/flow feature vectors, F=47 features)
  Encoder: Linear(F→32) → ReLU → Linear(32→24) → ReLU → Linear(24→Z)
  Latent:  z ∈ ℝ^{E × Z}  where Z=16  (bottleneck — the "fingerprint")
  Decoder: Linear(Z→24) → ReLU → Linear(24→32) → ReLU → Linear(32→F)
  Loss:    MSE(input, reconstruction)  — spike = anomaly

  Contrastive Negative Sampling:
    If a 'negative' (attack) sample z_neg is provided, we add:
    max(0, margin - ||z_pos - z_neg||₂) to the loss.
    This pushes attack embeddings away from the normal manifold without
    requiring ground-truth labels at inference time.

Layer 2 — Contextual Validator (Spatio-Temporal Graph Neural Network)
  Architecture: GraphSAGE (Hamilton et al., 2017) — inductive variant

  WHY GraphSAGE over GCN?
  ├── GCN is TRANSDUCTIVE: requires all nodes present at training time.
  │   New IP addresses (devices added post-training) would crash inference.
  ├── GraphSAGE is INDUCTIVE: it learns an aggregation FUNCTION over
  │   neighbourhoods, not fixed node embeddings. Any new node with features
  │   can be embedded without retraining.
  └── This is critical for networks where devices are added/removed hourly.

  SAGEConv operation (per layer):
    h_v^{(l+1)} = W_self · h_v^l  +  W_neigh · mean_{u ∈ N(v)}(h_u^l)
    then LayerNorm + ReLU (for training stability)

  Temporal Approximation:
    For the hackathon demo, temporal structure is approximated by processing
    consecutive graph snapshots and maintaining a hidden state buffer.
    A full ST-GNN with LSTM cells is the production upgrade path.

  Output: per-node anomaly score ∈ [0, 1] via sigmoid on a final linear head.
"""

import math
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Sparse Mean Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def sparse_mean_aggregate(
    x:          torch.Tensor,   # [N, F]  node feature matrix
    edge_index: torch.Tensor,   # [2, E]  COO sparse adjacency
) -> torch.Tensor:
    """
    Compute the mean of neighbour features for each node using sparse indexing.

    This is the AGGREGATE step of GraphSAGE:
        m_v = mean_{u ∈ N(v)} h_u

    Implementation uses torch.scatter_add for efficiency — avoids materialising
    a dense adjacency matrix (which would be O(N²) and impractical for large
    graphs).

    Tensor shapes
    -------------
    x           : [N, F]  — current node embeddings
    edge_index  : [2, E]  — row 0 = source nodes, row 1 = destination nodes
    returns     : [N, F]  — mean neighbour embedding for each node
    """
    src, dst = edge_index[0], edge_index[1]   # each is [E]
    N, feat_dim = x.shape

    # Accumulate source features into destination nodes
    # out[dst[i]] += x[src[i]]  for each edge i
    agg = torch.zeros(N, feat_dim, device=x.device, dtype=x.dtype)
    agg.scatter_add_(
        dim=0,
        index=dst.unsqueeze(1).expand(-1, feat_dim),   # [E, F]
        src=x[src],                              # [E, F]
    )

    # Count in-degree per node for mean normalisation
    deg = torch.zeros(N, device=x.device, dtype=x.dtype)
    deg.scatter_add_(
        dim=0,
        index=dst,
        src=torch.ones(len(dst), device=x.device, dtype=x.dtype),
    )
    # Avoid division by zero for isolated nodes
    deg = deg.clamp(min=1.0)

    return agg / deg.unsqueeze(1)   # [N, F]


# ─────────────────────────────────────────────────────────────────────────────
# SAGEConv Layer (Manual Implementation — Inductive)
# ─────────────────────────────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    Single GraphSAGE convolutional layer.

    Forward pass:
        h_v' = LayerNorm( ReLU( W_self·h_v + W_neigh·mean_{N(v)}(h_u) + b ) )

    Parameters
    ----------
    in_dim  : input node feature dimension
    out_dim : output node embedding dimension
    bias    : whether to include a bias term

    Mathematical note on weight matrix sizes
    -----------------------------------------
    W_self  ∈ ℝ^{out_dim × in_dim}
    W_neigh ∈ ℝ^{out_dim × in_dim}
    The two transforms are kept separate (not concatenated like some variants)
    to allow the model to independently weight self-information vs neighbour
    information.  This is the "mean aggregator" variant from the original paper.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

        # Kaiming initialisation: appropriate for ReLU activations
        nn.init.kaiming_uniform_(self.W_self.weight,  a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_neigh.weight, a=math.sqrt(5))

    def forward(
        self,
        x:          torch.Tensor,   # [N, in_dim]
        edge_index: torch.Tensor,   # [2, E]
    ) -> torch.Tensor:              # [N, out_dim]
        """
        x          : current node embeddings  [N, in_dim]
        edge_index : graph connectivity        [2, E]
        """
        # Self transform
        self_term = self.W_self(x)                               # [N, out_dim]

        # Neighbour mean aggregation + transform
        neigh_mean = sparse_mean_aggregate(x, edge_index)       # [N, in_dim]
        neigh_term = self.W_neigh(neigh_mean)                   # [N, out_dim]

        out = self_term + neigh_term
        if self.bias is not None:
            out = out + self.bias

        return self.norm(F.relu(out))                           # [N, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Unsupervised Autoencoder (Statistical Tripwire)
# ─────────────────────────────────────────────────────────────────────────────

class FlowAutoencoder(nn.Module):
    """
    Unsupervised flow-level anomaly detector.

    Architecture
    ------------
    Encoder:  F → 32 → 24 → Z      (compression)
    Latent:   Z = LATENT_DIM = 16
    Decoder:  Z → 24 → 32 → F      (reconstruction)

    Dropout(0.2) after each hidden layer prevents the model from memorising
    specific flow patterns — forcing it to learn the underlying distribution.

    Inference: anomaly_score = MSE(x, reconstruct(x))
    A high score indicates the flow's statistical physics deviate from the
    normal manifold encoded in the weights.
    """

    def __init__(
        self,
        feature_dim:  int = cfg.FEATURE_DIM,
        encoder_dims: list = None,
        latent_dim:   int  = cfg.LATENT_DIM,
        decoder_dims: list = None,
        dropout:      float = 0.2,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = cfg.ENCODER_DIMS
        if decoder_dims is None:
            decoder_dims = cfg.DECODER_DIMS

        # ── Encoder ─────────────────────────────────────────────────────────
        enc_layers = []
        prev = feature_dim
        for dim in encoder_dims:
            enc_layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        enc_layers += [nn.Linear(prev, latent_dim)]  # No activation on bottleneck
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ─────────────────────────────────────────────────────────
        dec_layers = []
        prev = latent_dim
        for dim in decoder_dims:
            dec_layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        dec_layers += [nn.Linear(prev, feature_dim), nn.Sigmoid()]  # Sigmoid: outputs in [0,1]
        self.decoder = nn.Sequential(*dec_layers)

        logger.info(
            f"FlowAutoencoder: {feature_dim}→{encoder_dims}→{latent_dim}"
            f"→{decoder_dims}→{feature_dim}  |  params={self.count_params():,}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress flow features to latent fingerprint.
        x : [B, F]  →  z : [B, Z]
        The latent vector z IS the "attack fingerprint" distributed via Federation.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct flow features from latent representation.
        z : [B, Z]  →  x_hat : [B, F]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        Returns (x_hat, z) — both needed for combined loss computation.
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_loss(
        self,
        x:     torch.Tensor,           # [B, F]  original
        x_hat: torch.Tensor,           # [B, F]  reconstructed
        z:     torch.Tensor,           # [B, Z]  latent
        z_neg: Optional[torch.Tensor] = None,   # [B, Z]  optional attack latent
        margin: float = cfg.CONTRASTIVE_MARGIN,
    ) -> torch.Tensor:
        """
        Combined MSE + Contrastive loss.

        MSE term:
            L_recon = (1/B) Σ ||x - x_hat||²₂

        Contrastive term (Negative Sampling — only used if z_neg provided):
            L_contrast = max(0, margin - ||z - z_neg||₂)

            Goal: push attack latents (z_neg) at least `margin` distance away
            from the normal latent distribution.  This makes the MSE spike on
            attacks even more pronounced, reducing false negatives.
        """
        l_recon = F.mse_loss(x_hat, x)

        if z_neg is not None:
            # L2 distance between positive (normal) and negative (attack) latents
            dist    = torch.norm(z - z_neg, p=2, dim=1)           # [B]
            l_cont  = F.relu(margin - dist).mean()                 # hinge loss
            return l_recon + 0.1 * l_cont                          # weighted sum
        return l_recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample reconstruction error (no grad — inference only).
        Returns : [B]  float tensor of MSE scores per flow.
        """
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            # Per-sample MSE (mean over feature dimension)
            scores = ((x - x_hat) ** 2).mean(dim=1)   # [B]
        return scores

    def explain_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Per-feature mean absolute residual for explainability.

        Returns the mean |x - x_hat| averaged over the batch dimension,
        giving a [F] numpy array where large values indicate which features
        the model found hardest to reconstruct (i.e. most anomalous).

        Used by ae_explainer.explain_ae() to infer attack category and
        generate the human-readable operator explanation panel.
        """
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            # Mean absolute error per feature dimension [F]
            feature_residuals = (x - x_hat).abs().mean(dim=0)   # [F]
        return feature_residuals.cpu().numpy()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Spatio-Temporal GNN (Contextual Validator)
# ─────────────────────────────────────────────────────────────────────────────

class AuraSTGNN(nn.Module):
    """
    Contextual Validator: GraphSAGE-based node anomaly scorer.

    Only invoked when Layer 1 (Autoencoder) flags an anomaly.  This model
    answers the question: "Is this connection TOPOLOGICALLY anomalous?"

    Architecture
    ------------
    2 × SAGEConv layers (message-passing depth = 2-hop neighbourhood)
    → Linear head with sigmoid → per-node anomaly score ∈ [0, 1]

    Why 2-hop?
    ----------
    Most lateral movement patterns are detectable within 2 hops:
    Compromised workstation → internal pivot → critical server.
    Deeper (3+ hop) patterns would require more layers and risk over-smoothing
    (all nodes converging to similar embeddings).

    Inductive Guarantee
    -------------------
    Because SAGEConv learns an aggregation function (not node embeddings),
    new IP-to-node mappings are handled naturally.  A device that appears
    for the first time will have its features aggregated from its neighbours
    without requiring any embedding lookup table.
    """

    def __init__(
        self,
        in_dim:     int = cfg.GNN_INPUT_DIM,
        hidden_dim: int = cfg.GNN_HIDDEN_DIM,
        out_dim:    int = cfg.GNN_OUTPUT_DIM,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_dim,     hidden_dim)
        self.conv2   = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head: maps node embedding → anomaly score
        self.head = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),   # Output ∈ (0, 1) = probability of anomaly
        )

        logger.info(
            f"AuraSTGNN (GraphSAGE-inductive): {in_dim}→{hidden_dim}→{out_dim}"
            f"→1  |  params={self.count_params():,}"
        )

    def forward(
        self,
        x:          torch.Tensor,   # [N, in_dim]  node feature matrix
        edge_index: torch.Tensor,   # [2, E]        graph topology
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through 2 GraphSAGE layers.

        Returns
        -------
        scores : [N]        per-node anomaly probability
        embeds : [N, out_dim]  node embeddings (for federation fingerprinting)
        """
        h = self.conv1(x, edge_index)          # [N, hidden_dim]
        h = self.dropout(h)
        h = self.conv2(h, edge_index)          # [N, out_dim]

        scores = self.head(h).squeeze(-1)      # [N]
        return scores, h

    def topology_anomaly_score(
        self,
        x:          torch.Tensor,
        edge_index:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-only wrapper.  Returns per-node anomaly scores ∈ [0, 1].
        A score near 1.0 indicates the node's neighbourhood pattern is
        inconsistent with trained normal topology.
        """
        with torch.no_grad():
            scores, _ = self.forward(x, edge_index)
        return scores

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Model Bundle (serialised together for Federated Learning)
# ─────────────────────────────────────────────────────────────────────────────

class AURAModelBundle(nn.Module):
    """
    Container for both Layer 1 and Layer 2 models.

    In Federated Learning, the clients and server operate on this bundle as a
    single unit — ensuring that both the statistical tripwire and the topological
    validator are kept in synchrony across all federation participants.

    The Flower FL client will extract parameters from this combined module.
    """

    def __init__(self):
        super().__init__()
        self.autoencoder = FlowAutoencoder()
        self.stgnn       = AuraSTGNN()

    def forward(self, x, edge_index):
        """Not called directly — models are invoked separately in the pipeline."""
        raise NotImplementedError("Use autoencoder and stgnn attributes directly.")

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AURA Models — Sanity Check ===\n")

    N = cfg.NUM_SYNTHETIC_NODES    # number of nodes
    num_feats = cfg.FEATURE_DIM    # 47 features (NF-UNSW-NB15-v3)
    E = 40                         # 40 synthetic edges

    # Synthetic graph
    x          = torch.randn(N, num_feats)
    edge_index = torch.randint(0, N, (2, E))

    # ── Autoencoder ──────────────────────────────────────────────────────────
    ae = FlowAutoencoder()
    x_flows = torch.randn(E, num_feats)     # E flow feature vectors
    x_hat, z = ae(x_flows)
    print(f"Autoencoder  |  input {x_flows.shape} → latent {z.shape} → recon {x_hat.shape}")

    loss = ae.reconstruction_loss(x_flows, x_hat, z)
    print(f"  MSE loss (normal): {loss.item():.6f}")

    z_neg  = torch.randn(E, cfg.LATENT_DIM)  # Simulated attack latents
    loss_c = ae.reconstruction_loss(x_flows, x_hat, z, z_neg=z_neg)
    print(f"  Combined loss (with contrastive): {loss_c.item():.6f}")

    scores = ae.anomaly_score(x_flows)
    print(f"  Anomaly scores: min={scores.min():.4f}  max={scores.max():.4f}")

    # ── STGNN ─────────────────────────────────────────────────────────────────
    gnn = AuraSTGNN()
    node_scores, embeds = gnn(x, edge_index)
    print(f"\nSTGNN (GraphSAGE)  |  x {x.shape} → scores {node_scores.shape}  embeds {embeds.shape}")
    print(f"  Node anomaly scores: min={node_scores.min():.4f}  max={node_scores.max():.4f}")

    # ── Bundle ────────────────────────────────────────────────────────────────
    bundle = AURAModelBundle()
    print(f"\nAURAModelBundle total params: {bundle.total_params():,}")

    print("\n✓ Model tests passed.")
