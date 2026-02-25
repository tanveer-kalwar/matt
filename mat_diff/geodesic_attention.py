"""
Geodesic Attention mechanism for tabular diffusion.

Feature-group attention with Mahalanobis distance kernel:
    Splits d_model into n_heads groups, applies attention across groups
    using a learnable Mahalanobis metric initialized from FIM.

    This captures inter-feature dependencies weighted by statistical
    geometry (Fisher Information), unlike cross-sample attention which
    computes meaningless distances between random batch samples.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeodesicAttentionBlock(nn.Module):
    """Feature-group attention with geodesic (Mahalanobis) distance kernel.

    Instead of B×B cross-sample attention (which is meaningless for
    random mini-batches), we reshape (B, d_model) → (B, n_heads, d_head)
    and apply attention across the n_heads dimension for each sample.
    This captures feature-group interactions weighted by the data geometry.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        init_fim: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        # Learnable metric per head (Cholesky factor for PD guarantee)
        self.L = nn.Parameter(torch.zeros(n_heads, self.d_head, self.d_head))
        if init_fim is not None:
            self._init_from_fim(init_fim)
        else:
            for h in range(n_heads):
                nn.init.eye_(self.L.data[h])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _init_from_fim(self, fim: torch.Tensor):
        """Initialize metric from FIM with data-driven regularization."""
        fim = fim.float()

        for h in range(self.n_heads):
            start = h * self.d_head
            end = start + self.d_head
            fim_h = fim[start:end, start:end]

            eigvals = torch.linalg.eigvalsh(fim_h)
            min_eig = eigvals.min().item()
            if min_eig <= 0:
                reg = abs(min_eig) + 1e-10
                fim_h = fim_h + reg * torch.eye(self.d_head, device=fim.device)

            try:
                L_h = torch.linalg.cholesky(fim_h)
            except RuntimeError:
                L_h = torch.eye(self.d_head, device=fim.device)
            self.L.data[h] = L_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature-group Geodesic Attention.

        Input: (B, d_model)
        Reshape to (B, n_heads, d_head)
        Attention across n_heads (feature groups) for each sample.
        """
        B, D = x.shape
        residual = x

        # Project
        Q = self.W_q(x).view(B, self.n_heads, self.d_head)  # (B, H, d_head)
        K = self.W_k(x).view(B, self.n_heads, self.d_head)
        V = self.W_v(x).view(B, self.n_heads, self.d_head)

        # Compute Mahalanobis distance between feature groups
        # For each sample, compute H×H attention matrix
        # Q_i, K_j are feature groups; distance uses metric L[i] (query's metric)
        outputs = []
        for b_start in range(0, B, 64):
            b_end = min(b_start + 64, B)
            Q_chunk = Q[b_start:b_end]  # (chunk, H, d_head)
            K_chunk = K[b_start:b_end]
            V_chunk = V[b_start:b_end]
            chunk_size = b_end - b_start

            # Compute pairwise distances between all head pairs
            # Using mean metric across heads for stable cross-group comparison
            L_mean = self.L.mean(dim=0)  # (d_head, d_head)

            # Transform Q and K through the metric
            Q_t = torch.matmul(Q_chunk, L_mean)  # (chunk, H, d_head)
            K_t = torch.matmul(K_chunk, L_mean)  # (chunk, H, d_head)

            # Standard dot-product attention on transformed features
            # (chunk, H, d_head) × (chunk, d_head, H) → (chunk, H, H)
            attn = torch.bmm(Q_t, K_t.transpose(1, 2)) / math.sqrt(self.d_head)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Apply attention: (chunk, H, H) × (chunk, H, d_head) → (chunk, H, d_head)
            out = torch.bmm(attn, V_chunk)
            outputs.append(out.reshape(chunk_size, D))

        out = torch.cat(outputs, dim=0)
        out = self.W_out(out)
        return self.layer_norm(out + residual)
