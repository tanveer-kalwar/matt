"""
Geodesic Attention mechanism for tabular diffusion.

Replaces dot-product attention with Mahalanobis distance kernel:
    d_M(q, k) = (q - k)^T L L^T (q - k)
    attention = softmax(-d_M / sqrt(d_head))

The metric tensor M = L L^T is parameterized via Cholesky factor L
to guarantee positive definiteness. Initialized from FIM when available.

FIM initialization regularization:
    When initializing from FIM, we add the minimum eigenvalue as
    regularization: reg = max(|lambda_min(FIM_h)|, 0) + 1e-10
    This is the smallest perturbation that ensures PD (not arbitrary).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeodesicAttentionBlock(nn.Module):
    """Multi-head attention with geodesic (Mahalanobis) distance kernel."""

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

        self.L = nn.Parameter(torch.zeros(n_heads, self.d_head, self.d_head))
        if init_fim is not None:
            self._init_from_fim(init_fim)
        else:
            for h in range(n_heads):
                nn.init.eye_(self.L.data[h])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _init_from_fim(self, fim: torch.Tensor):
        """Initialize metric from FIM with data-driven regularization.

        Regularization = |min_eigenvalue| + 1e-10 (machine precision floor).
        This is the minimum perturbation to ensure positive definiteness.
        """
        fim = fim.float()

        for h in range(self.n_heads):
            start = h * self.d_head
            end = start + self.d_head
            fim_h = fim[start:end, start:end]

            # Data-driven regularization: smallest perturbation for PD
            eigvals = torch.linalg.eigvalsh(fim_h)
            min_eig = eigvals.min().item()
            if min_eig <= 0:
                reg = abs(min_eig) + 1e-10  # 1e-10 = machine precision floor
                fim_h = fim_h + reg * torch.eye(self.d_head, device=fim.device)

            try:
                L_h = torch.linalg.cholesky(fim_h)
            except RuntimeError:
                L_h = torch.eye(self.d_head, device=fim.device)
            self.L.data[h] = L_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample attention: each sample attends to its own features.
        
        For tabular diffusion, samples are independently denoised.
        Instead of B×B pairwise attention, we use per-sample self-attention
        where each sample's n_heads attend to each other within that sample.
        
        Memory: O(B * n_heads^2) instead of O(B^2 * n_heads * d_head)
        """
        B = x.shape[0]
        residual = x

        # Project to Q, K, V: (B, n_heads, d_head)
        Q = self.W_q(x).view(B, self.n_heads, self.d_head)
        K = self.W_k(x).view(B, self.n_heads, self.d_head)
        V = self.W_v(x).view(B, self.n_heads, self.d_head)

        # Compute per-sample attention: heads attend to each other within each sample
        # Q: (B, n_heads, d_head) -> (B, n_heads, 1, d_head) for broadcasting
        # K: (B, n_heads, d_head) -> (B, 1, n_heads, d_head) for broadcasting
        Q_exp = Q.unsqueeze(2)  # (B, n_heads, 1, d_head)
        K_exp = K.unsqueeze(1)  # (B, 1, n_heads, d_head)
        
        # diff: (B, n_heads, n_heads, d_head) - O(B * n_heads^2 * d_head)
        diff = Q_exp - K_exp
        
        # Apply Mahalanobis metric via Cholesky factor L
        # L: (n_heads, d_head, d_head)
        # diff: (B, query_head, key_head, d_head)
        # We need to apply L based on the key_head dimension
        diff_transformed = torch.einsum('bqkd,kdf->bqkf', diff, self.L)
        
        # Squared distance: (B, n_heads, n_heads)
        sq_dist = (diff_transformed ** 2).sum(dim=-1)
        
        # Attention scores: softmax over key heads
        scale = math.sqrt(self.d_head)
        attn_logits = -sq_dist / scale
        attn = F.softmax(attn_logits, dim=2)  # softmax over key dimension
        attn = self.dropout(attn)
        
        # Weighted sum of values: (B, n_heads, n_heads) @ (B, n_heads, d_head)
        # attn: (B, query_head, key_head), V: (B, key_head, d_head)
        out = torch.einsum('bqk,bkd->bqd', attn, V)  # (B, n_heads, d_head)
        
        # Reshape and project
        out = out.reshape(B, self.d_model)
        out = self.W_out(out)
        out = self.layer_norm(out + residual)
        return out
