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
        """Cross-sample Geodesic Attention with Mahalanobis kernel.
        
        Memory-efficient implementation with chunking for large batches.
        """
        B, D = x.shape
        residual = x

        # Project to Q, K, V
        Q = self.W_q(x).view(B, self.n_heads, self.d_head)
        K = self.W_k(x).view(B, self.n_heads, self.d_head)
        V = self.W_v(x).view(B, self.n_heads, self.d_head)

        # Transpose for multi-head processing
        Q = Q.transpose(0, 1)  # (n_heads, B, d_head)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # For small batches, use full attention
        if B <= 128:
            outputs = []
            for h in range(self.n_heads):
                Q_h = Q[h]  # (B, d_head)
                K_h = K[h]
                V_h = V[h]
                
                # Mahalanobis distance: (B, B)
                diff = Q_h.unsqueeze(1) - K_h.unsqueeze(0)  # (B, B, d_head)
                diff_t = torch.matmul(diff, self.L[h])
                sq_dist = (diff_t ** 2).sum(dim=-1)
                
                # Attention
                scale = math.sqrt(self.d_head)
                attn = F.softmax(-sq_dist / scale, dim=1)
                attn = self.dropout(attn)
                
                out_h = torch.matmul(attn, V_h)
                outputs.append(out_h)
        
        # For large batches, use chunked attention to save memory
        else:
            chunk_size = 64
            outputs = []
            
            for h in range(self.n_heads):
                Q_h = Q[h]
                K_h = K[h]
                V_h = V[h]
                
                out_chunks = []
                for i in range(0, B, chunk_size):
                    Q_chunk = Q_h[i:i+chunk_size]
                    
                    # Compute attention for this chunk
                    diff = Q_chunk.unsqueeze(1) - K_h.unsqueeze(0)
                    diff_t = torch.matmul(diff, self.L[h])
                    sq_dist = (diff_t ** 2).sum(dim=-1)
                    
                    attn = F.softmax(-sq_dist / math.sqrt(self.d_head), dim=1)
                    attn = self.dropout(attn)
                    
                    out_chunk = torch.matmul(attn, V_h)
                    out_chunks.append(out_chunk)
                
                outputs.append(torch.cat(out_chunks, dim=0))
        
        # Concatenate heads and project
        out = torch.cat(outputs, dim=1)  # (B, d_model)
        out = self.W_out(out)
        return self.layer_norm(out + residual)

