"""
Feature-Cross Attention for Tabular Diffusion.

Based on FT-Transformer (Gorishniy et al., NeurIPS 2021):
    Treats each feature as a token and applies cross-attention.
    This naturally captures feature interactions without requiring
    FIM projection or dimension matching.

For ablation: StandardAttentionBlock uses simple MLP (no cross-feature attention).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StandardAttentionBlock(nn.Module):
    """MLP block WITHOUT cross-feature attention (ablation baseline).
    
    This is equivalent to processing features independently, 
    missing inter-feature dependencies.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 init_fim: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        # Simple MLP instead of attention
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model)
        return self.layer_norm(x + self.mlp(x))


class GeodesicAttentionBlock(nn.Module):
    """Feature-Cross Attention Block.
    
    Reshapes input (B, d_model) -> (B, n_heads, d_head) and applies
    attention across the n_heads dimension. This treats each "head"
    as a feature group and learns inter-group relationships.
    
    The FIM initialization (if provided) is used to initialize the
    value projection, biasing the model toward data geometry.
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
        self.scale = math.sqrt(self.d_head)

        # Standard Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        # Initialize V projection from FIM if provided
        if init_fim is not None:
            self._init_from_fim(init_fim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _init_from_fim(self, fim: torch.Tensor):
        """Initialize V projection to emphasize FIM structure."""
        try:
            fim = fim.float()
            # Use FIM eigenstructure to initialize W_v
            eigvals, eigvecs = torch.linalg.eigh(fim)
            eigvals = eigvals.clamp(min=1e-6)
            # Scale eigenvectors by sqrt(eigenvalues) for importance weighting
            scaled_vecs = eigvecs * torch.sqrt(eigvals).unsqueeze(0)
            # Project to d_model if sizes match
            if scaled_vecs.shape[0] == self.d_model:
                with torch.no_grad():
                    self.W_v.weight.copy_(scaled_vecs.T)
        except Exception:
            pass  # Fall back to default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature-cross attention.
        
        Input: (B, d_model)
        Reshape to (B, n_heads, d_head)
        Attention across n_heads (feature groups)
        """
        B, D = x.shape
        residual = x

        # Project to Q, K, V
        Q = self.W_q(x).view(B, self.n_heads, self.d_head)  # (B, H, d_head)
        K = self.W_k(x).view(B, self.n_heads, self.d_head)
        V = self.W_v(x).view(B, self.n_heads, self.d_head)

        # Attention scores: (B, H, H) - attention between feature groups
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.bmm(attn, V)  # (B, H, d_head)
        out = out.reshape(B, D)
        out = self.W_out(out)

        return self.layer_norm(out + residual)
