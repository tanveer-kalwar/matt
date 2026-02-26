"""
MAT-Diff Denoiser: MLP backbone with Geodesic Attention blocks.

Architecture:
    Input → Project → [AdaLN conditioning from Time+Class+Curvature] →
    GeodesicAttention → MLP Block → GeodesicAttention → MLP Block →
    Output Head → ε̂ (predicted noise)

Uses Adaptive Layer Normalization (AdaLN) from DiT (Peebles & Xie, 2023)
for conditioning instead of simple additive embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .geodesic_attention import GeodesicAttentionBlock


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings (from DDPM)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class AdaLN(nn.Module):
    """Adaptive Layer Normalization.

    Modulates LayerNorm output with learned scale and shift from conditioning.
    From DiT (Peebles & Xie, ICCV 2023).
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )
        # Initialize to identity: scale=1, shift=0
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class MLPBlock(nn.Module):
    """MLP block with AdaLN conditioning and residual connection."""

    def __init__(self, d_model: int, d_hidden: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.adaln = AdaLN(d_model, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.adaln(x + self.net(x), cond)


class MATDiffDenoiser(nn.Module):
    """MAT-Diff denoiser with geodesic attention and AdaLN conditioning.

    Args:
        d_in: Input feature dimension (number of tabular features).
        num_classes: Number of classes (0 for unconditional).
        d_model: Internal representation dimension.
        d_hidden: Hidden dimension in MLP blocks.
        n_blocks: Number of (GeodesicAttention + MLP) blocks.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        dim_t: Timestep embedding dimension.
        use_curvature: Whether to condition on Fisher curvature.
        init_fim: Optional initial FIM tensor for geodesic attention.
    """

    def __init__(
        self,
        d_in: int,
        num_classes: int,
        d_model: int = 256,
        d_hidden: int = 512,
        n_blocks: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        dim_t: int = 128,
        use_curvature: bool = True,
        init_fim: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.use_curvature = use_curvature

        # Conditioning dimension
        cond_dim = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.SiLU(),
        )

        # Time embedding → conditioning vector
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Class embedding
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, d_model)
        else:
            self.class_embed = None

        # Curvature embedding
        if use_curvature and num_classes > 0:
            self.curvature_proj = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.SiLU(),
                nn.Linear(d_model // 4, d_model),
            )
        else:
            self.curvature_proj = None

        # Conditioning fusion: combine time + class + curvature → single cond vector
        n_cond_sources = 1  # time is always present
        if num_classes > 0:
            n_cond_sources += 1
        if use_curvature and num_classes > 0:
            n_cond_sources += 1
        self.cond_fusion = nn.Sequential(
            nn.Linear(d_model * n_cond_sources, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Core blocks: GeodesicAttention + conditioned MLP
        self.attn_blocks = nn.ModuleList()
        self.mlp_blocks = nn.ModuleList()
        self.attn_adalns = nn.ModuleList()
        for _ in range(n_blocks):
            self.attn_blocks.append(
                GeodesicAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    init_fim=init_fim,
                )
            )
            self.attn_adalns.append(AdaLN(d_model, cond_dim))
            self.mlp_blocks.append(MLPBlock(d_model, d_hidden, cond_dim, dropout))

        # Output head
        self.output_norm = AdaLN(d_model, cond_dim)
        self.output_proj = nn.Linear(d_model, d_in)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        curvature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN conditioning."""
        # Project input
        h = self.input_proj(x)

        # Build conditioning vector
        cond_parts = []

        # Time embedding (always present)
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        cond_parts.append(t_emb)

        # Class embedding
        if self.class_embed is not None and y is not None:
            y_int = y.long().view(-1)
            y_int = torch.clamp(y_int, 0, self.num_classes - 1)
            c_emb = self.class_embed(y_int)
            cond_parts.append(c_emb)

        # Curvature embedding
        if self.curvature_proj is not None and curvature is not None:
            curv_input = curvature.float().unsqueeze(-1)
            curv_emb = self.curvature_proj(curv_input)
            cond_parts.append(curv_emb)

        # Fuse conditioning
        cond = self.cond_fusion(torch.cat(cond_parts, dim=-1))

        # Process through blocks with AdaLN conditioning
        for attn, adaln, mlp in zip(self.attn_blocks, self.attn_adalns, self.mlp_blocks):
            h = adaln(attn(h), cond)  # GeodesicAttention has internal residual+LN
            h = mlp(h, cond)  # MLP with AdaLN (has internal residual)

        # Output
        h = self.output_norm(h, cond)
        return self.output_proj(h)

