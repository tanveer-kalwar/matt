"""
MAT-Diff Denoiser: MLP backbone with Geodesic Attention blocks.

Architecture:
    Input → Project → [TimeEmbed + ClassEmbed + CurvatureEmbed] →
    GeodesicAttention → MLP Block → GeodesicAttention → MLP Block →
    Output Head → ε̂ (predicted noise)

The curvature embedding injects per-class geometric information,
allowing the denoiser to adapt its behavior based on manifold complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .geodesic_attention import GeodesicAttentionBlock


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings (from DDPM).

    Args:
        timesteps: 1-D Tensor of N timestep indices.
        dim: Embedding dimension.
        max_period: Controls minimum frequency.

    Returns:
        Tensor of shape (N, dim).
    """
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


class MLPBlock(nn.Module):
    """Standard MLP block with residual connection.

    Architecture: Linear → LayerNorm → SiLU → Dropout → Linear → Residual
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MATDiffDenoiser(nn.Module):
    """MAT-Diff denoiser with geodesic attention and curvature conditioning.

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

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Class embedding (for conditional generation)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, d_model)
        else:
            self.class_embed = None

        # Curvature embedding (novel: injects geometric complexity info)
        if use_curvature and num_classes > 0:
            self.curvature_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.curvature_proj = None

        # Core blocks: alternating GeodesicAttention and MLP
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict({
                    "attention": GeodesicAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        init_fim=init_fim,
                    ),
                    "mlp": MLPBlock(d_model, d_hidden, dropout),
                })
            )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_in),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        curvature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: predict noise ε given noisy x_t, timestep t, class y.

        Args:
            x: Noisy input of shape (B, d_in).
            timesteps: Timestep indices of shape (B,).
            y: Class labels of shape (B,) or None.
            curvature: Per-sample curvature values of shape (B,) or None.

        Returns:
            Predicted noise of shape (B, d_in).
        """
        # Project input
        h = self.input_proj(x)

        # Add time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        h = h + t_emb

        # Add class embedding
        if self.class_embed is not None and y is not None:
            y_int = y.long().view(-1)
            y_int = torch.clamp(y_int, 0, self.num_classes - 1)
            c_emb = self.class_embed(y_int)
            h = h + F.silu(c_emb)

        # Add curvature embedding (novel)
        if self.curvature_proj is not None and curvature is not None:
            curv_input = curvature.float().unsqueeze(-1)  # (B, 1)
            curv_emb = self.curvature_proj(curv_input)
            h = h + curv_emb

        # Process through Geodesic Attention + MLP blocks
        for block in self.blocks:
            h = block["attention"](h)
            h = block["mlp"](h)

        # Predict noise
        return self.output_head(h)
