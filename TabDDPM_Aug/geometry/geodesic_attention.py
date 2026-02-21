"""
Geodesic Attention for TabDDPM.
Replaces standard dot-product attention with manifold-aware similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GeodesicAttention(nn.Module):
    """
    Attention mechanism using geodesic distances on learned manifold.
    
    Standard Attention: softmax(Q·K^T / √d)
    Geodesic Attention: softmax(-geodesic_distance(Q, K))
    """
    
    def __init__(self, embed_dim, num_heads=4, metric='exponential'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.metric = metric
        
        # Learnable metric tensor
        self.metric_tensor = nn.Parameter(
            torch.eye(embed_dim, dtype=torch.float32)
        )
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def geodesic_distance(self, x, y):
        """
        Compute Riemannian distance using learned metric.
        
        d_M(x,y) = √((x-y)^T M (x-y))
        """
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        
        if self.metric == 'exponential':
            # Exponential map approximation (faster)
            dist = torch.norm(diff, dim=-1)
            return dist
        
        elif self.metric == 'riemann':
            # Full Riemannian metric
            M = self.metric_tensor @ self.metric_tensor.T
            quad_form = torch.einsum('...i,ij,...j->...', diff, M, diff)
            dist = torch.sqrt(quad_form + 1e-6)
            return dist
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            
        Returns:
            attended: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Compute geodesic distance matrix
        dist_matrix = self.geodesic_distance(Q, K)
        
        # Attention weights from NEGATIVE distance
        attn_weights = F.softmax(-dist_matrix / np.sqrt(self.embed_dim), dim=-1)
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, V)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output
