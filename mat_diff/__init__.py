"""
MAT-Diff: Manifold-Aligned Tabular Diffusion for
Geometry-Guided Imbalanced Multi-Class Classification
"""

__version__ = "1.0.0"

from .fisher import FisherInformationEstimator
from .geodesic_attention import GeodesicAttentionBlock, StandardAttentionBlock
from .spectral_scheduler import SpectralCurriculumScheduler
from .riemannian_privacy import RiemannianPrivacyFilter
from .manifold_diffusion import MATDiffPipeline
from .denoiser import MATDiffDenoiser

__all__ = [
    "FisherInformationEstimator",
    "GeodesicAttentionBlock",
    "StandardAttentionBlock",
    "SpectralCurriculumScheduler",
    "RiemannianPrivacyFilter",
    "MATDiffPipeline",
    "MATDiffDenoiser",
]

