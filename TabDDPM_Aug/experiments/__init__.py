"""
Experimental analysis modules for TabDDPM-Aug.

Modules:
    ablation: Strategy comparison (DBHA vs FDHA, with/without DCR)
    sensitivity: Hyperparameter robustness analysis
"""

from .ablation import generate_ablation_variant
from .sensitivity import run_sensitivity_overgeneration, run_sensitivity_threshold

__all__ = [
    'generate_ablation_variant',
    'run_sensitivity_overgeneration',
    'run_sensitivity_threshold'
]
