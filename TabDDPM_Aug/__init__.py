"""
TabDDPM-Aug: Adaptive Hybrid Augmentation for Imbalanced Tabular Data

Modules:
    config: Dataset-specific hyperparameters
    data_loader: Dataset loading and preprocessing
    generators: Data augmentation methods
    evaluation: Quality assessment metrics
    experiments: Ablation and sensitivity studies
"""

from .config import get_config, DATASET_CONFIGS
from .data_loader import load_dataset, prepare_data

__all__ = [
    'get_config',
    'DATASET_CONFIGS',
    'load_dataset',
    'prepare_data'
]
