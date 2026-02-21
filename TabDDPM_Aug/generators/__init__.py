"""
Synthetic data generation methods.

Modules:
    smote_baseline: SMOTE and ADASYN implementations
    gan_baselines: GAN-based methods (CTGAN, TVAE, Copula, CTAB-GAN+)
    tabddpm_baseline: Original TabDDPM implementation
    tabddpm_aug: TabDDPM-Aug (adaptive hybrid augmentation)
"""

from .smote_baseline import smote_adasyn_generator, SMOTE_AVAILABLE
from .gan_baselines import (
    copula_generator,
    ctgan_generator,
    tvae_generator,
    ctabgan_plus_generator
)
from .tabddpm_baseline import tabddpm_generator_original_baseline
from .tabddpm_aug import (
    tabddpm_aug_final,
    tabddpm_aug_ensemble_generator,
    dcr_filtering,
    find_hard_samples
)

__all__ = [
    'smote_adasyn_generator',
    'SMOTE_AVAILABLE',
    'copula_generator',
    'ctgan_generator',
    'tvae_generator',
    'ctabgan_plus_generator',
    'tabddpm_generator_original_baseline',
    'tabddpm_aug_final',
    'tabddpm_aug_ensemble_generator',
    'dcr_filtering',
    'find_hard_samples'
]
