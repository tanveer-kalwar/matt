"""
Evaluation metrics for synthetic data quality assessment.

Modules:
    fidelity: Distributional similarity metrics (KS, MMD, JS, WD)
    privacy: Membership inference attack evaluation
    utility: Classification performance metrics
"""

from .fidelity import (
    compute_ks_statistics,
    compute_mmd,
    compute_correlation_distance,
    compute_dcr_statistics,
    calculate_js_divergence,
    compute_wasserstein_distance
)

from .privacy import membership_inference_attack

from .utility import (
    evaluate_comprehensive,
    evaluate_simple,
    compute_synthetic_data_quality_score
)

__all__ = [
    'compute_ks_statistics',
    'compute_mmd',
    'compute_correlation_distance',
    'compute_dcr_statistics',
    'calculate_js_divergence',
    'compute_wasserstein_distance',
    'membership_inference_attack',
    'evaluate_comprehensive',
    'evaluate_simple',
    'compute_synthetic_data_quality_score'
]
