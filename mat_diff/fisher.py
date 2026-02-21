"""
Fisher Information Matrix estimation for multi-class tabular data.

Provides per-class curvature estimates that guide:
    - Loss weighting during diffusion training
    - Augmentation allocation (more samples for geometrically complex classes)
    - Spectral curriculum phase boundaries
"""

import numpy as np
import torch
from typing import Dict


class FisherInformationEstimator:
    """Estimate per-class Fisher Information from tabular data."""

    def __init__(self):
        self.fim_matrices: Dict[int, np.ndarray] = {}
        self.curvatures: Dict[int, float] = {}
        self.eigenspectra: Dict[int, np.ndarray] = {}
        self.class_means: Dict[int, np.ndarray] = {}
        self.class_covs: Dict[int, np.ndarray] = {}
        self.n_classes = 0
        self._regularizations: Dict[int, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherInformationEstimator":
        """Estimate per-class FIM with data-driven regularization.

        Regularization: lambda_c = median(diag(Sigma_c)) / n_c
        Ledoit-Wolf type shrinkage scaled by sample size.
        As n_c grows, regularization vanishes (statistical consistency).
        """
        classes = np.unique(y)
        self.n_classes = len(classes)
        d = X.shape[1]

        for c in classes:
            X_c = X[y == c]
            n_c = len(X_c)
            mu_c = np.mean(X_c, axis=0)
            self.class_means[int(c)] = mu_c

            if n_c < 2:
                # Degenerate: use global covariance scaled estimate
                reg = 1.0 / (d * max(1, len(X)))
                cov_c = np.eye(d) * reg
            else:
                cov_c = np.cov(X_c, rowvar=False)
                if cov_c.ndim == 0:
                    cov_c = np.array([[cov_c]])

                # Data-driven regularization:
                # lambda = median(diag(Sigma)) / n_c
                diag_var = np.diag(cov_c)
                diag_var = np.maximum(diag_var, 0)
                median_var = np.median(diag_var) if len(diag_var) > 0 else 1.0
                reg = max(median_var / n_c, 1e-10)  # 1e-10 is machine-precision floor
                cov_c = cov_c + np.eye(d) * reg

            self._regularizations[int(c)] = reg
            self.class_covs[int(c)] = cov_c

            try:
                fim_c = np.linalg.inv(cov_c)
            except np.linalg.LinAlgError:
                fim_c = np.linalg.pinv(cov_c)

            self.fim_matrices[int(c)] = fim_c

            eigenvalues = np.linalg.eigvalsh(fim_c)
            eigenvalues = np.maximum(eigenvalues, 0)
            self.eigenspectra[int(c)] = np.sort(eigenvalues)[::-1]
            self.curvatures[int(c)] = float(np.sum(eigenvalues))

        return self

    def get_loss_weights(self) -> Dict[int, float]:
        """Per-class loss weights proportional to curvature.

        Normalization: weights sum to n_classes, so average weight = 1.0.
        This preserves the overall loss magnitude while re-distributing
        capacity toward geometrically complex classes.
        """
        total = sum(self.curvatures.values())
        if total < 1e-12:
            return {c: 1.0 for c in self.curvatures}
        return {c: (curv / total) * self.n_classes
                for c, curv in self.curvatures.items()}

    def get_augmentation_allocation(
        self, total_budget: int, class_counts: Dict[int, int]
    ) -> Dict[int, int]:
        """Allocate synthetic samples proportional to deficit × curvature."""
        majority_count = max(class_counts.values())
        allocation = {}
        curvature_needs = {}

        for c, count in class_counts.items():
            deficit = majority_count - count
            if deficit <= 0:
                allocation[c] = 0
                continue
            curv = self.curvatures.get(c, 1.0)
            curvature_needs[c] = deficit * curv

        if not curvature_needs:
            return allocation

        total_need = sum(curvature_needs.values())
        for c, need in curvature_needs.items():
            raw = int(total_budget * need / total_need)
            deficit = majority_count - class_counts[c]
            allocation[c] = min(raw, deficit)

        return allocation

    def get_curvature_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return curvatures as a tensor indexed by class label."""
        max_label = max(self.curvatures.keys())
        curv = torch.zeros(max_label + 1, device=device)
        for c, v in self.curvatures.items():
            curv[c] = v
        return curv
