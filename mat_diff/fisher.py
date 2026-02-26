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
        self._class_counts: Dict[int, int] = {}

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
            self._class_counts[int(c)] = n_c
            mu_c = np.mean(X_c, axis=0)
            self.class_means[int(c)] = mu_c

            if n_c < 2:
                reg = 1.0 / (d * max(1, len(X)))
                cov_c = np.eye(d) * reg
            else:
                cov_c = np.cov(X_c, rowvar=False)
                if cov_c.ndim == 0:
                    cov_c = np.array([[cov_c]])

                diag_var = np.diag(cov_c)
                diag_var = np.maximum(diag_var, 0)
                median_var = np.median(diag_var) if len(diag_var) > 0 else 1.0
                reg = max(median_var / n_c, 1e-10)
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
        """Per-class loss weights: inverse-frequency × curvature modulation.

        Base weight = n_total / (n_classes * n_c)  [standard inverse frequency]
        Curvature modulation = (curv_c / mean_curv) ^ 0.5  [stronger curvature scaling]
        Final weight = base * modulation, normalized so mean = 1.0

        The 0.5 exponent provides meaningful curvature adjustment that actually
        differentiates between geometrically simple and complex classes.
        """
        if not self._class_counts:
            return {c: 1.0 for c in self.curvatures}

        n_total = sum(self._class_counts.values())
        n_cls = len(self._class_counts)

        # Inverse frequency base weights
        base_weights = {}
        for c, n_c in self._class_counts.items():
            base_weights[c] = n_total / (n_cls * max(n_c, 1))

        # Curvature modulation (meaningful strength)
        mean_curv = np.mean(list(self.curvatures.values())) if self.curvatures else 1.0
        mean_curv = max(mean_curv, 1e-12)

        raw_weights = {}
        for c in self._class_counts:
            curv_ratio = self.curvatures.get(c, mean_curv) / mean_curv
            modulation = curv_ratio ** 0.5  # stronger than 0.25, weaker than linear
            raw_weights[c] = base_weights[c] * modulation

        # Normalize: mean weight = 1.0
        mean_w = np.mean(list(raw_weights.values()))
        if mean_w < 1e-12:
            return {c: 1.0 for c in self._class_counts}
        return {c: w / mean_w for c, w in raw_weights.items()}

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

