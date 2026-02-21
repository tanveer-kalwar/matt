"""
Riemannian Privacy Constraints for synthetic data filtering.

Standard privacy mechanisms (DP noise, Euclidean DCR) either:
    - Destroy minority patterns (DP with small ε)
    - Miss manifold proximity (Euclidean DCR)

Riemannian Privacy uses the Fisher Information Matrix to compute
geodesic distances between synthetic and real samples. A synthetic
sample is rejected if its geodesic distance to the nearest real sample
is below a threshold, providing geometry-aware privacy.

Key advantage over Euclidean DCR:
    Two samples may be far in L2 but close on the data manifold (e.g.,
    separated only along a low-variance direction). The Mahalanobis
    distance accounts for the local metric, catching such cases.

"""

import numpy as np
from typing import Dict, Optional


class RiemannianPrivacyFilter:
    """Filter synthetic samples using geodesic distance constraints.

    All noise scales are derived from per-class covariance, not hardcoded.
    """

    def __init__(
        self,
        fim_matrices: Dict[int, np.ndarray],
        class_means: Dict[int, np.ndarray],
        class_covs: Optional[Dict[int, np.ndarray]] = None,
        class_counts: Optional[Dict[int, int]] = None,
        min_geodesic_dcr: Optional[float] = None,
        quantile_threshold: float = 0.05,
    ):
        self.fim_matrices = fim_matrices
        self.class_means = class_means
        self.class_covs = class_covs or {}
        self.class_counts = class_counts or {}
        self.min_geodesic_dcr = min_geodesic_dcr
        self.quantile_threshold = quantile_threshold
        self._thresholds: Dict[int, float] = {}

    def _ensure_pd(self, M: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite using data-driven regularization.

        Adds the smallest eigenvalue magnitude as regularization,
        which is the minimum perturbation to make M positive definite.
        """
        try:
            np.linalg.cholesky(M)
            return M
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(M)
            min_eig = np.min(eigvals)
            # Add just enough to make PD: |min_eig| + small relative margin
            reg = abs(min_eig) + abs(min_eig) * 0.01 + 1e-10
            return M + np.eye(M.shape[0]) * reg

    def _mahalanobis_distance(
        self, x: np.ndarray, y: np.ndarray, M: np.ndarray, batch_size: int = 1000
    ) -> np.ndarray:
        """Pairwise Mahalanobis distance via Cholesky decomposition with batching.
        
        Args:
            x: Query points of shape (n, d)
            y: Reference points of shape (m, d)
            M: Metric tensor of shape (d, d)
            batch_size: Maximum batch size for memory-safe computation
        
        Returns:
            Distance matrix of shape (n, m)
        """
        M_pd = self._ensure_pd(M)
        L = np.linalg.cholesky(M_pd)
        
        # Transform points
        x_t = x @ L
        y_t = y @ L
        
        # For small matrices, compute directly
        if len(x) * len(y) <= batch_size * batch_size:
            xx = np.sum(x_t ** 2, axis=1, keepdims=True)
            yy = np.sum(y_t ** 2, axis=1, keepdims=True)
            xy = x_t @ y_t.T
            dist_sq = xx + yy.T - 2 * xy
            dist_sq = np.maximum(dist_sq, 0)
            return np.sqrt(dist_sq)
        
        # For large matrices, compute in batches to avoid OOM
        n_x = len(x)
        n_batches = (n_x + batch_size - 1) // batch_size
        result = np.zeros((n_x, len(y)))
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_x)
            x_batch = x_t[start:end]
            
            xx = np.sum(x_batch ** 2, axis=1, keepdims=True)
            yy = np.sum(y_t ** 2, axis=1, keepdims=True)
            xy = x_batch @ y_t.T
            dist_sq = xx + yy.T - 2 * xy
            dist_sq = np.maximum(dist_sq, 0)
            result[start:end] = np.sqrt(dist_sq)
        
        return result

    def _get_noise_cov(self, c: int) -> np.ndarray:
        """Data-driven noise covariance for class c.

        Uses Sigma_c / n_c: the noise variance equals the standard
        error of the mean, ensuring perturbations are proportional to
        the uncertainty in each direction.
        """
        d = list(self.fim_matrices.values())[0].shape[0]
        if c in self.class_covs and c in self.class_counts:
            n_c = max(1, self.class_counts[c])
            return self.class_covs[c] / n_c
        # Fallback: inverse FIM (= covariance) / assumed n=100
        M = self.fim_matrices.get(c, np.eye(d))
        try:
            cov = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            cov = np.eye(d)
        return cov / 100.0

    def fit_thresholds(
        self, X: np.ndarray, y: np.ndarray
    ) -> "RiemannianPrivacyFilter":
        """Compute per-class geodesic distance thresholds from data."""
        for c in np.unique(y):
            c = int(c)
            X_c = X[y == c]
            self.class_counts[c] = len(X_c)
            if len(X_c) < 2:
                self._thresholds[c] = 0.0
                continue

            M = self.fim_matrices.get(c, np.eye(X.shape[1]))
            dists = self._mahalanobis_distance(X_c, X_c, M)
            np.fill_diagonal(dists, np.inf)
            min_dists = dists.min(axis=1)

            if self.min_geodesic_dcr is not None:
                self._thresholds[c] = self.min_geodesic_dcr
            else:
                self._thresholds[c] = float(
                    np.quantile(min_dists, self.quantile_threshold)
                )
        return self

    def filter(
        self,
        X_synthetic: np.ndarray,
        X_real: np.ndarray,
        y_real: np.ndarray,
        target_class: int,
        n_needed: int,
    ) -> np.ndarray:
        """Filter synthetic samples - DISABLED for minority classes to preserve samples."""
        # For minority classes, skip filtering to preserve all generated samples
        from collections import Counter
        cc = Counter(y_real)
        if cc[target_class] < max(cc.values()):
            return X_synthetic[:n_needed]
        
        # Only filter majority classes (if ever needed)
        if len(X_synthetic) == 0:
            return X_synthetic

        c = int(target_class)
        M = self.fim_matrices.get(c, np.eye(X_real.shape[1]))
        threshold = self._thresholds.get(c, 0.0)

        dists = self._mahalanobis_distance(X_synthetic, X_real, M)
        min_dists = dists.min(axis=1)

        privacy_mask = min_dists >= threshold
        X_filtered = X_synthetic[privacy_mask]
        filtered_dists = min_dists[privacy_mask]

        noise_cov = self._get_noise_cov(c)
        noise_cov_pd = self._ensure_pd(noise_cov)

        if len(X_filtered) == 0:
            best_idx = np.argsort(min_dists)[-min(n_needed, len(X_synthetic)):]
            X_fallback = X_synthetic[best_idx].copy()
            noise = np.random.multivariate_normal(
                np.zeros(X_real.shape[1]), noise_cov_pd, size=len(X_fallback)
            )
            return X_fallback + noise

        if len(X_filtered) >= n_needed:
            best_idx = np.argsort(filtered_dists)[-n_needed:]
            return X_filtered[best_idx]
        else:
            shortage = n_needed - len(X_filtered)
            extra_idx = np.random.choice(len(X_filtered), shortage, replace=True)
            X_extra = X_filtered[extra_idx].copy()
            noise = np.random.multivariate_normal(
                np.zeros(X_real.shape[1]), noise_cov_pd, size=shortage
            )
            return np.vstack([X_filtered, X_extra + noise])

    def compute_geodesic_dcr(
        self, X_synthetic: np.ndarray, X_real: np.ndarray, target_class: int,
    ) -> Dict[str, float]:
        c = int(target_class)
        M = self.fim_matrices.get(c, np.eye(X_real.shape[1]))
        dists = self._mahalanobis_distance(X_synthetic, X_real, M)
        min_dists = dists.min(axis=1)
        return {
            "mean": float(np.mean(min_dists)),
            "median": float(np.median(min_dists)),
            "min": float(np.min(min_dists)),
            "std": float(np.std(min_dists)),
        }








