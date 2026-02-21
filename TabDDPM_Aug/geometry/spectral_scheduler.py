"""
Spectral Curriculum Scheduling for diffusion training.
Analyzes data manifold eigenspectrum to derive adaptive noise schedules.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

class SpectralCurriculumScheduler:
    """
    Derive training phases from manifold spectral properties.
    
    Core Idea:
    - Low eigenvalues → smooth manifold regions → early training
    - High eigenvalues → complex curvature → late training
    """
    
    def __init__(self, n_phases=3):
        self.n_phases = n_phases
        self.eigenvalues = None
        self.phase_boundaries = None
    
    def analyze_manifold(self, X):
        """
        Compute eigenspectrum of data covariance (or graph Laplacian).
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            eigenvalues: Sorted spectrum λ_1 ≥ λ_2 ≥ ... ≥ λ_d
        """
        # Option 1: PCA-based (for high-dimensional data)
        pca = PCA(n_components=min(50, X.shape[1]))
        pca.fit(X)
        eigenvalues = pca.explained_variance_
        
        # Option 2: Graph Laplacian (for manifold learning)
        # Uncomment for stronger geometric guarantees:
        # D = squareform(pdist(X[:1000]))  # Subsample for speed
        # sigma = np.median(D)
        # W = np.exp(-D**2 / (2*sigma**2))
        # L = np.diag(W.sum(axis=1)) - W
        # eigenvalues = np.linalg.eigvalsh(L)
        
        self.eigenvalues = np.sort(eigenvalues)[::-1]
        return self.eigenvalues
    
    def compute_phase_boundaries(self):
        """
        Split training into phases based on spectral gaps.
        
        Returns:
            phase_boundaries: [(epoch_start, epoch_end, noise_level), ...]
        """
        if self.eigenvalues is None:
            raise ValueError("Must call analyze_manifold first")
        
        # Find spectral gaps (large eigenvalue drops)
        gaps = np.diff(self.eigenvalues)
        gap_indices = np.argsort(gaps)[:self.n_phases-1]
        
        # Convert to training epoch boundaries
        # Assumption: 1000 epochs total
        total_epochs = 1000
        boundaries = []
        
        prev_epoch = 0
        for i, gap_idx in enumerate(sorted(gap_indices)):
            # Map eigenvalue index to epoch
            epoch = int((gap_idx / len(self.eigenvalues)) * total_epochs)
            
            # Noise level inversely proportional to eigenvalue magnitude
            noise_level = 1.0 - (i / self.n_phases)
            
            boundaries.append((prev_epoch, epoch, noise_level))
            prev_epoch = epoch
        
        # Final phase
        boundaries.append((prev_epoch, total_epochs, 0.1))
        
        self.phase_boundaries = boundaries
        return boundaries
    
    def get_noise_schedule(self, epoch):
        """
        Return adaptive noise level for current epoch.
        
        Traditional: β_t = linear or cosine schedule
        Spectral: β_t adapts to manifold complexity
        """
        if self.phase_boundaries is None:
            self.compute_phase_boundaries()
        
        for start, end, noise in self.phase_boundaries:
            if start <= epoch < end:
                # Smooth transition within phase
                progress = (epoch - start) / (end - start)
                return noise * (1 - 0.5 * progress)
        
        return 0.01  # Minimum noise
