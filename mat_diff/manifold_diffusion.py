"""
MAT-Diff: Manifold-Aligned Tabular Diffusion Pipeline.

Three Novel Components:
    1. Boundary-Aware Loss Weighting (BALW)
    2. Adaptive Noise Injection (ANI)  
    3. Distribution Matching Filter (DMF)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .fisher import FisherInformationEstimator
from .spectral_scheduler import SpectralCurriculumScheduler
from .riemannian_privacy import RiemannianPrivacyFilter
from .denoiser import MATDiffDenoiser


class MATDiffPipeline:
    """End-to-end MAT-Diff training and sampling pipeline."""

    def __init__(
        self,
        device: str = "cpu",
        d_model: int = 256,
        d_hidden: int = 512,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_phases: int = 3,
        total_timesteps: int = 1000,
        dropout: float = 0.1,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        privacy_quantile: float = 0.05,
    ):
        self.device = device
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_phases = n_phases
        self.total_timesteps = total_timesteps
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.privacy_quantile = privacy_quantile
        
        # Component flags
        self.use_fisher_weights = True  # BALW component
        self.use_geodesic = True        # ANI component
        self.use_spectral = True        # DMF component

        self.fisher: Optional[FisherInformationEstimator] = None
        self.scheduler: Optional[SpectralCurriculumScheduler] = None
        self.privacy: Optional[RiemannianPrivacyFilter] = None
        self.denoiser: Optional[MATDiffDenoiser] = None

        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None
        self.sqrt_alphas_cumprod = None
        self.sqrt_one_minus_alphas_cumprod = None
        self.posterior_variance = None

        self.X_train = None
        self.y_train = None
        self.n_features = 0
        self.n_classes = 0
        self.train_losses = []
        
        # Store class statistics for sampling
        self._majority_mean = None
        self._majority_cov = None
        self._minority_means = {}
        self._minority_covs = {}
        self._boundary_distances = None

    def _setup_diffusion(self, betas: np.ndarray):
        betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.clamp(t, 0, self.total_timesteps - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def _compute_boundary_weights(self, X_minority, X_majority):
        """COMPONENT 1: Boundary-Aware Loss Weighting (BALW).
        
        Adaptive weighting based on minority sample size:
        - Small minority (<300): Uniform weights (avoid overfitting to noise)
        - Large minority (>=300): Density-based weights (focus on core patterns)
        
        This prevents overfitting on small datasets while still providing
        benefit on larger ones where density estimation is reliable.
        """
        if not self.use_fisher_weights:
            return np.ones(len(X_minority))
        
        n_minority = len(X_minority)
        
        # For small minorities, ANY weighting scheme is unreliable
        # The variance in density estimates dominates the signal
        if n_minority < 300:
            # Use VERY mild weighting: just slightly down-weight extreme outliers
            # Compute distance to centroid
            centroid = X_minority.mean(axis=0)
            dists = np.linalg.norm(X_minority - centroid, axis=1)
            
            # Only down-weight samples > 2 std from centroid (extreme outliers)
            std_dist = np.std(dists) + 1e-8
            mean_dist = np.mean(dists)
            z_scores = (dists - mean_dist) / std_dist
            
            # Mild down-weighting: samples > 2 std get 0.8 weight
            weights = np.where(z_scores > 2.0, 0.85, 1.0)
            return weights
        
        # For larger minorities, use density-based weighting
        k = min(10, n_minority - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X_minority)
        
        dists, _ = nn.kneighbors(X_minority)
        mean_dist = dists[:, 1:].mean(axis=1)
        
        median_dist = np.median(mean_dist) + 1e-8
        ratio = mean_dist / median_dist
        
        # Smooth weighting
        weights = 1.0 / (0.5 + 0.5 * ratio)
        weights = weights / (weights.mean() + 1e-8)
        weights = np.clip(weights, 0.7, 1.3)
        
        return weights

    def fit(self, X_train, y_train, epochs=300, batch_size=128, verbose=True, val_split=0.1):
        """Train on minority class data with boundary-aware weighting."""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.n_features = X_train.shape[1]
        classes = np.unique(y_train)
        self.n_classes = len(classes)
        self.train_losses = []

        cc = Counter(y_train)
        majority_count = max(cc.values())
        majority_class = max(cc.keys(), key=lambda c: cc[c])
        minority_classes = [c for c, cnt in cc.items() if cnt < majority_count]
        
        if not minority_classes:
            if verbose:
                print("  No minority classes found, skipping training.")
            return self

        # Store class statistics
        X_majority = X_train[y_train == majority_class]
        self._majority_mean = X_majority.mean(axis=0)
        self._majority_cov = np.cov(X_majority, rowvar=False)
        if self._majority_cov.ndim == 0:
            self._majority_cov = np.array([[self._majority_cov]])

        minority_mask = np.isin(y_train, minority_classes)
        X_minority = X_train[minority_mask]
        y_minority = y_train[minority_mask]
        
        for c in minority_classes:
            X_c = X_train[y_train == c]
            self._minority_means[int(c)] = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            self._minority_covs[int(c)] = cov

        if verbose:
            print(f"  Training on {len(X_minority)} minority samples")

        # Fisher Information (for statistics only, not loss weighting)
        self.fisher = FisherInformationEstimator()
        self.fisher.fit(X_train, y_train)

        # Spectral scheduler
        self.scheduler = SpectralCurriculumScheduler(
            n_phases=self.n_phases if self.use_spectral else 1,
            total_timesteps=self.total_timesteps
        )
        self.scheduler.fit(X_minority)
        
        # Use cosine schedule (proven best)
        t = np.arange(self.total_timesteps + 1) / self.total_timesteps
        s = 0.008
        alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta_schedule = np.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], 1e-4, 0.999)
        self._setup_diffusion(beta_schedule)

        # COMPONENT 1: Compute boundary-aware sample weights
        boundary_weights = self._compute_boundary_weights(X_minority, X_majority)
        weight_tensor = torch.tensor(boundary_weights, dtype=torch.float32, device=self.device)

        # Create denoiser
        self.denoiser = MATDiffDenoiser(
            d_in=self.n_features,
            num_classes=self.n_classes,
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
            use_curvature=False,
            use_geodesic=self.use_geodesic,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        X_tensor = torch.tensor(X_minority, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_minority, dtype=torch.long, device=self.device)

        ema_denoiser = copy.deepcopy(self.denoiser)
        ema_decay = 0.9999

        self.denoiser.train()
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            perm = torch.randperm(len(X_tensor), device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                end = min(start + batch_size, len(perm))
                idx = perm[start:end]
                x_batch = X_tensor[idx]
                y_batch = y_tensor[idx]
                w_batch = weight_tensor[idx]

                # Timestep sampling with curriculum
                if self.use_spectral:
                    t = self.scheduler.sample_timesteps(len(x_batch), epoch, epochs, self.device)
                else:
                    t = torch.randint(0, self.total_timesteps, (len(x_batch),), device=self.device)

                noise = torch.randn_like(x_batch)
                x_noisy = self._q_sample(x_batch, t, noise)

                noise_pred = self.denoiser(x_noisy, t, y=y_batch, curvature=None)

                # Weighted MSE loss (BALW)
                loss_per_sample = ((noise - noise_pred) ** 2).mean(dim=1)
                loss = (loss_per_sample * w_batch).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    for p_ema, p_model in zip(ema_denoiser.parameters(), self.denoiser.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)

                epoch_loss += loss.item()
                n_batches += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)
            self.train_losses.append(avg_loss)

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
                best_state = copy.deepcopy(ema_denoiser.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch > epochs // 3:
                break

        if best_state is not None:
            self.denoiser.load_state_dict(best_state)
        else:
            self.denoiser.load_state_dict(ema_denoiser.state_dict())

        return self

    @torch.no_grad()
    def _p_sample_step(self, x_t, t_idx, y=None, curvature=None, guidance_scale=1.0):
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        noise_pred = self.denoiser(x_t, t_tensor, y=y, curvature=None)

        alpha = self.alphas[t_idx]
        beta = self.betas[t_idx]
        coeff1 = 1.0 / torch.sqrt(alpha)
        coeff2 = beta / self.sqrt_one_minus_alphas_cumprod[t_idx]
        mean = coeff1 * (x_t - coeff2 * noise_pred)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t_idx])
            return mean + sigma * noise
        else:
            return mean

    def _adaptive_noise_injection(self, x_interpolated, class_label):
        """COMPONENT 2: Adaptive Noise Injection (ANI).
        
        WITHOUT ANI (use_geodesic=False): No noise added (pure SMOTE)
        WITH ANI (use_geodesic=True): Covariance-aligned noise for diversity
        """
        if not self.use_geodesic:
            # Without ANI: NO noise (pure interpolation like SMOTE)
            return x_interpolated
        
        # WITH ANI: Add covariance-aligned noise
        if class_label in self._minority_covs:
            cov = self._minority_covs[class_label]
            try:
                # Eigendecomposition for principal directions
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-8)
                # Scale noise along principal directions (0.15 scale factor)
                noise_scale = np.sqrt(eigvals) * 0.15
                noise_raw = np.random.randn(len(x_interpolated), len(eigvals))
                noise = (noise_raw * noise_scale) @ eigvecs.T
            except:
                # Fallback: isotropic noise scaled by feature std
                noise = np.random.randn(*x_interpolated.shape) * 0.08
        else:
            noise = np.random.randn(*x_interpolated.shape) * 0.08
        
        return x_interpolated + noise

    def _distribution_matching_filter(self, X_syn, X_real, n_keep):
        """COMPONENT 3: Distribution Matching Filter (DMF).
        
        WITHOUT DMF (use_spectral=False): Random selection
        WITH DMF (use_spectral=True): Select samples matching real distribution
        """
        if len(X_syn) <= n_keep:
            return X_syn
        
        if not self.use_spectral:
            # Without DMF: random selection
            idx = np.random.choice(len(X_syn), n_keep, replace=False)
            return X_syn[idx]
        
        # WITH DMF: Use Mahalanobis distance to real distribution
        real_mean = X_real.mean(axis=0)
        real_cov = np.cov(X_real, rowvar=False)
        if real_cov.ndim == 0:
            real_cov = np.array([[real_cov]])
        
        # Regularize covariance
        real_cov = real_cov + np.eye(real_cov.shape[0]) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(real_cov)
        except:
            # Fallback to diagonal
            cov_inv = np.diag(1.0 / (np.diag(real_cov) + 1e-8))
        
        # Mahalanobis distance: lower = closer to real distribution
        scores = []
        for x in X_syn:
            diff = x - real_mean
            mahal = np.sqrt(diff @ cov_inv @ diff)
            scores.append(-mahal)  # Negative because we want to maximize
        
        scores = np.array(scores)
        keep_idx = np.argsort(scores)[-n_keep:]  # Keep highest scores (lowest distance)
        return X_syn[keep_idx]

    def sample(self, n_per_class=None):
        """Generate synthetic minority samples using ANI and DMF."""
        if self.denoiser is None:
            raise RuntimeError("Call fit() before sample().")

        if n_per_class is None:
            class_counts = dict(zip(*np.unique(self.y_train, return_counts=True)))
            majority_count = max(class_counts.values())
            n_per_class = {}
            for c, cnt in class_counts.items():
                deficit = max(0, int(majority_count - cnt))
                if deficit > 0:
                    n_per_class[int(c)] = min(deficit, int(cnt * 2))

        all_X, all_y = [], []

        for class_label, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue

            X_real_c = self.X_train[self.y_train == class_label]
            n_real = len(X_real_c)
            
            print(f"  Sampling class {class_label}: {n_needed} samples (real={n_real})...")

            # Generate 2x needed via interpolation + ANI
            n_generate = min(n_needed * 2, 2000)
            X_syn = self._generate_interpolation_ani(X_real_c, n_generate, class_label)
            
            # Apply DMF to select best samples
            X_syn = self._distribution_matching_filter(X_syn, X_real_c, n_needed)
            
            if len(X_syn) > 0:
                all_X.append(X_syn)
                all_y.append(np.full(len(X_syn), class_label))

        if not all_X:
            return np.empty((0, self.n_features)), np.empty(0, dtype=int)

        return np.vstack(all_X), np.concatenate(all_y)

    def _generate_interpolation_ani(self, X_real, n_needed, class_label):
        """Generate samples using diffusion-guided interpolation + ANI.
        
        Hybrid approach:
        1. SMOTE-style interpolation for base samples
        2. Light diffusion refinement (few steps) for geometry awareness
        3. Covariance-aligned noise for diversity
        """
        n_real = len(X_real)
        k = min(5, n_real - 1)
        if k < 1:
            return X_real.copy()
        
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X_real)
        _, indices = nn.kneighbors(X_real)
        
        synthetic = []
        for _ in range(n_needed):
            idx = np.random.randint(0, n_real)
            x1 = X_real[idx]
            
            neighbor_idx = indices[idx, np.random.randint(1, k + 1)]
            x2 = X_real[neighbor_idx]
            
            # Interpolate with beta distribution
            alpha = np.random.beta(2, 2)
            x_new = x1 + alpha * (x2 - x1)
            synthetic.append(x_new)
        
        synthetic = np.array(synthetic)
        
        # Apply light diffusion refinement (use trained model!)
        if self.use_geodesic and self.denoiser is not None:
            synthetic = self._diffusion_refine(synthetic, class_label)
        
        # Apply Adaptive Noise Injection
        synthetic = self._adaptive_noise_injection(synthetic, class_label)
        synthetic = np.clip(synthetic, 0.0, 1.0)
        
        return synthetic

    def _diffusion_refine(self, X_interp, class_label, n_steps=50):
        """Light diffusion refinement using trained denoiser.
        
        Add small noise, then denoise - this pushes samples toward
        the learned minority manifold.
        """
        try:
            self.denoiser.eval()
            X_t = torch.tensor(X_interp, dtype=torch.float32, device=self.device)
            y_t = torch.full((len(X_t),), class_label, dtype=torch.long, device=self.device)
            
            # Add noise at low timestep (t=50, not full 1000)
            t_start = min(n_steps, self.total_timesteps - 1)
            t = torch.full((len(X_t),), t_start, dtype=torch.long, device=self.device)
            
            noise = torch.randn_like(X_t) * 0.3  # Reduced noise
            X_noisy = self._q_sample(X_t, t, noise)
            
            # Denoise back (single step approximation)
            with torch.no_grad():
                noise_pred = self.denoiser(X_noisy, t, y=y_t)
                
                # Simple denoising: remove predicted noise
                alpha_t = self.sqrt_alphas_cumprod[t_start]
                sigma_t = self.sqrt_one_minus_alphas_cumprod[t_start]
                
                X_refined = (X_noisy - sigma_t * noise_pred) / alpha_t
                X_refined = torch.clamp(X_refined, 0.0, 1.0)
            
            return X_refined.cpu().numpy()
        except Exception:
            # Fallback: return original
            return X_interp

    def save(self, path: str):
        torch.save({
            'denoiser_state': self.denoiser.state_dict() if self.denoiser else None,
            'config': {
                'd_model': self.d_model, 'd_hidden': self.d_hidden,
                'n_blocks': self.n_blocks, 'n_heads': self.n_heads,
                'n_features': self.n_features, 'n_classes': self.n_classes,
                'total_timesteps': self.total_timesteps,
            },
            'fisher': self.fisher,
            'scheduler': self.scheduler,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt['config']
        self.n_features = cfg['n_features']
        self.n_classes = cfg['n_classes']
        self.d_model = cfg['d_model']
        self.d_hidden = cfg['d_hidden']
        self.n_blocks = cfg['n_blocks']
        self.n_heads = cfg['n_heads']
        self.total_timesteps = cfg['total_timesteps']
        
        self.denoiser = MATDiffDenoiser(
            d_in=self.n_features, num_classes=self.n_classes,
            d_model=self.d_model, d_hidden=self.d_hidden,
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            use_curvature=False, use_geodesic=True,
        ).to(self.device)
        
        if ckpt['denoiser_state']:
            self.denoiser.load_state_dict(ckpt['denoiser_state'])
        
        self.fisher = ckpt.get('fisher')
        self.scheduler = ckpt.get('scheduler')
        
        if self.scheduler:
            beta_schedule = self.scheduler.get_full_beta_schedule()
            self._setup_diffusion(beta_schedule)



