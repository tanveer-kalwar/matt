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
        
        # Component flags - SEPARATE so ablations are clean
        self.use_fisher_weights = True   # BALW component
        self.use_geodesic = True         # ANI component (noise injection)
        self.use_spectral = True         # Curriculum scheduler
        self.use_dmf = True              # Distribution Matching Filter (SEPARATE!)

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
        
        Compute ACTUAL Fisher Information-based weights.
        Higher weight = higher uncertainty/gradient norm = more informative.
        """
        if not self.use_fisher_weights:
            return np.ones(len(X_minority))
        
        # Combine classes temporarily to compute per-sample gradients
        X_combined = np.vstack([X_majority, X_minority])
        y_combined = np.hstack([np.zeros(len(X_majority)), np.ones(len(X_minority))])
        
        # Simple logistic regression to get decision boundary
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_combined, y_combined)
        
        # Compute decision function values (distance to boundary)
        # For minority samples only
        decision_vals = clf.decision_function(X_minority)
        
        # Samples NEAR boundary (|decision| ≈ 0) are most informative
        # Use inverse absolute decision value (clipped)
        boundary_dist = np.abs(decision_vals) + 0.1  # +0.1 for stability
        
        # Inverse: closer to boundary = higher weight
        weights = 1.0 / boundary_dist
        
        # Normalize to mean 1
        weights = weights / (weights.mean() + 1e-8)
        
        # Clip to prevent extreme weights
        weights = np.clip(weights, 0.3, 3.0)
        
        return weights

    def fit(self, X_train, y_train, epochs=300, batch_size=128, verbose=True, val_split=0.1):
        """Train separate denoiser for each minority class."""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.n_features = X_train.shape[1]
        classes = np.unique(y_train)
        self.n_classes = len(classes)
        self.train_losses = []
        
        # Store class statistics
        self._minority_means = {}
        self._minority_covs = {}
        self.denoisers = {}  # NEW: per-class denoisers
        
        cc = Counter(y_train)
        majority_count = max(cc.values())
        majority_class = max(cc.keys(), key=lambda c: cc[c])
        minority_classes = [c for c, cnt in cc.items() if cnt < majority_count]
        
        if not minority_classes:
            if verbose:
                print("  No minority classes found, skipping training.")
            return self

        # Store majority statistics for boundary computation
        X_majority = X_train[y_train == majority_class]
        self._majority_mean = X_majority.mean(axis=0)
        self._majority_cov = np.cov(X_majority, rowvar=False)
        if self._majority_cov.ndim == 0:
            self._majority_cov = np.array([[self._majority_cov]])

        if verbose:
            print(f"  Found {len(minority_classes)} minority classes")
            
        # Setup diffusion schedule (shared across all classes)
        self.scheduler = SpectralCurriculumScheduler(
            n_phases=self.n_phases if self.use_spectral else 1,
            total_timesteps=self.total_timesteps
        )
        
        # Use cosine schedule
        t = np.arange(self.total_timesteps + 1) / self.total_timesteps
        s = 0.008
        alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta_schedule = np.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], 1e-4, 0.999)
        self._setup_diffusion(beta_schedule)

        # Train a separate denoiser for EACH minority class
        for class_label in minority_classes:
            X_c = X_train[y_train == class_label]
            n_c = len(X_c)
            
            if n_c < 2:
                if verbose:
                    print(f"  Skipping class {class_label}: only {n_c} samples")
                continue
                
            self._minority_means[int(class_label)] = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            self._minority_covs[int(class_label)] = cov
            
            if verbose:
                print(f"  Training denoiser for class {class_label} ({n_c} samples)...")

            # Compute boundary weights for this class
            boundary_weights = self._compute_boundary_weights(X_c, X_majority)
            weight_tensor = torch.tensor(boundary_weights, dtype=torch.float32, device=self.device)

            # Create UNCONDITIONAL denoiser for this class
            denoiser_c = MATDiffDenoiser(
                d_in=self.n_features,
                num_classes=0,  # UNCONDITIONAL - single class
                d_model=self.d_model,
                d_hidden=self.d_hidden,
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                dropout=self.dropout,
                use_curvature=False,
                use_geodesic=self.use_geodesic,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                denoiser_c.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            X_tensor = torch.tensor(X_c, dtype=torch.float32, device=self.device)

            ema_denoiser = copy.deepcopy(denoiser_c)
            ema_decay = 0.9999

            denoiser_c.train()
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
                    w_batch = weight_tensor[idx]

                    # Timestep sampling with curriculum
                    if self.use_spectral:
                        t = self.scheduler.sample_timesteps(len(x_batch), epoch, epochs, self.device)
                    else:
                        t = torch.randint(0, self.total_timesteps, (len(x_batch),), device=self.device)

                    noise = torch.randn_like(x_batch)
                    x_noisy = self._q_sample(x_batch, t, noise)

                    # Pass y=None (unconditional)
                    noise_pred = denoiser_c(x_noisy, t, y=None, curvature=None)

                    # Weighted MSE loss (BALW)
                    loss_per_sample = ((noise - noise_pred) ** 2).mean(dim=1)
                    loss = (loss_per_sample * w_batch).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(denoiser_c.parameters(), max_norm=1.0)
                    optimizer.step()

                    # EMA update
                    with torch.no_grad():
                        for p_ema, p_model in zip(ema_denoiser.parameters(), denoiser_c.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)

                    epoch_loss += loss.item()
                    n_batches += 1

                lr_scheduler.step()
                avg_loss = epoch_loss / max(1, n_batches)

                if avg_loss < best_loss - 1e-5:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(ema_denoiser.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience and epoch > epochs // 3:
                    break

            # Load best state
            if best_state is not None:
                denoiser_c.load_state_dict(best_state)
            else:
                denoiser_c.load_state_dict(ema_denoiser.state_dict())
                
            # Store trained denoiser
            self.denoisers[int(class_label)] = denoiser_c
            
            if verbose:
                print(f"    Final loss: {best_loss:.6f}")

        # Keep reference to last denoiser for backward compatibility
        if self.denoisers:
            self.denoiser = list(self.denoisers.values())[-1]
            
        return self

    @torch.no_grad()
    def _p_sample_step(self, x_t, t_idx, class_label, guidance_scale=1.0):
        """Single step for a specific class denoiser."""
        if int(class_label) not in self.denoisers:
            raise ValueError(f"No denoiser for class {class_label}")
            
        denoiser = self.denoisers[int(class_label)]
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        noise_pred = denoiser(x_t, t_tensor, y=None, curvature=None)

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
        WITH ANI (use_geodesic=True): SMALL covariance-aligned noise for diversity
        """
        if not self.use_geodesic:
            # Without ANI: NO noise (pure interpolation like SMOTE)
            return x_interpolated
        
        # WITH ANI: Add SMALL covariance-aligned noise (reduced from 0.15 to 0.05)
        if class_label in self._minority_covs:
            cov = self._minority_covs[class_label]
            try:
                # Eigendecomposition for principal directions
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-8)
                # REDUCED: Scale noise along principal directions (0.05 scale factor)
                noise_scale = np.sqrt(eigvals) * 0.05  # WAS 0.15, NOW 0.05
                noise_raw = np.random.randn(len(x_interpolated), len(eigvals))
                noise = (noise_raw * noise_scale) @ eigvecs.T
            except:
                # Fallback: isotropic noise scaled by feature std (reduced)
                noise = np.random.randn(*x_interpolated.shape) * 0.03  # WAS 0.08, NOW 0.03
        else:
            noise = np.random.randn(*x_interpolated.shape) * 0.03  # WAS 0.08, NOW 0.03
        
        return x_interpolated + noise

    def _distribution_matching_filter(self, X_syn, X_real, n_keep):
        """COMPONENT 3: Distribution Matching Filter (DMF).
        
        WITHOUT DMF (use_dmf=False): Random selection
        WITH DMF (use_dmf=True): Select diverse samples near real distribution
        """
        if len(X_syn) <= n_keep:
            return X_syn
        
        if not self.use_dmf:
            idx = np.random.choice(len(X_syn), n_keep, replace=False)
            return X_syn[idx]
        
        # WITH DMF: Use combination of distance AND diversity
        real_mean = X_real.mean(axis=0)
        real_cov = np.cov(X_real, rowvar=False)
        if real_cov.ndim == 0:
            real_cov = np.array([[real_cov]])
        
        # Regularize covariance heavily for stability
        real_cov = real_cov + np.eye(real_cov.shape[0]) * 0.01  # WAS 1e-6, NOW 0.01
        
        try:
            # Use pseudo-inverse for stability
            from numpy.linalg import pinv
            cov_inv = pinv(real_cov)
        except:
            cov_inv = np.diag(1.0 / (np.diag(real_cov) + 0.01))
        
        # Compute Mahalanobis distance
        mahal_dist = []
        for x in X_syn:
            diff = x - real_mean
            d = np.sqrt(diff @ cov_inv @ diff)
            mahal_dist.append(d)
        mahal_dist = np.array(mahal_dist)
        
        # GOOD samples are those WITHIN reasonable range (not too close, not too far)
        # Keep samples with distance between 25th and 75th percentile of real samples
        real_dists = []
        for x in X_real:
            diff = x - real_mean
            d = np.sqrt(diff @ cov_inv @ diff)
            real_dists.append(d)
        real_dists = np.array(real_dists)
        
        p25, p75 = np.percentile(real_dists, [25, 75])
        
        # Score: prefer samples within [p25, p75*1.5] range, penalize outliers
        scores = np.zeros(len(X_syn))
        in_range = (mahal_dist >= p25) & (mahal_dist <= p75 * 1.5)  # Slightly wider
        scores[in_range] = 1.0
        
        # Also penalize being too close (duplicates)
        too_close = mahal_dist < p25 * 0.5
        scores[too_close] = 0.3
        
        # Select top scoring, but ensure diversity with k-means
        n_candidates = min(n_keep * 3, len(X_syn))
        candidate_idx = np.argsort(scores)[-n_candidates:]
        
        if len(candidate_idx) <= n_keep:
            return X_syn[candidate_idx]
        
        # Final selection: k-means++ style diversity selection
        selected = [candidate_idx[0]]
        candidates = set(candidate_idx[1:])
        
        while len(selected) < n_keep and candidates:
            # Find furthest from already selected
            max_min_dist = -1
            best_idx = None
            for idx in candidates:
                min_dist = min([np.linalg.norm(X_syn[idx] - X_syn[s]) for s in selected])
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
            else:
                break
        
        return X_syn[selected]

    def sample(self, n_per_class=None):
        """Generate synthetic samples using hybrid SMOTE + diffusion refinement."""
        if not self.denoisers:
            raise RuntimeError("Call fit() before sample().")

        if n_per_class is None:
            class_counts = dict(zip(*np.unique(self.y_train, return_counts=True)))
            majority_count = max(class_counts.values())
            n_per_class = {}
            for c, cnt in class_counts.items():
                deficit = max(0, int(majority_count - cnt))
                if deficit > 0:
                    n_per_class[int(c)] = deficit

        all_X, all_y = [], []

        for class_label, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue

            if int(class_label) not in self.denoisers:
                print(f"  Warning: No denoiser for class {class_label}, skipping...")
                continue

            X_real_c = self.X_train[self.y_train == class_label]
            n_real = len(X_real_c)
            
            print(f"  Sampling class {class_label}: {n_needed} samples (real={n_real})...")

            # Generate 2x needed for DMF selection
            n_generate = min(n_needed * 2, 1000)
            
            # HYBRID: SMOTE base + light diffusion refinement
            X_base = self._smote_base(X_real_c, n_generate)
            X_refined = self._light_refinement(X_base, class_label)
            X_final = self._adaptive_noise_injection(X_refined, class_label)
            X_final = np.clip(X_final, 0.0, 1.0)
            
            # Apply DMF to select best samples
            X_syn_filtered = self._distribution_matching_filter(X_final, X_real_c, n_needed)
            
            if len(X_syn_filtered) > 0:
                all_X.append(X_syn_filtered)
                all_y.append(np.full(len(X_syn_filtered), class_label))

        if not all_X:
            return np.empty((0, self.n_features)), np.empty(0, dtype=int)

        return np.vstack(all_X), np.concatenate(all_y)

    def _smote_base(self, X_real, n_generate):
        """Generate base samples using SMOTE interpolation."""
        n_real = len(X_real)
        if n_real < 2:
            return np.tile(X_real, (n_generate // max(1, n_real) + 1, 1))[:n_generate]
        
        k = min(5, n_real - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X_real)
        _, indices = nn.kneighbors(X_real)
        
        synthetic = []
        for _ in range(n_generate):
            i = np.random.randint(0, n_real)
            j = indices[i, np.random.randint(1, k + 1)]
            
            # Random interpolation
            alpha = np.random.random()
            x_new = X_real[i] + alpha * (X_real[j] - X_real[i])
            synthetic.append(x_new)
        
        return np.array(synthetic)

    def _light_refinement(self, X_base, class_label, n_steps=10):
        """Multi-step diffusion refinement with better blend ratio."""
        if int(class_label) not in self.denoisers:
            return X_base
            
        try:
            denoiser = self.denoisers[int(class_label)]
            denoiser.eval()
            
            X_t = torch.tensor(X_base, dtype=torch.float32, device=self.device)
            
            # Start from moderate noise (not too high, not too low)
            t_start = min(50, self.total_timesteps // 4)  # t=50 for T=200
            X_current = X_t.clone()
            
            # Add initial noise
            noise = torch.randn_like(X_current) * 0.2
            t = torch.full((len(X_current),), t_start, dtype=torch.long, device=self.device)
            X_current = self._q_sample(X_current, t, noise)
            
            # Multi-step denoising (10 steps instead of 1)
            with torch.no_grad():
                for t_idx in reversed(range(0, t_start, max(1, t_start // n_steps))):
                    t_batch = torch.full((len(X_current),), t_idx, dtype=torch.long, device=self.device)
                    noise_pred = denoiser(X_current, t_batch, y=None)
                    
                    # DDPM update
                    alpha = self.alphas[t_idx]
                    alpha_bar = self.alphas_cumprod[t_idx]
                    beta = self.betas[t_idx]
                    
                    coef1 = 1.0 / torch.sqrt(alpha)
                    coef2 = beta / torch.sqrt(1.0 - alpha_bar)
                    mean = coef1 * (X_current - coef2 * noise_pred)
                    
                    if t_idx > 0:
                        noise = torch.randn_like(X_current) * 0.1  # Small noise
                        sigma = torch.sqrt(beta * 0.5)  # Reduced variance
                        X_current = mean + sigma * noise
                    else:
                        X_current = mean
            
            X_denoised = torch.clamp(X_current, 0.0, 1.0)
            
            # BETTER blend: 85% denoised, 15% original (trust diffusion more)
            X_blended = 0.85 * X_denoised.cpu().numpy() + 0.15 * X_base
            
            return np.clip(X_blended, 0.0, 1.0)
            
        except Exception as e:
            print(f"    Refinement failed: {e}")
            return X_base

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
        """Save all per-class denoisers."""
        denoiser_states = {k: v.state_dict() for k, v in self.denoisers.items()} if self.denoisers else {}
        torch.save({
            'denoiser_states': denoiser_states,
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
        
        # Reconstruct per-class denoisers
        self.denoisers = {}
        denoiser_states = ckpt.get('denoiser_states', {})
        for class_label, state in denoiser_states.items():
            denoiser_c = MATDiffDenoiser(
                d_in=self.n_features, num_classes=0,  # Unconditional
                d_model=self.d_model, d_hidden=self.d_hidden,
                n_blocks=self.n_blocks, n_heads=self.n_heads,
                use_curvature=False, use_geodesic=True,
            ).to(self.device)
            denoiser_c.load_state_dict(state)
            self.denoisers[int(class_label)] = denoiser_c
        
        # Backward compatibility
        if self.denoisers:
            self.denoiser = list(self.denoisers.values())[-1]
        
        self.fisher = ckpt.get('fisher')
        self.scheduler = ckpt.get('scheduler')
        
        if self.scheduler:
            beta_schedule = self.scheduler.get_full_beta_schedule()
            self._setup_diffusion(beta_schedule)








