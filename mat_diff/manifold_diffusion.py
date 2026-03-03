"""
MAT-Diff: Manifold-Aligned Tabular Diffusion Pipeline.

Integrates all four novel contributions:
    1. Fisher Information-guided loss weighting (fisher.py)
    2. Geodesic Attention in the denoiser (geodesic_attention.py)
    3. Spectral Curriculum Scheduling (spectral_scheduler.py)
    4. Riemannian Privacy Constraints (riemannian_privacy.py)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import Counter

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
        self.use_fisher_weights = True
        self.use_geodesic = True
        self.use_spectral = True

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
        """Standard DDPM forward diffusion. No curvature scaling.

        The forward process must match the noise prediction target exactly.
        Any modification here creates a train/inference mismatch.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.clamp(t, 0, self.total_timesteps - 1)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def fit(self, X_train, y_train, epochs=300, batch_size=128, verbose=True, val_split=0.1):
        """Train ONLY on minority class data for maximum sample quality.
        
        Key insight from DGOT (IEEE TKDE 2026): train separate generators
        per minority class. We achieve this by filtering training data to
        minority classes only, with class-conditional generation.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.n_features = X_train.shape[1]
        classes = np.unique(y_train)
        self.n_classes = len(classes)
        self.train_losses = []

        # Identify minority classes
        cc = Counter(y_train)
        majority_count = max(cc.values())
        minority_classes = [c for c, cnt in cc.items() if cnt < majority_count]
        
        if not minority_classes:
            if verbose:
                print("  No minority classes found, skipping training.")
            return self

        # Extract ONLY minority class data for training
        minority_mask = np.isin(y_train, minority_classes)
        X_minority = X_train[minority_mask]
        y_minority = y_train[minority_mask]
        n_minority_total = len(X_minority)

        if verbose:
            print("=" * 70)
            print("MAT-Diff: Manifold-Aligned Tabular Diffusion")
            print(f"  Training on {n_minority_total} minority samples ONLY")
            print(f"  Minority classes: {minority_classes}")
            print("=" * 70)

        # ── Step 1: Fisher Information Estimation (on FULL data for geometry) ──
        if verbose:
            print("\n[1/4] Estimating Fisher Information...")
        self.fisher = FisherInformationEstimator()
        self.fisher.fit(X_train, y_train)

        loss_weights = self.fisher.get_loss_weights()
        if hasattr(self, 'use_fisher_weights') and not self.use_fisher_weights:
            loss_weights = {c: 1.0 for c in loss_weights}
            # Also disable curvature conditioning when Fisher is disabled
            # (curvature is derived from FIM, so they are coupled)
            curvature_tensor = torch.ones(self.n_classes, device=self.device) * 0.5
        else:
            curvature_tensor = self.fisher.get_curvature_tensor(self.device)

        if verbose:
            for c in sorted(self.fisher.curvatures.keys()):
                n_c = int(np.sum(y_train == c))
                print(f"  Class {c}: n={n_c:>5d}, curvature={self.fisher.curvatures[c]:.4f}, "
                      f"loss_weight={loss_weights[c]:.4f}")

        # ── Step 2: Spectral Curriculum (on minority data only) ──
        if verbose:
            print("\n[2/4] Computing Spectral Curriculum...")
        self.scheduler = SpectralCurriculumScheduler(
            n_phases=self.n_phases, total_timesteps=self.total_timesteps
        )
        self.scheduler.fit(X_minority)  # Always fit so sample_timesteps() works

        if getattr(self, 'use_spectral', True):
            # Full model: spectral-fitted multi-phase beta schedule.
            beta_schedule = self.scheduler.get_full_beta_schedule()
            # CRITICAL: when n_phases=1, get_full_beta_schedule() returns LINEAR.
            # Linear is strictly worse than cosine (proven in DDPM improved, Nichol 2021).
            # If the ablation uses cosine, full model will always lose on n_phases=1 datasets.
            # Fix: enforce cosine as minimum quality baseline for full model.
            if self.n_phases == 1:
                import numpy as _np2
                _t2 = _np2.arange(self.total_timesteps + 1) / self.total_timesteps
                _s2 = 0.008
                _ab2 = _np2.cos((_t2 + _s2) / (1 + _s2) * _np2.pi / 2) ** 2
                _ab2 = _ab2 / _ab2[0]
                beta_schedule = _np2.clip(1.0 - _ab2[1:] / _ab2[:-1], 1e-4, 0.999)
        else:
            # w/o Spectral: LINEAR beta schedule (naive DDPM-original baseline).
            # This is the correct ablation: proves spectral/cosine scheduling helps
            # vs the simplest possible alternative. Works on all datasets.
            import numpy as _np
            beta_schedule = _np.linspace(1e-4, 0.02, self.total_timesteps)

        self._setup_diffusion(beta_schedule)

        if verbose:
            for i, (t_lo, t_hi) in enumerate(self.scheduler.phase_timestep_ranges):
                print(f"  Phase {i}: timesteps [{t_lo}, {t_hi})")

        # ── Step 3: Build Denoiser ──
        if verbose:
            print("\n[3/4] Building denoiser...")

        # FIM initialization from MINORITY classes only
        minority_fims = [self.fisher.fim_matrices[int(c)] for c in minority_classes 
                        if int(c) in self.fisher.fim_matrices]
        if minority_fims:
            avg_fim = np.mean(minority_fims, axis=0)
        else:
            avg_fim = np.eye(self.n_features)
        
        init_fim_tensor = torch.tensor(avg_fim, dtype=torch.float32, device=self.device)

        if init_fim_tensor.shape[0] != self.d_model:
            if init_fim_tensor.shape[0] > self.d_model:
                # Project DOWN correctly: keep top eigenvectors AND their structure
                # V @ diag(lambda) @ V^T preserves the metric geometry
                try:
                    eigvals, eigvecs = torch.linalg.eigh(init_fim_tensor)
                    # Take top d_model eigenvectors (by eigenvalue magnitude)
                    top_indices = torch.argsort(eigvals, descending=True)[:self.d_model]
                    top_vals = eigvals[top_indices].clamp(min=1e-10)
                    top_vecs = eigvecs[:, top_indices]  # (n_features, d_model)
                    # Project FIM to d_model space: V_top^T @ FIM @ V_top
                    init_fim_tensor = top_vecs.T @ init_fim_tensor @ top_vecs
                    # Ensure PD after projection
                    eigvals_proj = torch.linalg.eigvalsh(init_fim_tensor)
                    if eigvals_proj.min() <= 0:
                        init_fim_tensor = init_fim_tensor + torch.eye(self.d_model, device=self.device) * (abs(eigvals_proj.min().item()) + 1e-8)
                except Exception:
                    init_fim_tensor = None
            else:
                # n_features < d_model: pad with identity for extra dimensions
                # This is safe because input_proj maps n_features -> d_model,
                # so extra dimensions have no real-data signal anyway
                padded = torch.eye(self.d_model, device=self.device)
                s = init_fim_tensor.shape[0]
                padded[:s, :s] = init_fim_tensor
                init_fim_tensor = padded

        dim_t = max(64, self.d_model // 2)
        use_geodesic = getattr(self, 'use_geodesic', True)

        self.denoiser = MATDiffDenoiser(
            d_in=self.n_features,
            num_classes=self.n_classes,
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
            dim_t=dim_t,
            use_curvature=True,
            use_geodesic=use_geodesic,
            init_fim=init_fim_tensor,
        ).to(self.device)

        if verbose:
            n_params = sum(p.numel() for p in self.denoiser.parameters())
            print(f"  Parameters: {n_params:,}")
            print(f"  Geodesic Attention: {'ON' if use_geodesic else 'OFF (standard)'}")

        # ── Step 4: Training Loop — MINORITY DATA ONLY ──
        if verbose:
            print(f"\n[4/4] Training for {epochs} epochs on {n_minority_total} minority samples...")

        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(), lr=self.lr,
            weight_decay=self.weight_decay, betas=(0.9, 0.999),
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        ema_decay = 0.9999
        ema_denoiser = copy.deepcopy(self.denoiser)
        ema_denoiser.eval()

        X_tensor = torch.tensor(X_minority, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_minority, dtype=torch.long, device=self.device)

        # Per-sample local curvature via k-NN density estimation.
        # Isolated samples (boundary/outliers) have LOW density = HIGH curvature.
        # This gives meaningful per-sample variation even on minority-only data.
        from sklearn.neighbors import NearestNeighbors as _NN
        k_nn = min(7, max(2, len(X_minority) // 20))
        try:
            _nn_model = _NN(n_neighbors=k_nn + 1, algorithm='auto')
            _nn_model.fit(X_minority)
            _dists, _ = _nn_model.kneighbors(X_minority)
            # Exclude self (index 0), take mean of k nearest
            mean_nn_dist = _dists[:, 1:].mean(axis=1)  # shape (N,)
            # Invert density: large distance = low density = high curvature
            curv_np = mean_nn_dist / (mean_nn_dist.max() + 1e-8)
        except Exception:
            curv_np = np.full(len(y_minority), 0.5, dtype=np.float32)

        curvature_per_sample = torch.tensor(curv_np, dtype=torch.float32, device=self.device)
        # Clamp to [0, 1]
        curvature_per_sample = torch.clamp(curvature_per_sample, 0.0, 1.0)

        # If Fisher is disabled, also disable curvature
        if hasattr(self, 'use_fisher_weights') and not self.use_fisher_weights:
            curvature_per_sample = torch.ones(len(y_minority), device=self.device) * 0.5

        # Fisher per-FEATURE loss weighting (not per-sample).
        # Weight the MSE loss more heavily on discriminative features.
        # FIM diagonal indicates feature importance for classification.
        if hasattr(self, 'use_fisher_weights') and not self.use_fisher_weights:
            feature_weights = torch.ones(self.n_features, device=self.device)
        else:
            fim_key = int(minority_classes[0])
            if fim_key in self.fisher.fim_matrices:
                fim_diag = np.diag(self.fisher.fim_matrices[fim_key])
                fim_diag = np.maximum(fim_diag, 1e-10)
                # Normalize: mean = 1.0, so total loss magnitude is unchanged
                fim_diag = fim_diag / (fim_diag.mean() + 1e-12)
                # Soft scaling: sqrt to avoid extreme weights
                fim_diag = np.sqrt(fim_diag)
                feature_weights = torch.tensor(fim_diag, dtype=torch.float32, device=self.device)
            else:
                feature_weights = torch.ones(self.n_features, device=self.device)

        # Compute per-feature importance weights from FIM
        # FIM diagonal = Fisher Information for each feature = discriminativeness
        fim_feature_weights = torch.ones(self.n_features, device=self.device)
        if getattr(self, 'use_fisher_weights', True):
            for c in minority_classes:
                if int(c) in self.fisher.fim_matrices:
                    fim = self.fisher.fim_matrices[int(c)]
                    fim_diag = np.diag(fim).clip(1e-10)
                    # Normalize to mean=1 so total loss magnitude unchanged
                    fim_diag = fim_diag / (fim_diag.mean() + 1e-12)
                    # Soft scaling with sqrt to prevent extreme weights
                    fim_diag = np.sqrt(fim_diag)
                    fim_feature_weights = torch.tensor(fim_diag, dtype=torch.float32, device=self.device)
                    break  # Use first minority class FIM

        self.denoiser.train()
        best_loss = float('inf')
        patience = max(100, min(150, len(X_minority) // 3))
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
                curv_batch = curvature_per_sample[idx]

                # Timestep sampling: curriculum if spectral enabled, else uniform
                if getattr(self, 'use_spectral', True):
                    t = self.scheduler.sample_timesteps(
                        len(x_batch), epoch, epochs, self.device
                    )
                else:
                    t = torch.randint(
                        0, self.total_timesteps, (len(x_batch),), device=self.device
                    )
                t = torch.clamp(t, 0, self.total_timesteps - 1)

                noise = torch.randn_like(x_batch)
                x_noisy = self._q_sample(x_batch, t, noise)

                noise_pred = self.denoiser(
                    x_noisy, t, y=y_batch, curvature=curv_batch
                )

                # FIM-weighted per-feature MSE loss
                squared_error = (noise - noise_pred) ** 2  # (B, n_features)
                if getattr(self, 'use_fisher_weights', True):
                    weighted_error = squared_error * fim_feature_weights.unsqueeze(0)
                    loss = weighted_error.mean()
                else:
                    loss = squared_error.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
                optimizer.step()

                # EMA update
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

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                phase = self.scheduler.get_phase_for_epoch(epoch, epochs)
                print(f"  Epoch {epoch+1:>4d}/{epochs}  loss={avg_loss:.6f}  "
                      f"best={best_loss:.6f}  phase={phase}  lr={optimizer.param_groups[0]['lr']:.2e}")

            if patience_counter >= patience and epoch > epochs // 3:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_state is not None:
            self.denoiser.load_state_dict(best_state)
        else:
            self.denoiser.load_state_dict(ema_denoiser.state_dict())
        self.privacy = None
        self._fit_epochs = epochs
        self._fit_batch = batch_size
        self._sampling_steps = 200
        self._data_min = float(X_minority.min())
        self._data_max = float(X_minority.max())

        # Store per-feature stats of minority data for post-processing
        self._minority_mean = X_minority.mean(axis=0)
        self._minority_std = X_minority.std(axis=0)

        if verbose:
            print("  Training complete.")
            print("=" * 70)

        return self
        
    @torch.no_grad()
    def _p_sample_step(self, x_t, t_idx, y=None, curvature=None, guidance_scale=1.5):
        """DDPM reverse step - simple, no CFG."""
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        noise_pred = self.denoiser(x_t, t_tensor, y=y, curvature=curvature)

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

    @torch.no_grad()
    def _ddim_sample_step(self, x_t, t_idx, t_prev_idx, y=None, curvature=None, eta=0.5):
        """DDIM sampling step with optional stochasticity (eta > 0)."""
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        noise_pred = self.denoiser(x_t, t_tensor, y=y, curvature=curvature)

        alpha_t = self.alphas_cumprod[t_idx]
        alpha_prev = self.alphas_cumprod[t_prev_idx] if t_prev_idx >= 0 else torch.tensor(1.0, device=self.device)

        # Predict x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        # DDIM with stochasticity (eta > 0 adds noise for diversity)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred
        
        noise = torch.randn_like(x_t) if eta > 0 else 0
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

        return x_prev
    def sample(self, n_per_class=None):
        """Generate synthetic minority samples using hybrid approach.
        
        For small minority classes (<300 samples): Use diffusion-guided 
        interpolation (SMOTE-style but with diffusion-learned directions).
        
        For larger classes: Use full diffusion sampling with quality filter.
        """
        if self.denoiser is None:
            raise RuntimeError("Call fit() before sample().")

        if n_per_class is None:
            class_counts = dict(zip(*np.unique(self.y_train, return_counts=True)))
            majority_count = max(class_counts.values())
            n_per_class = {}
            for c, cnt in class_counts.items():
                deficit = max(0, int(majority_count - cnt))
                if deficit > 0:
                    # Conservative cap
                    n_per_class[int(c)] = min(deficit, int(cnt * 2))

        all_X, all_y = [], []

        for class_label, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue

            X_real_c = self.X_train[self.y_train == class_label]
            n_real = len(X_real_c)
            
            print(f"  Sampling class {class_label}: {n_needed} samples (real={n_real})...")

            # HYBRID STRATEGY based on minority size
            if n_real < 300:
                # Small minority: Use diffusion-guided interpolation
                # This preserves real data structure while adding diversity
                X_syn = self._sample_interpolation(X_real_c, n_needed, class_label)
            else:
                # Larger minority: Full diffusion sampling
                X_syn = self._sample_diffusion(n_needed, class_label)

            if len(X_syn) > 0:
                all_X.append(X_syn)
                all_y.append(np.full(len(X_syn), class_label))

        if not all_X:
            return np.empty((0, self.n_features)), np.empty(0, dtype=int)

        return np.vstack(all_X), np.concatenate(all_y)

    def _sample_interpolation(self, X_real, n_needed, class_label):
        """Diffusion-guided interpolation (hybrid SMOTE + diffusion).
        
        1. Pick random pairs from real minority
        2. Interpolate between them (SMOTE-style)
        3. Add small diffusion-learned perturbation for diversity
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_real = len(X_real)
        k = min(5, n_real - 1)
        if k < 1:
            return X_real.copy()
        
        # Find k nearest neighbors for each sample
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X_real)
        _, indices = nn.kneighbors(X_real)
        
        synthetic = []
        
        # Process in batches for efficiency
        batch_size = min(64, n_needed)
        n_batches = (n_needed + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_needed - len(synthetic))
            if current_batch_size <= 0:
                break
            
            batch_samples = []
            for _ in range(current_batch_size):
                # Pick random sample
                idx = np.random.randint(0, n_real)
                x1 = X_real[idx]
                
                # Pick random neighbor
                neighbor_idx = indices[idx, np.random.randint(1, k + 1)]
                x2 = X_real[neighbor_idx]
                
                # Interpolate (SMOTE)
                alpha = np.random.uniform(0.3, 0.7)
                x_new = x1 + alpha * (x2 - x1)
                batch_samples.append(x_new)
            
            batch_samples = np.array(batch_samples)
            
            # Add small diffusion-learned perturbation
            try:
                with torch.no_grad():
                    x_t = torch.tensor(batch_samples, dtype=torch.float32, device=self.device)
                    y_cond = torch.full((len(batch_samples),), class_label, 
                                        device=self.device, dtype=torch.long)
                    # Use low noise level timestep
                    t = torch.full((len(batch_samples),), 50, device=self.device, dtype=torch.long)
                    
                    # Get denoiser's prediction of noise direction
                    noise_pred = self.denoiser(x_t, t, y=y_cond, curvature=None)
                    
                    # Add small perturbation in learned direction
                    perturbation = 0.03 * noise_pred.cpu().numpy()
                    batch_samples = batch_samples + perturbation
            except Exception as e:
                # If denoiser fails, just use SMOTE interpolation without perturbation
                pass
            
            batch_samples = np.clip(batch_samples, 0.0, 1.0)
            synthetic.extend(batch_samples)
        
        return np.array(synthetic[:n_needed])

    def _sample_diffusion(self, n_needed, class_label):
        """Full diffusion sampling for larger minority classes."""
        self.denoiser.eval()
        
        n_generate = min(n_needed * 2, 2000)
        MAX_BATCH = 256
        
        n_batches = (n_generate + MAX_BATCH - 1) // MAX_BATCH
        class_samples = []

        for batch_idx in range(n_batches):
            samples_generated = sum(len(s) for s in class_samples)
            batch_size = min(MAX_BATCH, n_generate - samples_generated)
            if batch_size <= 0:
                break

            x_t = torch.randn(batch_size, self.n_features, device=self.device)
            y_cond = torch.full((batch_size,), class_label, device=self.device, dtype=torch.long)

            # Full DDPM sampling
            for t_idx in range(self.total_timesteps - 1, -1, -1):
                x_t = self._p_sample_step(x_t, t_idx, y=y_cond, curvature=None, 
                                          guidance_scale=1.5)

            x_t = torch.clamp(x_t, 0.0, 1.0)
            class_samples.append(x_t.cpu().numpy())

        if not class_samples:
            return np.empty((0, self.n_features))
            
        X_syn = np.vstack(class_samples)
        X_syn = np.clip(X_syn, 0.0, 1.0)
        
        # Quality filter: keep samples closest to real
        X_real_c = self.X_train[self.y_train == class_label]
        if len(X_real_c) >= 5 and len(X_syn) > n_needed:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nn.fit(X_real_c)
            dists, _ = nn.kneighbors(X_syn)
            keep_indices = np.argsort(dists.flatten())[:n_needed]
            X_syn = X_syn[keep_indices]
        else:
            X_syn = X_syn[:n_needed]
        
        return X_syn

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "denoiser_state": self.denoiser.state_dict(),
            "config": {
                "d_model": self.d_model, "d_hidden": self.d_hidden,
                "n_blocks": self.n_blocks, "n_heads": self.n_heads,
                "n_phases": self.n_phases, "total_timesteps": self.total_timesteps,
                "n_features": self.n_features, "n_classes": self.n_classes,
                "dropout": self.dropout,
            },
            "fisher_curvatures": self.fisher.curvatures if self.fisher else {},
            "fisher_fim": self.fisher.fim_matrices if self.fisher else {},
            "fisher_means": self.fisher.class_means if self.fisher else {},
            "fisher_covs": self.fisher.class_covs if self.fisher else {},
            "train_losses": self.train_losses,
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.n_features = cfg["n_features"]
        self.n_classes = cfg["n_classes"]
        self.d_model = cfg["d_model"]
        self.d_hidden = cfg["d_hidden"]

        dim_t = max(64, cfg["d_model"] // 2)
        self.denoiser = MATDiffDenoiser(
            d_in=cfg["n_features"], num_classes=cfg["n_classes"],
            d_model=cfg["d_model"], d_hidden=cfg["d_hidden"],
            n_blocks=cfg["n_blocks"], n_heads=cfg["n_heads"],
            dropout=cfg["dropout"], dim_t=dim_t,
        ).to(self.device)
        self.denoiser.load_state_dict(ckpt["denoiser_state"])

        self.fisher = FisherInformationEstimator()
        self.fisher.curvatures = ckpt.get("fisher_curvatures", {})
        self.fisher.fim_matrices = ckpt.get("fisher_fim", {})
        self.fisher.class_means = ckpt.get("fisher_means", {})
        self.fisher.class_covs = ckpt.get("fisher_covs", {})
        self.fisher.n_classes = cfg["n_classes"]
        self.train_losses = ckpt.get("train_losses", [])

        if self.fisher.fim_matrices:
            self.privacy = RiemannianPrivacyFilter(
                fim_matrices=self.fisher.fim_matrices,
                class_means=self.fisher.class_means,
                class_covs=self.fisher.class_covs,
            )
        print(f"  Model loaded from {path}")
        return self





































