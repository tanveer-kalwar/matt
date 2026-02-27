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
        """Train with focus on minority class quality."""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        X_tr = X_train
        y_tr = y_train
        self.n_features = X_tr.shape[1]
        classes = np.unique(y_tr)
        self.n_classes = len(classes)
        self.train_losses = []

        if verbose:
            print("=" * 70)
            print("MAT-Diff: Manifold-Aligned Tabular Diffusion")
            print("=" * 70)

        # ── Step 1: Fisher Information Estimation ──
        if verbose:
            print("\n[1/4] Estimating Fisher Information...")
        self.fisher = FisherInformationEstimator()
        self.fisher.fit(X_tr, y_tr)

        loss_weights = self.fisher.get_loss_weights()
        if hasattr(self, 'use_fisher_weights') and not self.use_fisher_weights:
            loss_weights = {c: 1.0 for c in loss_weights}
        curvature_tensor = self.fisher.get_curvature_tensor(self.device)

        if verbose:
            for c in sorted(self.fisher.curvatures.keys()):
                n_c = int(np.sum(y_tr == c))
                print(
                    f"  Class {c}: n={n_c:>5d}, "
                    f"curvature={self.fisher.curvatures[c]:.4f}, "
                    f"loss_weight={loss_weights[c]:.4f}"
                )

        # ── Step 2: Spectral Curriculum Scheduling ──
        if verbose:
            print("\n[2/4] Computing Spectral Curriculum...")
        self.scheduler = SpectralCurriculumScheduler(
            n_phases=self.n_phases, total_timesteps=self.total_timesteps
        )
        self.scheduler.fit(X_tr)

        beta_schedule = self.scheduler.get_full_beta_schedule()
        self._setup_diffusion(beta_schedule)

        if verbose:
            for i, (t_lo, t_hi) in enumerate(self.scheduler.phase_timestep_ranges):
                print(f"  Phase {i}: timesteps [{t_lo}, {t_hi})")

        # ── Step 3: Build Denoiser ──
        if verbose:
            print("\n[3/4] Building denoiser...")

        avg_fim = np.mean(list(self.fisher.fim_matrices.values()), axis=0)
        init_fim_tensor = torch.tensor(
            avg_fim, dtype=torch.float32, device=self.device
        )

        if init_fim_tensor.shape[0] != self.d_model:
            eigvals = torch.linalg.eigvalsh(init_fim_tensor)
            median_eig = torch.median(torch.abs(eigvals)).item()
            fill_value = max(median_eig, 1e-10)
            padded = torch.eye(self.d_model, device=self.device) * fill_value
            s = min(init_fim_tensor.shape[0], self.d_model)
            padded[:s, :s] = init_fim_tensor[:s, :s]
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

        # ── Step 4: Training Loop ──
        if verbose:
            print(f"\n[4/4] Training for {epochs} epochs...")

        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        # EMA for stable sampling
        ema_decay = 0.9999
        ema_denoiser = copy.deepcopy(self.denoiser)
        ema_denoiser.eval()

        X_tensor = torch.tensor(X_tr, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_tr, dtype=torch.long, device=self.device)

        # Compute per-sample curvature (for conditioning)
        curvature_per_sample = torch.zeros(len(y_tr), device=self.device)
        for i, label in enumerate(y_tr):
            curvature_per_sample[i] = curvature_tensor[int(label)]
        if curvature_per_sample.max() > 0:
            curvature_per_sample = torch.log1p(curvature_per_sample)
            curv_min = curvature_per_sample.min()
            curv_range = curvature_per_sample.max() - curv_min
            if curv_range > 0:
                curvature_per_sample = (curvature_per_sample - curv_min) / curv_range

        weight_per_sample = torch.ones(len(y_tr), device=self.device)
        for i, label in enumerate(y_tr):
            weight_per_sample[i] = loss_weights[int(label)]

        self.denoiser.train()

        # Balanced sampling: each class seen equally per epoch
        # NO minority boost — it causes overfitting on small minority sets
        unique_labels = torch.unique(y_tensor).tolist()
        class_indices = {}
        for c in unique_labels:
            class_indices[int(c)] = torch.where(y_tensor == c)[0]
        
        class_counts = {int(c): len(v) for c, v in class_indices.items()}
        max_class_size = max(class_counts.values())
        
        # Equal representation per class (standard balanced sampling)
        samples_per_class = {c: max_class_size for c in class_counts}

        best_loss = float('inf')
        patience = 80
        patience_counter = 0

        for epoch in range(epochs):
            balanced_idx = []
            for c in unique_labels:
                cidx = class_indices[int(c)]
                if len(cidx) == 0:
                    continue
                n_samples = samples_per_class[int(c)]
                sampled = cidx[torch.randint(0, len(cidx), (n_samples,), device=self.device)]
                balanced_idx.append(sampled)
            balanced_idx = torch.cat(balanced_idx)
            perm = balanced_idx[torch.randperm(len(balanced_idx), device=self.device)]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                end = min(start + batch_size, len(perm))
                idx = perm[start:end]
                x_batch = X_tensor[idx]
                y_batch = y_tensor[idx]
                curv_batch = curvature_per_sample[idx]
                w_batch = weight_per_sample[idx]

                t = self.scheduler.sample_timesteps(
                    len(x_batch), epoch, epochs, self.device
                )
                t = torch.clamp(t, 0, self.total_timesteps - 1)

                noise = torch.randn_like(x_batch)
                x_noisy = self._q_sample(x_batch, t, noise)

                # NO CFG dropping — always condition on class
                # CFG requires large datasets to learn good unconditional model.
                # With tiny minority classes, unconditional model is dominated by majority,
                # and guidance amplifies majority-biased noise.
                y_cfg = y_batch.clone()
                curv_cfg = curv_batch.clone()

                noise_pred = self.denoiser(
                    x_noisy, t, y=y_cfg, curvature=curv_cfg
                )

                # Weighted MSE loss
                loss_per_sample = ((noise - noise_pred) ** 2).mean(dim=1)
                loss = (loss_per_sample * w_batch).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.denoiser.parameters(), max_norm=1.0
                )
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
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                phase = self.scheduler.get_phase_for_epoch(epoch, epochs)
                print(
                    f"  Epoch {epoch+1:>4d}/{epochs}  "
                    f"loss={avg_loss:.6f}  best={best_loss:.6f}  "
                    f"phase={phase}  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if patience_counter >= patience and epoch > epochs // 3:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Use EMA weights
        self.denoiser.load_state_dict(ema_denoiser.state_dict())
        self.privacy = None
        self._fit_epochs = epochs
        self._fit_batch = batch_size
        self._sampling_steps = 200

        # Store data statistics for proper sample clipping
        self._data_min = float(X_tr.min())
        self._data_max = float(X_tr.max())

        if verbose:
            print("  Training complete.")
            print("=" * 70)

        return self
        
    @torch.no_grad()
    def _p_sample_step(self, x_t, t_idx, y=None, curvature=None, guidance_scale=1.5):
        """Single reverse diffusion step with classifier-free guidance.

        guidance_scale=1.5 is the standard for conditional diffusion (TabDDPM uses 1-3).
        Fixed scale, NOT adaptive. Adaptive scaling caused instability.
        """
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        # Pure conditional prediction — no CFG
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
    def _ddim_sample_step(self, x_t, t_idx, t_prev_idx, y=None, curvature=None, guidance_scale=1.5):
        """DDIM deterministic sampling step."""
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        # Pure conditional prediction — no CFG
        noise_pred = self.denoiser(x_t, t_tensor, y=y, curvature=curvature)

        # DDIM update
        alpha_t = self.alphas_cumprod[t_idx]
        alpha_prev = self.alphas_cumprod[t_prev_idx] if t_prev_idx >= 0 else torch.tensor(1.0, device=self.device)

        # Predict x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Clamp x0 to training data range — NO margin
        # QuantileTransformer maps to [0, 1], samples must stay in [0, 1]
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred

        # DDIM deterministic step
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        return x_prev

    def sample(self, n_per_class=None):
        """Generate synthetic minority samples using DDIM."""
        if self.denoiser is None:
            raise RuntimeError("Call fit() before sample().")

        if n_per_class is None:
            class_counts = dict(zip(*np.unique(self.y_train, return_counts=True)))
            majority_count = max(class_counts.values())
            n_per_class = {
                int(c): max(0, int(majority_count - cnt))
                for c, cnt in class_counts.items()
            }
            if self.fisher is not None:
                total_deficit = sum(n_per_class.values())
                alloc = self.fisher.get_augmentation_allocation(
                    total_deficit, {int(k): int(v) for k, v in class_counts.items()}
                )
                for c in n_per_class:
                    if c in alloc:
                        deficit = int(majority_count - class_counts.get(c, 0))
                        n_per_class[c] = max(alloc[c], deficit)

        sampling_steps = getattr(self, '_sampling_steps', 200)
        step_size = max(1, self.total_timesteps // sampling_steps)
        ddim_timesteps = list(range(0, self.total_timesteps, step_size))
        if ddim_timesteps[-1] != self.total_timesteps - 1:
            ddim_timesteps.append(self.total_timesteps - 1)
        ddim_timesteps = sorted(ddim_timesteps)

        all_X, all_y = [], []
        MAX_BATCH = 512

        # Data range for clipping
        data_min = getattr(self, '_data_min', 0.0)
        data_max = getattr(self, '_data_max', 1.0)

        for class_label, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue
            print(f"  Sampling class {class_label}: {n_needed} samples...")

            self.denoiser.eval()

            # Curvature normalization
            import math as _math
            curv_val = self.fisher.curvatures.get(class_label, 1.0)
            all_curvs = list(self.fisher.curvatures.values())
            log_curvs = [_math.log1p(c) for c in all_curvs]
            log_min = min(log_curvs)
            log_range = max(log_curvs) - log_min
            curv_norm = (_math.log1p(curv_val) - log_min) / log_range if log_range > 0 else 0.5

            # Fixed guidance = 1.0 for ALL classes
            # Higher guidance amplifies noise in the conditional prediction,
            # which is catastrophic for minority classes with limited training data.
            # DGOT and TabDDPM do NOT use classifier-free guidance at all.
            guidance = 1.0

            n_batches = (n_needed + MAX_BATCH - 1) // MAX_BATCH
            class_samples = []

            for batch_idx in range(n_batches):
                samples_generated = sum(len(s) for s in class_samples)
                batch_size = min(MAX_BATCH, n_needed - samples_generated)
                if batch_size <= 0:
                    break

                x_t = torch.randn(batch_size, self.n_features, device=self.device)
                y_cond = torch.full((batch_size,), class_label, device=self.device, dtype=torch.long)
                curvature = torch.full((batch_size,), curv_norm, device=self.device, dtype=torch.float32)

                # DDIM reverse diffusion
                for i in reversed(range(len(ddim_timesteps))):
                    t = ddim_timesteps[i]
                    t_prev = ddim_timesteps[i - 1] if i > 0 else -1
                    x_t = self._ddim_sample_step(x_t, t, t_prev, y=y_cond, 
                                                  curvature=curvature, guidance_scale=guidance)

                # Clamp to [0, 1] — QuantileTransformer range
                x_t = torch.clamp(x_t, 0.0, 1.0)
                class_samples.append(x_t.cpu().numpy())

            X_syn = np.vstack(class_samples)
            all_X.append(X_syn)
            all_y.append(np.full(len(X_syn), class_label))

        if not all_X:
            return np.empty((0, self.n_features)), np.empty(0, dtype=int)

        return np.vstack(all_X), np.concatenate(all_y)

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







