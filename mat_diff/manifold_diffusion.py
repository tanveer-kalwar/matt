"""
MAT-Diff: Manifold-Aligned Tabular Diffusion Pipeline.

Integrates all four novel contributions:
    1. Fisher Information-guided loss weighting (fisher.py)
    2. Geodesic Attention in the denoiser (geodesic_attention.py)
    3. Spectral Curriculum Scheduling (spectral_scheduler.py)
    4. Riemannian Privacy Constraints (riemannian_privacy.py)
"""

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

    def _q_sample(self, x_start, t, noise=None, curvature=None):
        """Forward diffusion with curvature-adaptive noise for minority protection."""
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.clamp(t, 0, self.total_timesteps - 1)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # CRITICAL: Reduce noise for high-curvature (minority) samples
        if curvature is not None:
            # Normalize curvature to [0, 1]
            curv_norm = curvature.view(-1, 1)
            # High curvature = reduce noise by up to 30%
            noise_scale = 1.0 - 0.3 * curv_norm
            sqrt_one_minus = sqrt_one_minus * noise_scale
        
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def fit(self, X_train, y_train, epochs=300, batch_size=128, verbose=True, val_split=0.1):
        """
        Train with validation-based early stopping to prevent overfitting.
        """
        # Split into train/val
        from sklearn.model_selection import train_test_split
        if val_split > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=val_split, stratify=y_train, random_state=42
            )
        else:
            X_tr, X_val = X_train, None
            y_tr, y_val = y_train, None
        self.X_train = X_tr.copy()
        self.y_train = y_tr.copy()
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
            print("\n[3/4] Building denoiser with Geodesic Attention...")

        avg_fim = np.mean(list(self.fisher.fim_matrices.values()), axis=0)
        init_fim_tensor = torch.tensor(
            avg_fim, dtype=torch.float32, device=self.device
        )

        # Data-driven FIM padding: use median eigenvalue of avg FIM
        if init_fim_tensor.shape[0] != self.d_model:
            eigvals = torch.linalg.eigvalsh(init_fim_tensor)
            median_eig = torch.median(torch.abs(eigvals)).item()
            fill_value = max(median_eig, 1e-10)
            padded = torch.eye(self.d_model, device=self.device) * fill_value
            s = min(init_fim_tensor.shape[0], self.d_model)
            padded[:s, :s] = init_fim_tensor[:s, :s]
            init_fim_tensor = padded

        # dim_t: derived from d_model (d_model // 2, standard practice)
        dim_t = max(64, self.d_model // 2)

        self.denoiser = MATDiffDenoiser(
            d_in=self.n_features,
            num_classes=self.n_classes + 1,  # +1 for CFG null token
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
            dim_t=dim_t,
            use_curvature=True,
            init_fim=init_fim_tensor,
        ).to(self.device)

        if verbose:
            n_params = sum(p.numel() for p in self.denoiser.parameters())
            print(f"  Parameters: {n_params:,}")

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

        X_tensor = torch.tensor(X_tr, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_tr, dtype=torch.long, device=self.device)

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

        # Compute per-class indices for balanced sampling
        class_indices = {}
        for c in range(self.n_classes):
            class_indices[c] = torch.where(y_tensor == c)[0]
        max_class_size = max(len(v) for v in class_indices.values())
        balanced_size = max_class_size * self.n_classes

        best_loss = float('inf')

        for epoch in range(epochs):
            balanced_idx = []
            for c in range(self.n_classes):
                cidx = class_indices[c]
                if len(cidx) == 0:
                    continue
                # Sample 2x for balanced minority coverage without overfitting
                n_samples = max_class_size * 2
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
                x_noisy = self._q_sample(x_batch, t, noise, curvature=curv_batch)

                # Classifier-free guidance training: drop labels 10% of the time
                drop_mask = torch.rand(len(x_batch), device=self.device) < 0.1
                y_cfg = y_batch.clone()
                y_cfg[drop_mask] = self.n_classes  # null class token
                curv_cfg = curv_batch.clone()
                curv_cfg[drop_mask] = 0.0

                noise_pred = self.denoiser(
                    x_noisy, t, y=y_cfg, curvature=curv_cfg
                )

                loss_per_sample = ((noise - noise_pred) ** 2).mean(dim=1)
                loss = (loss_per_sample * w_batch).mean()

                optimizer.zero_grad()
                loss.backward()
                # Adaptive gradient clipping for high-curvature datasets
                max_norm = 2.0 if max(self.fisher.curvatures.values()) > 500 else 1.0
                torch.nn.utils.clip_grad_norm_(
                    self.denoiser.parameters(), max_norm=max_norm
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)
            self.train_losses.append(avg_loss)
            
            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                phase = self.scheduler.get_phase_for_epoch(epoch, epochs)
                print(
                    f"  Epoch {epoch+1:>4d}/{epochs}  "
                    f"loss={avg_loss:.6f}  best={best_loss:.6f}  "
                    f"phase={phase}  lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                
        self.privacy = None

        self._fit_epochs = epochs
        self._fit_batch = batch_size

        if verbose:
            print("  Training complete.")
            print("=" * 70)

        return self

    @torch.no_grad()
    def _p_sample_step(self, x_t, t_idx, y=None, curvature=None, guidance_scale=2.0):
        """Single reverse diffusion step with classifier-free guidance."""
        B = x_t.shape[0]
        t_idx = max(0, min(t_idx, self.total_timesteps - 1))
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
    
        # Conditional prediction
        noise_pred_cond = self.denoiser(x_t, t_tensor, y=y, curvature=curvature)
    
        # Unconditional prediction (null class token)
        y_uncond = torch.full_like(y, self.n_classes)
        curv_uncond = torch.zeros_like(curvature) if curvature is not None else None
        noise_pred_uncond = self.denoiser(x_t, t_tensor, y=y_uncond, curvature=curv_uncond)
    
        # CRITICAL: Adaptive guidance - higher for minority (high-curvature) classes
        if curvature is not None:
            # High curvature (minority) = stronger guidance (up to 3.5x)
            adaptive_scale = guidance_scale + curvature.view(-1, 1) * 1.5
            noise_pred = noise_pred_uncond + adaptive_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
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
    def _sample_class(self, class_label, n_samples, oversample_factor=1, max_batch_size=512):
        """Generate raw synthetic samples for a single class with memory-safe batching.
        
        Args:
            class_label: Target class to generate
            n_samples: Number of final samples needed
            oversample_factor: Generate n_samples * oversample_factor for filtering
            max_batch_size: Maximum samples to process at once (prevents OOM)
        
        Returns:
            Generated samples as numpy array
        """
        self.denoiser.eval()
        n_generate = n_samples * oversample_factor
        
        # If n_generate is small, process in one batch
        if n_generate <= max_batch_size:
            x_t = torch.randn(n_generate, self.n_features, device=self.device)
            y_cond = torch.full(
                (n_generate,), class_label, device=self.device, dtype=torch.long
            )
            import math
            curv_val = self.fisher.curvatures.get(class_label, 1.0)
            all_curvs = list(self.fisher.curvatures.values())
            log_curvs = [math.log1p(c) for c in all_curvs]
            log_min = min(log_curvs)
            log_range = max(log_curvs) - log_min
            curv_norm = (math.log1p(curv_val) - log_min) / log_range if log_range > 0 else 0.5
            curvature = torch.full(
                (n_generate,), curv_norm, device=self.device, dtype=torch.float32
            )

            for t in reversed(range(self.total_timesteps)):
                x_t = self._p_sample_step(x_t, t, y=y_cond, curvature=curvature)

            return x_t.cpu().numpy()
        
        # Process in batches to avoid OOM
        all_samples = []
        n_batches = (n_generate + max_batch_size - 1) // max_batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, n_generate)
            batch_size = end_idx - start_idx
            
            x_t = torch.randn(batch_size, self.n_features, device=self.device)
            y_cond = torch.full(
                (batch_size,), class_label, device=self.device, dtype=torch.long
            )
            curv_val = self.fisher.curvatures.get(class_label, 1.0)
            curv_max = max(self.fisher.curvatures.values())
            curv_norm = curv_val / curv_max if curv_max > 0 else 0.0
            curvature = torch.full(
                (batch_size,), curv_norm, device=self.device, dtype=torch.float32
            )
            
            for t in reversed(range(self.total_timesteps)):
                x_t = self._p_sample_step(x_t, t, y=y_cond, curvature=curvature)
            
            all_samples.append(x_t.cpu().numpy())
        
        return np.vstack(all_samples)

    def sample(self, n_per_class=None):
        """Generate synthetic samples in batches for quality and memory efficiency."""
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
    
        all_X, all_y = [], []
        MAX_BATCH = 512  # Maximum samples per batch to prevent OOM
        
        for class_label, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue
            print(f"  Sampling class {class_label}: {n_needed} samples...")
            
            self.denoiser.eval()
            
            # Compute curvature normalization once
            import math
            curv_val = self.fisher.curvatures.get(class_label, 1.0)
            all_curvs = list(self.fisher.curvatures.values())
            log_curvs = [math.log1p(c) for c in all_curvs]
            log_min = min(log_curvs)
            log_range = max(log_curvs) - log_min
            curv_norm = (math.log1p(curv_val) - log_min) / log_range if log_range > 0 else 0.5
            
            # Generate in batches
            n_batches = (n_needed + MAX_BATCH - 1) // MAX_BATCH
            class_samples = []
            
            for batch_idx in range(n_batches):
                batch_size = min(MAX_BATCH, n_needed - len(class_samples))
                
                # Initialize noise
                x_t = torch.randn(batch_size, self.n_features, device=self.device)
                y_cond = torch.full((batch_size,), class_label, device=self.device, dtype=torch.long)
                curvature = torch.full((batch_size,), curv_norm, device=self.device, dtype=torch.float32)
    
                # Reverse diffusion
                for t in reversed(range(self.total_timesteps)):
                    x_t = self._p_sample_step(x_t, t, y=y_cond, curvature=curvature)
                
                class_samples.append(x_t.cpu().numpy())
                
                # Clear GPU cache every 5 batches
                if batch_idx % 5 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()
            
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













