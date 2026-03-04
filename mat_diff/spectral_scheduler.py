"""
Spectral Curriculum Scheduling for Tabular Diffusion.

Curriculum learning for diffusion: train on "easier" noise levels first,
then gradually cover all timesteps.

Key insight: The beta SCHEDULE should be fixed (cosine, proven best).
The CURRICULUM is implemented via biased timestep sampling during training.

Phases are derived from data eigenspectrum:
    - Phase 1 (coarse): Focus on high-noise timesteps (global structure)
    - Phase 2 (medium): Expand to medium timesteps
    - Phase 3 (fine): Cover all timesteps including low-noise (details)
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

# DDPM constants
BETA_MIN = 1e-4
BETA_MAX = 0.02


class SpectralCurriculumScheduler:
    """Spectral-aware curriculum for diffusion training."""

    def __init__(
        self,
        n_phases: int = 3,
        total_timesteps: int = 1000,
        energy_thresholds: Optional[List[float]] = None,
    ):
        self.n_phases = max(1, n_phases)
        self.total_timesteps = total_timesteps

        if energy_thresholds is None:
            self.energy_thresholds = [
                (i + 1) / self.n_phases for i in range(self.n_phases - 1)
            ]
        else:
            self.energy_thresholds = energy_thresholds

        self.singular_values: Optional[np.ndarray] = None
        self.spectral_energy: Optional[np.ndarray] = None
        self.phase_boundaries: List[int] = []
        self.phase_timestep_ranges: List[Tuple[int, int]] = []
        self.phase_energy_fractions: List[float] = []

    def fit(self, X: np.ndarray) -> "SpectralCurriculumScheduler":
        """Analyze data eigenspectrum to determine curriculum phases."""
        X_centered = X - X.mean(axis=0)
        try:
            _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            S = np.ones(min(X.shape))
        
        self.singular_values = S
        energy = np.cumsum(S ** 2)
        total_e = energy[-1] + 1e-12
        self.spectral_energy = energy / total_e

        # Compute phase boundaries from energy thresholds
        self.phase_boundaries = []
        for threshold in self.energy_thresholds:
            idx = int(np.searchsorted(self.spectral_energy, threshold))
            self.phase_boundaries.append(min(idx, len(S) - 1))

        self._compute_timestep_ranges()
        return self

    def _compute_timestep_ranges(self):
        """Assign timestep ranges to each curriculum phase.
        
        Curriculum strategy: Easy → Hard
        - Early phases: High timesteps (noisy, easier denoising)
        - Late phases: Low timesteps (clean, harder denoising)
        """
        if self.n_phases == 1:
            # Ablation: use full range
            self.phase_timestep_ranges = [(0, self.total_timesteps)]
            self.phase_energy_fractions = [1.0]
            return
            
        steps_per_phase = max(1, self.total_timesteps // self.n_phases)
        
        self.phase_timestep_ranges = []
        self.phase_energy_fractions = []
        
        for i in range(self.n_phases):
            # Phase i: range of timesteps
            # Phase 0 = highest timesteps (easiest), Phase n-1 = all timesteps
            if i == self.n_phases - 1:
                # Last phase: full range
                t_low, t_high = 0, self.total_timesteps
            else:
                # Earlier phases: progressively higher timesteps only
                t_low = self.total_timesteps - (i + 1) * steps_per_phase
                t_high = self.total_timesteps - i * steps_per_phase
            
            # Clamp
            t_low = max(0, min(t_low, self.total_timesteps - 1))
            t_high = max(t_low + 1, min(t_high, self.total_timesteps))
            
            self.phase_timestep_ranges.append((t_low, t_high))
            self.phase_energy_fractions.append(1.0 / self.n_phases)

    def get_full_beta_schedule(self) -> np.ndarray:
        """Return COSINE beta schedule (proven best, used for all variants)."""
        t = np.arange(self.total_timesteps + 1) / self.total_timesteps
        s = 0.008  # Offset from Nichol & Dhariwal
        alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = np.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], BETA_MIN, 0.999)
        return betas.astype(np.float64)

    def get_phase_for_epoch(self, epoch: int, total_epochs: int) -> int:
        """Determine curriculum phase based on training progress."""
        if self.n_phases <= 1:
            return 0
        fraction = epoch / max(1, total_epochs)
        return min(int(fraction * self.n_phases), self.n_phases - 1)

    def get_timestep_range_for_epoch(self, epoch: int, total_epochs: int) -> Tuple[int, int]:
        """Get the timestep range for current curriculum phase."""
        phase = self.get_phase_for_epoch(epoch, total_epochs)
        if phase >= len(self.phase_timestep_ranges):
            return (0, self.total_timesteps)
        return self.phase_timestep_ranges[phase]

    def sample_timesteps(
        self, batch_size: int, epoch: int, total_epochs: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Sample timesteps with curriculum bias."""
        if self.n_phases <= 1 or len(self.phase_timestep_ranges) == 0:
            # Ablation: uniform sampling
            return torch.randint(0, self.total_timesteps, (batch_size,), 
                                 device=device).long()

        t_low, t_high = self.get_timestep_range_for_epoch(epoch, total_epochs)
        
        # Ensure valid range
        t_low = max(0, min(t_low, self.total_timesteps - 1))
        t_high = max(t_low + 1, min(t_high, self.total_timesteps))
        
        if t_high - t_low < 2:
            # Range too small, use uniform
            return torch.randint(0, self.total_timesteps, (batch_size,), device=device).long()

        # Split: 70% curriculum, 30% uniform (adjusted from 60/40 for more focus)
        n_curriculum = int(batch_size * 0.7)
        n_uniform = batch_size - n_curriculum

        # Curriculum samples from current phase
        t_curriculum = torch.randint(t_low, t_high, (n_curriculum,), device=device)
        
        # Uniform samples from full range (ensures coverage)
        t_uniform = torch.randint(0, self.total_timesteps, (n_uniform,), device=device)
        
        # Combine and shuffle
        t = torch.cat([t_curriculum, t_uniform])
        t = t[torch.randperm(len(t), device=device)]
        
        return torch.clamp(t.long(), 0, self.total_timesteps - 1)

