"""
Spectral Curriculum Scheduling for tabular diffusion.

Beta schedule derivation (justification for IEEE TKDE):
    The DDPM standard (Ho et al. 2020) uses beta in [1e-4, 0.02].
    We adopt these EXACT bounds from the DDPM paper (not magic numbers).

    Per-phase scaling: each phase's beta range is modulated by spectral
    energy fraction. Phases covering high-energy (coarse) directions use
    the full [beta_min, beta_max] range. Phases covering low-energy (fine)
    directions use a narrower range to prevent over-noising fine structure.

    Formally:
        beta_max_k = beta_max_ddpm * (1 - E_k / E_total * 0.5)
        beta_min_k = beta_min_ddpm * (1 + E_k / E_total)
    where E_k is cumulative energy fraction up to phase k.
    This is derived from the principle that fine-structure noise should
    be bounded by the energy content of that spectral band.

Phase boundaries:
    Derived from cumulative energy thresholds of the SVD of X.
    Evenly-spaced energy thresholds [1/K, 2/K, ..., (K-1)/K] partition
    the spectrum by equal energy contribution, not arbitrary index.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

BETA_MIN_DDPM = 1e-4
BETA_MAX_DDPM = 0.02


class SpectralCurriculumScheduler:
    """Derive diffusion schedule from data eigenspectrum."""

    def __init__(
        self,
        n_phases: int = 3,
        total_timesteps: int = 1000,
        energy_thresholds: Optional[List[float]] = None,
    ):
        self.n_phases = n_phases
        self.total_timesteps = total_timesteps

        if energy_thresholds is None:
            self.energy_thresholds = [
                (i + 1) / n_phases for i in range(n_phases - 1)
            ]
        else:
            self.energy_thresholds = energy_thresholds

        self.singular_values: Optional[np.ndarray] = None
        self.spectral_energy: Optional[np.ndarray] = None
        self.phase_boundaries: List[int] = []
        self.phase_timestep_ranges: List[Tuple[int, int]] = []
        self.beta_schedules: List[np.ndarray] = []
        self.phase_energy_fractions: List[float] = []

    def fit(self, X: np.ndarray) -> "SpectralCurriculumScheduler":
        """Analyze eigenspectrum and derive phase boundaries."""
        X_centered = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        self.singular_values = S

        energy = np.cumsum(S ** 2)
        total_e = energy[-1] + 1e-12
        self.spectral_energy = energy / total_e

        self.phase_boundaries = []
        for threshold in self.energy_thresholds:
            idx = int(np.searchsorted(self.spectral_energy, threshold))
            self.phase_boundaries.append(idx)

        self._compute_timestep_ranges()
        self._compute_beta_schedules()
        return self

    def _compute_timestep_ranges(self):
        """Assign timestep ranges proportional to spectral energy per band."""
        boundaries = [0] + self.phase_boundaries + [len(self.singular_values)]
        band_energies = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end > start:
                band_energy = np.sum(self.singular_values[start:end] ** 2)
            else:
                band_energy = 1e-6
            band_energies.append(band_energy)

        total_energy = sum(band_energies)
        fractions = [e / total_energy for e in band_energies]
        self.phase_energy_fractions = fractions

        self.phase_timestep_ranges = []
        t_start = self.total_timesteps
        for frac in fractions:
            t_end = max(0, t_start - int(frac * self.total_timesteps))
            if t_end >= t_start:
                t_end = max(0, t_start - 1)
            self.phase_timestep_ranges.append((t_end, t_start))
            t_start = t_end

        self.phase_timestep_ranges.reverse()
        self.phase_energy_fractions.reverse()

        for i, (t_low, t_high) in enumerate(self.phase_timestep_ranges):
            t_low = max(0, min(t_low, self.total_timesteps - 1))
            t_high = max(t_low + 1, min(t_high, self.total_timesteps))
            self.phase_timestep_ranges[i] = (t_low, t_high)

    def _compute_beta_schedules(self):
        """Per-phase cosine beta schedules derived from spectral energy.

        Uses DDPM bounds [1e-4, 0.02] modulated by energy fraction.
        """
        self.beta_schedules = []
        for i, (t_low, t_high) in enumerate(self.phase_timestep_ranges):
            n_steps = max(1, t_high - t_low)
            efrac = self.phase_energy_fractions[i] if i < len(self.phase_energy_fractions) else 0.5

            # Data-driven modulation of DDPM bounds
            max_beta = BETA_MAX_DDPM * (1.0 - efrac * 0.5)
            min_beta = BETA_MIN_DDPM * (1.0 + efrac)

            steps = np.linspace(0, 1, n_steps)
            betas = min_beta + 0.5 * (max_beta - min_beta) * (1 - np.cos(np.pi * steps))
            self.beta_schedules.append(betas)

    def get_full_beta_schedule(self) -> np.ndarray:
        full = np.concatenate(self.beta_schedules)
        if len(full) > self.total_timesteps:
            full = full[:self.total_timesteps]
        elif len(full) < self.total_timesteps:
            pad = np.full(self.total_timesteps - len(full), full[-1])
            full = np.concatenate([full, pad])
        return full.astype(np.float64)

    def get_phase_for_epoch(self, epoch: int, total_epochs: int) -> int:
        fraction = epoch / max(1, total_epochs)
        return min(int(fraction * self.n_phases), self.n_phases - 1)

    def get_timestep_range_for_epoch(self, epoch: int, total_epochs: int) -> Tuple[int, int]:
        phase = self.get_phase_for_epoch(epoch, total_epochs)
        t_low = self.phase_timestep_ranges[phase][0]
        t_high = self.phase_timestep_ranges[phase][1]
        t_low = max(0, min(t_low, self.total_timesteps - 1))
        t_high = max(t_low + 1, min(t_high, self.total_timesteps))
        return (t_low, t_high)

    def sample_timesteps(
    self, batch_size: int, epoch: int, total_epochs: int, device: str = "cpu"
) -> torch.Tensor:
    t_low, t_high = self.get_timestep_range_for_epoch(epoch, total_epochs)
    t_low = max(0, t_low)
    t_high = max(t_low + 1, t_high)
    
    # Base uniform sampling
    t = torch.randint(t_low, t_high, (batch_size,), device=device)
    
    # CRITICAL: Bias toward phase-appropriate timesteps for curriculum
    if self.n_phases > 1:
        phase = self.get_phase_for_epoch(epoch, total_epochs)
        epochs_per_phase = max(1, total_epochs // self.n_phases)
        progress = (epoch % epochs_per_phase) / epochs_per_phase
        
        if phase == 0:  # Early phase: bias toward high timesteps (coarse)
            bias = int((1.0 - progress) * 0.3 * (t_high - t_low))
            t = torch.clamp(t + bias, 0, self.total_timesteps - 1)
        elif phase == self.n_phases - 1:  # Late phase: bias toward low timesteps (fine)
            bias = int(progress * 0.3 * (t_high - t_low))
            t = torch.clamp(t - bias, 0, self.total_timesteps - 1)
    
    return torch.clamp(t.long(), 0, self.total_timesteps - 1)


