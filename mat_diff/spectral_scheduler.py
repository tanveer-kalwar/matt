    def sample_timesteps(
        self, batch_size: int, epoch: int, total_epochs: int, device: str = "cpu"
        ) -> torch.Tensor:
            """Sample timesteps uniformly from the full range.

            The curriculum is encoded in the beta schedule itself (computed
            from spectral energy). Additional timestep biasing causes
            under-training on mid-range timesteps and hurts sample quality.
            """
            t_low, t_high = self.get_timestep_range_for_epoch(epoch, total_epochs)
            t_low = max(0, t_low)
            t_high = max(t_low + 1, t_high)

            # Uniform sampling within the phase range (no bias)
            t = torch.randint(t_low, t_high, (batch_size,), device=device)
            return torch.clamp(t.long(), 0, self.total_timesteps - 1)
