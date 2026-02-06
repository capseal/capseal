"""
BICEP EEG Module - Stochastic Multi-Future Simulator for EEG
Implements Ornstein-Uhlenbeck SDE with event jumps for ERP modeling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class OUParams:
    """Parameters for Ornstein-Uhlenbeck process"""
    def __init__(self, theta: float = 2.0, mu0: float = 0.0, sigma: float = 0.5):
        self.theta = theta  # Mean reversion rate
        self.mu0 = mu0      # Long-term mean
        self.sigma = sigma  # Volatility


class EEGSDE(nn.Module):
    """
    EEG-specific SDE simulator using Ornstein-Uhlenbeck process
    Models both oscillatory dynamics (SSVEP) and event-related jumps (ERP)
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        ou_params: Optional[OUParams] = None,
        jump_rate: float = 0.1,
        jump_scale: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.dt = dt
        self.ou = ou_params if ou_params else OUParams()
        self.jump_rate = jump_rate
        self.jump_scale = jump_scale
        self.device = device
        
        # Learnable parameters for adaptive SDE
        self.theta_scale = nn.Parameter(torch.tensor(1.0))
        self.sigma_scale = nn.Parameter(torch.tensor(1.0))
        
    @torch.no_grad()
    def simulate_paths(
        self,
        x: torch.Tensor,
        N_paths: int = 64,
        use_antithetic: bool = True,
        adaptive_budget: Optional[float] = 2e7
    ) -> torch.Tensor:
        """
        Simulate N stochastic paths from base EEG trial using Euler-Maruyama

        Args:
            x: Base EEG trial [B, C, T]
            N_paths: Number of paths to simulate
            use_antithetic: Use antithetic sampling for variance reduction
            adaptive_budget: Maximum element budget for memory safety

        Returns:
            Simulated paths [N_paths, B, C, T]
        """
        B, C, T = x.shape

        # Adaptive compute: reduce paths if exceeding budget
        if adaptive_budget is not None:
            total_elements = B * C * T * N_paths
            if total_elements > adaptive_budget:
                N_paths = max(1, int(adaptive_budget / (B * C * T)))
                use_antithetic = N_paths > 1  # Only use if we have enough paths

        dt = torch.tensor(self.dt, device=x.device)

        # Adaptive parameters with stability bounds
        theta = torch.clamp(self.ou.theta * self.theta_scale.abs(), min=0.1, max=10.0)
        sigma = torch.clamp(self.ou.sigma * self.sigma_scale.abs(), min=0.01, max=2.0)
        mu0 = self.ou.mu0

        # Precompute OU constants for efficiency
        exp_theta_dt = torch.exp(-theta * dt)
        
        # Initialize paths
        if use_antithetic and N_paths % 2 == 0:
            # Antithetic sampling for variance reduction
            N_half = N_paths // 2
            x0 = x.unsqueeze(0).repeat(N_half, 1, 1, 1)
            
            # Generate base noise
            eps = torch.randn_like(x0)
            
            # Create antithetic pairs
            eps_full = torch.cat([eps, -eps], dim=0)
            
            # Euler-Maruyama with OU dynamics
            dW = torch.sqrt(dt) * eps_full * sigma
            drift = theta * (mu0 - x0.repeat(2, 1, 1, 1)) * dt
            paths = x.unsqueeze(0).repeat(N_paths, 1, 1, 1) + drift + dW
            
        else:
            x0 = x.unsqueeze(0).repeat(N_paths, 1, 1, 1)
            
            # Standard Euler-Maruyama
            eps = torch.randn_like(x0)
            dW = torch.sqrt(dt) * eps * sigma
            drift = theta * (mu0 - x0) * dt
            paths = x0 + drift + dW
        
        # Add event-related jumps (ERP simulation)
        if self.jump_rate > 0 and self.jump_scale > 0:
            # Poisson-distributed jumps
            jump_mask = torch.rand(N_paths, B, C, 1, device=x.device) < self.jump_rate
            jump_magnitude = self.jump_scale * torch.randn(N_paths, B, C, 1, device=x.device)
            
            # Random jump times
            jump_times = torch.randint(T//4, 3*T//4, (N_paths, B, C, 1), device=x.device)
            
            # Vectorized jump series creation
            jump_series = torch.zeros_like(paths)

            # Create time grid once
            t_grid = torch.arange(T, device=x.device).float().view(1, 1, 1, -1)

            # Expand jump_times and jump_magnitude for broadcasting
            jump_times_expanded = jump_times.float().unsqueeze(-1).expand(N_paths, B, C, 1, T)
            jump_magnitude_expanded = jump_magnitude.expand(N_paths, B, C, 1)

            # Vectorized Gaussian ERP computation
            erp_shapes = torch.exp(-0.5 * ((t_grid - jump_times_expanded[..., 0:1]) / 10)**2)

            # Apply jumps with mask
            jump_series = (jump_mask.unsqueeze(-1) * jump_magnitude_expanded.unsqueeze(-1) * erp_shapes).squeeze(3)
            
            paths = paths + jump_series
        
        return paths
    
    def compute_path_statistics(
        self, 
        paths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute statistics across simulated paths
        
        Returns:
            mean, std, quantiles of paths
        """
        mean = paths.mean(dim=0)
        std = paths.std(dim=0)
        
        # Compute quantiles for uncertainty bands
        quantiles = torch.quantile(paths, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=paths.device), dim=0)
        
        return mean, std, quantiles


class OscillatorySDEVariant(EEGSDE):
    """
    Extended SDE for SSVEP modeling with explicit frequency components
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        frequencies: List[float] = [10.0, 15.0, 20.0],  # Hz
        **kwargs
    ):
        super().__init__(dt=dt, **kwargs)
        self.frequencies = frequencies
        
        # Learnable frequency modulation
        self.freq_weights = nn.Parameter(torch.ones(len(frequencies)))
        
    @torch.no_grad()
    def simulate_paths(
        self,
        x: torch.Tensor,
        N_paths: int = 64,
        use_antithetic: bool = True,
        adaptive_budget: Optional[float] = 2e7
    ) -> torch.Tensor:
        """
        Simulate with added oscillatory components for SSVEP
        """
        # Base OU paths with adaptive budget
        paths = super().simulate_paths(x, N_paths, use_antithetic, adaptive_budget)
        
        B, C, T = x.shape
        
        # Add oscillatory components
        for i, freq in enumerate(self.frequencies):
            t_grid = torch.arange(T, device=x.device).float() * self.dt
            
            # Random phase for each path/channel
            phases = torch.rand(N_paths, B, C, 1, device=x.device) * 2 * np.pi
            
            # Oscillation with frequency modulation
            oscillation = self.freq_weights[i].abs() * torch.sin(
                2 * np.pi * freq * t_grid.view(1, 1, 1, -1) + phases
            )
            
            # Add with random amplitude modulation
            amp_mod = 0.1 * torch.randn(N_paths, B, C, 1, device=x.device).abs()
            paths = paths + amp_mod * oscillation
        
        return paths


class AdaptiveBICEP(nn.Module):
    """
    Adaptive BICEP that learns SDE parameters from data
    """

    def __init__(
        self,
        base_sde: Optional[EEGSDE] = None,
        n_ensemble: int = 5,
        max_budget: float = 2e7
    ):
        super().__init__()

        if base_sde is None:
            base_sde = OscillatorySDEVariant()

        self.max_budget = max_budget

        # Ensemble of SDEs with different parameters
        self.sde_ensemble = nn.ModuleList([
            OscillatorySDEVariant(
                dt=0.01,
                ou_params=OUParams(theta=2.0 + 0.5 * i, sigma=0.5 + 0.1 * i),
                device=base_sde.device
            ) for i in range(n_ensemble)
        ])

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(n_ensemble) / n_ensemble)

        # Low-rank noise projector for correlated paths (reduces RNG calls)
        self.noise_rank = min(8, n_ensemble)
        self.noise_proj = nn.Parameter(torch.randn(self.noise_rank, n_ensemble) / np.sqrt(self.noise_rank))
        
    def forward(
        self,
        x: torch.Tensor,
        N_paths_per_sde: int = 20
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate ensemble of paths with learned weighting and adaptive compute

        Returns:
            Weighted paths and (aleatoric, epistemic) uncertainty estimates
        """
        B, C, T = x.shape

        # Adaptive paths per SDE based on budget
        n_sde = len(self.sde_ensemble)
        total_elements = B * C * T * N_paths_per_sde * n_sde
        if total_elements > self.max_budget:
            N_paths_per_sde = max(1, int(self.max_budget / (B * C * T * n_sde)))

        all_paths = []
        weights = torch.softmax(self.ensemble_weights, dim=0)

        # Generate low-rank correlated noise for all SDEs at once
        base_noise = torch.randn(self.noise_rank, B, C, T, device=x.device)

        for i, sde in enumerate(self.sde_ensemble):
            # Use correlated noise via projection
            paths = sde.simulate_paths(x, N_paths=N_paths_per_sde, adaptive_budget=self.max_budget / n_sde)
            all_paths.append(paths * weights[i])

        # Combine weighted paths
        combined_paths = torch.cat(all_paths, dim=0)

        # Split uncertainty: aleatoric (within SDE) vs epistemic (between SDEs)
        path_mean = combined_paths.mean(dim=0)

        # Aleatoric: average variance within each SDE
        aleatoric_vars = []
        for i in range(n_sde):
            start_idx = i * N_paths_per_sde
            end_idx = (i + 1) * N_paths_per_sde
            sde_paths = combined_paths[start_idx:end_idx]
            aleatoric_vars.append(sde_paths.var(dim=0))
        aleatoric_uncertainty = torch.stack(aleatoric_vars).mean(dim=0)

        # Epistemic: variance of means between SDEs
        sde_means = [combined_paths[i*N_paths_per_sde:(i+1)*N_paths_per_sde].mean(dim=0) for i in range(n_sde)]
        epistemic_uncertainty = torch.stack(sde_means).var(dim=0)

        return combined_paths, (aleatoric_uncertainty, epistemic_uncertainty)

    def simulate_paths(
        self,
        x: torch.Tensor,
        N_paths: int = 64,
        adaptive_budget: Optional[float] = None
    ) -> torch.Tensor:
        """Compatibility wrapper so AdaptiveBICEP matches ``EEGSDE`` API."""

        n_sde = max(1, len(self.sde_ensemble))
        paths_per = max(1, N_paths // n_sde)

        if adaptive_budget is not None:
            prev_budget = self.max_budget
            self.max_budget = min(self.max_budget, adaptive_budget)
        else:
            prev_budget = None

        paths, _ = self.forward(x, N_paths_per_sde=paths_per)

        if prev_budget is not None:
            self.max_budget = prev_budget

        return paths
    
    def get_calibrated_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Get calibrated uncertainty estimates via Monte Carlo
        """
        uncertainties = []
        
        for _ in range(n_samples // 20):
            _, var = self.forward(x, N_paths_per_sde=20)
            uncertainties.append(var)
        
        # Average uncertainties
        return torch.stack(uncertainties).mean(dim=0)
