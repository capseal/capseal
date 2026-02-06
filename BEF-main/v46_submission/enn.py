"""
Entangled Neural Network (ENN) for EEG
True multi-state entanglement with delayed collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SpatialCNN(nn.Module):
    """Braindecode-style spatial feature extractor for EEG channels"""
    
    def __init__(
        self, 
        in_chans: int = 129,
        sfreq: int = 100,
        embed_dim: int = 64,
        n_filters: int = 40
    ):
        super().__init__()
        
        # Temporal convolution to extract frequency patterns
        self.temporal_conv = nn.Conv2d(1, n_filters, kernel_size=(1, 25), padding=(0, 12))
        self.temporal_bn = nn.BatchNorm2d(n_filters)
        
        # Spatial convolution to combine channels
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, kernel_size=(in_chans, 1))
        self.spatial_bn = nn.BatchNorm2d(n_filters)
        
        # Separable convolutions for deeper feature extraction
        self.separable = nn.Sequential(
            # Depthwise
            nn.Conv2d(n_filters, n_filters, kernel_size=(1, 15), 
                     padding=(0, 7), groups=n_filters),
            # Pointwise
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(0.25)
        )
        
        # Final projection to embedding dimension
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(n_filters * 2, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG data [B, C, T] 
        Returns:
            features: [B, T', embed_dim] where T' is reduced time dimension
        """
        # Add channel dimension for 2D convs
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]
        
        # Temporal filtering
        out = self.temporal_conv(x)
        out = self.temporal_bn(out)
        out = F.elu(out)
        
        # Spatial filtering
        out = self.spatial_conv(out)
        out = self.spatial_bn(out)
        out = F.elu(out)
        
        # Separable convolutions
        out = self.separable(out)  # [B, 80, 1, T/4]
        
        # Pool over spatial dimension, keep time
        out = self.pool(out)  # [B, 80, 1, T/4]
        out = out.squeeze(2).transpose(1, 2)  # [B, T/4, 80]
        
        # Project to embedding
        out = self.proj(out)  # [B, T/4, embed_dim]
        
        return out


class EntangledRNNCell(nn.Module):
    """
    True entangled RNN cell with K latent states
    Implements: h_{t+1} = tanh(W_x x_t + E h_t - λ h_t + b)
    where E = L L^T is PSD entanglement matrix
    """
    
    def __init__(
        self,
        input_dim: int,
        K: int = 16,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        
        # Input projection
        self.W_x = nn.Linear(input_dim, K, bias=False)
        
        # Entanglement factor L for E = L L^T
        # Initialize near identity for stable training
        L_init = torch.eye(K) + 0.1 * torch.randn(K, K)
        self.L = nn.Parameter(L_init)
        
        # Decay rate (learned)
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(0.1)))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(K))
        
        # Optional layer norm for stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(K)
        
    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Single step of entangled RNN
        
        Args:
            x_t: Input at time t [B, input_dim]
            h_t: Hidden state at time t [B, K]
        Returns:
            h_{t+1}: Next hidden state [B, K]
        """
        # Compute entanglement matrix E = L L^T (guaranteed PSD)
        E = torch.matmul(self.L, self.L.T)
        
        # Input contribution
        Wx = self.W_x(x_t)  # [B, K]
        
        # Entanglement contribution
        Eh = torch.matmul(h_t, E.T)  # [B, K]
        
        # Decay
        lambda_val = torch.exp(self.log_lambda)
        decay = lambda_val * h_t
        
        # Combine
        pre_activation = Wx + Eh - decay + self.bias
        
        # Optional normalization
        if self.use_layer_norm:
            pre_activation = self.ln(pre_activation)
        
        # Non-linearity
        h_next = torch.tanh(pre_activation)
        
        return h_next
    
    def get_entanglement_matrix(self) -> torch.Tensor:
        """Get the current entanglement matrix E"""
        with torch.no_grad():
            return torch.matmul(self.L, self.L.T)


class CollapseLayer(nn.Module):
    """
    Collapse K entangled states to output via attention mechanism
    """
    
    def __init__(self, K: int, output_dim: int, temperature: float = 1.0):
        super().__init__()
        self.K = K
        self.output_dim = output_dim
        self.temperature = temperature
        
        # Query for attention-based collapse
        self.W_q = nn.Linear(K, K)
        
        # Output projection after collapse
        self.W_out = nn.Linear(K, output_dim)
        
        # Optional: learned temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(
        self,
        psi: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Collapse entangled state to output
        
        Args:
            psi: Entangled state [B, K]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Collapsed output [B, output_dim]
            alpha: Optional attention weights [B, K]
        """
        # Compute attention scores per-state (vector over K)
        # Use a simple gated projection to keep consistent with training
        query = self.W_q(psi)  # [B, K]
        scores = query * psi    # [B, K]

        # Temperature (clamped at eval to avoid argmax collapse)
        temp = torch.exp(self.log_temp)
        if not self.training:
            try:
                tmin = float(os.getenv("BEF_TEMP_MIN", "0.5"))
                tmax = float(os.getenv("BEF_TEMP_MAX", "2.0"))
                temp = torch.clamp(temp, min=tmin, max=tmax)
            except Exception:
                pass

        # Optional uniform attention (eval-time fallback) via env toggle
        use_uniform = False
        try:
            # Prefer a generic switch; fall back to C1-specific name
            use_uniform = os.getenv("BEF_POOL_AVG", "0") == "1" or os.getenv("C1_BEF_POOL_AVG", "0") == "1"
        except Exception:
            pass

        if not use_uniform:
            alpha = F.softmax(scores / temp, dim=-1)  # [B, K]
        else:
            alpha = torch.full_like(scores, 1.0 / scores.shape[-1])  # uniform over K
        
        # Weighted collapse (element-wise gating)
        collapsed = alpha * psi  # [B, K]
        
        # Output projection
        output = self.W_out(collapsed)  # [B, output_dim]
        
        if return_attention:
            return output, alpha
        return output, None


class ENNEncoder(nn.Module):
    """
    Complete ENN encoder: CNN front-end + Entangled RNN + Collapse
    """
    
    def __init__(
        self,
        in_chans: int = 129,
        sfreq: int = 100,
        embed_dim: int = 64,
        K: int = 16,
        output_dim: int = 1,
        n_layers: int = 2
    ):
        super().__init__()
        
        # Spatial feature extraction
        self.front = SpatialCNN(in_chans, sfreq, embed_dim)
        
        # Stack of entangled RNN cells
        self.K = K
        self.n_layers = n_layers
        
        self.enn_cells = nn.ModuleList([
            EntangledRNNCell(embed_dim if i == 0 else K, K)
            for i in range(n_layers)
        ])
        
        # Collapse mechanism
        self.collapse = CollapseLayer(K, output_dim)
        
        # Regularization strength
        self.kl_weight = 0.01
        
    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ENN
        
        Args:
            x: EEG data [B, C, T]
            return_trajectory: Whether to return hidden state trajectory
            
        Returns:
            Z: Final latent embedding [B, K]
            alpha: Collapse attention weights [B, K]
            trajectory: Optional hidden states over time
        """
        # Extract spatial-temporal features
        feats = self.front(x)  # [B, T', embed_dim]
        B, T, E = feats.shape
        
        # Initialize hidden states for all layers
        h = [torch.zeros(B, self.K, device=x.device) for _ in range(self.n_layers)]
        
        trajectory = [] if return_trajectory else None
        
        # Process sequence through entangled RNN layers
        for t in range(T):
            x_t = feats[:, t, :]  # [B, embed_dim]
            
            # Forward through layers
            for layer_idx, cell in enumerate(self.enn_cells):
                if layer_idx == 0:
                    h[layer_idx] = cell(x_t, h[layer_idx])
                else:
                    h[layer_idx] = cell(h[layer_idx-1], h[layer_idx])
            
            if return_trajectory:
                trajectory.append(h[-1].clone())
        
        # Final entangled state
        Z = h[-1]  # [B, K]
        
        # Compute collapse attention
        _, alpha = self.collapse(Z, return_attention=True)
        
        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=1)  # [B, T', K]
        
        return Z, alpha, trajectory
    
    def get_output(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get collapsed output and uncertainty
        
        Returns:
            mean: Predicted output [B, output_dim]
            uncertainty: Entropy of collapse distribution [B, 1]
        """
        Z, alpha, _ = self.forward(x)
        
        # Collapse to output
        output, _ = self.collapse(Z)
        
        # Compute uncertainty as entropy of alpha
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1, keepdim=True)
        
        return output, entropy
    
    def regularization_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        KL regularization to encourage decisive collapse
        """
        # Target is one-hot (maximum entropy reduction)
        uniform = torch.ones_like(alpha) / alpha.shape[-1]
        kl = F.kl_div(torch.log(alpha + 1e-8), uniform, reduction='batchmean')
        
        return self.kl_weight * kl


class MultiScaleENN(ENNEncoder):
    """
    ENN with multi-scale temporal processing
    """
    
    def __init__(
        self,
        scales: list = [1, 2, 4],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.scales = scales
        
        # Multiple ENN paths for different temporal scales
        self.scale_enns = nn.ModuleList([
            EntangledRNNCell(self.front.proj.out_features, self.K)
            for _ in scales
        ])
        
        # Fusion layer
        self.scale_fusion = nn.Linear(self.K * len(scales), self.K)
        
    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-scale forward pass
        """
        # Extract features
        feats = self.front(x)  # [B, T', embed_dim]
        B, T, E = feats.shape
        
        # Process at multiple scales
        scale_outputs = []
        
        for scale, enn_cell in zip(self.scales, self.scale_enns):
            h = torch.zeros(B, self.K, device=x.device)
            
            # Process with temporal stride
            for t in range(0, T, scale):
                if t < T:
                    x_t = feats[:, t, :]
                    h = enn_cell(x_t, h)
            
            scale_outputs.append(h)
        
        # Fuse multi-scale representations
        multi_scale = torch.cat(scale_outputs, dim=-1)  # [B, K * n_scales]
        Z = self.scale_fusion(multi_scale)  # [B, K]
        
        # Collapse attention
        _, alpha = self.collapse(Z, return_attention=True)
        
        return Z, alpha, None
