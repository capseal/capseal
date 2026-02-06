"""
BEF Pipeline: Unified BICEP → ENN → Fusion Alpha for EEG
Complete implementation with uncertainty quantification and transfer learning
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from contextlib import nullcontext
import numpy as np

from .bicep_eeg import EEGSDE, OscillatorySDEVariant, AdaptiveBICEP
from .enn import ENNEncoder, MultiScaleENN
from .fusion_alpha import FusionAlphaGNN, HierarchicalFusionAlpha, build_sensor_graph, GraphCache


class BEF_EEG(nn.Module):
    """
    Complete BEF pipeline for EEG decoding
    BICEP: Multi-future stochastic simulation
    ENN: Entangled multi-state encoding
    Fusion Alpha: Graph-based contradiction resolution
    """
    
    def __init__(
        self,
        # Data parameters
        in_chans: int = 129,
        sfreq: int = 100,
        
        # BICEP parameters (v35a_DIAG_v2: reduced for speed)
        n_paths: int = 32,  # Was 64, halved to avoid timeout
        use_oscillatory_sde: bool = True,
        sde_frequencies: List[float] = [10.0, 15.0, 20.0],
        
        # ENN parameters (MUST match trained weights - DO NOT CHANGE)
        K: int = 8,  # Trained with K=8, verified from c1_bef.pt
        embed_dim: int = 64,
        enn_layers: int = 2,
        use_multiscale: bool = False,
        
        # Fusion Alpha parameters
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        use_hierarchical: bool = False,
        
        # Output parameters
        output_dim: int = 1,
        
        # Training parameters
        dropout: float = 0.0,  # Default to 0 for inference
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        adaptive_budget: float = 1e7  # Element budget (v35a_DIAG_v2: reduced from 2e7 for speed)
    ):
        super().__init__()

        # Allow environment overrides for diagnostics/runtime tuning
        try:
            if output_dim == 1:  # C1 instance
                n_paths = int(os.getenv("C1_BEF_N_PATHS", str(n_paths)))
            elif output_dim == 4:  # C2 instance
                n_paths = int(os.getenv("C2_BEF_N_PATHS", str(n_paths)))
        except Exception:
            pass

        # Env-controlled knobs
        try:
            if output_dim == 1:
                n_paths = int(os.getenv("C1_BEF_N_PATHS", str(n_paths)))
            elif output_dim == 4:
                n_paths = int(os.getenv("C2_BEF_N_PATHS", str(n_paths)))
        except Exception:
            pass
        self.n_paths = n_paths
        self.K = K
        self.output_dim = output_dim
        self.device = device
        self.adaptive_budget = adaptive_budget

        # Graph cache for Fusion Alpha
        self.graph_cache = GraphCache(max_size=50)
        
        # BICEP: Adaptive stochastic simulator
        self.use_adaptive_bicep = True
        if self.use_adaptive_bicep:
            self.bicep = AdaptiveBICEP(
                base_sde=OscillatorySDEVariant(
                    dt=1/sfreq,
                    frequencies=sde_frequencies,
                    device=device
                ) if use_oscillatory_sde else EEGSDE(dt=1/sfreq, device=device),
                n_ensemble=3,  # MUST match trained weights (do NOT change)
                max_budget=adaptive_budget
            )
        elif use_oscillatory_sde:
            self.bicep = OscillatorySDEVariant(
                dt=1/sfreq,
                frequencies=sde_frequencies,
                device=device
            )
        else:
            self.bicep = EEGSDE(dt=1/sfreq, device=device)
        
        # ENN: Entangled encoder
        if use_multiscale:
            self.enn = MultiScaleENN(
                in_chans=in_chans,
                sfreq=sfreq,
                embed_dim=embed_dim,
                K=K,
                output_dim=output_dim,
                n_layers=enn_layers,
                scales=[1, 2, 4]
            )
        else:
            self.enn = ENNEncoder(
                in_chans=in_chans,
                sfreq=sfreq,
                embed_dim=embed_dim,
                K=K,
                output_dim=output_dim,
                n_layers=enn_layers
            )

        # Optional: disable stochasticity at eval via env toggles
        # C1_BEF_DISABLE_STOCH=1 disables OU noise, jumps, and oscillations
        try:
            disable_stoch = False
            if output_dim == 1:
                # Default to deterministic for C1 unless explicitly overridden
                disable_stoch = os.getenv("C1_BEF_DISABLE_STOCH", "1") == "1"
            elif output_dim == 4:
                disable_stoch = os.getenv("C2_BEF_DISABLE_STOCH", "0") == "1"
            if disable_stoch:
                if self.use_adaptive_bicep and hasattr(self, 'bicep') and hasattr(self.bicep, 'sde_ensemble'):
                    for sde in self.bicep.sde_ensemble:
                        # Zero OU volatility and disable jumps/oscillations
                        try:
                            sde.sigma_scale.data = torch.zeros_like(sde.sigma_scale.data)
                        except Exception:
                            pass
                        if hasattr(sde, 'ou'):
                            try:
                                sde.ou.sigma = 0.0
                                sde.jump_rate = 0.0
                                sde.jump_scale = 0.0
                            except Exception:
                                pass
                        # Zero oscillatory amplitudes if present
                        if hasattr(sde, 'freq_weights'):
                            try:
                                sde.freq_weights.data = torch.zeros_like(sde.freq_weights.data)
                            except Exception:
                                pass
                print(f"BEF_EEG: stochastic components disabled (output_dim={output_dim})")
        except Exception:
            pass
        
        # Fusion Alpha: Graph fusion (allow env override for time budget)
        try:
            fusion_max_time = float(os.getenv("FUSION_MAX_TIME_S", "2.0"))
        except Exception:
            fusion_max_time = 2.0
        if use_hierarchical:
            self.fusion = HierarchicalFusionAlpha(
                node_feat_dim=K + 2,  # K states + mean/std from BICEP
                hidden_dim=gnn_hidden,
                output_dim=output_dim,
                n_layers=gnn_layers,
                dropout=dropout,
                max_time=fusion_max_time
            )
        else:
            self.fusion = FusionAlphaGNN(
                node_feat_dim=K + 2,
                hidden_dim=gnn_hidden,
                output_dim=output_dim,
                n_layers=gnn_layers,
                dropout=dropout,
                max_time=fusion_max_time
            )
        
        # Task-specific heads and bypasses
        self.use_multi_task = False
        # Optional bypass of fusion for C1 to A/B variance of ENN Z directly
        self._bypass_fusion = False
        if output_dim == 1 and os.getenv("C1_BEF_BYPASS_FUSION", "0") == "1":
            self._bypass_fusion = True
        self._bypass_head = None
        
    def extract_channel_features(
        self,
        enn_Z: torch.Tensor,
        enn_alpha: torch.Tensor,
        bicep_stats: Tuple[torch.Tensor, torch.Tensor],
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract per-channel features for Fusion Alpha nodes
        
        Args:
            enn_Z: ENN latent states [B, K]
            enn_alpha: ENN attention weights [B, K] 
            bicep_stats: (mean, std) from BICEP paths [B, C, T]
            x: Original EEG [B, C, T]
            
        Returns:
            Node features [B, C, K+2]
        """
        B, C, T = x.shape
        
        # Get BICEP statistics per channel
        bicep_mean, bicep_std = bicep_stats
        channel_mean = bicep_mean.mean(dim=-1)  # [B, C]
        channel_std = bicep_std.mean(dim=-1)    # [B, C]
        
        # Project ENN states to channels
        # Simple approach: broadcast ENN state to all channels
        # (Could be improved with learned projection)
        enn_broadcast = enn_Z.unsqueeze(1).expand(B, C, self.K)  # [B, C, K]
        
        # Weight by attention for different channels
        # (Heuristic: channels with higher variance get more uncertain states)
        channel_uncertainty = channel_std / (channel_std.mean(dim=1, keepdim=True) + 1e-6)
        weighted_enn = enn_broadcast * channel_uncertainty.unsqueeze(-1)
        
        # Concatenate features
        node_features = torch.cat([
            weighted_enn,                    # [B, C, K]
            channel_mean.unsqueeze(-1),      # [B, C, 1]
            channel_std.unsqueeze(-1)        # [B, C, 1]
        ], dim=-1)  # [B, C, K+2]
        
        return node_features
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        mc_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Full BEF forward pass with split uncertainty

        Args:
            x: EEG data [B, C, T]
            return_intermediates: Return intermediate representations
            mc_samples: MC dropout samples for uncertainty

        Returns:
            Dictionary with predictions and split uncertainties
        """
        B, C, T = x.shape

        # Optional channel trim via env (comma-separated indices), preserves behavior but reduces work
        try:
            keep_env = os.getenv("KEEP_CHANNELS") or os.getenv("C1_KEEP_CHANNELS")
            if keep_env:
                idx = [int(s.strip()) for s in keep_env.split(",") if s.strip()]
                if idx:
                    x = x[:, idx, :]
                    C = x.shape[1]
        except Exception:
            pass

        # Adaptive compute check (allow env override for budget)
        total_elements = B * C * T * self.n_paths
        try:
            if self.output_dim == 1:
                self.adaptive_budget = float(os.getenv("C1_BEF_MAX_BUDGET", str(self.adaptive_budget)))
            elif self.output_dim == 4:
                self.adaptive_budget = float(os.getenv("C2_BEF_MAX_BUDGET", str(self.adaptive_budget)))
        except Exception:
            pass
        if total_elements > self.adaptive_budget:
            actual_n_paths = max(1, int(self.adaptive_budget / (B * C * T)))
        else:
            actual_n_paths = self.n_paths

        # Stage 1: BICEP - Generate stochastic futures with split uncertainty
        with torch.no_grad():
            if self.use_adaptive_bicep:
                paths, (aleatoric_bicep, epistemic_bicep) = self.bicep(x, N_paths_per_sde=actual_n_paths // 3)
            else:
                paths = self.bicep.simulate_paths(x, N_paths=actual_n_paths, adaptive_budget=self.adaptive_budget)
                # Simple uncertainty split for non-adaptive
                path_var = paths.var(dim=0)
                aleatoric_bicep = 0.7 * path_var  # Heuristic split
                epistemic_bicep = 0.3 * path_var

        # Compute path statistics
        path_mean = paths.mean(dim=0)  # [B, C, T]
        path_std = paths.std(dim=0)    # [B, C, T]
        
        # Stage 2: ENN - Process through entangled network (optional AMP at eval)
        use_amp = (self.device == "cuda" and os.getenv("EVAL_AMP", "1") == "1")
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with autocast_ctx:
            # We process the mean path (could also process multiple paths)
            enn_Z, enn_alpha, trajectory = self.enn(path_mean, return_trajectory=True)
            # Also get ENN output and uncertainty
            enn_output, enn_entropy = self.enn.get_output(path_mean)
        
        # Optional bypass: directly map ENN Z to output (C1 only) to A/B collapse/fusion
        if self._bypass_fusion and self.output_dim == 1:
            # Z is [B, K]; ensure 2D
            Z_red = enn_Z
            while Z_red.dim() > 2:
                Z_red = Z_red.mean(dim=-1)
            if self._bypass_head is None:
                in_features = Z_red.shape[-1]
                self._bypass_head = nn.Linear(in_features, 1).to(x.device)
                print(f"BEF_EEG: bypass fusion enabled (C1), head in_features={in_features}")
            rt = self._bypass_head(Z_red).view(B, 1)
            return {
                'rt': rt,
                'rt_prediction': rt,
                'prediction': rt,
                'aleatoric_uncertainty': torch.zeros_like(rt),
                'epistemic_uncertainty': torch.zeros_like(rt),
                'total_uncertainty': torch.zeros_like(rt),
                'attention_weights': torch.ones(B, x.shape[1], device=x.device) / max(1, x.shape[1])
            }

        # Stage 3: Fusion Alpha - Graph-based fusion with caching
        # Build sensor graph with cache
        A = build_sensor_graph(x, k=min(8, C-1), use_correlation=True, cache=self.graph_cache)
        
        # Extract node features
        node_features = self.extract_channel_features(
            enn_Z, enn_alpha, (path_mean, path_std), x
        )
        
        # Graph fusion with MC dropout
        with autocast_ctx:
            fusion_out = self.fusion(
                node_features, A,
                mc_samples=mc_samples
            )
        
        # Combine outputs with proper uncertainty split
        # Aleatoric: inherent data uncertainty (from BICEP within-SDE variance + ENN entropy)
        aleatoric_total = aleatoric_bicep.mean() * 0.3 + enn_entropy * 0.7

        # Epistemic: model uncertainty (from BICEP between-SDE variance + fusion dropout)
        epistemic_total = epistemic_bicep.mean() * 0.3 + fusion_out['uncertainty'] * 0.7

        # Prepare output based on task dimension
        if self.output_dim == 1:
            # Challenge 1: RT regression - ensure [B,1] shape
            rt = fusion_out['logits'].view(B, 1).float()
            results = {
                'rt': rt,
                'rt_prediction': rt,
                'prediction': rt,
                'aleatoric_uncertainty': aleatoric_total,
                'epistemic_uncertainty': epistemic_total,
                'total_uncertainty': aleatoric_total + epistemic_total,
                'attention_weights': fusion_out.get('attention', torch.ones(B, C, device=x.device) / C)
            }
        elif self.output_dim == 4:
            # Challenge 2: 4-factor regression - ensure [B,4] shape
            factors = fusion_out['logits'].view(B, 4).float()
            results = {
                'psycho_predictions': factors,
                'prediction': factors,
                'aleatoric_uncertainty': aleatoric_total,
                'epistemic_uncertainty': epistemic_total,
                'total_uncertainty': aleatoric_total + epistemic_total,
                'attention_weights': fusion_out.get('attention', torch.ones(B, C, device=x.device) / C)
            }
        else:
            # Default case
            results = {
                'prediction': fusion_out['logits'],
                'aleatoric_uncertainty': aleatoric_total,
                'epistemic_uncertainty': epistemic_total,
                'total_uncertainty': aleatoric_total + epistemic_total,
                'attention_weights': fusion_out.get('attention', torch.ones(B, C, device=x.device) / C)
            }
        
        if return_intermediates:
            results.update({
                'bicep_mean': path_mean,
                'bicep_std': path_std,
                'enn_Z': enn_Z,
                'enn_alpha': enn_alpha,
                'enn_trajectory': trajectory,
                'graph_adjacency': A
            })
        
        return results
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task: str = "regression",
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BEF loss with regularization
        """
        losses = {}
        
        if task == "regression":
            # Main regression loss (supports scalar [B,1] or vector [B,D])
            pred = outputs['prediction']
            if pred.ndim == 2 and pred.size(-1) > 1:
                # Vector regression [B,D]
                if targets.ndim == 1:
                    tgt = targets.unsqueeze(-1)
                else:
                    tgt = targets
                losses['mse'] = F.mse_loss(pred, tgt)
                # Optional uncertainty-weighted term when shapes match [B,D]
                unc = outputs.get('total_uncertainty')
                if isinstance(unc, torch.Tensor) and unc.shape == pred.shape:
                    weighted = (pred - tgt) ** 2 / (2 * unc.clamp_min(1e-6)) + 0.5 * torch.log(unc.clamp_min(1e-6))
                    losses['nll'] = weighted.mean()
                else:
                    losses['nll'] = torch.tensor(0.0, device=pred.device)
            else:
                # Scalar regression; squeeze to [B]
                pred_s = pred.squeeze(-1)
                tgt_s = targets.squeeze(-1)
                losses['mse'] = F.mse_loss(pred_s, tgt_s)
                unc = outputs.get('total_uncertainty')
                if isinstance(unc, torch.Tensor):
                    unc_s = unc.squeeze(-1)
                    weighted = (pred_s - tgt_s) ** 2 / (2 * unc_s.clamp_min(1e-6)) + 0.5 * torch.log(unc_s.clamp_min(1e-6))
                    losses['nll'] = weighted.mean()
                else:
                    losses['nll'] = torch.tensor(0.0, device=pred.device)
            
        elif task == "classification":
            # Binary classification
            logits = outputs['prediction'].squeeze(-1)
            targets_squeezed = targets.squeeze(-1)
            losses['bce'] = F.binary_cross_entropy_with_logits(logits, targets_squeezed.float())
        elif task == "multiclass":
            # Multi-class classification
            logits = outputs['prediction']  # Shape: [batch, num_classes]
            targets_long = targets.long().squeeze(-1)
            losses['ce'] = F.cross_entropy(logits, targets_long, weight=class_weights)
        
        # Regularization
        enn_alpha = outputs.get('enn_alpha')
        if enn_alpha is not None:
            losses['enn_reg'] = self.enn.regularization_loss(enn_alpha)
        else:
            losses['enn_reg'] = torch.tensor(0.0, device=targets.device)
        
        # Total loss
        losses['total'] = losses.get('mse', 0) + losses.get('bce', 0) + losses.get('ce', 0) + 0.01 * losses['enn_reg']
        
        return losses
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with calibrated split uncertainty

        Returns:
            mean_pred, aleatoric_uncertainty, epistemic_uncertainty
        """
        predictions = []
        aleatorics = []
        epistemics = []

        # Ensure minimum samples for proper uncertainty estimation
        min_samples = 5  # Minimum for meaningful variance computation
        actual_samples = max(min_samples, min(n_samples, max(5, int(self.adaptive_budget / (x.shape[0] * x.shape[1] * x.shape[2] * 10)))))

        # Multiple forward passes for epistemic uncertainty
        for _ in range(actual_samples):
            out = self.forward(x, mc_samples=1)
            predictions.append(out['prediction'])
            aleatorics.append(out['aleatoric_uncertainty'])
            epistemics.append(out['epistemic_uncertainty'])

        # Stack tensors (shape [n_samples, batch_size, D])
        predictions = torch.stack(predictions)  # [S, B, D]
        aleatorics = torch.stack(aleatorics)    # [S, B, D]
        epistemics = torch.stack(epistemics)    # [S, B, D]

        # Compute mean and uncertainties
        mean_pred = predictions.mean(dim=0)       # [B, D]
        mean_aleatoric = aleatorics.mean(dim=0)   # [B, D]

        # Epistemic uncertainty from prediction variance across MC samples
        if predictions.shape[0] > 1:
            epistemic_from_var = predictions.var(dim=0, unbiased=True)  # [B, D]
            mean_epistemic = epistemics.mean(dim=0) + epistemic_from_var
        else:
            mean_epistemic = epistemics.mean(dim=0)

        return mean_pred, mean_aleatoric, mean_epistemic


class MultiTaskBEF(BEF_EEG):
    """
    Multi-task BEF for Challenge 1 & 2
    """
    
    def __init__(
        self,
        n_psycho_factors: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_multi_task = True
        
        # Additional task heads
        self.rt_head = nn.Linear(kwargs.get('gnn_hidden', 64), 1)
        self.success_head = nn.Linear(kwargs.get('gnn_hidden', 64), 1)
        
        # Psychopathology heads (Challenge 2)
        self.psycho_heads = nn.ModuleList([
            nn.Linear(kwargs.get('gnn_hidden', 64), 1)
            for _ in range(n_psycho_factors)
        ])
        
    def forward_multitask(
        self,
        x: torch.Tensor,
        task: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass
        """
        # Get base BEF outputs
        base_out = self.forward(x, return_intermediates=False, mc_samples=1)
        
        # Extract fusion embeddings from the fusion output
        # Need to get it directly from fusion module
        with torch.no_grad():
            paths = self.bicep.simulate_paths(x, N_paths=self.n_paths)
        path_mean = paths.mean(dim=0)
        enn_Z, enn_alpha, _ = self.enn(path_mean)
        A = build_sensor_graph(x, k=8, use_correlation=True)
        node_features = self.extract_channel_features(
            enn_Z, enn_alpha, (path_mean, paths.std(dim=0)), x
        )
        fusion_out = self.fusion(node_features, A, mc_samples=1)
        fusion_embedding = fusion_out['global_embedding']
        
        outputs = {}
        
        if task in ["challenge1", "all"]:
            # Reaction time regression
            outputs['rt_prediction'] = self.rt_head(fusion_embedding)
            outputs['rt_uncertainty'] = base_out['total_uncertainty']
            
            # Success/failure classification
            outputs['success_logits'] = self.success_head(fusion_embedding)
            
        if task in ["challenge2", "all"]:
            # Psychopathology factors
            psycho_preds = []
            for head in self.psycho_heads:
                psycho_preds.append(head(fusion_embedding))
            
            outputs['psycho_predictions'] = torch.cat(psycho_preds, dim=-1)
            outputs['psycho_uncertainty'] = base_out['total_uncertainty'].expand(-1, len(self.psycho_heads))
        
        return outputs


class PretrainableBEF(BEF_EEG):
    """
    BEF with pretraining capabilities
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Contrastive learning head for pretraining
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.K, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def compute_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        SimCLR-style contrastive loss for pretraining
        """
        # Project embeddings
        h1 = self.contrastive_head(z1)
        h2 = self.contrastive_head(z2)
        
        # Normalize
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        
        # Compute similarity
        sim = torch.matmul(h1, h2.T) / temperature
        
        # Contrastive loss
        labels = torch.arange(h1.shape[0], device=h1.device)
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def pretrain_forward(
        self,
        x: torch.Tensor,
        augment: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining
        """
        if augment:
            # Create two augmented views
            x1 = x + 0.1 * torch.randn_like(x)  # Noise augmentation
            x2 = x * (0.9 + 0.2 * torch.rand_like(x))  # Amplitude augmentation
        else:
            x1 = x2 = x
        
        # Get ENN embeddings for both views
        z1, _, _ = self.enn(x1)
        z2, _, _ = self.enn(x2)
        
        # Compute contrastive loss
        loss = self.compute_contrastive_loss(z1, z2)
        
        return {'contrastive_loss': loss, 'z1': z1, 'z2': z2}
