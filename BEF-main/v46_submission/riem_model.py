"""
Riemannian (log-Euclidean) covariance features + Ridge - FIXED VERSION
Adds safety clips + better scaler handling to prevent explosion
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import torch


def _logeuclid_cov(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute log-Euclidean covariance features for a single window."""
    C, T = x.shape
    # remove per-channel mean
    xm = x - x.mean(axis=1, keepdims=True)
    cov = (xm @ xm.T) / max(T, 1)
    # jitter for SPD
    cov = cov + (eps * np.eye(C, dtype=cov.dtype))
    # eigendecomposition
    lam, V = np.linalg.eigh(cov)
    lam = np.clip(lam, eps, None)
    loglam = np.log(lam)
    logC = (V * loglam[None, :]) @ V.T
    return logC


def _vec_upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat)
    return mat[iu]


def cov_tangent_features(batch: np.ndarray, channels: Optional[np.ndarray], eps: float) -> np.ndarray:
    """Compute log-Euclidean tangent features for a batch of windows."""
    if channels is not None:
        x = batch[:, channels, :]
    else:
        x = batch
    feats: List[np.ndarray] = []
    for i in range(x.shape[0]):
        logC = _logeuclid_cov(x[i], eps=eps)
        feats.append(_vec_upper(logC))
    return np.stack(feats, axis=0).astype(np.float32)


@dataclass
class RiemannianRidge:
    channels: Optional[np.ndarray]
    eps: float
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: float
    ridge_alpha: float

    @classmethod
    def from_json(cls, path: Path) -> "RiemannianRidge":
        cfg = json.loads(Path(path).read_text())
        if cfg.get("type") != "riem_ridge":
            raise ValueError(f"Expected type='riem_ridge', got {cfg.get('type')}")
        channels = cfg.get("channels")
        ch = np.asarray(channels, dtype=np.int64) if channels is not None else None
        scaler = cfg.get("scaler", {})
        ridge = cfg.get("ridge", {})
        return cls(
            channels=ch,
            eps=float(cfg.get("eps", 1e-6)),
            scaler_mean=np.asarray(scaler.get("mean_", []), dtype=np.float32),
            scaler_std=np.asarray(scaler.get("scale_", []), dtype=np.float32),
            ridge_coef=np.asarray(ridge.get("coef_", []), dtype=np.float32).reshape(1, -1),
            ridge_intercept=float(ridge.get("intercept_", 0.0)),
            ridge_alpha=float(ridge.get("alpha", 0.0)),
        )

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) float32 torch tensor -> returns (B, 1) torch tensor

        FIXED VERSION: Clips features BEFORE scaler to prevent explosion
        """
        # Remember original device for output
        original_device = x.device
        arr = x.detach().cpu().numpy().astype(np.float32)

        # Compute tangent space features
        feats = cov_tangent_features(arr, self.channels, eps=self.eps)

        # === SAFETY CLIP 1: Raw features (prevents extreme covariance values) ===
        feats = np.clip(feats, -5.0, 5.0)

        # === SAFETY: Validate shape and sanitize scaler parameters ===
        if self.scaler_mean.shape[0] != feats.shape[1]:
            raise ValueError(f"Scaler dimension mismatch: expected {feats.shape[1]}, got {self.scaler_mean.shape[0]}")

        # Sanitize scaler stats (handle NaN/inf in scaler itself)
        safe_mean = np.nan_to_num(self.scaler_mean, nan=0.0, posinf=0.0, neginf=0.0)
        safe_std = np.nan_to_num(self.scaler_std, nan=1.0, posinf=1.0, neginf=1.0)

        # Replace near-zero std with 1.0, cap extremes
        safe_std = np.where(safe_std < 1e-6, 1.0, safe_std)
        safe_std = np.clip(safe_std, 1e-6, 1e6)  # Prevent both explosion and suppression

        # Standardize with safe scaler
        xf = (feats - safe_mean) / safe_std
        xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)

        # === SAFETY CLIP 2: Scaled features (catch any remaining explosion) ===
        xf = np.clip(xf, -10.0, 10.0)

        # Ridge regression
        y = xf @ self.ridge_coef.T + self.ridge_intercept
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # === SAFETY CLIP 3: Final output (ensure reasonable reaction times) ===
        y = np.clip(y, 0.1, 3.0)

        # Return tensor on same device as input
        return torch.from_numpy(y).to(original_device)
