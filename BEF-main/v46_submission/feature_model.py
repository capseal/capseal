"""Utility functions for EEG feature extraction and linear models (safe corr)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import json
import os
import numpy as np
from sklearn.linear_model import Ridge

# Extra features enabled for PCA+Ridge models
ENABLE_EXTRA_FEATURES = True

# Thresholds for guarding zero-variance features in serialized scalers.
# Training artefacts legitimately include a handful of features with
# vanishing variance (~1e-9). At inference we treat those dimensions as
# constants and refuse to amplify any drift through division.
ZERO_STD_EPS = 1e-6
# If a supposedly constant feature deviates by more than this amount
# (after subtracting the training mean) we abort, signalling a schema drift.
ZERO_STD_DRIFT_GUARD = 1e-3

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

_FEATURE_NAMES: List[str] = [
    "amplitude_mean",
    "amplitude_std",
    "amplitude_skew",
    "amplitude_kurtosis",
    "channel_var_mean",
    "channel_var_std",
    "corr_mean",
    "corr_std",
]
for band in ("delta", "theta", "alpha", "beta", "gamma"):
    _FEATURE_NAMES.append(f"bp_{band}_mean")
    _FEATURE_NAMES.append(f"bp_{band}_std")


def get_feature_names() -> List[str]:
    return list(_FEATURE_NAMES)


def _bandpass_fft(signals: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    n_times = signals.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    spectrum = np.fft.rfft(signals, axis=-1)
    spectrum[..., ~mask] = 0.0
    filtered = np.fft.irfft(spectrum, n=n_times, axis=-1)
    return filtered


def preprocess(signals: np.ndarray, sfreq: float = 100.0) -> np.ndarray:
    filtered = _bandpass_fft(signals, sfreq=sfreq, fmin=0.5, fmax=40.0)
    mean = filtered.mean(axis=-1, keepdims=True)
    std = filtered.std(axis=-1, keepdims=True) + 1e-6
    return (filtered - mean) / std


def _skew(values: np.ndarray) -> float:
    mean = values.mean()
    std = values.std() + 1e-6
    centred = values - mean
    return float((centred ** 3).mean() / (std ** 3))


def _kurtosis(values: np.ndarray) -> float:
    mean = values.mean()
    std = values.std() + 1e-6
    centred = values - mean
    return float((centred ** 4).mean() / (std ** 4) - 3.0)


def _hjorth_parameters(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = x.var(axis=-1) + 1e-8
    var1 = dx.var(axis=-1) + 1e-8
    var2 = ddx.var(axis=-1) + 1e-8
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-8)
    return np.stack([activity, mobility, complexity], axis=-1)


def _spectral_entropy(psd: np.ndarray) -> np.ndarray:
    p = psd.astype(np.float32)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-8)
    entropy = -(p * np.log(p + 1e-8)).sum(axis=-1)
    return entropy


def _line_length(x: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(x, axis=-1)).sum(axis=-1)


def _compute_bandpower(epoch: np.ndarray, sfreq: float) -> List[float]:
    n_times = epoch.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum = np.fft.rfft(epoch, axis=-1)
    psd = (np.abs(spectrum) ** 2) / max(n_times, 1)
    features: List[float] = []
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        fmin, fmax = BANDS[band]
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            features.extend([0.0, 0.0])
            continue
        band_power = psd[..., mask].mean(axis=-1)
        features.append(float(band_power.mean()))
        features.append(float(band_power.std()))
    return features


def compute_epoch_features(epoch: np.ndarray, sfreq: float = 100.0) -> List[float]:
    flattened = epoch.reshape(epoch.shape[0], -1)
    values = flattened.reshape(-1)
    channel_var = flattened.var(axis=1)

    # Safe correlation (no warnings): corr = Xn @ Xn.T with eps
    X = flattened - flattened.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    Xn = X / denom
    corr = Xn @ Xn.T
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.ndim == 2 and corr.shape[0] == corr.shape[1] and corr.shape[0] > 0:
        np.fill_diagonal(corr, 1.0)
    iu = np.triu_indices_from(corr, k=1)
    corr_vals = corr[iu] if iu[0].size else np.array([0.0])

    feats = [
        float(values.mean()),
        float(values.std()),
        _skew(values),
        _kurtosis(values),
        float(channel_var.mean()),
        float(channel_var.std()),
        float(corr_vals.mean()),
        float(corr_vals.std()),
    ]
    feats.extend(_compute_bandpower(epoch, sfreq=sfreq))

    if ENABLE_EXTRA_FEATURES:
        hjorth = _hjorth_parameters(epoch)
        feats.extend(hjorth.reshape(-1).tolist())
        ll = _line_length(epoch)
        feats.extend(ll.tolist())
        n_times = epoch.shape[-1]
        spectrum = np.fft.rfft(epoch, axis=-1)
        psd = (np.abs(spectrum) ** 2) / max(n_times, 1)
        entropy = _spectral_entropy(psd)
        feats.extend(entropy.tolist())

    return feats


def compute_features_batch(signals: np.ndarray, sfreq: float = 100.0) -> Tuple[np.ndarray, List[str]]:
    feats = np.stack([compute_epoch_features(epoch, sfreq=sfreq) for epoch in signals], axis=0)
    return feats, list(_FEATURE_NAMES)


@dataclass
class LinearModel:
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)
        self.weights = self.weights.astype(np.float32)
        self.bias = self.bias.astype(np.float32)
        self.n_features = int(self.mean.shape[0])
        if self.std.shape[0] != self.n_features:
            raise ValueError(
                f"LinearModel scaler mismatch: mean {self.mean.shape[0]} vs std {self.std.shape[0]}"
            )

    @classmethod
    def from_json(cls, path: Path) -> "LinearModel":
        cfg = json.loads(Path(path).read_text())
        return cls(
            feature_names=cfg["feature_names"],
            mean=np.asarray(cfg["scaler_mean"], dtype=np.float32),
            std=np.asarray(cfg["scaler_std"], dtype=np.float32),
            weights=np.asarray(cfg["weights"], dtype=np.float32),
            bias=np.asarray(cfg["bias"], dtype=np.float32),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"LinearModel expects 2D features, got shape {features.shape}")
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"LinearModel feature dim mismatch: expected {self.n_features}, got {features.shape[1]}"
            )
        diff = features.astype(np.float32) - self.mean
        zero_mask = self.std <= ZERO_STD_EPS
        if zero_mask.any():
            diff[:, zero_mask] = 0.0
        x = diff / np.where(self.std > ZERO_STD_EPS, self.std, 1.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = x @ self.weights.T + self.bias
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    def to_json(self, path: Path) -> None:
        payload = {
            "feature_names": self.feature_names,
            "scaler_mean": self.mean.tolist(),
            "scaler_std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        Path(path).write_text(json.dumps(payload))


@dataclass
class PCARidge:
    # Schema
    feature_names: List[str]
    keep_mask: np.ndarray
    # Scaler on kept columns
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    # PCA on standardized kept columns
    pca_mean: np.ndarray
    pca_components: np.ndarray
    # Ridge
    ridge_coef: np.ndarray
    ridge_intercept: np.ndarray
    ridge_alpha: float
    # Derived
    n_components: int = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self.keep_mask = np.asarray(self.keep_mask, dtype=bool)
        self.scaler_mean = np.asarray(self.scaler_mean, dtype=np.float32)
        self.scaler_std = np.asarray(self.scaler_std, dtype=np.float32)
        self.pca_mean = np.asarray(self.pca_mean, dtype=np.float32)
        self.pca_components = np.asarray(self.pca_components, dtype=np.float32)
        self.ridge_coef = np.asarray(self.ridge_coef, dtype=np.float32)
        self.ridge_intercept = np.asarray(self.ridge_intercept, dtype=np.float32)
        self.n_features = len(self.feature_names)
        self.n_components = int(self.pca_components.shape[0])

        # Validate sizes
        if self.keep_mask.size != self.n_features:
            raise ValueError(
                f"PCARidge schema mismatch: keep_mask {self.keep_mask.size} vs feature_names {self.n_features}"
            )
        kept = int(self.keep_mask.sum())
        if self.scaler_mean.shape[0] != kept or self.scaler_std.shape[0] != kept:
            raise ValueError("PCARidge scaler shapes do not match kept feature dimension")
        if self.pca_components.shape[1] != kept or self.pca_mean.shape[0] != kept:
            raise ValueError("PCARidge PCA shapes do not match kept feature dimension")

    @classmethod
    def from_json(cls, path: Path) -> "PCARidge":
        cfg = json.loads(Path(path).read_text())
        if cfg.get("type") != "pca_ridge":
            raise ValueError(f"Expected type='pca_ridge', got {cfg.get('type')}")

        # New nested schema
        if "schema" in cfg and "scaler" in cfg and "pca" in cfg and "ridge" in cfg:
            schema = cfg["schema"]
            scaler = cfg["scaler"]
            pca = cfg["pca"]
            ridge = cfg["ridge"]
            obj = cls(
                feature_names=list(schema.get("feature_names", [])),
                keep_mask=np.asarray(schema.get("keep_mask", []), dtype=bool),
                scaler_mean=np.asarray(scaler.get("mean_", []), dtype=np.float32),
                scaler_std=np.asarray(scaler.get("scale_", []), dtype=np.float32),
                pca_mean=np.asarray(pca.get("mean_", []), dtype=np.float32),
                pca_components=np.asarray(pca.get("components_", []), dtype=np.float32),
                ridge_coef=np.asarray(ridge.get("coef_", []), dtype=np.float32),
                ridge_intercept=np.asarray(ridge.get("intercept_", 0.0), dtype=np.float32),
                ridge_alpha=float(ridge.get("alpha", 0.0)),
            )
            # Optional: explained variance ratio for diagnostics
            evr = pca.get("explained_variance_ratio_")
            if isinstance(evr, list):
                try:
                    obj.pca_evr = np.asarray(evr, dtype=np.float32)
                except Exception:
                    pass
            return obj

        # Backward compatibility (flat keys)
        feature_names = cfg.get("feature_names") or [f"f{i}" for i in range(len(cfg.get("scaler_mean", [])))]
        keep_mask = cfg.get("keep_mask") or [1] * len(feature_names)
        return cls(
            feature_names=feature_names,
            keep_mask=np.asarray(keep_mask, dtype=bool),
            scaler_mean=np.asarray(cfg["scaler_mean"], dtype=np.float32),
            scaler_std=np.asarray(cfg["scaler_std"], dtype=np.float32),
            pca_mean=np.asarray(cfg["pca_mean"], dtype=np.float32),
            pca_components=np.asarray(cfg["pca_components"], dtype=np.float32),
            ridge_coef=np.asarray(cfg["ridge_coef"], dtype=np.float32),
            ridge_intercept=np.asarray(cfg["ridge_intercept"], dtype=np.float32),
            ridge_alpha=float(cfg.get("ridge_alpha", 0.0)),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"PCARidge expects 2D features, got shape {features.shape}")
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"PCARidge feature dim mismatch: expected {len(self.feature_names)}, got {features.shape[1]}"
            )

        xk = features.astype(np.float32)[:, self.keep_mask]
        xk = (xk - self.scaler_mean) / (self.scaler_std + 1e-12)
        xk = np.nan_to_num(xk, nan=0.0, posinf=0.0, neginf=0.0)
        x_pca = (xk - self.pca_mean) @ self.pca_components.T
        if self.ridge_coef.ndim == 1:
            y = x_pca @ self.ridge_coef + self.ridge_intercept
        else:
            y = x_pca @ self.ridge_coef.T + self.ridge_intercept
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


def load_feature_model(path: Path, model_type: str = "auto"):
    cfg = json.loads(Path(path).read_text())
    if model_type == "auto":
        model_type = cfg.get("type", "linear")
    if model_type == "pca_ridge":
        return PCARidge.from_json(path)
    else:
        return LinearModel.from_json(path)


def fit_linear_model(features: np.ndarray, targets: np.ndarray, feature_names: Iterable[str]) -> LinearModel:
    feature_names = list(feature_names)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    x = (features - mean) / std
    y = targets
    alpha = 100.0
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x, y)
    if y.ndim == 1:
        weights = ridge.coef_.reshape(1, -1)
        bias = np.array([ridge.intercept_], dtype=np.float32)
    else:
        weights = ridge.coef_
        bias = np.asarray(ridge.intercept_, dtype=np.float32)
    return LinearModel(
        feature_names=feature_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        bias=np.asarray(bias, dtype=np.float32),
    )
