#!/usr/bin/env python3
"""
EEG Foundation Challenge 2025 Submission Script.

This module provides the PyTorch inference entry-point for both Challenge 1 (reaction time)
and Challenge 2 (psychopathology factors). The current release focuses on stabilising the
Challenge 1 stack by ensuring all ensemble heads execute correctly while maintaining
deterministic, GPU-aware inference.

Key enhancements in this revision:
    * Riemannian head execution (tensor ↔ ndarray bridge + diagnostics)
    * Defensive fusion logic that zero-fills missing heads
    * Feature branch cooling (clip + gain)
    * Reduced affine defaults (A=0.40, B=1.70)
    * BEF simple head disabled by default (can be toggled via BEF_SIMPLE_HEAD)

Constraints respected:
    * Single GPU (≤20 GB)
    * Offline-safe (no runtime downloads)
    * Deterministic torch / numpy seeds
    * Type hints, docstrings, and logging for operators

Author: EEG Foundation Challenge 2025 ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]

import importlib


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEED = int(os.getenv("GLOBAL_SEED", "1337"))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


def _ensure_sys_path(directory: Path) -> None:
    """Ensure the given directory is present on sys.path for dynamic imports."""
    resolved = str(directory.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _extract_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Return a state_dict from various checkpoint formats if available."""
    if isinstance(checkpoint, dict):
        # Direct state_dict (all tensor keys)
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            if any("." in key for key in checkpoint.keys()):
                return checkpoint

        # Common wrapper key
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict

    return None


def _load_module_from_artifact(path: Path, device: torch.device, model_name: str) -> nn.Module:
    """
    Load a model from an artifact path, supporting TorchScript, pickled modules, and raw state_dicts.

    Parameters
    ----------
    path:
        Path to the model artifact (.pt / .pth).
    device:
        Target device for the loaded model.
    model_name:
        Logical name (e.g. "c1_bef") used for logging and architecture selection.

    Returns
    -------
    nn.Module
        An evaluation-mode PyTorch module resident on the requested device.

    Raises
    ------
    RuntimeError
        If the artifact format is not recognised or compatible.
    """
    LOGGER.info("Loading model '%s' from %s", model_name, path)

    # Attempt TorchScript first for backwards compatibility.
    try:
        module = torch.jit.load(str(path), map_location=device)
        LOGGER.info("Loaded '%s' as TorchScript module", model_name)
        return module.to(device).eval()
    except (RuntimeError, ValueError) as jit_error:
        LOGGER.debug("TorchScript load failed for '%s': %s", model_name, jit_error)

    # Fallback: torch.load (supports pickled nn.Module or dict checkpoints).
    try:
        # PyTorch 2.6+ changed default to weights_only=True, but our trusted checkpoints need False
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except Exception as load_error:
        raise RuntimeError(f"Failed to torch.load artifact for '{model_name}': {load_error}") from load_error

    if isinstance(checkpoint, nn.Module):
        LOGGER.info("Loaded '%s' as pickled nn.Module", model_name)
        return checkpoint.to(device).eval()

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        module = checkpoint["model"]
        if not isinstance(module, nn.Module):
            raise RuntimeError(f"Checkpoint['model'] for '{model_name}' is not an nn.Module")
        LOGGER.info("Loaded '%s' from checkpoint['model']", model_name)
        return module.to(device).eval()

    # Handle training checkpoint with 'model_state_dict' key
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        LOGGER.info("Detected training checkpoint with 'model_state_dict' for '%s'", model_name)
    else:
        state_dict = _extract_state_dict(checkpoint)
    if state_dict is not None:
        LOGGER.info("Detected state_dict format for '%s' (sample keys: %s)", model_name, list(state_dict)[:5])

        if "bef" in model_name.lower():
            model = _instantiate_bef_model(model_name=model_name, device=device, artifact_dir=path.parent)
        elif "eegnet" in model_name.lower():
            model = _instantiate_eegnet_model(device=device, artifact_dir=path.parent)
        else:
            raise RuntimeError(f"State_dict detected but loader does not know how to instantiate '{model_name}'")

        try:
            # Use strict=False to allow architecture evolution (e.g., missing readout_norm layers)
            load_result = model.load_state_dict(state_dict, strict=False)
            # PyTorch <2.0 returns None, >=2.0 returns _IncompatibleKeys; both are handled here.
            if load_result is not None:
                missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
                if missing:
                    LOGGER.warning("Missing keys in state_dict for '%s': %s", model_name, missing)
                if unexpected:
                    LOGGER.warning("Unexpected keys in state_dict for '%s': %s", model_name, unexpected)
        except RuntimeError as state_error:
            raise RuntimeError(f"Failed to load state_dict for '{model_name}': {state_error}") from state_error

        LOGGER.info("State_dict loaded successfully for '%s'", model_name)
        return model.to(device).eval()

    raise RuntimeError(
        f"Unknown checkpoint format for {model_name} at {path}. "
        f"Available keys: {list(checkpoint)[:5] if isinstance(checkpoint, dict) else type(checkpoint)}"
    )


def _instantiate_bef_model(model_name: str, device: torch.device, artifact_dir: Path) -> nn.Module:
    """
    Instantiate the BEF_EEG architecture with challenge-specific output dimensions.

    Parameters
    ----------
    model_name:
        Identifier used to choose between C1 and C2 output dimensions.
    device:
        Device where the model should reside.
    artifact_dir:
        Directory containing the architecture source files.

    Returns
    -------
    nn.Module
        Freshly constructed BEF_EEG model on the requested device.

    Raises
    ------
    RuntimeError
        If the BEF_EEG class cannot be imported.
    """
    _ensure_sys_path(artifact_dir)

    # Locate the BEF_EEG definition (in pipeline.py).
    candidate_modules = ("pipeline", "bicep_eeg", "fusion_alpha")
    bef_class = None
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            bef_class = getattr(module, "BEF_EEG")
            LOGGER.info("BEF_EEG imported from %s.py", module_name)
            break
        except (ImportError, AttributeError) as e:
            LOGGER.debug("Could not import BEF_EEG from %s: %s", module_name, e)
            continue

    if bef_class is None:
        raise RuntimeError(
            "Failed to import BEF_EEG architecture. Expected pipeline.py, bicep_eeg.py, or fusion_alpha.py alongside the checkpoint."
        )

    model_name_lower = model_name.lower()
    if "c1" in model_name_lower:
        output_dim = 1
    elif "c2" in model_name_lower:
        output_dim = 4
    else:
        raise RuntimeError(f"Cannot infer output_dim for '{model_name}'. Expected model name to contain 'c1' or 'c2'.")

    # Instantiate with validated configuration (v35a baseline).
    # BEF_EEG uses in_chans (not n_chans), and doesn't take n_times or n_ensemble
    model = bef_class(
        in_chans=129,
        sfreq=100,
        K=8,
        output_dim=output_dim,
        device=str(device),
    )

    LOGGER.info("Instantiated BEF_EEG for '%s' with output_dim=%d", model_name, output_dim)
    return model.to(device)


def _instantiate_eegnet_model(device: torch.device, artifact_dir: Path) -> nn.Module:
    """
    Instantiate the EEGNetC1 architecture from the artifact directory.

    Parameters
    ----------
    device:
        Target device.
    artifact_dir:
        Directory where the EEGNet definition lives.

    Returns
    -------
    nn.Module
        EEGNetC1 instance on the requested device.

    Raises
    ------
    RuntimeError
        If EEGNetC1 cannot be imported.
    """
    _ensure_sys_path(artifact_dir)

    try:
        eegnet_module = importlib.import_module("c1_eegnet")
        eegnet_class = getattr(eegnet_module, "EEGNetC1")
    except (ImportError, AttributeError) as import_error:
        raise RuntimeError(
            f"Failed to import EEGNetC1 architecture from {artifact_dir}: {import_error}"
        ) from import_error

    model = eegnet_class(
        n_chans=129,
        n_samples=200,
    )

    LOGGER.info("Instantiated EEGNetC1 model")
    return model.to(device)


@dataclass
class StackConfig:
    """Configuration for linear fusion of model heads."""
    weights: torch.Tensor
    bias: float
    components: List[str]


def _ensure_dir(path: Union[str, Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Expected path '{resolved}' to exist.")
    return resolved


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_joblib_load(path: Path) -> Any:
    if joblib is None:
        raise ImportError("joblib is required to load non-Torch models.")
    return joblib.load(path)


def _as_column_tensor(t: Tensor, name: str) -> Tensor:
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    elif t.ndim > 2:
        # Flatten while preserving batch dimension; keep first column if wider.
        batch = t.shape[0]
        t = t.reshape(batch, -1)
    if t.shape[1] != 1:
        LOGGER.warning("Component '%s' produced width %d; keeping first column only.", name, t.shape[1])
        t = t[:, :1]
    return t.contiguous()


def _to_device(t: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
    if t is None:
        return None
    return t.to(device=device, dtype=torch.float32, non_blocking=True)


class Challenge1Model(nn.Module):
    """
    Challenge 1 reaction time model stack.

    Components:
        * BEF_EEG backbone (Torch)
        * EEGNetC1 (Torch) - currently weight 0.0 but still available for future tuning
        * Feature regressor (Torch / joblib)
        * Riemannian ensemble (joblib / JSON)

    The forward pass produces (batch_size, 1) reaction time predictions.
    """

    clamp_min: float
    clamp_max: float

    def __init__(
        self,
        weights_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        stack_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = _ensure_dir(weights_dir)

        # Blend configuration
        self.stack_config = self._load_stack_config(stack_override)
        self.stack_components: List[str] = self.stack_config.components
        self.stack_weights: Tensor = self.stack_config.weights.to(self.device)
        self.stack_bias: float = self.stack_config.bias

        # Affine & clamp parameters (env overridable)
        self.affine_a: float = float(os.getenv("C1_AFFINE_A", "0.40"))
        self.affine_b: float = float(os.getenv("C1_AFFINE_B", "1.70"))
        self.clamp_min = float(os.getenv("C1_CLAMP_MIN", "0.12"))
        self.clamp_max = float(os.getenv("C1_CLAMP_MAX", "2.50"))

        # Feature cooling
        self.feature_clip_value: float = float(os.getenv("FEATURE_CLIP_VALUE", "5.0"))
        self.feature_gain: float = float(os.getenv("FEATURE_GAIN", "0.2"))

        # Optional diagnostics
        self.verbose = os.getenv("C1_VERBOSE", "0") == "1"

        # Model components
        simple_head = bool(int(os.getenv("BEF_SIMPLE_HEAD", "0")))
        self.bef_model = self._load_bef_model(simple_head)
        self.eeg_model = self._load_eegnet_model()
        self.feature_model = self._load_feature_regressor()
        self.feature_scaler = self._load_feature_scaler()
        self.riem_model = self._load_riemannian_model()

        # Internal buffers for diagnostics
        self._last_pre_affine: Optional[Tensor] = None
        self._last_post_affine: Optional[Tensor] = None
        self._last_output: Optional[Tensor] = None

        self.to(self.device)
        LOGGER.info("Challenge1Model initialised with components: %s", self.stack_components)

    # ----------------------------------------------------------------------------------
    # Loading utilities
    # ----------------------------------------------------------------------------------

    def _load_stack_config(self, override: Optional[Dict[str, Any]]) -> StackConfig:
        cfg_path = self.weights_dir / "c1_stack.json"
        data = _load_json(cfg_path)

        if override is not None:
            LOGGER.warning("Applying stack override for testing: %s", override)
            data.update(override)

        components = data.get("components")
        weights = data.get("w")
        bias = data.get("b", 0.0)

        if not isinstance(components, list) or not isinstance(weights, list):
            raise ValueError("Invalid stack configuration: 'components' and 'w' must be lists.")

        if len(components) != len(weights):
            raise ValueError("Length mismatch between components and weights.")

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return StackConfig(weights=weights_tensor, bias=float(bias), components=components)

    def _load_bef_model(self, simple_head: bool) -> Optional[nn.Module]:
        path = self.weights_dir / "c1_bef.pt"
        if not path.exists():
            LOGGER.warning("BEF model not found at %s", path)
            return None

        model = _load_module_from_artifact(
            path=path,
            device=self.device,
            model_name="c1_bef",
        )

        # Configure simple_head if supported
        if hasattr(model, "set_simple_head"):
            model.set_simple_head(simple_head)
            LOGGER.info("Set simple_head=%s via set_simple_head()", simple_head)
        elif hasattr(model, "simple_head"):
            model.simple_head = simple_head
            LOGGER.info("Set simple_head=%s via attribute", simple_head)
        else:
            LOGGER.debug("BEF model has no simple_head configuration")

        model.eval()
        return model

    def _load_eegnet_model(self) -> Optional[nn.Module]:
        path = self.weights_dir / "c1_eegnet.pt"
        if not path.exists():
            LOGGER.info("EEGNet branch skipped (file missing).")
            return None

        model = _load_module_from_artifact(
            path=path,
            device=self.device,
            model_name="c1_eegnet",
        )
        model.eval()
        return model

    def _load_feature_regressor(self) -> Optional[Any]:
        candidates = [
            self.weights_dir / "c1_pca_ridge.json",  # Primary: PCA+Ridge JSON model
            self.weights_dir / "c1_feature.pt",
            self.weights_dir / "c1_feature.pkl",
            self.weights_dir / "c1_feature.joblib",
        ]
        for path in candidates:
            if not path.exists():
                continue
            LOGGER.info("Loading feature regressor from %s", path)
            if path.suffix == ".pt":
                model = torch.load(str(path), map_location=self.device)
                if isinstance(model, dict) and "model" in model:
                    model = model["model"]
                if isinstance(model, nn.Module):
                    model.eval()
                return model
            elif path.suffix == ".json":
                data = _load_json(path)
                # PCA+Ridge JSON format (flat structure: StandardScaler -> PCA -> Ridge)
                class JsonFeatureWrapper:
                    """PCA+Ridge wrapper matching flat JSON structure with full safety."""

                    def __init__(self, cfg: Dict[str, Any]) -> None:
                        # Load from FLAT structure (NOT nested)
                        self.scaler_mean = np.asarray(cfg.get('scaler_mean', []), dtype=np.float32)
                        self.scaler_std = np.asarray(cfg.get('scaler_std', []), dtype=np.float32)
                        self.pca_mean = np.asarray(cfg.get('pca_mean', []), dtype=np.float32)
                        self.pca_components = np.asarray(cfg.get('pca_components', []), dtype=np.float32)
                        self.ridge_coef = np.asarray(cfg.get('ridge_coef', []), dtype=np.float32)
                        self.ridge_intercept = float(cfg.get('ridge_intercept', 0.0))

                        # Validate shapes (663 features -> 64 PCA components -> 1 output)
                        if self.scaler_mean.size == 0 or self.pca_components.size == 0:
                            raise ValueError(f"Empty arrays in JSON: scaler_mean={self.scaler_mean.shape}, "
                                           f"pca_components={self.pca_components.shape}")

                    def predict(self, X: np.ndarray) -> np.ndarray:
                        """
                        Full sklearn pipeline: StandardScaler -> PCA -> Ridge

                        Args:
                            X: Input features (B, 663)

                        Returns:
                            Predictions (B, 1)
                        """
                        # 1. StandardScaler transform
                        safe_mean = np.nan_to_num(self.scaler_mean, nan=0.0, posinf=0.0, neginf=0.0)
                        safe_std = np.nan_to_num(self.scaler_std, nan=1.0, posinf=1.0, neginf=1.0)
                        safe_std = np.where(safe_std < 1e-6, 1.0, safe_std)
                        safe_std = np.clip(safe_std, 1e-6, 1e6)

                        X_scaled = (X - safe_mean) / safe_std
                        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                        X_scaled = np.clip(X_scaled, -10.0, 10.0)

                        # 2. PCA transform
                        X_centered = X_scaled - self.pca_mean
                        X_pca = X_centered @ self.pca_components.T  # (B, 663) @ (64, 663).T -> (B, 64)
                        X_pca = np.clip(X_pca, -5.0, 5.0)

                        # 3. Ridge predict
                        y = X_pca @ self.ridge_coef + self.ridge_intercept  # (B, 64) @ (64,) + scalar -> (B,)
                        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                        # Reshape to (B, 1) and ensure float32
                        return y.reshape(-1, 1).astype(np.float32)

                LOGGER.info("Loaded PCA+Ridge feature model from JSON")
                return JsonFeatureWrapper(data)
            else:
                return _maybe_joblib_load(path)
        LOGGER.info("Feature regressor not found; feature branch will fallback to zeros.")
        return None

    def _load_feature_scaler(self) -> Optional[Any]:
        candidates = [
            self.weights_dir / "c1_feature_scaler.pkl",
            self.weights_dir / "c1_feature_scaler.joblib",
        ]
        for path in candidates:
            if path.exists():
                LOGGER.info("Loading feature scaler from %s", path)
                return _maybe_joblib_load(path)
        return None

    def _load_riemannian_model(self) -> Optional[Any]:
        candidates = [
            self.weights_dir / "c1_riem_ridge.json",  # Primary: ridge-based JSON model
            self.weights_dir / "c1_riem.pkl",
            self.weights_dir / "c1_riem.joblib",
            self.weights_dir / "c1_riem.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            LOGGER.info("Loading Riemannian model from %s", path)
            if path.suffix in {".pkl", ".joblib"}:
                return _maybe_joblib_load(path)
            if path.suffix == ".json":
                data = _load_json(path)

                class JsonRiemannianWrapper:
                    """Log-Euclidean Riemannian + Ridge wrapper for JSON config."""

                    def __init__(self, cfg: Dict[str, Any]) -> None:
                        if cfg.get("type") != "riem_ridge":
                            raise ValueError(f"Expected type='riem_ridge', got {cfg.get('type')}")
                        channels = cfg.get("channels")
                        self.channels = np.asarray(channels, dtype=np.int64) if channels is not None else None
                        self.eps = float(cfg.get("eps", 1e-6))

                        scaler = cfg.get("scaler", {})
                        self.scaler_mean = np.asarray(scaler.get("mean_", []), dtype=np.float32)
                        self.scaler_std = np.asarray(scaler.get("scale_", []), dtype=np.float32)

                        ridge = cfg.get("ridge", {})
                        self.ridge_coef = np.asarray(ridge.get("coef_", []), dtype=np.float32).reshape(1, -1)
                        self.ridge_intercept = float(ridge.get("intercept_", 0.0))

                    def _logeuclid_cov(self, x: np.ndarray) -> np.ndarray:
                        """Compute log-Euclidean covariance features for a single window."""
                        C, T = x.shape
                        xm = x - x.mean(axis=1, keepdims=True)
                        cov = (xm @ xm.T) / max(T, 1)
                        cov = cov + (self.eps * np.eye(C, dtype=cov.dtype))
                        lam, V = np.linalg.eigh(cov)
                        lam = np.clip(lam, self.eps, None)
                        loglam = np.log(lam)
                        logC = (V * loglam[None, :]) @ V.T
                        return logC

                    def _vec_upper(self, mat: np.ndarray) -> np.ndarray:
                        iu = np.triu_indices_from(mat)
                        return mat[iu]

                    def _cov_tangent_features(self, batch: np.ndarray) -> np.ndarray:
                        """Compute log-Euclidean tangent features for a batch."""
                        x = batch[:, self.channels, :] if self.channels is not None else batch
                        feats = []
                        for i in range(x.shape[0]):
                            logC = self._logeuclid_cov(x[i])
                            feats.append(self._vec_upper(logC))
                        return np.stack(feats, axis=0).astype(np.float32)

                    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
                        """Predict from EEG array (B, C, T) -> (B, 1)."""
                        # Compute tangent space features
                        feats = self._cov_tangent_features(batch)

                        # Safety clip raw features
                        feats = np.clip(feats, -5.0, 5.0)

                        # Sanitize scaler parameters
                        safe_mean = np.nan_to_num(self.scaler_mean, nan=0.0, posinf=0.0, neginf=0.0)
                        safe_std = np.nan_to_num(self.scaler_std, nan=1.0, posinf=1.0, neginf=1.0)
                        safe_std = np.where(safe_std < 1e-6, 1.0, safe_std)
                        safe_std = np.clip(safe_std, 1e-6, 1e6)

                        # Standardize
                        xf = (feats - safe_mean) / safe_std
                        xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
                        xf = np.clip(xf, -10.0, 10.0)

                        # Ridge regression
                        y = xf @ self.ridge_coef.T + self.ridge_intercept
                        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        y = np.clip(y, 0.1, 3.0)

                        return y

                wrapper = JsonRiemannianWrapper(data)
                LOGGER.info("Loaded Riemannian ridge model from JSON: channels=%s, eps=%g",
                           wrapper.channels if wrapper.channels is not None else "all", wrapper.eps)
                return wrapper
        LOGGER.info("Riemannian model not present; branch will fill zeros.")
        return None

    # ----------------------------------------------------------------------------------
    # Forward helpers
    # ----------------------------------------------------------------------------------

    def _extract_eeg_tensor(self, batch: Union[Tensor, Dict[str, Tensor]]) -> Optional[Tensor]:
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, dict):
            eeg = batch.get("eeg") or batch.get("x")
            return eeg
        LOGGER.error("Unsupported batch type for EEG extraction: %s", type(batch))
        return None

    def _extract_feature_tensor(self, batch: Union[Tensor, Dict[str, Tensor]], eeg_tensor: Optional[Tensor] = None) -> Optional[Tensor]:
        # First, check if pre-computed features are provided in batch dict
        if isinstance(batch, dict):
            features = batch.get("features") or batch.get("feat")
            if features is not None:
                return features

        # If no pre-computed features, compute on-the-fly from EEG tensor
        if eeg_tensor is not None:
            try:
                # Import feature extraction function
                from feature_model import compute_features_batch

                # Convert EEG tensor to numpy (B, C, T)
                eeg_np = eeg_tensor.detach().cpu().numpy().astype(np.float32)

                # Compute features on-the-fly (sfreq=100 Hz is fixed by competition spec)
                features_np, _ = compute_features_batch(eeg_np, sfreq=100.0)

                # Convert back to tensor
                features_tensor = torch.from_numpy(features_np).to(self.device, dtype=torch.float32)
                return features_tensor
            except Exception as exc:
                LOGGER.warning("On-the-fly feature extraction failed (%s); feature branch will use zeros.", exc)
                return None

        return None

    def forward(self, batch: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """
        Forward pass through the Challenge 1 ensemble.

        Args:
            batch: Either a raw EEG tensor (B, C, T) or dict with keys `eeg` and optional `features`.

        Returns:
            Tensor of shape (B, 1) with clamped reaction time predictions.
        """
        eeg_tensor = self._extract_eeg_tensor(batch)
        if eeg_tensor is None:
            raise ValueError("EEG tensor must be provided for Challenge 1 inference.")

        eeg_tensor = _to_device(eeg_tensor, self.device)
        if eeg_tensor is None:
            raise RuntimeError("Failed to move EEG tensor to target device.")

        batch_size = eeg_tensor.shape[0]
        if batch_size == 0:
            LOGGER.warning("Received empty batch; returning empty tensor.")
            empty = torch.empty((0, 1), device=self.device, dtype=torch.float32)
            self._record_stats(empty, empty, empty)
            return empty

        preds: Dict[str, Tensor] = {}
        # BEF branch
        if "bef" in self.stack_components and self.bef_model is not None:
            with torch.no_grad():
                bef_out = self.bef_model(eeg_tensor)
            # BEF returns dict with 'prediction' key
            if isinstance(bef_out, dict):
                bef_out = bef_out['prediction']
            preds["bef"] = _as_column_tensor(bef_out, "bef").to(self.device)

        # EEGNet branch
        if "eeg" in self.stack_components and self.eeg_model is not None:
            with torch.no_grad():
                eeg_out = self.eeg_model(eeg_tensor)
            preds["eeg"] = _as_column_tensor(eeg_out, "eeg").to(self.device)

        # Feature branch
        if "feature" in self.stack_components:
            feature_tensor = self._extract_feature_tensor(batch, eeg_tensor=eeg_tensor)
            feature_pred = self._run_feature_branch(feature_tensor, batch_size)
            preds["feature"] = feature_pred

        # Riemannian branch (critical fix)
        if "riem" in self.stack_components and self.riem_model is not None:
            preds["riem"] = self._run_riemannian_branch(eeg_tensor, batch_size)

        # Defensive fill for missing components
        ordered_preds: List[Tensor] = []
        for idx, name in enumerate(self.stack_components):
            if name not in preds:
                print(f"FUSION_WARN: missing '{name}' -> filled zeros")
                preds[name] = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
            ordered_preds.append(preds[name])

        # Weighted fusion
        fused = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
        for idx, name in enumerate(self.stack_components):
            fused = fused + ordered_preds[idx] * self.stack_weights[idx]

        if self.stack_bias:
            fused = fused + self.stack_bias

        pre_affine = fused.clone()
        post_affine = fused * self.affine_a + self.affine_b
        clamped = torch.clamp(post_affine, min=self.clamp_min, max=self.clamp_max)

        self._record_stats(pre_affine, post_affine, clamped)

        if self.verbose:
            pa_mean = float(pre_affine.mean().item())
            pa_std = float(pre_affine.std(unbiased=False).item())
            LOGGER.info("Fusion pre-affine mean=%.4f std=%.4f", pa_mean, pa_std)

        return clamped

    # ----------------------------------------------------------------------------------
    # Branch helpers
    # ----------------------------------------------------------------------------------

    def _run_feature_branch(self, feature_tensor: Optional[Tensor], batch_size: int) -> Tensor:
        if self.feature_model is None:
            return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

        if feature_tensor is None:
            LOGGER.warning("Feature branch requested but no features provided; filling zeros.")
            return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

        features = feature_tensor.detach().cpu().float().numpy()
        if self.feature_scaler is not None:
            try:
                features = self.feature_scaler.transform(features)
            except Exception as exc:  # pragma: no cover - scaler optional
                LOGGER.warning("Feature scaler transform failed (%s); proceeding without scaling.", exc)

        # Model inference
        if isinstance(self.feature_model, nn.Module):
            with torch.no_grad():
                feat_out = self.feature_model(torch.from_numpy(features).to(self.device))
            feat_out = feat_out.to(torch.float32)
        else:
            feat_pred = self.feature_model.predict(features)  # type: ignore[attr-defined]
            feat_out = torch.from_numpy(np.asarray(feat_pred, dtype=np.float32)).to(self.device)

        feat_out = _as_column_tensor(feat_out, "feature")
        # Feature cooling
        feat_out = torch.clamp(feat_out, -self.feature_clip_value, self.feature_clip_value)
        feat_out = feat_out * self.feature_gain
        return feat_out

    def _run_riemannian_branch(self, eeg_tensor: Tensor, batch_size: int) -> Tensor:
        try:
            if isinstance(eeg_tensor, torch.Tensor):
                eeg_np = eeg_tensor.detach().to("cpu").numpy()
            else:
                eeg_np = eeg_tensor  # type: ignore[assignment]

            with torch.no_grad():
                y_riem = self.riem_model.predict_batch(eeg_np)  # type: ignore[attr-defined]

            y_riem = np.asarray(y_riem)
            if y_riem.ndim == 1:
                y_riem = y_riem[:, None]

            y_riem_t = torch.from_numpy(y_riem).to(self.device, dtype=torch.float32)

            if os.getenv("RIEM_VERBOSE", "0") == "1":
                mean = float(y_riem_t.mean().item())
                std = float(y_riem_t.std(unbiased=False).item())
                print(f"RIEM_DIAG: batch_mean={mean:.6f} batch_std={std:.6f} shape={tuple(y_riem_t.shape)}")

            return _as_column_tensor(y_riem_t, "riem")
        except Exception as exc:
            print(f"RIEM_WARN: predict_batch failed ({type(exc).__name__}: {exc}); using zeros")
            return torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

    # ----------------------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------------------

    def _record_stats(self, pre_affine: Tensor, post_affine: Tensor, output: Tensor) -> None:
        self._last_pre_affine = pre_affine.detach()
        self._last_post_affine = post_affine.detach()
        self._last_output = output.detach()

    def last_forward_stats(self) -> Dict[str, float]:
        """Return aggregated statistics from the most recent forward pass."""
        if self._last_output is None:
            raise RuntimeError("Forward pass has not been executed yet.")
        pre = self._last_pre_affine
        post = self._last_post_affine
        out = self._last_output
        assert pre is not None and post is not None

        return {
            "pre_affine_mean": float(pre.mean().item()),
            "pre_affine_std": float(pre.std(unbiased=False).item()),
            "post_affine_mean": float(post.mean().item()),
            "post_affine_std": float(post.std(unbiased=False).item()),
            "output_mean": float(out.mean().item()),
            "output_std": float(out.std(unbiased=False).item()),
        }


class Challenge2Model(nn.Module):
    """
    Placeholder Challenge 2 model.

    NOTE: The Challenge 2 pipeline remains unchanged from previous releases.
    This class simply loads the BEF backbone and calibrated regressor as before.
    """

    def __init__(self, weights_dir: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = _ensure_dir(weights_dir)
        self.bef_model = self._load_bef_model()
        self.calib_path = self.weights_dir / "c2_calib.json"
        self.calibrator = self._load_calibrator()
        self.to(self.device)

    def _load_bef_model(self) -> nn.Module:
        path = self.weights_dir / "c2_bef.pt"
        if not path.exists():
            raise FileNotFoundError(f"C2 BEF model missing at {path}")

        model = _load_module_from_artifact(
            path=path,
            device=self.device,
            model_name="c2_bef",
        )
        model.eval()
        return model

    def _load_calibrator(self) -> Optional[Any]:
        if not self.calib_path.exists():
            LOGGER.warning("C2 calibrator not found; outputs will be uncalibrated.")
            return None
        data = _load_json(self.calib_path)
        return data

    def forward(self, batch: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        if isinstance(batch, dict):
            eeg = batch.get("eeg") or batch.get("x")
        else:
            eeg = batch
        if eeg is None:
            raise ValueError("C2 model expects EEG tensor.")
        eeg = _to_device(eeg, self.device)
        assert eeg is not None
        with torch.no_grad():
            preds = self.bef_model(eeg)
        # BEF returns dict with 'prediction' key
        if isinstance(preds, dict):
            preds = preds['prediction']
        # Extract Externalizing factor (column 2 of 4: p_factor, internalizing, externalizing, attention)
        if preds.shape[-1] == 4:
            preds = preds[:, 2:3]  # Shape: (B, 4) -> (B, 1)
        preds = _as_column_tensor(preds, "c2_bef")
        return preds


def get_model(task: str, weights_dir: Union[str, Path]) -> nn.Module:
    """
    Factory for challenge models.

    Args:
        task: Either "c1" or "c2".
        weights_dir: Directory containing model artefacts.

    Returns:
        Instantiated nn.Module ready for inference.
    """
    task_lower = task.lower()
    if task_lower == "c1":
        return Challenge1Model(weights_dir)
    if task_lower == "c2":
        return Challenge2Model(weights_dir)
    raise ValueError(f"Unknown task '{task}'. Expected 'c1' or 'c2'.")
# ============================================================================
# COMPETITION API WRAPPER (Required by Codabench)
# ============================================================================

class Submission:
    """
    Competition API wrapper for Challenge 1 and Challenge 2 models.

    This class provides the interface expected by the Codabench evaluation script.
    """

    def __init__(self, SFREQ: int, DEVICE: torch.device) -> None:
        """
        Initialize submission with competition parameters.

        Args:
            SFREQ: Sampling frequency (expected: 100 Hz)
            DEVICE: torch.device for model placement
        """
        self.sfreq = SFREQ
        self.device = DEVICE

        # Models will be lazy-loaded in get_model_* methods
        self._model_c1: Optional[Challenge1Model] = None
        self._model_c2: Optional[Challenge2Model] = None

        # Weights directory - Codabench extracts zip contents to /app/input/res/
        # The actual files are in /app/input/res/v35c_PRODUCTION/ or just use __file__ parent
        self.weights_dir = Path(__file__).parent
        
        # Fallback: try common Codabench paths
        if not (self.weights_dir / "c1_stack.json").exists():
            candidates = [
                Path("/app/input/res/v35c_PRODUCTION"),
                Path("/app/input/res"),
                Path(__file__).parent / "v35c_PRODUCTION",
            ]
            for candidate in candidates:
                if (candidate / "c1_stack.json").exists():
                    self.weights_dir = candidate
                    break

        LOGGER.info(f"Submission initialized: sfreq={SFREQ}, device={DEVICE}, weights_dir={self.weights_dir}")

    def get_model_challenge_1(self) -> nn.Module:
        """
        Get Challenge 1 model (reaction time prediction).

        Returns:
            nn.Module: Challenge 1 model in eval mode
        """
        if self._model_c1 is None:
            LOGGER.info("Loading Challenge 1 model...")
            self._model_c1 = Challenge1Model(
                weights_dir=self.weights_dir,
                device=self.device
            )
            self._model_c1.eval()
            LOGGER.info("Challenge 1 model loaded and ready")

        return self._model_c1

    def get_model_challenge_2(self) -> nn.Module:
        """
        Get Challenge 2 model (psychopathology prediction).

        Returns:
            nn.Module: Challenge 2 model in eval mode
        """
        if self._model_c2 is None:
            LOGGER.info("Loading Challenge 2 model...")
            self._model_c2 = Challenge2Model(
                weights_dir=self.weights_dir,
                device=self.device
            )
            self._model_c2.eval()
            LOGGER.info("Challenge 2 model loaded and ready")

        return self._model_c2
