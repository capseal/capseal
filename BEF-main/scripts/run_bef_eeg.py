#!/usr/bin/env python3
"""Utility to run the BEF (BICEP → ENN → FusionAlpha) EEG model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from v46_submission import BEF_EEG
except ImportError as exc:  # pragma: no cover - packaging guard
    raise SystemExit(
        "Unable to import v46_submission package. Ensure the directory exists and contains __init__.py"
    ) from exc


def _load_signals(path: Path, sfreq: float = 100.0) -> torch.Tensor:
    """Load EEG data from .npy/.npz/.pt or generate dummy data if absent."""
    if path is None:
        raise ValueError("Input path must be provided")

    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")

    data: Any
    if path.suffix in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):  # type: ignore[attr-defined]
            # Use first array
            first_key = list(data.files)[0]
            data = data[first_key]
    elif path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported EEG file extension: {path.suffix}")

    if isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        arr = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(arr)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected data with shape [B, C, T], got {tuple(tensor.shape)}")

    # Basic normalization similar to feature_model.preprocess
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, keepdim=True).clamp(min=1e-6)
    tensor = (tensor - mean) / std
    return tensor


def _resolve_state_dict(artifact: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(artifact, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            # Heuristic: treat as state dict
            return checkpoint  # type: ignore[return-value]
    raise RuntimeError(f"Unsupported checkpoint format at {artifact}")


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    signals = _load_signals(Path(args.input)).to(device)

    model = BEF_EEG(
        in_chans=signals.shape[1],
        output_dim=args.output_dim,
        device=str(device),
        n_paths=args.n_paths,
        use_multiscale=args.use_multiscale,
    ).to(device)

    if args.checkpoint:
        state_dict = _resolve_state_dict(Path(args.checkpoint), device)
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    with torch.no_grad():
        outputs = model(signals, return_intermediates=args.return_intermediates, mc_samples=args.mc_samples)

    # Collate tensors to CPU for serialization
    json_ready: Dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            json_ready[key] = value.cpu().numpy().tolist()
        else:
            json_ready[key] = value

    Path(args.output).write_text(json.dumps(json_ready, indent=2))
    print(f"Saved BEF outputs to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BEF EEG pipeline on preprocessed trials")
    parser.add_argument("--input", required=True, help="Path to EEG tensor (.npy/.npz/.pt) shaped [B,C,T]")
    parser.add_argument("--checkpoint", help="Path to BEF checkpoint (.pt) e.g. v46_submission/c1_bef.pt")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "bef_outputs.json"))
    parser.add_argument("--output-dim", type=int, default=1, help="Prediction dimension (1 for C1, 4 for C2)")
    parser.add_argument("--n-paths", type=int, default=32, help="Number of stochastic paths per SDE")
    parser.add_argument("--mc-samples", type=int, default=1, help="MC samples for fusion uncertainty")
    parser.add_argument("--use-multiscale", action="store_true", help="Enable multiscale ENN encoder")
    parser.add_argument("--return-intermediates", action="store_true", help="Emit intermediate tensors")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
