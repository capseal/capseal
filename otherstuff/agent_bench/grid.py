#!/usr/bin/env python3
"""Grid generation for AgentEvalBench v1.

Parameter space (1024 points = 4^5):
    tool_noise:       0, 1, 2, 3       (int) - Noise magnitude for lookup()
    verify_flip:      0.0, 0.05, 0.10, 0.20 (float) - Probability verify() lies
    hint_ambiguity:   0, 1, 2, 3       (int) - Offset range for hint target
    distractor_count: 0, 2, 4, 6       (int) - Number of distractor tokens
    memory_tokens:    16, 32, 64, 128  (int) - Token limit (not chars)

Usage:
    python -m agent_bench.grid --out artifacts/agent_grid_v1.npz
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Dict, Any

import numpy as np


# Parameter value definitions
TOOL_NOISE_VALUES = [0, 1, 2, 3]
VERIFY_FLIP_VALUES = [0.0, 0.05, 0.10, 0.20]
HINT_AMBIGUITY_VALUES = [0, 1, 2, 3]
DISTRACTOR_COUNT_VALUES = [0, 2, 4, 6]
MEMORY_TOKENS_VALUES = [16, 32, 64, 128]

GRID_VERSION = "agent_grid_v1"


def generate_grid() -> Dict[str, np.ndarray]:
    """Generate the full parameter grid (1024 points).
    
    Returns:
        Dict with per-param arrays and metadata:
        - tool_noise: int array of shape (1024,)
        - verify_flip: float array of shape (1024,)
        - hint_ambiguity: int array of shape (1024,)
        - distractor_count: int array of shape (1024,)
        - memory_tokens: int array of shape (1024,)
        - grid_version: str
        - n_points: int
    """
    # Generate all combinations
    all_combos = list(product(
        TOOL_NOISE_VALUES,
        VERIFY_FLIP_VALUES,
        HINT_AMBIGUITY_VALUES,
        DISTRACTOR_COUNT_VALUES,
        MEMORY_TOKENS_VALUES,
    ))
    
    n_points = len(all_combos)
    assert n_points == 4 ** 5 == 1024, f"Expected 1024 points, got {n_points}"
    
    # Unpack into arrays
    tool_noise = np.array([c[0] for c in all_combos], dtype=np.int32)
    verify_flip = np.array([c[1] for c in all_combos], dtype=np.float64)
    hint_ambiguity = np.array([c[2] for c in all_combos], dtype=np.int32)
    distractor_count = np.array([c[3] for c in all_combos], dtype=np.int32)
    memory_tokens = np.array([c[4] for c in all_combos], dtype=np.int32)
    
    return {
        "tool_noise": tool_noise,
        "verify_flip": verify_flip,
        "hint_ambiguity": hint_ambiguity,
        "distractor_count": distractor_count,
        "memory_tokens": memory_tokens,
        "grid_version": np.array(GRID_VERSION),
        "n_points": np.array(n_points),
    }


def save_grid(path: Path) -> None:
    """Generate and save the grid to an NPZ file."""
    grid = generate_grid()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **grid)
    print(f"[Grid] Saved {grid['n_points']} points to {path}")
    print(f"[Grid] Version: {GRID_VERSION}")


def load_grid(path: Path) -> Dict[str, np.ndarray]:
    """Load a grid from an NPZ file."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def get_params_for_idx(grid: Dict[str, np.ndarray], idx: int) -> Dict[str, Any]:
    """Get parameter dict for a specific grid index."""
    return {
        "tool_noise": int(grid["tool_noise"][idx]),
        "verify_flip": float(grid["verify_flip"][idx]),
        "hint_ambiguity": int(grid["hint_ambiguity"][idx]),
        "distractor_count": int(grid["distractor_count"][idx]),
        "memory_tokens": int(grid["memory_tokens"][idx]),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate AgentEvalBench grid")
    parser.add_argument("--out", type=str, required=True, help="Output path for grid.npz")
    args = parser.parse_args()
    
    save_grid(Path(args.out))


if __name__ == "__main__":
    main()
