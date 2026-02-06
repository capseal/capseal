#!/usr/bin/env python3
"""
Canonical State Commitment for BICEP v2 — thin re-export.

Delegates to canonical.bicep_state (the single source of truth).
This module exists for backward compatibility within verifier-independent/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.bicep_state import (
    canonical_json_bytes,
    quantize,
    dequantize,
    total_leaves,
    leaf_index,
    leaf_to_path_channel,
    build_merkle_tree,
    compute_merkle_proof,
    verify_merkle_proof,
    compute_challenge_seed,
    sample_audit_indices,
    generate_audit_opening,
)


def quantize_state_tensor(state_f: list[float], precision_bits: int = 24) -> list[int]:
    """Quantize an entire state tensor (already linearized)."""
    return [quantize(v, precision_bits) for v in state_f]


def merkle_root(leaves_q: list[int]) -> str:
    """Compute Merkle root only (no level storage)."""
    root, _ = build_merkle_tree(leaves_q)
    return root


__all__ = [
    "canonical_json_bytes",
    "quantize", "dequantize", "quantize_state_tensor",
    "leaf_index", "leaf_to_path_channel", "total_leaves",
    "build_merkle_tree", "merkle_root",
    "compute_merkle_proof", "verify_merkle_proof",
    "compute_challenge_seed", "sample_audit_indices",
    "generate_audit_opening",
]
