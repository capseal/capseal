#!/usr/bin/env python3
"""
State Audit Infrastructure for bicep_v2 (thin wrapper).

This module re-exports the canonical primitives from canonical.bicep_state
so both the executor and the verifier share a single source of truth.
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
    leaf_index,
    leaf_to_path_channel,
    total_leaves,
    build_merkle_tree,
    compute_merkle_proof,
    verify_merkle_proof,
    compute_challenge_seed,
    sample_audit_indices,
    sde_step_em,
    sde_step_with_jump,
    generate_audit_opening,
    verify_audit_opening,
    AddressableRNG,
)

__all__ = [
    "canonical_json_bytes",
    "quantize",
    "dequantize",
    "leaf_index",
    "leaf_to_path_channel",
    "total_leaves",
    "build_merkle_tree",
    "compute_merkle_proof",
    "verify_merkle_proof",
    "compute_challenge_seed",
    "sample_audit_indices",
    "sde_step_em",
    "sde_step_with_jump",
    "generate_audit_opening",
    "verify_audit_opening",
    "AddressableRNG",
]
