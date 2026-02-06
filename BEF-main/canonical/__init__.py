"""
Canonical primitives shared by executor and verifier.

Exports canonical JSON, quantization, RNG, Merkle, and audit helpers
for the bicep_v2 state audit contract.
"""

from .bicep_state import (
    canonical_json_bytes,
    AddressableRNG,
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
)

__all__ = [
    "canonical_json_bytes",
    "AddressableRNG",
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
]

