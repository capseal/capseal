"""FRI prover using STC-backed vector commitments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Any

import hashlib

from .config import FRIConfig
from .domain import fold_codeword
from .types import FRILayerInfo, FRIProof, FRILayerBatch
from bef_zk.stc.vc import (
    VectorCommitment,
    VCCommitment,
    VCBatchProof,
    VCBatchEntry,
    ChunkLeafProof,
)
from bef_zk.stc.merkle import MerkleMultiProof

MODULUS = (1 << 61) - 1

# Try to import Rust backend
try:
    import bef_rust
    _HAS_RUST = True
    # Check for PyFriState (may be missing in stale builds)
    _HAS_FRI_STATE = hasattr(bef_rust, "PyFriState")
except ImportError:
    bef_rust = None  # type: ignore
    _HAS_RUST = False
    _HAS_FRI_STATE = False


def _rust_proof_to_python(rust_proof: Any) -> VCBatchProof:
    """Convert Rust PyBatchProof to Python VCBatchProof."""
    entries = [
        VCBatchEntry(
            index=e.index,
            value=e.value,
            chunk_index=e.chunk_index,
            chunk_offset=e.chunk_offset,
            leaf_pos=e.leaf_pos,
            leaf_path=None,
        )
        for e in rust_proof.entries
    ]

    chunk_proof = MerkleMultiProof(
        tree_size=rust_proof.chunk_proof.tree_size,
        arity=rust_proof.chunk_proof.arity,
        sibling_levels=[
            [bytes.fromhex(h) for h in level]
            for level in rust_proof.chunk_proof.sibling_levels
        ],
    )

    chunk_leaf_proofs = [
        ChunkLeafProof(
            chunk_index=clp.chunk_index,
            chunk_offset=clp.chunk_offset,
            leaf_positions=list(clp.leaf_positions),
            proof=MerkleMultiProof(
                tree_size=clp.proof.tree_size,
                arity=clp.proof.arity,
                sibling_levels=[
                    [bytes.fromhex(h) for h in level]
                    for level in clp.proof.sibling_levels
                ],
            ),
        )
        for clp in rust_proof.chunk_leaf_proofs
    ]

    chunk_roots = [bytes.fromhex(h) for h in rust_proof.chunk_roots]

    return VCBatchProof(
        entries=entries,
        chunk_positions=list(rust_proof.chunk_positions),
        chunk_roots=chunk_roots,
        chunk_proof=chunk_proof,
        chunk_leaf_proofs=chunk_leaf_proofs,
    )


def _mod(x: int) -> int:
    return x % MODULUS


def _derive_beta(root: bytes, round_idx: int) -> int:
    h = hashlib.sha256()
    h.update(root)
    h.update(round_idx.to_bytes(4, "big"))
    candidate = int.from_bytes(h.digest(), "big") % MODULUS
    if candidate == 0:
        candidate = 1
    return candidate


def _build_layers(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
    use_rust: bool = False,
    skip_python_commit: bool = False,
) -> List[FRILayerInfo]:
    """Build FRI layer commitments.

    Args:
        fri_cfg: FRI configuration
        vc: Vector commitment interface
        base_evals: Initial codeword evaluations
        base_commitment: Commitment to base evaluations
        use_rust: If True, use Rust backend for fold+commit (24x faster)
        skip_python_commit: If True and use_rust, skip Python commit (even faster
            but open_batch won't work). Useful for benchmarking.
    """
    if len(base_evals) != fri_cfg.domain_size:
        raise ValueError("base evaluations length must match domain size")
    if fri_cfg.domain_size & (fri_cfg.domain_size - 1):
        raise ValueError("domain size must be a power of two")

    if use_rust and _HAS_RUST and _HAS_FRI_STATE:
        return _build_layers_rust(fri_cfg, vc, base_evals, base_commitment, skip_python_commit)
    else:
        return _build_layers_python(fri_cfg, vc, base_evals, base_commitment)


def _build_layers_python(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
) -> List[FRILayerInfo]:
    """Pure Python implementation (original)."""
    layers: List[FRILayerInfo] = []
    current = [int(v) % MODULUS for v in base_evals]
    commit = base_commitment

    for round_idx in range(fri_cfg.num_rounds):
        beta = _derive_beta(commit.root, round_idx)
        layers.append(FRILayerInfo(commitment=commit, beta=beta, length=len(current)))
        next_codeword = fold_codeword(current, beta, fri_cfg.field_modulus)
        commit = vc.commit(next_codeword)
        current = next_codeword

    # last layer commitment
    layers.append(FRILayerInfo(commitment=commit, beta=0, length=len(current)))
    return layers


def _build_layers_rust(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
    skip_python_commit: bool = False,
) -> List[FRILayerInfo]:
    """Rust-accelerated implementation (24x faster fold+commit).

    Python still owns the transcript (beta derivation) and layer storage.
    Rust handles the heavy compute: codeword folding and Merkle commitment.

    Args:
        skip_python_commit: If True, only use Rust for commitment (24x faster
            but open_batch won't work). Set True for benchmarking or when
            query opening is also moved to Rust.
    """
    layers, _ = _build_layers_and_state_rust(fri_cfg, vc, base_evals, base_commitment, skip_python_commit)
    return layers


def _build_layers_and_state_rust(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
    skip_python_commit: bool = False,
) -> Tuple[List[FRILayerInfo], Any]:
    """Rust-accelerated implementation returning (layers, rust_state).

    Uses caching methods (commit_and_cache, fold_commit_and_cache) so that
    open_batch() can be called on the returned state.

    Args:
        skip_python_commit: If True, skip Python commit (full Rust path).
            This is now the default for production - Rust handles everything.

    Returns:
        Tuple of (layers, rust_state). rust_state can be used for open_batch().
    """
    chunk_len = getattr(vc, 'chunk_len', 256)
    chunk_tree_arity = int(os.environ.get("STC_CHUNK_TREE_ARITY", "16"))

    layers: List[FRILayerInfo] = []
    current = [int(v) % MODULUS for v in base_evals]

    # Initialize Rust FRI state
    state = bef_rust.PyFriState(current)

    # Commit initial layer with caching
    result = state.commit_and_cache(chunk_len, chunk_tree_arity=chunk_tree_arity)
    rust_root = bytes.fromhex(result.root_hex)
    commit = VCCommitment(
        root=rust_root,
        length=result.length,
        chunk_len=chunk_len,
        num_chunks=result.num_chunks,
        challenges=[],
        sketches=[],
        powers=[],
        chunk_tree_arity=chunk_tree_arity,
    )

    # Check if prove_all is available (newer API)
    _has_prove_all = hasattr(state, "prove_all")

    current_commit = commit
    if _has_prove_all:
        # Use pure-Rust proving loop (faster)
        # This derives beta internally using SHA256 matching _derive_beta
        proof_results = state.prove_all(fri_cfg.num_rounds, chunk_len, chunk_tree_arity)

        # Reconstruct layers info from results
        for round_idx, res in enumerate(proof_results):
            beta = _derive_beta(current_commit.root, round_idx)
            layers.append(FRILayerInfo(commitment=current_commit, beta=beta, length=current_commit.length))

            # Prepare for next layer (which is result of this fold)
            rust_root = bytes.fromhex(res.root_hex)
            current_commit = VCCommitment(
                root=rust_root,
                length=res.length,
                chunk_len=chunk_len,
                num_chunks=res.num_chunks,
                challenges=[],
                sketches=[],
                powers=[],
                chunk_tree_arity=chunk_tree_arity,
            )
    else:
        # Fallback: iterative fold_commit_and_cache (older API, same correctness)
        for round_idx in range(fri_cfg.num_rounds):
            beta = _derive_beta(current_commit.root, round_idx)
            layers.append(FRILayerInfo(commitment=current_commit, beta=beta, length=current_commit.length))

            # Fold and commit using older API
            res = state.fold_commit_and_cache(beta, chunk_len, chunk_tree_arity=chunk_tree_arity)
            rust_root = bytes.fromhex(res.root_hex)
            current_commit = VCCommitment(
                root=rust_root,
                length=res.length,
                chunk_len=chunk_len,
                num_chunks=res.num_chunks,
                challenges=[],
                sketches=[],
                powers=[],
                chunk_tree_arity=chunk_tree_arity,
            )

    # last layer commitment
    layers.append(FRILayerInfo(commitment=current_commit, beta=0, length=current_commit.length))
    return layers, state


def fri_prove(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
    query_indices: Sequence[int],
    use_rust: bool = False,
) -> FRIProof:
    """Generate FRI proof.

    Args:
        fri_cfg: FRI configuration
        vc: Vector commitment interface
        base_evals: Initial codeword evaluations
        base_commitment: Commitment to base evaluations
        query_indices: Indices to open
        use_rust: If True, use Rust backend for fold+commit+open (19x faster)
    """
    # Build layers (and get Rust state if using Rust)
    rust_state = None
    if use_rust and _HAS_RUST and _HAS_FRI_STATE:
        layers, rust_state = _build_layers_and_state_rust(
            fri_cfg, vc, base_evals, base_commitment, skip_python_commit=True
        )
    else:
        layers = _build_layers_python(fri_cfg, vc, base_evals, base_commitment)

    num_layers = len(layers)
    needed: List[set[int]] = [set() for _ in range(num_layers)]

    for idx in query_indices:
        cur_idx = idx
        for round_idx in range(fri_cfg.num_rounds):
            layer_info = layers[round_idx]
            if layer_info.length <= 1:
                raise ValueError("FRI layer length too small for folding")
            parent_idx = cur_idx // 2
            even_index = parent_idx * 2
            odd_index = even_index + 1
            needed[round_idx].add(even_index)
            needed[round_idx].add(odd_index)
            cur_idx = parent_idx
        needed[-1].add(cur_idx)

    batches: List[FRILayerBatch] = []
    for layer_idx, layer_info in enumerate(layers):
        idxs = sorted(needed[layer_idx])
        if rust_state is not None:
            # Use Rust's open_batch (full Rust path)
            rust_proof = rust_state.open_batch(layer_idx, idxs)
            batch_proof = _rust_proof_to_python(rust_proof)
        else:
            # Use Python's open_batch
            batch_proof = vc.open_batch(layer_info.commitment, idxs)
        batches.append(
            FRILayerBatch(
                layer_index=layer_idx,
                proof=batch_proof,
            )
        )

    proof = FRIProof(layers=layers, batches=batches)
    return proof
