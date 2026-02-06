"""FRI verifier for STC-backed commitments."""
from __future__ import annotations

from typing import Sequence

import hashlib

from .config import FRIConfig
from .types import FRIProof
from bef_zk.stc.vc import VectorCommitment, VCCommitment

MODULUS = (1 << 61) - 1


def _mod(x: int) -> int:
    return x % MODULUS


def _derive_beta(root: bytes, round_idx: int) -> int:
    h = hashlib.sha256()
    h.update(root)
    h.update(round_idx.to_bytes(4, "big"))
    beta = int.from_bytes(h.digest(), "big") % MODULUS
    if beta == 0:
        beta = 1
    return beta


def fri_verify(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_commitment: VCCommitment,
    proof: FRIProof,
    expected_query_points: Sequence[int],
    expected_values: Sequence[int],
) -> bool:
    if len(expected_query_points) != len(expected_values):
        return False
    if len(proof.layers) != fri_cfg.num_rounds + 1:
        return False
    if proof.layers[0].commitment.root != base_commitment.root:
        return False

    if len(proof.batches) != len(proof.layers):
        return False

    betas = []
    for r in range(fri_cfg.num_rounds):
        layer_info = proof.layers[r]
        beta = _derive_beta(layer_info.commitment.root, r)
        betas.append(beta)
    betas.append(0)

    layer_values: list[dict[int, int]] = [{} for _ in proof.layers]
    for batch in proof.batches:
        if batch.layer_index >= len(proof.layers):
            return False
        layer_info = proof.layers[batch.layer_index]
        if not vc.verify_batch(layer_info.commitment, batch.proof):
            return False
        value_map = layer_values[batch.layer_index]
        for entry in batch.proof.entries:
            val = entry.value % MODULUS
            existing = value_map.get(entry.index)
            if existing is not None and existing != val:
                return False
            value_map[entry.index] = val

    for q_idx, base_index in enumerate(expected_query_points):
        if q_idx >= len(expected_values):
            return False
        expected_value = expected_values[q_idx] % MODULUS
        cur_idx = base_index
        folded = None

        for round_idx in range(fri_cfg.num_rounds):
            layer_map = layer_values[round_idx]
            beta = betas[round_idx]
            parent_idx = cur_idx // 2
            even_index = parent_idx * 2
            odd_index = even_index + 1
            if even_index not in layer_map or odd_index not in layer_map:
                return False
            v_even = layer_map[even_index]
            v_odd = layer_map[odd_index]
            if round_idx == 0:
                actual = v_even if cur_idx % 2 == 0 else v_odd
                if actual != expected_value:
                    return False
            folded = _mod(v_even + beta * v_odd)
            cur_idx = parent_idx

        final_map = layer_values[-1]
        if cur_idx not in final_map:
            return False
        if final_map[cur_idx] != folded:
            return False

    return True
