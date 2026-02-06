"""Polynomial commitment using STC-backed vector commitments + FRI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from bef_zk.fri.config import FRIConfig
from bef_zk.fri.prover import fri_prove
from bef_zk.fri.verifier import fri_verify
from bef_zk.fri.types import FRIProof
from bef_zk.stc.vc import VectorCommitment, VCCommitment

MODULUS = (1 << 61) - 1


def _mod(x: int) -> int:
    return x % MODULUS


@dataclass
class PCCommitment:
    fri_params: FRIConfig
    base_commitment: VCCommitment


@dataclass
class PCProverState:
    commitment: PCCommitment
    values: List[int]


def pc_commit(
    values: Sequence[int],
    fri_params: FRIConfig,
    vc: VectorCommitment,
) -> PCProverState:
    if len(values) != fri_params.domain_size:
        raise ValueError("evaluation vector length must equal FRI domain size")
    vals = [_mod(v) for v in values]
    base_commit = vc.commit(vals)
    commitment = PCCommitment(fri_params=fri_params, base_commitment=base_commit)
    return PCProverState(commitment=commitment, values=vals)


def pc_open(
    state: PCProverState,
    query_indices: Sequence[int],
    vc: VectorCommitment,
) -> FRIProof:
    fri_params = state.commitment.fri_params
    proof = fri_prove(
        fri_cfg=fri_params,
        vc=vc,
        base_evals=state.values,
        base_commitment=state.commitment.base_commitment,
        query_indices=query_indices,
    )
    return proof


def pc_verify(
    commitment: PCCommitment,
    proof: FRIProof,
    query_indices: Sequence[int],
    expected_values: Sequence[int],
    vc: VectorCommitment,
) -> bool:
    return fri_verify(
        fri_cfg=commitment.fri_params,
        vc=vc,
        base_commitment=commitment.base_commitment,
        proof=proof,
        expected_query_points=query_indices,
        expected_values=[_mod(v) for v in expected_values],
    )
