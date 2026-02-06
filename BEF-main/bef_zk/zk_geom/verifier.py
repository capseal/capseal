"""Transparent verifier for geometry AIR using STC+FRI PC."""
from __future__ import annotations

from typing import Dict, List, Optional

import time

from ..air.geom_air import GeomAIRParams, GeomInitialState
from ..air.geom_constraints import (
    composition_value_from_row,
    eval_constraints_from_row,
    row_values_from_columns,
    ConstraintResidual,
)
from ..fri.config import FRIConfig
from bef_zk.pc.stc_fri_pc import pc_verify
from ..stc.vc import VectorCommitment
from bef_zk.transcript import Transcript
from .types import GeomProof, RowOpening
from .prover import (
    _derive_alpha_digest,
    _derive_alphas,
    _row_commitment_root,
    _sample_query_indices,
)
from .masking import mask_at_index, column_mask_at_index, serialize_statement
from .columns import column_names
from .backend import get_row_backend
from .trace_only import verify_trace_only as _verify_trace_only_program
def verify_proof_only(
    vc: VectorCommitment,
    proof: GeomProof,
    statement_hash: bytes | None = None,
) -> tuple[bool, Dict[str, float]]:
    statement = proof.statement
    params: GeomAIRParams = statement.params
    fri_cfg: FRIConfig = proof.pc_commitment.fri_params
    start = time.perf_counter()

    alpha_digest = _derive_alpha_digest(params)
    if alpha_digest != proof.alpha_digest:
        return False, {"time_verify_sec": time.perf_counter() - start}

    constraint_names = [
        "pc_step",
        "gas_step",
        "cnt_step",
        "m11_step",
        "m12_step",
        "m22_step",
    ] + [f"s_step_{j}" for j in range(params.num_challenges)] + [
        f"pow_step_{j}" for j in range(params.num_challenges)
    ] + ["m11_final", "m12_final", "m22_final", "cnt_final"]
    alphas = _derive_alphas(alpha_digest, constraint_names)

    sigma_expected = {
        "m11": statement.final_m11,
        "m12": statement.final_m12,
        "m22": statement.final_m22,
        "cnt": statement.final_cnt,
    }

    row_commitment = proof.row_commitment
    if row_commitment is None:
        return False, {"time_verify_sec": time.perf_counter() - start}
    if len(proof.row_openings) != len(proof.query_indices):
        return False, {"time_verify_sec": time.perf_counter() - start}
    if statement_hash is not None:
        stmt_hash = statement_hash
    else:
        stmt_hash = serialize_statement(statement)
    transcript = Transcript("geom_zk_v1")
    transcript.absorb_bytes(alpha_digest)
    transcript.absorb_bytes(_row_commitment_root(row_commitment))
    transcript.absorb_bytes(stmt_hash)
    expected_mask_digest = transcript.challenge_bytes()
    if proof.mask_digest != expected_mask_digest:
        return False, {"time_verify_sec": time.perf_counter() - start}
    mask_digest = proof.mask_digest

    row_backend = get_row_backend(row_commitment.backend, row_commitment.row_width)

    col_names = column_names(params)
    row_width = len(col_names)
    expected_values: List[int] = []
    trace_len = params.steps
    zero_residual = ConstraintResidual(
        pc_step=0,
        gas_step=0,
        cnt_step=0,
        m11_step=0,
        m12_step=0,
        m22_step=0,
        s_steps=[0] * params.num_challenges,
        pow_steps=[0] * params.num_challenges,
    )

    for idx, row_open in zip(proof.query_indices, proof.row_openings):
        if row_open.backend != row_commitment.backend:
            return False, {"time_verify_sec": time.perf_counter() - start}
        if idx != row_open.row_index:
            return False, {"time_verify_sec": time.perf_counter() - start}
        if len(row_open.row_values) != row_width:
            return False, {"time_verify_sec": time.perf_counter() - start}
        if not row_backend.verify_leaf(row_commitment, idx, row_open.row_values, row_open.proof):
            return False, {"time_verify_sec": time.perf_counter() - start}

        next_idx = row_open.next_index
        next_row_values = row_open.next_row_values
        next_proof = row_open.next_proof
        if next_idx is not None:
            if next_idx != idx + 1 or next_idx >= fri_cfg.domain_size:
                return False, {"time_verify_sec": time.perf_counter() - start}
            if next_row_values is None or next_proof is None:
                return False, {"time_verify_sec": time.perf_counter() - start}
            if len(next_row_values) != row_width:
                return False, {"time_verify_sec": time.perf_counter() - start}
            if not row_backend.verify_leaf(
                row_commitment,
                next_idx,
                next_row_values,
                next_proof,
            ):
                return False, {"time_verify_sec": time.perf_counter() - start}
        else:
            next_row_values = None

        col_values: Dict[str, int] = {}
        next_col_values: Dict[str, int] = {}
        for idx_col, name in enumerate(col_names):
            masked_val = row_open.row_values[idx_col]
            col_mask = column_mask_at_index(
                alpha_digest,
                name,
                idx,
                fri_cfg.field_modulus,
            )
            col_values[name] = (masked_val - col_mask) % fri_cfg.field_modulus
            if next_idx is not None and next_row_values is not None:
                masked_next = next_row_values[idx_col]
                next_col_values[name] = (
                    masked_next
                    - column_mask_at_index(
                        alpha_digest,
                        name,
                        next_idx,
                        fri_cfg.field_modulus,
                    )
                ) % fri_cfg.field_modulus
            else:
                next_col_values[name] = 0

        row_vals = row_values_from_columns(params, col_values, next_col_values)
        if idx >= trace_len - 1:
            residual = zero_residual
        else:
            residual = eval_constraints_from_row(params, row_vals)

        comp_value = composition_value_from_row(
            params,
            row_vals,
            residual,
            idx,
            trace_len,
            alphas,
            sigma_expected,
        )
        row_mask = mask_at_index(mask_digest, idx, fri_cfg.field_modulus)
        expected_values.append((comp_value + row_mask) % fri_cfg.field_modulus)

    transcript.absorb_bytes(proof.pc_commitment.base_commitment.root)
    computed_indices = _sample_query_indices(transcript, fri_cfg)
    if computed_indices != proof.query_indices:
        return False, {"time_verify_sec": time.perf_counter() - start}

    ok = pc_verify(
        commitment=proof.pc_commitment,
        proof=proof.fri_proof,
        query_indices=proof.query_indices,
        expected_values=expected_values,
        vc=vc,
    )
    return ok, {"time_verify_sec": time.perf_counter() - start}


def zk_verify_geom(
    program: Optional[List[int]],
    params: Optional[GeomAIRParams],
    init_state: Optional[GeomInitialState],
    vc: VectorCommitment,
    proof: GeomProof,
    statement_hash: bytes | None = None,
) -> tuple[bool, Dict[str, float]]:
    stats: Dict[str, float] = {}
    if program is not None and params is not None and init_state is not None:
        ok_trace, trace_stats = _verify_trace_only_program(program, params, init_state)
        stats.update({f"trace_{k}": v for k, v in trace_stats.items()})
        if not ok_trace:
            return False, stats
    ok_proof, proof_stats = verify_proof_only(vc, proof, statement_hash=statement_hash)
    stats.update(proof_stats)
    return ok_proof, stats
