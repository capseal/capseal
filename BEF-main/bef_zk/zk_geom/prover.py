"""Transparent Σ+FS prover for the geometry AIR."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Optional

import hashlib
import time

from ..air.geom_air import (
    GeomAIRParams,
    GeomInitialState,
    GeomTrace,
    trace_to_eval_table,
    simulate_trace,
)
from ..air.geom_constraints import build_composition_vector
from ..fri.config import FRIConfig
from bef_zk.pc.stc_fri_pc import pc_commit, pc_open
from ..stc.vc import VectorCommitment
from bef_zk.transcript import Transcript
from .types import GeomStatement, GeomProof, RowOpening
from .masking import (
    derive_column_masks,
    derive_mask_vector,
    serialize_statement,
)
from .columns import extract_masked_columns, column_names, build_row_matrix
from .backend import get_row_backend, RowCommitment
from .proof_stats import compute_proof_stats

MODULUS = (1 << 61) - 1


def _derive_alpha_digest(params: GeomAIRParams) -> bytes:
    h = hashlib.sha256()
    h.update(params.steps.to_bytes(8, "big"))
    for row in params.matrix:
        for entry in row:
            h.update(int(entry).to_bytes(8, "big", signed=False))
    for challenge in params.r_challenges:
        h.update(int(challenge).to_bytes(8, "big", signed=False))
    return h.digest()


def _derive_alphas(alpha_digest: bytes, names: Sequence[str]) -> Dict[str, int]:
    alphas: Dict[str, int] = {}
    ctr = 0
    for name in names:
        hh = hashlib.sha256(alpha_digest + ctr.to_bytes(4, "big")).digest()
        val = int.from_bytes(hh, "big") % MODULUS
        if val == 0:
            val = 1
        alphas[name] = val
        ctr += 1
    return alphas


def _row_commitment_root(commitment: RowCommitment) -> bytes:
    params = commitment.params or {}
    root_hex = params.get("root")
    if not root_hex:
        raise ValueError("row commitment missing root")
    return bytes.fromhex(root_hex)


def _sample_query_indices(tx: Transcript, fri_cfg: FRIConfig) -> List[int]:
    seen: set[int] = set()
    indices: List[int] = []
    while len(indices) < fri_cfg.num_queries:
        idx = tx.challenge_field(fri_cfg.domain_size)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    indices.sort()
    return indices


def zk_prove_geom(
    program: Sequence[int],
    params: GeomAIRParams,
    init_state: GeomInitialState,
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    row_backend: str = "geom_stc_fri",
    row_backend_params: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, float]] = None,
    statement_hash_fn: Optional[Callable[[RowCommitment, GeomStatement], bytes]] = None,
) -> GeomProof:
    return _zk_prove_geom_internal(
        program,
        params,
        init_state,
        fri_cfg,
        vc,
        row_backend=row_backend,
        row_backend_params=row_backend_params,
        profile=profile,
        statement_hash_fn=statement_hash_fn,
    )


def _zk_prove_geom_internal(
    program: Sequence[int],
    params: GeomAIRParams,
    init_state: GeomInitialState,
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    row_backend: str,
    row_backend_params: Optional[Dict[str, Any]],
    profile: Optional[Dict[str, float]],
    statement_hash_fn: Optional[Callable[[RowCommitment, GeomStatement], bytes]],
) -> GeomProof:
    t_start = time.perf_counter()
    trace: GeomTrace = simulate_trace(program, params, init_state)
    t_after_trace = time.perf_counter()
    final_row = trace.rows[-1]
    statement = GeomStatement(
        params=params,
        final_m11=final_row.m11,
        final_m12=final_row.m12,
        final_m22=final_row.m22,
        final_cnt=final_row.cnt,
    )

    alpha_digest = _derive_alpha_digest(params)
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
        "m11": final_row.m11,
        "m12": final_row.m12,
        "m22": final_row.m22,
        "cnt": final_row.cnt,
    }
    column_masks = derive_column_masks(alpha_digest, params, fri_cfg.domain_size)
    eval_table = trace_to_eval_table(
        trace,
        fri_cfg.domain_size,
        column_masks=column_masks,
    )
    masked_columns = extract_masked_columns(eval_table, params)
    composition = build_composition_vector(
        params,
        eval_table,
        alphas,
        sigma_expected=sigma_expected,
    )

    col_names = column_names(params)
    row_width = max(1, len(col_names))
    row_matrix = build_row_matrix(masked_columns, params)
    backend_impl = get_row_backend(
        row_backend,
        row_width=row_width,
        **(row_backend_params or {}),
    )
    row_commit_start = time.perf_counter()
    row_commitment = backend_impl.commit_rows(row_matrix)
    row_commit_end = time.perf_counter()

    if statement_hash_fn is not None:
        stmt_hash = statement_hash_fn(row_commitment, statement)
    else:
        stmt_hash = serialize_statement(statement)
    transcript = Transcript("geom_zk_v1")
    transcript.absorb_bytes(alpha_digest)
    transcript.absorb_bytes(_row_commitment_root(row_commitment))
    transcript.absorb_bytes(stmt_hash)
    mask_digest = transcript.challenge_bytes()
    mask_vec = derive_mask_vector(mask_digest, fri_cfg.domain_size)
    masked_composition = [
        (composition[i] + mask_vec[i]) % MODULUS for i in range(len(composition))
    ]

    pc_commit_start = time.perf_counter()
    pc_state = pc_commit(masked_composition, fri_params=fri_cfg, vc=vc)
    transcript.absorb_bytes(pc_state.commitment.base_commitment.root)
    query_indices = _sample_query_indices(transcript, fri_cfg)
    pc_commit_end = time.perf_counter()
    fri_start = time.perf_counter()
    fri_proof = pc_open(pc_state, query_indices=query_indices, vc=vc)

    row_openings: List[RowOpening] = []
    for idx in query_indices:
        row_values, proof_obj = backend_impl.open_row(row_commitment, idx)
        next_idx = idx + 1 if idx + 1 < fri_cfg.domain_size else None
        if next_idx is not None:
            next_values, next_proof = backend_impl.open_row(row_commitment, next_idx)
        else:
            next_values, next_proof = None, None
        row_openings.append(
            RowOpening(
                backend=backend_impl.name,
                row_index=idx,
                row_values=row_values,
                proof=proof_obj,
                next_index=next_idx,
                next_row_values=next_values,
                next_proof=next_proof,
            )
        )
    fri_end = time.perf_counter()

    proof = GeomProof(
        statement=statement,
        pc_commitment=pc_state.commitment,
        query_indices=query_indices,
        fri_proof=fri_proof,
        alpha_digest=alpha_digest,
        mask_digest=mask_digest,
        row_commitment=row_commitment,
        row_openings=row_openings,
    )

    proof_stats = compute_proof_stats(proof)

    if profile is not None:
        profile.update(
            {
                "time_trace_sec": t_after_trace - t_start,
                "time_row_commit_sec": row_commit_end - row_commit_start,
                "time_pc_commit_sec": pc_commit_end - pc_commit_start,
                "time_fri_sec": fri_end - fri_start,
                "time_total_sec": fri_end - t_start,
                "row_openings": len(row_openings),
                "backend": backend_impl.name,
            }
        )
        profile.update(proof_stats)

    return proof
