"""Geometry backend implementation for the TraceAdapter API."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from bef_zk.adapter import ProofArtifacts, TraceAdapter, TraceArtifacts, TraceCommitment
from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState, simulate_trace, trace_to_eval_table
from bef_zk.air.geom_constraints import build_composition_vector
from bef_zk.air.geom_trace_export import geom_trace_to_bef_trace, flatten_geom_row
from bef_zk.fri.config import FRIConfig
from bef_zk.pc.stc_fri_pc import pc_commit, pc_open
from bef_zk.spec import (
    StatementV1,
    TraceSpecV1,
    compute_statement_hash,
    compute_trace_spec_hash,
)
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.transcript import Transcript
from bef_zk.zk_geom.backend import get_row_backend, RowCommitment, RowOpening
from bef_zk.zk_geom.columns import column_names, build_row_matrix, extract_masked_columns
from bef_zk.zk_geom.masking import derive_column_masks, derive_mask_vector, serialize_statement
from bef_zk.zk_geom.proof_stats import compute_proof_stats
from bef_zk.zk_geom.serialization import proof_from_json, proof_to_bytes, proof_to_json
from bef_zk.zk_geom.types import GeomStatement, GeomProof
from bef_zk.zk_geom.verifier import zk_verify_geom
from scripts.geom_programs import GEOM_PROGRAM

ROOT = Path(__file__).parent.parent.resolve()

MODULUS = (1 << 61) - 1


@dataclass
class GeomContext:
    params: GeomAIRParams
    init: GeomInitialState
    trace: Any
    row_width: int
    fri_cfg: FRIConfig
    trace_time_sec: float
    trace_path: Path | None = None
    prepared: dict[str, Any] | None = None


def _derive_r_challenges(seed: int, m: int) -> list[int]:
    base = seed.to_bytes(32, "big", signed=False)
    out: list[int] = []
    counter = 0
    while len(out) < m:
        hh = hashlib.sha256(base + counter.to_bytes(4, "big")).digest()
        val = int.from_bytes(hh, "big") % MODULUS
        if val != 0:
            out.append(val)
        counter += 1
    return out


def _derive_alpha_digest(params: GeomAIRParams) -> bytes:
    h = hashlib.sha256()
    h.update(params.steps.to_bytes(8, "big"))
    for row in params.matrix:
        for entry in row:
            h.update(int(entry).to_bytes(8, "big", signed=False))
    for challenge in params.r_challenges:
        h.update(int(challenge).to_bytes(8, "big", signed=False))
    return h.digest()


def _derive_alphas(alpha_digest: bytes, names: list[str]) -> dict[str, int]:
    alphas: dict[str, int] = {}
    ctr = 0
    for name in names:
        hh = hashlib.sha256(alpha_digest + ctr.to_bytes(4, "big")).digest()
        val = int.from_bytes(hh, "big") % MODULUS
        if val == 0:
            val = 1
        alphas[name] = val
        ctr += 1
    return alphas


def _sample_query_indices(tx: Transcript, fri_cfg: FRIConfig) -> list[int]:
    seen: set[int] = set()
    indices: list[int] = []
    while len(indices) < fri_cfg.num_queries:
        idx = tx.challenge_field(fri_cfg.domain_size)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    indices.sort()
    return indices


def _build_fri_cfg(steps: int, num_queries: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    return FRIConfig(
        field_modulus=MODULUS,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=min(6, max_rounds),
        num_queries=num_queries,
    )


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class GeomTraceAdapter(TraceAdapter):
    name = "geom"
    PROGRAM = GEOM_PROGRAM

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        parser.add_argument("--steps", type=int, default=64)
        parser.add_argument("--num-challenges", type=int, default=2)
        parser.add_argument("--num-queries", type=int, default=8)
        parser.add_argument("--challenge-seed", type=int, default=42)
        parser.add_argument(
            "--stats-out",
            type=Path,
            help="write aggregated pipeline stats to this JSON file",
        )
        parser.add_argument(
            "--no-compressed",
            action="store_true",
            help="skip compressed SNARK (Nova only)",
        )
        parser.add_argument(
            "--run-nova",
            action="store_true",
            help="enable Nova recursion step (disabled by default)",
        )
        parser.add_argument(
            "--rust-stc",
            action="store_true",
            help="use experimental Rust STC backend",
        )
        parser.add_argument(
            "--gpu-da",
            action="store_true",
            help="use GPU acceleration for DA sketches",
        )

    def simulate_trace(self, args: Any) -> TraceArtifacts:
        params = GeomAIRParams(
            steps=args.steps,
            num_challenges=args.num_challenges,
            r_challenges=_derive_r_challenges(args.challenge_seed, args.num_challenges),
            matrix=[[2, 1], [1, 1]],
        )
        init = GeomInitialState()
        start = time.perf_counter()
        trace = simulate_trace(GEOM_PROGRAM, params, init)
        trace_time = time.perf_counter() - start
        row_width = len(flatten_geom_row(trace.rows[0]))
        schema_doc = {"columns": column_names(params)}
        schema_hash = hashlib.sha256(json.dumps(schema_doc, sort_keys=True).encode()).hexdigest()
        trace_spec = TraceSpecV1(
            spec_version="1.0",
            trace_format_id="GEOM_AIR_V1",
            record_schema_ref=f"sha256:{schema_hash}",
            encoding_id=args.encoding_id,
            field_modulus_id="goldilocks_61",
        )
        trace_spec_hash = compute_trace_spec_hash(trace_spec)
        bef_trace = geom_trace_to_bef_trace(trace, args.trace_id)
        self._emit_progress(
            "trace_simulated",
            {
                "trace_spec_hash": trace_spec_hash,
                "steps": params.steps,
                "num_challenges": params.num_challenges,
            },
        )
        ctx = GeomContext(
            params=params,
            init=init,
            trace=trace,
            row_width=row_width,
            fri_cfg=_build_fri_cfg(args.steps, args.num_queries),
            trace_time_sec=trace_time,
        )
        return TraceArtifacts(
            trace_id=args.trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            bef_trace=bef_trace,
            row_width=row_width,
            context=ctx,
            trace_time_sec=trace_time,
        )

    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict[str, Any]]:
        ctx: GeomContext = artifacts.context
        prepared = self._prepare_trace_state(ctx)
        return self._statement_public_inputs(prepared["statement"])

    def commit_to_trace(
        self,
        artifacts: TraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> TraceCommitment:
        ctx: GeomContext = artifacts.context
        prepared = self._prepare_trace_state(ctx)
        row_matrix = prepared["row_matrix"]
        backend_name = "geom_stc_rust" if getattr(self.args, "rust_stc", False) else "geom_stc_fri"
        row_backend = get_row_backend(
            backend_name,
            row_width=ctx.row_width,
            archive_dir=str(row_archive_dir),
            use_gpu=getattr(self.args, "gpu_da", False),
        )
        commit_start = time.perf_counter()
        row_commitment = row_backend.commit_rows(row_matrix)
        commit_elapsed = time.perf_counter() - commit_start
        (
            row_archive_artifact,
            chunk_handles,
            chunk_roots_hex,
            chunk_roots_digest,
            chunk_roots_paths,
        ) = self._materialize_row_archive(row_commitment, row_archive_dir)
        profile_data = {
            "time_trace_sec": ctx.trace_time_sec,
            "time_row_commit_sec": commit_elapsed,
        }
        self._emit_progress(
            "row_root_finalized",
            {
                "trace_root": row_commitment.params.get("root"),
                "num_chunks": len(chunk_handles),
            },
        )
        return TraceCommitment(
            row_commitment=row_commitment,
            row_archive_artifact=row_archive_artifact,
            chunk_handles=chunk_handles,
            chunk_roots_hex=chunk_roots_hex,
            chunk_roots_digest=chunk_roots_digest,
            chunk_roots_paths=chunk_roots_paths,
            profile_data=profile_data,
        )

    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: TraceCommitment,
        *,
        statement_hash: bytes,
        binding_hash: Optional[bytes] = None,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        ctx: GeomContext = artifacts.context
        prepared = self._prepare_trace_state(ctx)
        row_commitment = commitment.row_commitment
        if row_commitment is None:
            raise RuntimeError("trace commitment missing row commitment")
        vc = STCVectorCommitment(chunk_len=ctx.row_width)
        profile_data = dict(commitment.profile_data)
        binding_material = binding_hash or statement_hash
        transcript = Transcript("geom_zk_v1")
        transcript.absorb_bytes(prepared["alpha_digest"])
        transcript.absorb_bytes(bytes.fromhex(row_commitment.params.get("root", "")))
        transcript.absorb_bytes(binding_material)
        mask_digest = transcript.challenge_bytes()
        mask_vec = derive_mask_vector(mask_digest, ctx.fri_cfg.domain_size)
        composition = prepared["composition"]
        masked_composition = [
            (composition[i] + mask_vec[i]) % MODULUS for i in range(len(composition))
        ]
        pc_commit_start = time.perf_counter()
        pc_state = pc_commit(masked_composition, fri_params=ctx.fri_cfg, vc=vc)
        transcript.absorb_bytes(pc_state.commitment.base_commitment.root)
        query_indices = _sample_query_indices(transcript, ctx.fri_cfg)
        pc_commit_end = time.perf_counter()
        fri_start = time.perf_counter()
        fri_proof = pc_open(pc_state, query_indices=query_indices, vc=vc)
        row_openings = self._open_rows(row_commitment, query_indices, ctx)
        fri_end = time.perf_counter()
        proof = GeomProof(
            statement=prepared["statement"],
            pc_commitment=pc_state.commitment,
            query_indices=query_indices,
            fri_proof=fri_proof,
            alpha_digest=prepared["alpha_digest"],
            mask_digest=mask_digest,
            row_commitment=row_commitment,
            row_openings=row_openings,
        )
        proof_stats = compute_proof_stats(proof)
        profile_data.setdefault("time_trace_sec", ctx.trace_time_sec)
        profile_data.setdefault("time_row_commit_sec", 0.0)
        profile_data.update(
            {
                "time_pc_commit_sec": pc_commit_end - pc_commit_start,
                "time_fri_sec": fri_end - fri_start,
                "time_total_sec": profile_data.get("time_trace_sec", 0.0)
                + profile_data.get("time_row_commit_sec", 0.0)
                + (fri_end - pc_commit_start),
                "row_openings": len(row_openings),
                "backend": row_commitment.backend,
            }
        )
        profile_data.update(proof_stats)
        chunk_leaf_stats = self._compute_leaf_stats(profile_data)
        proof_json = proof_to_json(proof)
        proof_bytes = proof_to_bytes(proof, encoding_id=encoding_id)
        extra: dict[str, Any] | None = self._maybe_run_nova(trace_path, proof)
        return ProofArtifacts(
            proof_obj=proof,
            proof_json=proof_json,
            proof_bytes=proof_bytes,
            profile_data=profile_data,
            chunk_leaf_stats=chunk_leaf_stats,
            extra=extra,
        )

    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
        *,
        binding_hash: Optional[bytes] = None,
    ):
        ctx: GeomContext = artifacts.context
        start = time.perf_counter()
        ok, verify_stats = zk_verify_geom(
            GEOM_PROGRAM,
            ctx.params,
            ctx.init,
            STCVectorCommitment(chunk_len=ctx.row_width),
            proof_from_json(proof_json),
            statement_hash=binding_hash or statement_hash,
        )
        elapsed = time.perf_counter() - start
        return ok, verify_stats, elapsed

    def _prepare_trace_state(self, ctx: GeomContext) -> dict[str, Any]:
        if ctx.prepared:
            return ctx.prepared
        trace = ctx.trace
        final_row = trace.rows[-1]
        statement = GeomStatement(
            params=ctx.params,
            final_m11=final_row.m11,
            final_m12=final_row.m12,
            final_m22=final_row.m22,
            final_cnt=final_row.cnt,
        )
        alpha_digest = _derive_alpha_digest(ctx.params)
        constraint_names = [
            "pc_step",
            "gas_step",
            "cnt_step",
            "m11_step",
            "m12_step",
            "m22_step",
        ] + [f"s_step_{j}" for j in range(ctx.params.num_challenges)] + [
            f"pow_step_{j}" for j in range(ctx.params.num_challenges)
        ] + ["m11_final", "m12_final", "m22_final", "cnt_final"]
        alphas = _derive_alphas(alpha_digest, constraint_names)
        sigma_expected = {
            "m11": final_row.m11,
            "m12": final_row.m12,
            "m22": final_row.m22,
            "cnt": final_row.cnt,
        }
        column_masks = derive_column_masks(alpha_digest, ctx.params, ctx.fri_cfg.domain_size)
        eval_table = trace_to_eval_table(
            trace,
            ctx.fri_cfg.domain_size,
            column_masks=column_masks,
        )
        masked_columns = extract_masked_columns(eval_table, ctx.params)
        composition = build_composition_vector(
            ctx.params,
            eval_table,
            alphas,
            sigma_expected=sigma_expected,
        )
        row_matrix = build_row_matrix(masked_columns, ctx.params)
        prepared = {
            "statement": statement,
            "alpha_digest": alpha_digest,
            "eval_table": eval_table,
            "masked_columns": masked_columns,
            "composition": composition,
            "row_matrix": row_matrix,
        }
        ctx.prepared = prepared
        return prepared

    def _statement_public_inputs(self, statement: GeomStatement) -> list[dict[str, Any]]:
        return [
            {"name": "final_m11", "value": int(statement.final_m11)},
            {"name": "final_m12", "value": int(statement.final_m12)},
            {"name": "final_m22", "value": int(statement.final_m22)},
            {"name": "final_cnt", "value": int(statement.final_cnt)},
        ]

    def _materialize_row_archive(
        self,
        row_commitment: RowCommitment,
        row_archive_dir: Path,
    ) -> tuple[dict[str, Any], list[Any], list[str], str, dict[str, Path]]:
        chunk_handles = list(row_commitment.params.get("chunk_handles", []) or [])
        chunk_roots_hex = list(row_commitment.params.get("chunk_roots_hex", []) or [])
        row_commitment.params.pop("chunk_handles", None)
        row_commitment.params.pop("chunk_roots_hex", None)
        chunk_roots_path = row_archive_dir / "chunk_roots.json"
        chunk_roots_path.write_text(json.dumps(chunk_roots_hex, indent=2))
        chunk_roots_bin_path = row_archive_dir / "chunk_roots.bin"
        chunk_roots_bin_path.write_bytes(b"".join(bytes.fromhex(h) for h in chunk_roots_hex))
        chunk_roots_digest = _compute_file_hash(chunk_roots_bin_path)
        row_archive_artifact = {
            "mode": "LOCAL_FILE",
            "abs_path": str(row_archive_dir),
            "num_chunks": len(chunk_handles),
            "chunk_tree_arity": row_commitment.params.get("chunk_tree_arity"),
            "chunk_roots_path": str(chunk_roots_path),
            "chunk_roots_format": "hex_json_v1",
            "chunk_roots_bin_path": str(chunk_roots_bin_path),
            "chunk_roots_bin_format": "raw32_v1",
            "chunk_roots_digest": chunk_roots_digest,
        }
        row_archive_artifact["chunk_handles"] = chunk_handles
        return (
            row_archive_artifact,
            chunk_handles,
            chunk_roots_hex,
            chunk_roots_digest,
            {"json": chunk_roots_path, "bin": chunk_roots_bin_path},
        )

    def _open_rows(
        self,
        row_commitment: RowCommitment,
        indices: list[int],
        ctx: GeomContext,
    ) -> list[RowOpening]:
        backend_impl = get_row_backend(row_commitment.backend, row_width=row_commitment.row_width)
        openings: list[RowOpening] = []
        for idx in indices:
            row_values, proof_obj = backend_impl.open_row(row_commitment, idx)
            next_idx = idx + 1 if idx + 1 < ctx.fri_cfg.domain_size else None
            if next_idx is not None:
                next_vals, next_proof = backend_impl.open_row(row_commitment, next_idx)
            else:
                next_vals, next_proof = None, None
            openings.append(
                RowOpening(
                    backend=backend_impl.name,
                    row_index=idx,
                    row_values=row_values,
                    proof=proof_obj,
                    next_index=next_idx,
                    next_row_values=next_vals,
                    next_proof=next_proof,
                )
            )
        return openings

    @staticmethod
    def _compute_leaf_stats(profile_data: dict[str, float]) -> dict[str, int]:
        leaf_base = int(profile_data.get("fri_leaf_bytes_baseline", 0))
        leaf_actual = int(
            profile_data.get(
                "fri_leaf_bytes_actual",
                profile_data.get("fri_leaf_siblings", 0) * 32,
            )
        )
        leaf_saved = max(leaf_base - leaf_actual, 0)
        ratio = leaf_saved / leaf_base if leaf_base else 0.0
        profile_data.setdefault("leaf_auth_bytes_baseline", leaf_base)
        profile_data.setdefault("leaf_auth_bytes_actual", leaf_actual)
        profile_data.setdefault("leaf_auth_bytes_saved", leaf_saved)
        profile_data.setdefault("leaf_auth_savings_ratio", ratio)
        return {
            "leaf_auth_bytes_baseline": leaf_base,
            "leaf_auth_bytes_actual": leaf_actual,
            "leaf_auth_bytes_saved": leaf_saved,
            "leaf_auth_savings_ratio": ratio,
        }

    def _maybe_run_nova(self, trace_path: Path, proof: Any) -> dict[str, Any] | None:
        if not getattr(self.args, "run_nova", False):
            return None
        nova_dir = ROOT / "nova_stc"
        stats_path = trace_path.parent / "nova_stats.json"
        cmd = [
            "cargo",
            "run",
            "-p",
            "nova_stc",
            "--",
            "prove",
            "--chunks",
            str(trace_path),
            "--challenges",
            str(getattr(self.args, "num_challenges", 0)),
            "--stats-out",
            str(stats_path),
        ]
        if not getattr(self.args, "no_compressed", False):
            cmd.append("--compressed")
        subprocess.run(cmd, check=True, cwd=nova_dir)
        if not stats_path.exists():
            raise FileNotFoundError(f"Nova stats missing at {stats_path}")
        nova_stats = json.loads(stats_path.read_text())
        result = {
            "nova": {
                "stats_path": str(stats_path),
                **nova_stats,
            }
        }
        return result
