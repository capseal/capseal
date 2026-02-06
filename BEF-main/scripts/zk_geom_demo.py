#!/usr/bin/env python3
"""CLI demo for the geometry AIR proof using STC + FRI."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState, simulate_trace
from bef_zk.air.geom_trace_export import geom_trace_to_bef_trace
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment, VectorCommitment
from bef_zk.zk_geom.prover import zk_prove_geom
from bef_zk.zk_geom.verifier import zk_verify_geom
from bef_zk.zk_geom.serialization import proof_to_json
from bef_zk.zk_geom.trace_only import verify_trace_only
from bef_zk.zk_geom.backend import available_row_backends
from scripts.geom_programs import GEOM_PROGRAM


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry STC+FRI demo")
    parser.add_argument("mode", choices=["prove"], help="currently only prove demo")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--output", type=str, help="write proof JSON to this file")
    parser.add_argument("--trace-id", type=str, default="geom_demo", help="identifier embedded in exported traces")
    parser.add_argument(
        "--stc-trace-out",
        type=str,
        help="write bef_trace_v1 chunk log to this path before proving",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print one-line JSON with timing + proof size",
    )
    parser.add_argument(
        "--no-zk",
        action="store_true",
        help="run trace + constraint checks only (no STC/FRI)"
    )
    parser.add_argument(
        "--row-backend",
        choices=available_row_backends(),
        default="geom_stc_fri",
        help="row commitment backend to use",
    )
    parser.add_argument(
        "--chunk-tree-arity",
        type=int,
        help="override chunk tree arity for STC backend",
    )
    parser.add_argument(
        "--row-archive-dir",
        type=str,
        help="persist STC row chunks in this directory (cleared before proving)",
    )
    args = parser.parse_args()

    params = GeomAIRParams(
        steps=args.steps,
        num_challenges=2,
        r_challenges=[1234567, 89101112],
        matrix=[[2, 1], [1, 1]],
    )
    init = GeomInitialState()
    domain_size = 1 << (args.steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    fri_cfg = FRIConfig(
        field_modulus=(1 << 61) - 1,
        domain_size=domain_size,
        max_degree=args.steps - 1,
        num_rounds=min(6, max_rounds),
        num_queries=args.num_queries,
    )
    vc: VectorCommitment = STCVectorCommitment(chunk_len=256)

    if args.stc_trace_out:
        trace_snapshot = simulate_trace(GEOM_PROGRAM, params, init)
        bef = geom_trace_to_bef_trace(trace_snapshot, args.trace_id)
        with open(args.stc_trace_out, "w", encoding="utf-8") as fh:
            json.dump(bef, fh, indent=2)
        print(f"Wrote STC trace to {args.stc_trace_out}")

    if args.no_zk:
        ok_trace, trace_stats = verify_trace_only(GEOM_PROGRAM, params, init)
        record = {
            "mode": "no_zk_geom",
            "steps": params.steps,
            "trace_ok": ok_trace,
        }
        record.update(trace_stats)
        print(json.dumps(record))
        return

    profile_data = {} if args.profile else None
    row_backend_params = {}
    archive_dir = None
    if args.row_archive_dir:
        archive_dir = Path(args.row_archive_dir).expanduser().resolve()
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        row_backend_params["archive_dir"] = archive_dir
        print(f"Row archive will be stored in {archive_dir}")
    if args.chunk_tree_arity is not None and args.row_backend == "geom_stc_fri":
        row_backend_params["chunk_tree_arity"] = int(args.chunk_tree_arity)
    t0 = time.perf_counter()
    proof = zk_prove_geom(
        GEOM_PROGRAM,
        params,
        init,
        fri_cfg,
        vc,
        row_backend=args.row_backend,
        row_backend_params=row_backend_params or None,
        profile=profile_data,
    )
    t1 = time.perf_counter()
    ok, verify_stats = zk_verify_geom(GEOM_PROGRAM, params, init, vc, proof)
    t2 = time.perf_counter()

    proof_json = proof_to_json(proof)
    proof_size_bytes = len(proof_json.encode("utf-8"))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(proof_json)
        print(f"Wrote proof JSON to {args.output} ({proof_size_bytes} bytes)")

    print("Proof generated. Verifying...")
    print("Verifier result:", ok)

    if args.profile:
        if profile_data is None:
            profile_data = {}
        profile_data.update(
            {
                "steps": params.steps,
                "num_queries": len(proof.query_indices),
                "row_openings": len(proof.row_openings),
                "proof_size_bytes": proof_size_bytes,
                "proving_time_sec": t1 - t0,
                "verify_time_sec": t2 - t1,
                "verifier_ok": bool(ok),
            }
        )
        profile_data.update(verify_stats)
        print(json.dumps(profile_data))


if __name__ == "__main__":
    main()
