#!/usr/bin/env python3
"""Comprehensive regression checks for binary + multiproof pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.codec import canonical_decode
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.serialization import proof_from_bytes, proof_to_bytes
from bef_zk.zk_geom.verifier import zk_verify_geom
from scripts.geom_programs import GEOM_PROGRAM
from scripts.stc_da_sample import derive_seed_from_capsule, sample_chunk_indices
from scripts.verify_capsule import verify_capsule

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not existing else f"{repo}:{existing}"
    return env


def _run_pipeline(out_dir: Path, steps: int, num_queries: int) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_geom_pipeline.py"),
        "--steps",
        str(steps),
        "--num-queries",
        str(num_queries),
        "--output-dir",
        str(out_dir),
        "--skip-nova",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_python_env())


def _load_capsule_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_capsule_bin(path: Path) -> dict:
    return canonical_decode(path.read_bytes())  # type: ignore[return-value]


def _build_params(capsule: dict) -> GeomAIRParams:
    params = capsule["params"]
    return GeomAIRParams(
        steps=int(params["steps"]),
        num_challenges=int(params["num_challenges"]),
        r_challenges=[int(x) for x in params["r_challenges"]],
        matrix=[[2, 1], [1, 1]],
    )


def _verify_bytes(capsule: dict, proof_bytes: bytes) -> bool:
    params = _build_params(capsule)
    init = GeomInitialState()
    row_width = int(capsule["params"]["row_width"])
    vc = STCVectorCommitment(chunk_len=row_width)
    proof = proof_from_bytes(proof_bytes)
    ok, _ = zk_verify_geom(GEOM_PROGRAM, params, init, vc, proof)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression suite")
    parser.add_argument("--output-dir", type=Path, default=Path("out/regression"))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=16)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_pipeline(out_dir, args.steps, args.num_queries)

    capsule_json_path = out_dir / "strategy_capsule.json"
    capsule_bin_path = out_dir / "strategy_capsule.bin"
    proof_bin_path = out_dir / "geom_proof.bin"

    capsule_json = _load_capsule_json(capsule_json_path)
    capsule_bin = _load_capsule_bin(capsule_bin_path)

    # 1. Binary round-trip for proof
    proof_bytes = proof_bin_path.read_bytes()
    proof_obj = proof_from_bytes(proof_bytes)
    round_trip = proof_to_bytes(proof_obj)
    if round_trip != proof_bytes:
        raise RuntimeError("proof binary round-trip mismatch")
    print("[regression] Proof binary round-trip OK")

    # 2. Multi-proof correctness (ensure multiproof present) & verification
    has_multiproof = any(
        bool(batch.proof.chunk_leaf_proofs)
        for batch in proof_obj.fri_proof.batches
    )
    if not has_multiproof:
        raise RuntimeError("proof missing chunk leaf multiproofs")
    print("[regression] Multiproof presence confirmed")
    if not _verify_bytes(capsule_json, proof_bytes):
        raise RuntimeError("fresh proof failed verification")
    print("[regression] Proof verifies successfully")

    # 3. Sampler parity between JSON and BIN capsules
    seed_json = derive_seed_from_capsule(capsule_json)
    seed_bin = derive_seed_from_capsule(capsule_bin)
    if seed_json != seed_bin:
        raise RuntimeError("capsule hash-derived seeds differ between JSON and BIN")
    da_policy = capsule_json.get("da_policy")
    if da_policy:
        k = int(da_policy.get("k_samples", 0))
    else:
        k = int((capsule_json.get("da_profile") or {}).get("sampling", {}).get("k_min", 0))
    total_chunks = int(capsule_json.get("chunk_meta", {}).get("num_chunks", 0))
    samples_json = sample_chunk_indices(total_chunks, k, seed_json)
    samples_bin = sample_chunk_indices(total_chunks, k, seed_bin)
    if samples_json != samples_bin:
        raise RuntimeError("sampler indices differ between JSON and BIN capsules")
    print("[regression] Sampler parity confirmed")

    # 4. Tamper detection quick check
    offsets = [0, len(proof_bytes) // 2, len(proof_bytes) - 1]
    for idx, off in enumerate({max(0, o) for o in offsets}):
        mutated = bytearray(proof_bytes)
        mutated[off] ^= 0xAA
        ok = False
        try:
            ok = _verify_bytes(capsule_json, bytes(mutated))
        except Exception:
            ok = False
        if ok:
            raise RuntimeError(f"tampered proof #{idx} unexpectedly verified")
    print("[regression] Tamper detection succeeded")

    # 5. Capsule tamper check using CLI verifier
    mutated_capsule = out_dir / "strategy_capsule.tampered.bin"
    data = bytearray(capsule_bin_path.read_bytes())
    data[0] ^= 0x55
    mutated_capsule.write_bytes(bytes(data))
    try:
        verify_capsule(mutated_capsule)
    except Exception:
        print("[regression] Capsule tamper rejected")
    else:
        raise RuntimeError("tampered capsule unexpectedly verified")
    finally:
        mutated_capsule.unlink(missing_ok=True)

    print("All regression checks passed.")


if __name__ == "__main__":
    main()
