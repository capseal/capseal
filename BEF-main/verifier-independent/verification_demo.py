#!/usr/bin/env python3
"""
Production verification demo for bicep_v2.

Emits a canonical run directory, then verifies it through the production
verifier.py entrypoint. This is NOT a test — it exercises the same path
that a real BICEP run would use.

Canonical run directory layout:
    <run_dir>/
        trace.jsonl
        manifest.json
        commitments.json
        audit_openings/
            audit_step_0000.json
            audit_step_0001.json
            ...

Usage:
    python verifier-independent/verification_demo.py [--run-dir <path>]
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

# These imports are from sibling modules in verifier-independent/
from state_audit import canonical_json_bytes

from verifier import (
    verify_trace,
    verify_trace_correctness,
)

# Production emitter lives in bef_zk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bef_zk.capsule.bicep_v2_executor import emit_bicep_v2_run  # type: ignore


SEED = "cafebabe" * 8


def emit_run_directory(
    run_dir: Path,
    seed_hex: str = SEED,
    num_steps: int = 10,
    num_paths: int = 4,
    num_channels: int = 4,
    audit_k: int = 8,
) -> dict:
    """Emit a complete bicep_v2 run directory via the production emitter."""
    run_dir.mkdir(parents=True, exist_ok=True)
    emit_bicep_v2_run(
        run_dir,
        seed_hex=seed_hex,
        num_steps=num_steps,
        num_paths=num_paths,
        num_channels=num_channels,
        audit_k=audit_k,
    )

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest


def verify_run_directory(run_dir: Path, seed_hex: str = SEED) -> bool:
    """Verify a run directory through the production verifier entrypoint."""
    trace_path = run_dir / "trace.jsonl"
    commitments_path = run_dir / "commitments.json"
    manifest_path = run_dir / "manifest.json"
    openings_dir = run_dir / "audit_openings"

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 70)
    print("PRODUCTION VERIFICATION: bicep_v2")
    print("=" * 70)
    print(f"  Run directory: {run_dir}")
    print(f"  Spec: {manifest.get('transition_spec_id', '?')}")
    print(f"  State: {manifest.get('state_num_paths', '?')} paths x {manifest.get('state_num_channels', '?')} channels")
    print(f"  Audit k: {manifest.get('audit_k', '?')}")
    print()

    all_ok = True

    # Level 0: Integrity
    print("[Level 0] Hash chain integrity...")
    ok, msg = verify_trace(trace_path, commitments_path)
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    all_ok = all_ok and ok

    # Level 1 + 1.5: Correctness + Audit
    print("[Level 1+1.5] Correctness + state audit...")
    ok, msg = verify_trace_correctness(
        trace_path,
        seed_hex,
        transition_spec_id="bicep_v2",
        manifest=manifest,
        openings_path=openings_dir,
    )
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    all_ok = all_ok and ok

    print()
    print("=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED' if all_ok else 'VERIFICATION FAILED'}")
    print("=" * 70)

    return all_ok


def main():
    # Parse --run-dir
    run_dir = None
    if "--run-dir" in sys.argv:
        idx = sys.argv.index("--run-dir")
        if idx + 1 < len(sys.argv):
            run_dir = Path(sys.argv[idx + 1])

    if run_dir is None:
        tmp = tempfile.mkdtemp(prefix="bicep_v2_run_")
        run_dir = Path(tmp)
        print(f"Using temp directory: {run_dir}")

    print("Emitting run directory...")
    emit_run_directory(run_dir)
    print(f"Run directory emitted: {run_dir}")
    print()

    ok = verify_run_directory(run_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
