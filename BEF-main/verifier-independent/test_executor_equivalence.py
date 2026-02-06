#!/usr/bin/env python3
"""
Executor-vs-Generator Equivalence Test.

Proves that the production executor (bef_zk.capsule.bicep_v2_executor)
and the test generator (test_bicep_v2.generate_bicep_v2_trace) produce
byte-for-byte identical artifacts when given the same seed + parameters.

This is the #1 regression guard against "verifier passes demo, fails production."

Run: python verifier-independent/test_executor_equivalence.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Add repo root to path for canonical imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_audit import canonical_json_bytes
from test_bicep_v2 import generate_bicep_v2_trace

from bef_zk.capsule.bicep_v2_executor import emit_bicep_v2_run

from verifier import (
    verify_trace,
    verify_trace_correctness,
)


SEED = "cafebabe" * 8


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# =============================================================================
# EQUIVALENCE TESTS
# =============================================================================

def test_state_root_equivalence() -> TestResult:
    """Executor and generator produce identical state_root at every step."""
    # Generator path
    rows_gen, audits_gen, manifest_gen, mh_gen = generate_bicep_v2_trace(
        SEED, num_steps=8, num_paths=4, num_channels=4, audit_k=4,
    )

    # Executor path
    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8, num_paths=4,
                      num_channels=4, audit_k=4)

    with open(run_dir / "trace.jsonl") as f:
        rows_exec = [json.loads(line) for line in f if line.strip()]

    for t, (rg, re) in enumerate(zip(rows_gen, rows_exec)):
        gen_pre = rg["view_pre"]["state_root"]
        exec_pre = re["view_pre"]["state_root"]
        if gen_pre != exec_pre:
            return TestResult("state_root_equivalence", False,
                              f"state_root_pre mismatch at t={t}: gen={gen_pre[:16]}..., exec={exec_pre[:16]}...")

        gen_post = rg["view_post"]["state_root"]
        exec_post = re["view_post"]["state_root"]
        if gen_post != exec_post:
            return TestResult("state_root_equivalence", False,
                              f"state_root_post mismatch at t={t}: gen={gen_post[:16]}..., exec={exec_post[:16]}...")

    return TestResult("state_root_equivalence", True,
                      f"All {len(rows_gen)} steps match byte-for-byte")


def test_challenge_seed_equivalence() -> TestResult:
    """Executor and generator produce identical challenge_seed at every step."""
    rows_gen, audits_gen, _, _ = generate_bicep_v2_trace(
        SEED, num_steps=8, num_paths=4, num_channels=4, audit_k=4,
    )

    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8, num_paths=4,
                      num_channels=4, audit_k=4)

    exec_audits = []
    for t in range(8):
        with open(run_dir / "audit_openings" / f"audit_step_{t:04d}.json") as f:
            exec_audits.append(json.load(f))

    for t, (ag, ae) in enumerate(zip(audits_gen, exec_audits)):
        if ag["challenge_seed"] != ae["challenge_seed"]:
            return TestResult("challenge_seed_equivalence", False,
                              f"challenge_seed mismatch at t={t}")

    return TestResult("challenge_seed_equivalence", True,
                      f"All {len(audits_gen)} challenge seeds match")


def test_audit_indices_equivalence() -> TestResult:
    """Executor and generator produce identical audit indices."""
    _, audits_gen, _, _ = generate_bicep_v2_trace(
        SEED, num_steps=8, num_paths=4, num_channels=4, audit_k=4,
    )

    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8, num_paths=4,
                      num_channels=4, audit_k=4)

    for t in range(8):
        with open(run_dir / "audit_openings" / f"audit_step_{t:04d}.json") as f:
            ae = json.load(f)
        if audits_gen[t]["audit_indices"] != ae["audit_indices"]:
            return TestResult("audit_indices_equivalence", False,
                              f"audit_indices mismatch at t={t}")

    return TestResult("audit_indices_equivalence", True, "All 8 steps match")


def test_opening_payload_equivalence() -> TestResult:
    """Executor and generator produce identical opening payloads (value_q + proof)."""
    _, audits_gen, _, _ = generate_bicep_v2_trace(
        SEED, num_steps=8, num_paths=4, num_channels=4, audit_k=4,
    )

    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8, num_paths=4,
                      num_channels=4, audit_k=4)

    for t in range(8):
        with open(run_dir / "audit_openings" / f"audit_step_{t:04d}.json") as f:
            ae = json.load(f)

        ag = audits_gen[t]

        for side in ("openings_pre", "openings_post"):
            for i, (og, oe) in enumerate(zip(ag[side], ae[side])):
                if og["leaf_index"] != oe["leaf_index"]:
                    return TestResult("opening_payload_equivalence", False,
                                      f"leaf_index mismatch at t={t}, {side}[{i}]")
                if og["value_q"] != oe["value_q"]:
                    return TestResult("opening_payload_equivalence", False,
                                      f"value_q mismatch at t={t}, {side}[{i}]: gen={og['value_q']}, exec={oe['value_q']}")
                # Normalize tuple/list from JSON round-trip
                gen_proof = [list(p) if isinstance(p, tuple) else p for p in og["merkle_proof"]]
                exec_proof = [list(p) if isinstance(p, tuple) else p for p in oe["merkle_proof"]]
                if gen_proof != exec_proof:
                    return TestResult("opening_payload_equivalence", False,
                                      f"merkle_proof mismatch at t={t}, {side}[{i}]")

    return TestResult("opening_payload_equivalence", True,
                      "All opening payloads match byte-for-byte across 8 steps")


def test_trace_row_byte_equivalence() -> TestResult:
    """Entire trace rows are byte-for-byte identical (canonical JSON)."""
    rows_gen, _, _, _ = generate_bicep_v2_trace(
        SEED, num_steps=8, num_paths=4, num_channels=4, audit_k=4,
    )

    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8, num_paths=4,
                      num_channels=4, audit_k=4)

    with open(run_dir / "trace.jsonl") as f:
        rows_exec = [json.loads(line) for line in f if line.strip()]

    for t, (rg, re) in enumerate(zip(rows_gen, rows_exec)):
        gen_bytes = canonical_json_bytes(rg)
        exec_bytes = canonical_json_bytes(re)
        if gen_bytes != exec_bytes:
            return TestResult("trace_row_byte_equivalence", False,
                              f"Row t={t} canonical bytes differ")

    return TestResult("trace_row_byte_equivalence", True,
                      f"All {len(rows_gen)} rows identical in canonical form")


def test_executor_run_verifies() -> TestResult:
    """Production executor output passes full verifier.py pipeline."""
    tmp = tempfile.mkdtemp(prefix="equiv_")
    run_dir = Path(tmp)
    manifest = emit_bicep_v2_run(run_dir, seed_hex=SEED, num_steps=8,
                                  num_paths=4, num_channels=4, audit_k=4)

    # Level 0
    ok0, msg0 = verify_trace(run_dir / "trace.jsonl", run_dir / "commitments.json")
    if not ok0:
        return TestResult("executor_run_verifies", False, f"Level 0 failed: {msg0}")

    # Level 1 + 1.5
    ok1, msg1 = verify_trace_correctness(
        run_dir / "trace.jsonl",
        SEED,
        transition_spec_id="bicep_v2",
        manifest=manifest,
        openings_path=run_dir / "audit_openings",
    )
    if not ok1:
        return TestResult("executor_run_verifies", False, f"Level 1+1.5 failed: {msg1}")

    return TestResult("executor_run_verifies", True, msg1)


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests() -> list[TestResult]:
    tests = [
        test_state_root_equivalence,
        test_challenge_seed_equivalence,
        test_audit_indices_equivalence,
        test_opening_payload_equivalence,
        test_trace_row_byte_equivalence,
        test_executor_run_verifies,
    ]

    print("=" * 70)
    print("EXECUTOR vs GENERATOR EQUIVALENCE TESTS")
    print("=" * 70)
    print()

    results = []
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        print(result)

    print()
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")

    if failed > 0:
        print("\nFAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if all(r.passed for r in results) else 1)
