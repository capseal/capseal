#!/usr/bin/env python3
"""Boundary Condition Tests for Verification Infrastructure.

Tests edge cases where Merkle systems quietly lie:
- step=0 opening
- step=T-1 opening
- step exactly at chunk boundary (K, 2K, ...)
- K=1, K=2, K=1024
- trace length not multiple of K
- missing last checkpoint
- empty trace / 1-row trace

Run: PYTHONPATH=. python scripts/boundary_tests.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.capsule.manifest import create_manifest
from bef_zk.capsule.trace_chain import (
    TraceRow,
    TraceEmitter,
    CheckpointReceipt,
    open_row,
    verify_opening,
    verify_trace_against_commitments,
)


@dataclass
class BoundaryTestResult:
    name: str
    passed: bool
    message: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def run_boundary_test(
    name: str,
    num_steps: int,
    checkpoint_interval: int,
    test_steps: Optional[list[int]] = None,
    expect_failure: bool = False,
) -> BoundaryTestResult:
    """Run a boundary test with given parameters."""
    try:
        with tempfile.TemporaryDirectory(prefix=f"boundary_{name}_") as tmp:
            tmp_dir = Path(tmp)

            # Create manifest and emitter
            seed = bytes.fromhex("b0a1d4e1" * 8)  # Valid hex (32 bytes)
            manifest, rng = create_manifest(
                seed=seed,
                bicep_version="0.1.0",
                checkpoint_interval=checkpoint_interval,
            )

            emitter = TraceEmitter(
                manifest_hash=manifest.manifest_hash,
                checkpoint_interval=checkpoint_interval,
                output_dir=tmp_dir,
            )

            # Emit rows
            for t in range(num_steps):
                row = TraceRow(
                    t=t,
                    x_t=[rng.rand("input", t, i) for i in range(3)],
                    view_pre={"state": t},
                    view_post={"state": t + 1},
                    rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(3)],
                )
                emitter.emit(row)

            summary = emitter.finalize()
            trace_path = tmp_dir / "trace.jsonl"
            commitments_path = tmp_dir / "commitments.json"

            # Verify trace first
            valid, msg = verify_trace_against_commitments(trace_path, commitments_path)
            if not valid:
                if expect_failure:
                    return BoundaryTestResult(name, True, f"Expected failure: {msg}")
                return BoundaryTestResult(name, False, f"Trace verification failed: {msg}")

            # Test specific step openings
            if test_steps:
                for step in test_steps:
                    if step >= num_steps:
                        continue  # Skip invalid steps

                    try:
                        opening = open_row(step=step, trace_path=trace_path, checkpoints_dir=tmp_dir)
                        valid, msg = verify_opening(opening)
                        if not valid:
                            return BoundaryTestResult(name, False, f"Opening step={step} failed: {msg}")
                    except ValueError as e:
                        return BoundaryTestResult(name, False, f"Opening step={step} error: {e}")

            return BoundaryTestResult(
                name,
                True,
                f"T={num_steps}, K={checkpoint_interval}, checkpoints={summary['total_checkpoints']}"
            )

    except Exception as e:
        if expect_failure:
            return BoundaryTestResult(name, True, f"Expected failure: {e}")
        return BoundaryTestResult(name, False, f"Exception: {e}")


def test_empty_trace() -> BoundaryTestResult:
    """Test empty trace (0 rows).

    Empty trace is a valid edge case - should produce 0 checkpoints
    but finalize() should still work.
    """
    name = "empty_trace"
    try:
        with tempfile.TemporaryDirectory(prefix="boundary_empty_") as tmp:
            tmp_dir = Path(tmp)

            seed = bytes.fromhex("e0e0e0e0" * 8)
            manifest, rng = create_manifest(seed=seed, checkpoint_interval=4)

            emitter = TraceEmitter(
                manifest_hash=manifest.manifest_hash,
                checkpoint_interval=4,
                output_dir=tmp_dir,
            )

            # Don't emit any rows
            summary = emitter.finalize()

            # Verify summary
            if summary["total_steps"] != 0:
                return BoundaryTestResult(name, False, f"Expected 0 steps, got {summary['total_steps']}")

            if summary["total_checkpoints"] != 0:
                return BoundaryTestResult(name, False, f"Expected 0 checkpoints, got {summary['total_checkpoints']}")

            return BoundaryTestResult(name, True, "Empty trace handled correctly (0 steps, 0 checkpoints)")

    except Exception as e:
        return BoundaryTestResult(name, False, f"Exception: {e}")


def test_single_row() -> BoundaryTestResult:
    """Test single row trace."""
    return run_boundary_test(
        name="single_row",
        num_steps=1,
        checkpoint_interval=4,
        test_steps=[0],
    )


def test_two_rows() -> BoundaryTestResult:
    """Test two row trace."""
    return run_boundary_test(
        name="two_rows",
        num_steps=2,
        checkpoint_interval=4,
        test_steps=[0, 1],
    )


def test_step_zero_opening() -> BoundaryTestResult:
    """Test opening step=0."""
    return run_boundary_test(
        name="step_zero_opening",
        num_steps=10,
        checkpoint_interval=4,
        test_steps=[0],
    )


def test_step_last_opening() -> BoundaryTestResult:
    """Test opening step=T-1."""
    return run_boundary_test(
        name="step_last_opening",
        num_steps=10,
        checkpoint_interval=4,
        test_steps=[9],  # T-1
    )


def test_chunk_boundary_opening() -> BoundaryTestResult:
    """Test opening step exactly at chunk boundary (K, 2K, ...)."""
    return run_boundary_test(
        name="chunk_boundary_opening",
        num_steps=12,
        checkpoint_interval=4,
        test_steps=[0, 4, 8],  # First step of each chunk
    )


def test_chunk_last_step_opening() -> BoundaryTestResult:
    """Test opening step at end of chunk (K-1, 2K-1, ...)."""
    return run_boundary_test(
        name="chunk_last_step_opening",
        num_steps=12,
        checkpoint_interval=4,
        test_steps=[3, 7, 11],  # Last step of each chunk
    )


def test_k_equals_1() -> BoundaryTestResult:
    """Test K=1 (every step is a checkpoint)."""
    return run_boundary_test(
        name="k_equals_1",
        num_steps=5,
        checkpoint_interval=1,
        test_steps=[0, 2, 4],
    )


def test_k_equals_2() -> BoundaryTestResult:
    """Test K=2."""
    return run_boundary_test(
        name="k_equals_2",
        num_steps=7,
        checkpoint_interval=2,
        test_steps=[0, 1, 2, 3, 6],
    )


def test_k_large() -> BoundaryTestResult:
    """Test K=1024 (single checkpoint for small trace)."""
    return run_boundary_test(
        name="k_large_1024",
        num_steps=10,
        checkpoint_interval=1024,
        test_steps=[0, 5, 9],
    )


def test_not_multiple_of_k() -> BoundaryTestResult:
    """Test trace length not multiple of K."""
    return run_boundary_test(
        name="not_multiple_of_k",
        num_steps=11,  # Not multiple of 4
        checkpoint_interval=4,
        test_steps=[0, 4, 10],  # Including last row of partial chunk
    )


def test_k_equals_trace_length() -> BoundaryTestResult:
    """Test K equals trace length (single full chunk)."""
    return run_boundary_test(
        name="k_equals_trace_length",
        num_steps=8,
        checkpoint_interval=8,
        test_steps=[0, 4, 7],
    )


def test_k_greater_than_trace_length() -> BoundaryTestResult:
    """Test K greater than trace length."""
    return run_boundary_test(
        name="k_greater_than_trace",
        num_steps=5,
        checkpoint_interval=10,
        test_steps=[0, 2, 4],
    )


def test_all_steps_openable() -> BoundaryTestResult:
    """Test that every step in a trace can be opened."""
    name = "all_steps_openable"
    try:
        with tempfile.TemporaryDirectory(prefix="boundary_all_steps_") as tmp:
            tmp_dir = Path(tmp)

            seed = bytes.fromhex("a115e9a5" * 8)  # Valid hex (32 bytes)
            manifest, rng = create_manifest(seed=seed, checkpoint_interval=4)

            emitter = TraceEmitter(
                manifest_hash=manifest.manifest_hash,
                checkpoint_interval=4,
                output_dir=tmp_dir,
            )

            num_steps = 15
            for t in range(num_steps):
                row = TraceRow(
                    t=t,
                    x_t=[rng.rand("input", t, i) for i in range(3)],
                    view_pre={"state": t},
                    view_post={"state": t + 1},
                    rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(3)],
                )
                emitter.emit(row)

            emitter.finalize()
            trace_path = tmp_dir / "trace.jsonl"

            # Try to open every step
            for step in range(num_steps):
                opening = open_row(step=step, trace_path=trace_path, checkpoints_dir=tmp_dir)
                valid, msg = verify_opening(opening)
                if not valid:
                    return BoundaryTestResult(name, False, f"Step {step} failed: {msg}")

            return BoundaryTestResult(name, True, f"All {num_steps} steps openable and verifiable")

    except Exception as e:
        return BoundaryTestResult(name, False, f"Exception: {e}")


def test_merkle_proof_boundary_indices() -> BoundaryTestResult:
    """Test Merkle proofs at boundary leaf indices (0, K-1, power-of-2 edges)."""
    name = "merkle_proof_boundaries"
    try:
        with tempfile.TemporaryDirectory(prefix="boundary_merkle_") as tmp:
            tmp_dir = Path(tmp)

            seed = bytes.fromhex("ae4c1e00" * 8)  # Valid hex (32 bytes)
            manifest, rng = create_manifest(seed=seed, checkpoint_interval=8)

            emitter = TraceEmitter(
                manifest_hash=manifest.manifest_hash,
                checkpoint_interval=8,  # Power of 2
                output_dir=tmp_dir,
            )

            # Create exactly 8 rows (one full chunk)
            for t in range(8):
                row = TraceRow(
                    t=t,
                    x_t=[rng.rand("input", t, i) for i in range(3)],
                    view_pre={"state": t},
                    view_post={"state": t + 1},
                    rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(3)],
                )
                emitter.emit(row)

            emitter.finalize()
            trace_path = tmp_dir / "trace.jsonl"

            # Test boundary indices: 0, 1, 3, 4, 7 (first, second, mid, power-of-2, last)
            for step in [0, 1, 3, 4, 7]:
                opening = open_row(step=step, trace_path=trace_path, checkpoints_dir=tmp_dir)
                valid, msg = verify_opening(opening)
                if not valid:
                    return BoundaryTestResult(name, False, f"Step {step} (leaf_idx={opening.leaf_index}) failed")

            return BoundaryTestResult(name, True, "All boundary indices verified")

    except Exception as e:
        return BoundaryTestResult(name, False, f"Exception: {e}")


def run_all_boundary_tests() -> list[BoundaryTestResult]:
    """Run all boundary condition tests."""
    tests = [
        test_empty_trace,
        test_single_row,
        test_two_rows,
        test_step_zero_opening,
        test_step_last_opening,
        test_chunk_boundary_opening,
        test_chunk_last_step_opening,
        test_k_equals_1,
        test_k_equals_2,
        test_k_large,
        test_not_multiple_of_k,
        test_k_equals_trace_length,
        test_k_greater_than_trace_length,
        test_all_steps_openable,
        test_merkle_proof_boundary_indices,
    ]

    print("=" * 70)
    print("BOUNDARY CONDITION TESTS")
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
    results = run_all_boundary_tests()
    sys.exit(0 if all(r.passed for r in results) else 1)
