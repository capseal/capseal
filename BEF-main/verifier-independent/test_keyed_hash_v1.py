#!/usr/bin/env python3
"""
Semantic Mutation Tests for keyed_hash_v1 spec.

Tests that the verifier correctly rejects:
- Flipped hash bits
- Wrong genesis hash
- Broken hash chain (skip step)
- Modified output with correct hash (forgery attempt)
- Modified hash with correct output
- Continuity violations
- Wrong timestep
- Tampered RNG addresses

Run: python verifier-independent/test_keyed_hash_v1.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Import from sibling modules
from trace_generator import generate_trace_keyed_hash_v1, canonical_json_bytes
from verifier import verify_trace_correctness, AddressableRNG


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    expected_outcome: str  # "reject" or "accept"

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def run_mutation_test(
    name: str,
    mutator: Callable[[list[dict]], list[dict]],
    seed_hex: str,
    num_steps: int = 10,
    expected_outcome: str = "reject",
) -> TestResult:
    """Run a mutation test.

    Args:
        name: Test name
        mutator: Function that mutates the trace rows
        seed_hex: RNG seed
        num_steps: Number of trace steps
        expected_outcome: "reject" or "accept"
    """
    # Generate valid trace
    rows, manifest = generate_trace_keyed_hash_v1(seed_hex, num_steps)

    # Apply mutation
    mutated_rows = mutator(rows)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for row in mutated_rows:
            f.write(json.dumps(row) + '\n')
        trace_path = Path(f.name)

    try:
        # Verify
        success, msg = verify_trace_correctness(
            trace_path,
            seed_hex,
            transition_spec_id="keyed_hash_v1",
            manifest=manifest,
        )

        if expected_outcome == "reject":
            if not success:
                return TestResult(name, True, f"Correctly rejected: {msg}", expected_outcome)
            else:
                return TestResult(name, False, f"Should have been rejected but was accepted", expected_outcome)
        else:  # expected_outcome == "accept"
            if success:
                return TestResult(name, True, f"Correctly accepted: {msg}", expected_outcome)
            else:
                return TestResult(name, False, f"Should have been accepted but was rejected: {msg}", expected_outcome)

    finally:
        trace_path.unlink()


# =============================================================================
# MUTATION FUNCTIONS
# =============================================================================

def no_mutation(rows: list[dict]) -> list[dict]:
    """Control: no mutation."""
    return rows


def flip_hash_bit(rows: list[dict]) -> list[dict]:
    """Flip one bit in the middle row's hash."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    old_hash = rows[mid]["view_post"]["hash"]
    # Flip first character
    new_first = hex((int(old_hash[0], 16) ^ 1) % 16)[2:]
    rows[mid]["view_post"]["hash"] = new_first + old_hash[1:]
    return rows


def wrong_genesis_hash(rows: list[dict]) -> list[dict]:
    """Replace genesis hash with wrong value."""
    rows = copy.deepcopy(rows)
    rows[0]["view_pre"]["hash"] = "0" * 64  # Wrong genesis
    return rows


def break_hash_chain(rows: list[dict]) -> list[dict]:
    """Break hash chain by using wrong previous hash."""
    rows = copy.deepcopy(rows)
    if len(rows) > 2:
        # Use hash from 2 steps back
        rows[2]["view_pre"]["hash"] = rows[0]["view_post"]["hash"]
    return rows


def modify_output_keep_hash(rows: list[dict]) -> list[dict]:
    """Modify output but keep hash (forgery attempt)."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    rows[mid]["x_t"][0] += 0.001  # Modify output
    # Hash is now wrong because H(hash || x_t) changed
    return rows


def modify_hash_keep_output(rows: list[dict]) -> list[dict]:
    """Modify hash but keep output."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    # Change hash to something else
    rows[mid]["view_post"]["hash"] = hashlib.sha256(b"forged").hexdigest()
    return rows


def break_continuity_hash(rows: list[dict]) -> list[dict]:
    """Break continuity: view_post[t].hash != view_pre[t+1].hash."""
    rows = copy.deepcopy(rows)
    if len(rows) > 1:
        rows[0]["view_post"]["hash"] = hashlib.sha256(b"discontinuity").hexdigest()
    return rows


def break_continuity_t(rows: list[dict]) -> list[dict]:
    """Break continuity: view_post[t].t != view_pre[t+1].t."""
    rows = copy.deepcopy(rows)
    if len(rows) > 1:
        rows[0]["view_post"]["t"] = 999
    return rows


def wrong_timestep(rows: list[dict]) -> list[dict]:
    """Wrong timestep increment."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    rows[mid]["view_post"]["t"] = rows[mid]["view_pre"]["t"] + 2  # Should be +1
    return rows


def swap_adjacent_rows(rows: list[dict]) -> list[dict]:
    """Swap two adjacent rows."""
    rows = copy.deepcopy(rows)
    if len(rows) >= 2:
        rows[1], rows[2] = rows[2], rows[1]
    return rows


def delete_middle_row(rows: list[dict]) -> list[dict]:
    """Delete a row from the middle."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    del rows[mid]
    return rows


def duplicate_row(rows: list[dict]) -> list[dict]:
    """Duplicate a row."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    rows.insert(mid, copy.deepcopy(rows[mid]))
    return rows


def tamper_rng_address_tag(rows: list[dict]) -> list[dict]:
    """Tamper with RNG address tag."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    if rows[mid]["rand_addrs"]:
        rows[mid]["rand_addrs"][0]["tag"] = "wrong_tag"
    return rows


def tamper_rng_address_index(rows: list[dict]) -> list[dict]:
    """Tamper with RNG address index."""
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2
    if rows[mid]["rand_addrs"]:
        rows[mid]["rand_addrs"][0]["i"] = 9999
    return rows


def recompute_hash_after_output_change(rows: list[dict]) -> list[dict]:
    """Sophisticated attack: change output AND recompute hash.

    This should still fail because:
    1. Output doesn't match RNG
    2. Continuity breaks at next step
    """
    rows = copy.deepcopy(rows)
    mid = len(rows) // 2

    # Modify output
    rows[mid]["x_t"][0] += 0.001

    # Recompute hash (attacker knows the algorithm)
    pre_hash = rows[mid]["view_pre"]["hash"]
    x_t_bytes = canonical_json_bytes(rows[mid]["x_t"])
    new_hash = hashlib.sha256(pre_hash.encode() + x_t_bytes).hexdigest()
    rows[mid]["view_post"]["hash"] = new_hash

    # But now continuity breaks with next row!
    return rows


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests(seed_hex: str = "cafebabe" * 8) -> list[TestResult]:
    """Run all keyed_hash_v1 mutation tests."""
    tests = [
        # Control
        ("control_valid_trace", no_mutation, "accept"),

        # Hash tampering
        ("flip_hash_bit", flip_hash_bit, "reject"),
        ("wrong_genesis_hash", wrong_genesis_hash, "reject"),
        ("break_hash_chain", break_hash_chain, "reject"),
        ("modify_output_keep_hash", modify_output_keep_hash, "reject"),
        ("modify_hash_keep_output", modify_hash_keep_output, "reject"),

        # Continuity violations
        ("break_continuity_hash", break_continuity_hash, "reject"),
        ("break_continuity_t", break_continuity_t, "reject"),

        # Timestep violations
        ("wrong_timestep", wrong_timestep, "reject"),

        # Row tampering
        ("swap_adjacent_rows", swap_adjacent_rows, "reject"),
        ("delete_middle_row", delete_middle_row, "reject"),
        ("duplicate_row", duplicate_row, "reject"),

        # RNG address tampering
        ("tamper_rng_address_tag", tamper_rng_address_tag, "reject"),
        ("tamper_rng_address_index", tamper_rng_address_index, "reject"),

        # Sophisticated attacks
        ("recompute_hash_after_output_change", recompute_hash_after_output_change, "reject"),
    ]

    print("=" * 70)
    print("KEYED_HASH_V1 SEMANTIC MUTATION TESTS")
    print("=" * 70)
    print()

    results = []
    for name, mutator, expected in tests:
        result = run_mutation_test(name, mutator, seed_hex, expected_outcome=expected)
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
