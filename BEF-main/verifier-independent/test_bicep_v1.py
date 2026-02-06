#!/usr/bin/env python3
"""
Semantic Mutation Tests for bicep_v1 spec.

Tests that the verifier correctly rejects all mutations targeting:
- Quantized output integrity (x_t_q)
- Output chain binding (H(chain || x_t_q))
- rng_use_hash tamper evidence
- State root continuity
- Genesis state
- Quantization boundary attacks (float != quantize(dequantize(q)))

Run: python verifier-independent/test_bicep_v1.py
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

from trace_generator import (
    generate_trace_bicep_v1,
    canonical_json_bytes,
    quantize,
    dequantize,
    compute_output_chain_update,
    compute_rng_use_hash,
)
from verifier import verify_trace_correctness


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def run_test(
    name: str,
    mutator: Callable[[list[dict]], list[dict]],
    seed_hex: str = "cafebabe" * 8,
    num_steps: int = 10,
    expected: str = "reject",
) -> TestResult:
    rows, manifest = generate_trace_bicep_v1(seed_hex, num_steps)
    mutated = mutator(rows)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for row in mutated:
            f.write(json.dumps(row) + '\n')
        trace_path = Path(f.name)

    try:
        success, msg = verify_trace_correctness(
            trace_path, seed_hex,
            transition_spec_id="bicep_v1",
            manifest=manifest,
        )

        if expected == "reject":
            if not success:
                return TestResult(name, True, f"Correctly rejected: {msg}")
            return TestResult(name, False, "Should have been rejected but was accepted")
        else:
            if success:
                return TestResult(name, True, f"Correctly accepted: {msg}")
            return TestResult(name, False, f"Should have been accepted but was rejected: {msg}")
    finally:
        trace_path.unlink()


# =============================================================================
# MUTATIONS
# =============================================================================

def no_mutation(rows): return rows

def flip_quantized_value(rows):
    """Flip one quantized output value."""
    rows = copy.deepcopy(rows)
    rows[3]["x_t_q"][0] += 1  # Off by 1 LSB
    return rows

def wrong_quantization(rows):
    """Use wrong precision bits to quantize."""
    rows = copy.deepcopy(rows)
    # Re-quantize with 16 bits instead of 24
    val = dequantize(rows[5]["x_t_q"][0], 24)
    rows[5]["x_t_q"][0] = quantize(val, 16)
    return rows

def tamper_output_chain(rows):
    """Modify output_chain in view_post."""
    rows = copy.deepcopy(rows)
    rows[4]["view_post"]["output_chain"] = hashlib.sha256(b"tampered").hexdigest()
    return rows

def tamper_rng_use_hash(rows):
    """Modify rng_use_hash."""
    rows = copy.deepcopy(rows)
    rows[3]["rng_use_hash"] = hashlib.sha256(b"wrong").hexdigest()
    return rows

def wrong_genesis_state_root(rows):
    """Wrong genesis state_root."""
    rows = copy.deepcopy(rows)
    rows[0]["view_pre"]["state_root"] = "0" * 64
    return rows

def wrong_genesis_output_chain(rows):
    """Wrong genesis output_chain."""
    rows = copy.deepcopy(rows)
    rows[0]["view_pre"]["output_chain"] = "0" * 64
    return rows

def break_state_root_continuity(rows):
    """Break state_root continuity between rows."""
    rows = copy.deepcopy(rows)
    rows[4]["view_post"]["state_root"] = hashlib.sha256(b"discontinuous").hexdigest()
    return rows

def break_output_chain_continuity(rows):
    """Break output_chain continuity between rows."""
    rows = copy.deepcopy(rows)
    rows[2]["view_post"]["output_chain"] = hashlib.sha256(b"break").hexdigest()
    return rows

def swap_rows(rows):
    """Swap adjacent rows."""
    rows = copy.deepcopy(rows)
    rows[3], rows[4] = rows[4], rows[3]
    return rows

def delete_row(rows):
    """Delete a middle row."""
    rows = copy.deepcopy(rows)
    del rows[5]
    return rows

def duplicate_row(rows):
    """Duplicate a row."""
    rows = copy.deepcopy(rows)
    rows.insert(4, copy.deepcopy(rows[4]))
    return rows

def modify_rand_addrs_not_hash(rows):
    """Modify rand_addrs but don't update rng_use_hash (detected by hash check)."""
    rows = copy.deepcopy(rows)
    rows[3]["rand_addrs"][0]["tag"] = "wrong"
    # rng_use_hash is now stale
    return rows

def modify_rand_addrs_with_hash(rows):
    """Modify rand_addrs AND update rng_use_hash (detected by output check)."""
    rows = copy.deepcopy(rows)
    rows[3]["rand_addrs"][0]["tag"] = "wrong"
    rows[3]["rng_use_hash"] = compute_rng_use_hash(rows[3]["rand_addrs"])
    # Outputs no longer match RNG addresses
    return rows

def recompute_chain_after_output_flip(rows):
    """Sophisticated: flip output, recompute output_chain, still caught by output check."""
    rows = copy.deepcopy(rows)
    mid = 5
    rows[mid]["x_t_q"][0] += 1

    # Recompute output_chain (attacker knows algorithm)
    prev_chain = rows[mid]["view_pre"]["output_chain"]
    new_chain = compute_output_chain_update(prev_chain, rows[mid]["x_t_q"])
    rows[mid]["view_post"]["output_chain"] = new_chain

    # But continuity breaks with next row AND output doesn't match RNG
    return rows

def inject_float_not_quantized(rows):
    """Put a float value in x_t_q that isn't a valid quantized integer."""
    rows = copy.deepcopy(rows)
    rows[4]["x_t_q"][0] = 0.5  # Float in integer field
    return rows

def remove_rng_use_hash(rows):
    """Remove rng_use_hash entirely."""
    rows = copy.deepcopy(rows)
    rows[3]["rng_use_hash"] = ""
    return rows

def remove_x_t_q(rows):
    """Remove x_t_q field (fall back to empty)."""
    rows = copy.deepcopy(rows)
    rows[3]["x_t_q"] = []
    return rows


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests() -> list[TestResult]:
    tests = [
        # Control
        ("control_valid_trace", no_mutation, "accept"),

        # Quantized output integrity
        ("flip_quantized_value", flip_quantized_value, "reject"),
        ("wrong_quantization_bits", wrong_quantization, "reject"),
        ("inject_float_in_x_t_q", inject_float_not_quantized, "reject"),
        ("remove_x_t_q", remove_x_t_q, "reject"),

        # Output chain binding
        ("tamper_output_chain", tamper_output_chain, "reject"),
        ("break_output_chain_continuity", break_output_chain_continuity, "reject"),

        # rng_use_hash
        ("tamper_rng_use_hash", tamper_rng_use_hash, "reject"),
        ("remove_rng_use_hash", remove_rng_use_hash, "reject"),

        # State root
        ("wrong_genesis_state_root", wrong_genesis_state_root, "reject"),
        ("wrong_genesis_output_chain", wrong_genesis_output_chain, "reject"),
        ("break_state_root_continuity", break_state_root_continuity, "reject"),

        # Row tampering
        ("swap_rows", swap_rows, "reject"),
        ("delete_row", delete_row, "reject"),
        ("duplicate_row", duplicate_row, "reject"),

        # RNG address tampering
        ("modify_rand_addrs_not_hash", modify_rand_addrs_not_hash, "reject"),
        ("modify_rand_addrs_with_hash", modify_rand_addrs_with_hash, "reject"),

        # Sophisticated attacks
        ("recompute_chain_after_output_flip", recompute_chain_after_output_flip, "reject"),
    ]

    print("=" * 70)
    print("BICEP_V1 SEMANTIC MUTATION TESTS")
    print("=" * 70)
    print()

    results = []
    for name, mutator, expected in tests:
        result = run_test(name, mutator, expected=expected)
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
