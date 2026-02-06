#!/usr/bin/env python3
"""
End-to-end tests for bicep_v2: Merkle state + audit openings + SDE transition.

Tests:
1. Generate valid trace with real SDE state evolution
2. Verify audited openings pass
3. Mutation suite targeting audit proof fraud

Run: python verifier-independent/test_bicep_v2.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from state_audit import (
    AddressableRNG,
    quantize,
    dequantize,
    canonical_json_bytes,
    leaf_index,
    total_leaves,
    build_merkle_tree,
    compute_merkle_proof,
    verify_merkle_proof,
    compute_challenge_seed,
    sample_audit_indices,
    sde_step_em,
    generate_audit_opening,
    verify_audit_opening,
)


# =============================================================================
# TRACE GENERATOR for bicep_v2
# =============================================================================

def generate_bicep_v2_trace(
    seed_hex: str,
    num_steps: int = 10,
    num_paths: int = 4,
    num_channels: int = 4,
    audit_k: int = 8,
    precision_bits: int = 24,
    sde_params: dict = None,
) -> tuple[list[dict], list[dict], dict, str]:
    """Generate a complete bicep_v2 trace with audit openings.

    Returns:
        (rows, audits, manifest, manifest_hash)
    """
    if sde_params is None:
        sde_params = {
            "theta": 2.0, "mu0": 0.0, "sigma": 0.5, "dt": 0.01,
            "jump_rate": 0.0, "jump_scale": 0.0,
        }

    rng = AddressableRNG.from_hex(seed_hex)
    n_leaves = total_leaves(num_paths, num_channels)

    manifest = {
        "transition_spec_id": "bicep_v2",
        "state_view_schema_id": "bicep_state_v2",
        "transition_fn_id": "bicep_sde_em_v1",
        "output_fn_id": "bicep_features_v1",
        "seed_commitment": rng.seed_commitment,
        "x_quant_scheme_id": "fixed_point_v1",
        "x_quant_precision_bits": precision_bits,
        "sampling_scheme_id": "standard_v1",
        "state_layout_id": "flat_path_channel_v1",
        "state_num_paths": num_paths,
        "state_num_channels": num_channels,
        "sde_params": sde_params,
        "audit_k": audit_k,
    }
    manifest_hash = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()

    # Genesis state: each element initialized from RNG
    state_q = []
    for idx in range(n_leaves):
        val = rng.rand("state_init", 0, idx)
        state_q.append(quantize(val, precision_bits))

    # Genesis output_chain
    output_chain = hashlib.sha256(
        f"bicep_v2:output_chain:genesis:{rng.seed_commitment}".encode()
    ).hexdigest()

    # Build initial Merkle tree
    state_root, levels = build_merkle_tree(state_q)

    # Genesis head (for hash chain)
    head = hashlib.sha256(f"genesis:{manifest_hash}".encode()).hexdigest()

    rows = []
    audits = []

    for t in range(num_steps):
        # Outputs from RNG (quantized)
        x_t_q = [quantize(rng.rand("input", t, i), precision_bits)
                  for i in range(num_channels)]

        # RNG range addresses
        rand_addrs = [
            {"tag": "sde_noise", "t": t, "i_start": 0,
             "i_count": n_leaves, "layout_id": "flat_path_channel_v1"},
        ]
        rng_use_hash = hashlib.sha256(canonical_json_bytes(rand_addrs)).hexdigest()

        view_pre = {"state_root": state_root, "output_chain": output_chain}

        # SDE step for every leaf
        state_pre_q = state_q[:]
        levels_pre = levels

        state_post_q = []
        for idx in range(n_leaves):
            eps = rng.rand_normal("sde_noise", t, idx)
            new_val = sde_step_em(state_q[idx], sde_params, eps, precision_bits)
            state_post_q.append(new_val)

        state_q = state_post_q
        state_root_new, levels_new = build_merkle_tree(state_q)

        # Update output chain
        output_chain_new = hashlib.sha256(
            output_chain.encode() + canonical_json_bytes(x_t_q)
        ).hexdigest()

        view_post = {"state_root": state_root_new, "output_chain": output_chain_new}

        row = {
            "schema": "bicep_trace_v2",
            "t": t,
            "x_t_q": x_t_q,
            "view_pre": view_pre,
            "view_post": view_post,
            "rand_addrs": rand_addrs,
            "rng_use_hash": rng_use_hash,
        }
        rows.append(row)

        # Generate audit opening
        challenge_seed = compute_challenge_seed(
            manifest_hash, head, state_root, rng_use_hash,
        )
        audit_indices = sample_audit_indices(challenge_seed, n_leaves, audit_k)

        audit = generate_audit_opening(
            t, state_pre_q, state_post_q,
            levels_pre, levels_new,
            state_root, state_root_new,
            audit_indices, challenge_seed,
        )
        audits.append(audit)

        # Advance hash chain head
        row_digest = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        head = hashlib.sha256(f"{head}:{row_digest}".encode()).hexdigest()

        # Advance state
        state_root = state_root_new
        levels = levels_new
        output_chain = output_chain_new

    return rows, audits, manifest, manifest_hash


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


SEED = "cafebabe" * 8


def make_trace():
    """Generate a standard test trace."""
    return generate_bicep_v2_trace(SEED, num_steps=8, num_paths=4,
                                   num_channels=4, audit_k=4)


def verify_audit(rows, audits, manifest, manifest_hash, seed_hex=SEED):
    """Run full audit verification on all steps."""
    rng = AddressableRNG.from_hex(seed_hex)
    head = hashlib.sha256(f"genesis:{manifest_hash}".encode()).hexdigest()

    for t, (row, audit) in enumerate(zip(rows, audits)):
        success, msg = verify_audit_opening(
            audit, row, manifest, manifest_hash, head, rng,
        )
        if not success:
            return False, f"Step t={t}: {msg}"

        # Advance head
        row_digest = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        head = hashlib.sha256(f"{head}:{row_digest}".encode()).hexdigest()

    return True, f"All {len(rows)} steps audited successfully"


# =============================================================================
# TESTS
# =============================================================================

def test_valid_trace() -> TestResult:
    """Control: valid trace passes all audits."""
    rows, audits, manifest, mh = make_trace()
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("valid_trace", ok, msg)


def test_flip_opened_value() -> TestResult:
    """Flip one opened value_q in pre-state (fails Merkle opening)."""
    rows, audits, manifest, mh = make_trace()
    audits[3] = copy.deepcopy(audits[3])
    audits[3]["openings_pre"][0]["value_q"] += 1  # Off by 1
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("flip_opened_value", not ok, msg if not ok else "Should have failed")


def test_flip_post_value() -> TestResult:
    """Flip one post-state value (fails SDE transition or Merkle)."""
    rows, audits, manifest, mh = make_trace()
    audits[2] = copy.deepcopy(audits[2])
    audits[2]["openings_post"][0]["value_q"] += 1
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("flip_post_value", not ok, msg if not ok else "Should have failed")


def test_wrong_merkle_proof() -> TestResult:
    """Replace a Merkle proof sibling hash."""
    rows, audits, manifest, mh = make_trace()
    audits[1] = copy.deepcopy(audits[1])
    if audits[1]["openings_pre"][0]["merkle_proof"]:
        audits[1]["openings_pre"][0]["merkle_proof"][0] = ("0" * 64, "R")
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("wrong_merkle_proof", not ok, msg if not ok else "Should have failed")


def test_wrong_audit_indices() -> TestResult:
    """Use wrong audit indices (attacker cherry-picks easy leaves)."""
    rows, audits, manifest, mh = make_trace()
    audits[4] = copy.deepcopy(audits[4])
    # Shift all indices by 1
    audits[4]["audit_indices"] = [i + 1 for i in audits[4]["audit_indices"]]
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("wrong_audit_indices", not ok, msg if not ok else "Should have failed")


def test_wrong_challenge_seed() -> TestResult:
    """Forge challenge seed to control which leaves get checked."""
    rows, audits, manifest, mh = make_trace()
    audits[2] = copy.deepcopy(audits[2])
    audits[2]["challenge_seed"] = hashlib.sha256(b"attacker_controlled").hexdigest()
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("wrong_challenge_seed", not ok, msg if not ok else "Should have failed")


def test_state_root_mismatch() -> TestResult:
    """Change state_root in row but leave audit intact (openings fail)."""
    rows, audits, manifest, mh = make_trace()
    rows = copy.deepcopy(rows)
    rows[3]["view_pre"]["state_root"] = hashlib.sha256(b"wrong_root").hexdigest()
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("state_root_mismatch", not ok, msg if not ok else "Should have failed")


def test_swap_pre_post_openings() -> TestResult:
    """Swap pre and post openings (should fail transition check)."""
    rows, audits, manifest, mh = make_trace()
    audits[5] = copy.deepcopy(audits[5])
    audits[5]["openings_pre"], audits[5]["openings_post"] = \
        audits[5]["openings_post"], audits[5]["openings_pre"]
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("swap_pre_post_openings", not ok, msg if not ok else "Should have failed")


def test_valid_opening_wrong_step() -> TestResult:
    """Use audit opening from step 2 at step 4 (should fail on challenge seed)."""
    rows, audits, manifest, mh = make_trace()
    audits[4] = copy.deepcopy(audits[2])
    audits[4]["t"] = 4  # Claim it's for step 4
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("valid_opening_wrong_step", not ok, msg if not ok else "Should have failed")


def test_modify_sde_params() -> TestResult:
    """Use different SDE params than what was committed in manifest."""
    rows, audits, manifest, mh = make_trace()
    manifest = copy.deepcopy(manifest)
    manifest["sde_params"]["theta"] = 999.0  # Wrong params
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("modify_sde_params", not ok, msg if not ok else "Should have failed")


def test_output_chain_still_checked() -> TestResult:
    """Verify output_chain check still works (bicep_v1 feature)."""
    rows, audits, manifest, mh = make_trace()
    rows = copy.deepcopy(rows)
    rows[3]["x_t_q"][0] += 1  # Flip output
    # output_chain is now wrong, but we test at audit level
    # This should fail at challenge seed derivation since head changes
    ok, msg = verify_audit(rows, audits, manifest, mh)
    return TestResult("output_chain_still_checked", not ok,
                      msg if not ok else "Should have failed")


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests() -> list[TestResult]:
    tests = [
        test_valid_trace,
        test_flip_opened_value,
        test_flip_post_value,
        test_wrong_merkle_proof,
        test_wrong_audit_indices,
        test_wrong_challenge_seed,
        test_state_root_mismatch,
        test_swap_pre_post_openings,
        test_valid_opening_wrong_step,
        test_modify_sde_params,
        test_output_chain_still_checked,
    ]

    print("=" * 70)
    print("BICEP_V2 STATE AUDIT TESTS")
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
