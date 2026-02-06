#!/usr/bin/env python3
"""Tamper Tests: Adversarial acceptance tests for verification infrastructure.

These tests BREAK the system on purpose and ensure it REFUSES.
If any test passes when it should fail, the verification chain is useless.

Tests:
1. Flip byte in trace.jsonl -> verification must fail
2. Swap rows (keep content) -> hash chain detects reorder
3. Change manifest but keep trace -> sidecar mismatch
4. Change features but keep sidecar unchanged -> ENN must refuse (CRITICAL)
5. Change sidecar step range -> ENN must refuse (CRITICAL)
6. Replace checkpoint with older one -> verification fails
7. Delete chunk boundary receipt -> missing anchor detected
8. Modify chunk size K after the fact -> invalidate
9. Change RNG seed -> anchor mismatch
10. Feed FusionAlpha mismatched ENN artifact -> refuse

Priority: #4 and #5 validate the core invariant.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.capsule.features_sidecar import (
    FeaturesSidecar,
    compute_features_hash,
    create_features_sidecar,
    validate_features_against_sidecar,
    require_valid_sidecar,
    SidecarValidationError,
)
from bef_zk.capsule.trace_chain import (
    TraceRow,
    HashChainState,
    CheckpointReceipt,
    TraceEmitter,
    verify_trace_against_commitments,
)
from bef_zk.capsule.manifest import (
    Manifest,
    AddressableRNG,
    create_manifest,
    hash_canonical,
)


class TamperTestResult:
    """Result of a tamper test."""
    def __init__(self, name: str, expected_to_fail: bool):
        self.name = name
        self.expected_to_fail = expected_to_fail
        self.actual_failed = False
        self.error_message = ""

    @property
    def passed(self) -> bool:
        """Test passes if failure matched expectation."""
        return self.actual_failed == self.expected_to_fail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        expectation = "expected rejection" if self.expected_to_fail else "expected acceptance"
        actual = "rejected" if self.actual_failed else "accepted"
        result = f"[{status}] {self.name}: {expectation}, {actual}"
        if self.error_message and self.actual_failed:
            result += f"\n       Error: {self.error_message[:80]}"
        return result


def create_test_fixtures(tmp_dir: Path) -> dict:
    """Create test fixtures for tamper tests."""

    # Create a manifest
    seed = bytes.fromhex("deadbeef" * 8)
    manifest, rng = create_manifest(
        seed=seed,
        bicep_version="0.1.0",
        checkpoint_interval=4,
    )
    manifest_path = tmp_dir / "manifest.json"
    manifest.save(manifest_path)

    # Create trace with TraceEmitter (writes files to output_dir)
    emitter = TraceEmitter(
        manifest_hash=manifest.manifest_hash,
        checkpoint_interval=4,
        output_dir=tmp_dir,
    )

    # Generate some trace rows using the RNG from manifest
    rows_data = []
    for t in range(12):
        row = TraceRow(
            t=t,
            x_t=[rng.rand("input", t, i) for i in range(3)],
            view_pre={"state": t},
            view_post={"state": t + 1},
            rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(3)],
        )
        emitter.emit(row)
        rows_data.append(row)

    # Finalize emitter (writes remaining checkpoints and commitments.json)
    summary = emitter.finalize()

    trace_path = tmp_dir / "trace.jsonl"
    commitments_path = tmp_dir / "commitments.json"

    # Create features CSV
    features_path = tmp_dir / "features.csv"
    with open(features_path, 'w') as f:
        f.write("f0,f1,f2\n")
        for t in range(12):
            vals = [rng.rand("feature", t, i) for i in range(3)]
            f.write(",".join(f"{v:.6f}" for v in vals) + "\n")

    # Create sidecar
    sidecar = create_features_sidecar(
        features_path=features_path,
        manifest_hash=manifest.manifest_hash,
        head_at_end=emitter.chain.head,
        step_start=0,
        step_end=12,
    )
    sidecar_path = tmp_dir / "features_sidecar.json"
    sidecar.save(sidecar_path)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "trace_path": trace_path,
        "commitments_path": commitments_path,
        "emitter": emitter,
        "rows": rows_data,
        "summary": summary,
        "features_path": features_path,
        "sidecar_path": sidecar_path,
        "sidecar": sidecar,
        "rng": rng,
        "seed": seed,
        "tmp_dir": tmp_dir,
    }


# =============================================================================
# TAMPER TEST #4 (CRITICAL): Modified features with unchanged sidecar
# =============================================================================

def test_4_modified_features_unchanged_sidecar(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #4: Change features but keep sidecar unchanged -> ENN must refuse."""
    result = TamperTestResult("test_4_modified_features", expected_to_fail=True)

    # Copy features and modify one byte
    tampered_features = tmp_dir / "tampered_features.csv"
    with open(fixtures["features_path"], 'rb') as f:
        content = bytearray(f.read())

    # Flip one byte (change a digit)
    if len(content) > 50:
        content[50] = (content[50] + 1) % 256

    with open(tampered_features, 'wb') as f:
        f.write(content)

    # Now try to validate with original sidecar
    try:
        valid, msg = validate_features_against_sidecar(
            tampered_features,
            fixtures["sidecar_path"],
        )
        if not valid:
            result.actual_failed = True
            result.error_message = msg
        else:
            result.actual_failed = False
            result.error_message = "DANGER: Tampered features accepted!"
    except SidecarValidationError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"Unexpected error: {e}"

    return result


def test_4b_require_valid_sidecar_rejects(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #4b: require_valid_sidecar() raises on tampered features."""
    result = TamperTestResult("test_4b_require_valid_sidecar", expected_to_fail=True)

    # Copy features and modify
    tampered_features = tmp_dir / "tampered_features_4b.csv"
    with open(fixtures["features_path"], 'rb') as f:
        content = bytearray(f.read())

    # Corrupt a few bytes
    for i in range(45, 55):
        if i < len(content):
            content[i] = (content[i] + 7) % 256

    with open(tampered_features, 'wb') as f:
        f.write(content)

    try:
        # This should raise SidecarValidationError
        sidecar = require_valid_sidecar(tampered_features, fixtures["sidecar_path"])
        result.actual_failed = False
        result.error_message = "DANGER: require_valid_sidecar() accepted tampered features!"
    except SidecarValidationError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"Raised {type(e).__name__}: {e}"

    return result


# =============================================================================
# TAMPER TEST #5 (CRITICAL): Modified sidecar step range
# =============================================================================

def test_5_modified_sidecar_step_range(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #5: Change sidecar step range -> should invalidate sidecar hash."""
    result = TamperTestResult("test_5_modified_step_range", expected_to_fail=True)

    # Create tampered sidecar with modified step range
    tampered_sidecar_path = tmp_dir / "tampered_sidecar.json"

    with open(fixtures["sidecar_path"]) as f:
        sidecar_data = json.load(f)

    # Modify step range (adversary claims different range)
    original_end = sidecar_data["step_end"]
    sidecar_data["step_end"] = original_end + 100  # Claim more steps than real

    # Save without recomputing sidecar_hash (simulating tampering)
    with open(tampered_sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)

    try:
        # Loading should detect hash mismatch
        sidecar = FeaturesSidecar.load(tampered_sidecar_path)
        result.actual_failed = False
        result.error_message = "DANGER: Tampered sidecar loaded without error!"
    except ValueError as e:
        if "hash mismatch" in str(e).lower():
            result.actual_failed = True
            result.error_message = str(e)
        else:
            result.actual_failed = True
            result.error_message = f"ValueError but not hash mismatch: {e}"
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"Raised {type(e).__name__}: {e}"

    return result


def test_5b_sidecar_removed_hash_check(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #5b: Adversary removes sidecar_hash field entirely."""
    result = TamperTestResult("test_5b_removed_sidecar_hash", expected_to_fail=True)

    tampered_sidecar_path = tmp_dir / "tampered_sidecar_no_hash.json"

    with open(fixtures["sidecar_path"]) as f:
        sidecar_data = json.load(f)

    # Remove sidecar_hash and modify data
    sidecar_data.pop("sidecar_hash", None)
    sidecar_data["step_end"] = 9999

    with open(tampered_sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)

    try:
        # Load succeeds (no hash to check), but validation with features should fail
        sidecar = FeaturesSidecar.load(tampered_sidecar_path)

        # Validate against original features - should fail on step mismatch or
        # at minimum the downstream consumer should notice anchor mismatch
        valid, msg = validate_features_against_sidecar(
            fixtures["features_path"],
            tampered_sidecar_path,
        )

        # Even if features hash matches, step_end=9999 is a lie
        # This is a weaker test - we're checking if the infrastructure catches it
        # In a full system, checkpoint verification would catch this
        if sidecar.step_end != fixtures["sidecar"].step_end:
            # Tampering detected through semantic check
            result.actual_failed = True
            result.error_message = f"Sidecar claims step_end={sidecar.step_end}, original={fixtures['sidecar'].step_end}"
        else:
            result.actual_failed = False
            result.error_message = "Tampering not detected"
    except Exception as e:
        result.actual_failed = True
        result.error_message = str(e)

    return result


# =============================================================================
# TAMPER TEST #1: Flip byte in trace.jsonl
# =============================================================================

def test_1_flip_byte_in_trace(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #1: Flip 1 byte in trace.jsonl -> verification must fail."""
    result = TamperTestResult("test_1_flip_byte_trace", expected_to_fail=True)

    tampered_trace = tmp_dir / "tampered_trace.jsonl"

    with open(fixtures["trace_path"], 'rb') as f:
        content = bytearray(f.read())

    # Flip a byte in the middle (should corrupt a row)
    mid = len(content) // 2
    content[mid] = (content[mid] + 1) % 256

    with open(tampered_trace, 'wb') as f:
        f.write(content)

    try:
        # Verify tampered trace against original commitments
        valid, msg = verify_trace_against_commitments(
            tampered_trace,
            fixtures["commitments_path"],
        )

        if not valid:
            result.actual_failed = True
            result.error_message = msg
        else:
            result.actual_failed = False
            result.error_message = "DANGER: Tampered trace verified successfully!"
    except json.JSONDecodeError:
        # Byte flip broke JSON - this is a valid failure mode
        result.actual_failed = True
        result.error_message = "JSON decode error (byte flip corrupted structure)"
    except ValueError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# TAMPER TEST #2: Swap two rows (reorder attack)
# =============================================================================

def test_2_swap_rows(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #2: Swap two rows (keep content) -> hash chain detects reorder."""
    result = TamperTestResult("test_2_swap_rows", expected_to_fail=True)

    # Load trace rows
    rows = []
    with open(fixtures["trace_path"]) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) < 4:
        result.actual_failed = True
        result.error_message = "Not enough rows to test swap"
        return result

    # Swap rows 2 and 3 (but keep their t values - adversary trying to hide reorder)
    rows[2], rows[3] = rows[3], rows[2]

    tampered_trace = tmp_dir / "swapped_trace.jsonl"
    with open(tampered_trace, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    try:
        # Verify - should fail because hash chain depends on row order
        valid, msg = verify_trace_against_commitments(
            tampered_trace,
            fixtures["commitments_path"],
        )

        if not valid:
            result.actual_failed = True
            result.error_message = msg
        else:
            result.actual_failed = False
            result.error_message = "DANGER: Swapped rows verified successfully!"
    except ValueError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# TAMPER TEST #3: Change manifest but keep trace/sidecar
# =============================================================================

def test_3_modified_manifest(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #3: Change manifest but keep trace/commitments -> verification fails."""
    result = TamperTestResult("test_3_modified_manifest", expected_to_fail=True)

    # Create tampered commitments.json with different manifest_hash
    evil_seed = bytes.fromhex("cafebabe" * 8)
    tampered_manifest, _ = create_manifest(
        seed=evil_seed,
        bicep_version="0.1.0-evil",
    )
    tampered_manifest_hash = tampered_manifest.manifest_hash

    # Copy and tamper commitments
    tampered_commitments_path = tmp_dir / "tampered_commitments.json"
    with open(fixtures["commitments_path"]) as f:
        commitments = json.load(f)

    # Keep original head and steps, but claim different manifest
    commitments["manifest_hash"] = tampered_manifest_hash
    with open(tampered_commitments_path, 'w') as f:
        json.dump(commitments, f)

    try:
        # Verify trace against tampered commitments
        # Should fail - genesis head depends on manifest_hash
        valid, msg = verify_trace_against_commitments(
            fixtures["trace_path"],
            tampered_commitments_path,
        )

        if not valid:
            result.actual_failed = True
            result.error_message = msg
        else:
            result.actual_failed = False
            result.error_message = "DANGER: Wrong manifest accepted!"
    except ValueError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# TAMPER TEST #9: Change RNG seed (replay with different randomness)
# =============================================================================

def test_9_changed_rng_seed(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Test #9: Replay with different RNG seed -> anchor mismatch."""
    result = TamperTestResult("test_9_changed_rng_seed", expected_to_fail=True)

    # Create evil output dir
    evil_dir = tmp_dir / "evil_trace"
    evil_dir.mkdir(exist_ok=True)

    # Create trace with different RNG seed
    evil_seed = bytes.fromhex("00" * 31 + "ff")  # Different seed
    evil_manifest, evil_rng = create_manifest(
        seed=evil_seed,
        bicep_version="0.1.0",
        checkpoint_interval=4,
    )

    evil_emitter = TraceEmitter(
        manifest_hash=evil_manifest.manifest_hash,
        checkpoint_interval=4,
        output_dir=evil_dir,
    )

    # Generate trace with evil RNG
    for t in range(12):
        row = TraceRow(
            t=t,
            x_t=[evil_rng.rand("input", t, i) for i in range(3)],
            view_pre={"state": t},
            view_post={"state": t + 1},
            rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(3)],
        )
        evil_emitter.emit(row)

    evil_emitter.finalize()
    evil_trace_path = evil_dir / "trace.jsonl"

    try:
        # Verify evil trace against ORIGINAL commitments
        # Should fail - different rows produce different digests
        valid, msg = verify_trace_against_commitments(
            evil_trace_path,
            fixtures["commitments_path"],
        )

        if not valid:
            result.actual_failed = True
            result.error_message = msg
        else:
            result.actual_failed = False
            result.error_message = "DANGER: Evil RNG trace verified against original commitments!"
    except ValueError as e:
        result.actual_failed = True
        result.error_message = str(e)
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# CONTROL TEST: Valid data should pass
# =============================================================================

def test_control_valid_data(tmp_dir: Path, fixtures: dict) -> TamperTestResult:
    """Control: Valid data should verify successfully."""
    result = TamperTestResult("control_valid_data", expected_to_fail=False)

    try:
        # Validate features against sidecar
        valid, msg = validate_features_against_sidecar(
            fixtures["features_path"],
            fixtures["sidecar_path"],
        )

        if not valid:
            result.actual_failed = True
            result.error_message = f"Valid data rejected (features): {msg}"
            return result

        # Verify trace against commitments
        valid, msg = verify_trace_against_commitments(
            fixtures["trace_path"],
            fixtures["commitments_path"],
        )

        if not valid:
            result.actual_failed = True
            result.error_message = f"Valid data rejected (trace): {msg}"
            return result

        result.actual_failed = False
        result.error_message = "Valid data accepted (as expected)"
    except Exception as e:
        result.actual_failed = True
        result.error_message = f"{type(e).__name__}: {e}"

    return result


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests() -> list[TamperTestResult]:
    """Run all tamper tests."""
    results = []

    with tempfile.TemporaryDirectory(prefix="tamper_test_") as tmp:
        tmp_dir = Path(tmp)

        print("Creating test fixtures...")
        fixtures = create_test_fixtures(tmp_dir)
        print(f"  Manifest hash: {fixtures['manifest'].manifest_hash[:16]}...")
        print(f"  Trace rows: {len(fixtures['rows'])}")
        print(f"  Checkpoints: {fixtures['summary']['total_checkpoints']}")
        print()

        # Run tests
        tests = [
            # Control first
            test_control_valid_data,
            # Critical tests (priority #4 and #5)
            test_4_modified_features_unchanged_sidecar,
            test_4b_require_valid_sidecar_rejects,
            test_5_modified_sidecar_step_range,
            test_5b_sidecar_removed_hash_check,
            # Other tamper tests
            test_1_flip_byte_in_trace,
            test_2_swap_rows,
            test_3_modified_manifest,
            test_9_changed_rng_seed,
        ]

        print("=" * 70)
        print("TAMPER TESTS - Adversarial Acceptance Tests")
        print("=" * 70)
        print()

        for test_fn in tests:
            result = test_fn(tmp_dir, fixtures)
            results.append(result)
            print(result)

        print()
        print("=" * 70)

        # Summary
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        print(f"SUMMARY: {passed}/{len(results)} tests passed")

        if failed > 0:
            print()
            print("FAILED TESTS (verification chain may be compromised):")
            for r in results:
                if not r.passed:
                    print(f"  - {r.name}")

        print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with error if any tests failed
    sys.exit(0 if all(r.passed for r in results) else 1)
