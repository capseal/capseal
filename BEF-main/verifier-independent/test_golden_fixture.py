#!/usr/bin/env python3
"""
Golden fixture test for bicep_v2.

Regenerates the fixture from scratch and asserts byte-for-byte equality
with the checked-in golden run. This catches spec drift even if both
executor and generator drift together.

Run: python verifier-independent/test_golden_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bef_zk.capsule.bicep_v2_executor import emit_bicep_v2_run
from verifier import verify_trace, verify_trace_correctness

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden_bicep_v2"
SEED = "cafebabe" * 8
PARAMS = dict(seed_hex=SEED, num_steps=4, num_paths=2, num_channels=2, audit_k=2)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_golden_regeneration() -> bool:
    """Regenerate golden fixture and compare byte-for-byte."""
    tmp = Path(tempfile.mkdtemp(prefix="golden_regen_"))
    emit_bicep_v2_run(tmp, **PARAMS)

    files_to_check = [
        "manifest.json",
        "commitments.json",
        "trace.jsonl",
    ]
    # Add audit opening files
    for t in range(PARAMS["num_steps"]):
        files_to_check.append(f"audit_openings/audit_step_{t:04d}.json")

    all_ok = True
    for relpath in files_to_check:
        golden = GOLDEN_DIR / relpath
        regen = tmp / relpath

        if not golden.exists():
            print(f"  FAIL: golden file missing: {relpath}")
            all_ok = False
            continue
        if not regen.exists():
            print(f"  FAIL: regenerated file missing: {relpath}")
            all_ok = False
            continue

        gh = file_hash(golden)
        rh = file_hash(regen)
        if gh != rh:
            print(f"  FAIL: {relpath} hash mismatch")
            print(f"    golden: {gh[:32]}...")
            print(f"    regen:  {rh[:32]}...")
            all_ok = False
        else:
            print(f"  OK: {relpath} ({gh[:16]}...)")

    return all_ok


def test_golden_verifies() -> bool:
    """Golden fixture passes full verifier pipeline."""
    manifest_path = GOLDEN_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    ok0, msg0 = verify_trace(GOLDEN_DIR / "trace.jsonl", GOLDEN_DIR / "commitments.json")
    print(f"  Level 0: {'PASS' if ok0 else 'FAIL'}: {msg0}")

    ok1, msg1 = verify_trace_correctness(
        GOLDEN_DIR / "trace.jsonl", SEED,
        transition_spec_id="bicep_v2",
        manifest=manifest,
        openings_path=GOLDEN_DIR / "audit_openings",
    )
    print(f"  Level 1+1.5: {'PASS' if ok1 else 'FAIL'}: {msg1}")

    return ok0 and ok1


def main():
    print("=" * 70)
    print("GOLDEN FIXTURE TESTS (bicep_v2)")
    print("=" * 70)

    print("\n[1] Regeneration byte-for-byte equality:")
    ok1 = test_golden_regeneration()

    print("\n[2] Golden fixture verification:")
    ok2 = test_golden_verifies()

    print("\n" + "=" * 70)
    all_ok = ok1 and ok2
    print(f"RESULT: {'ALL PASSED' if all_ok else 'FAILED'}")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
