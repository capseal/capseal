#!/usr/bin/env python3
"""Tamper test for RISC0 backend - verifies that modified proof is rejected."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run_pipeline(receipt_dir: Path, output_dir: Path, policy: Path) -> bool:
    """Run the pipeline to generate a capsule."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--backend", "risc0",
            "--receipt-dir", str(receipt_dir),
            "--policy", str(policy),
            "--policy-id", "risc0_demo_policy",
            "--track-id", "risc0_test_track",
            "--trace-id", "risc0_tamper_test",
            "--output-dir", str(output_dir),
            "--verification-profile", "proof_only",
            "--allow-insecure-da-challenge",
        ],
        capture_output=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), **{k: v for k, v in __import__("os").environ.items()}},
    )
    return result.returncode == 0


def verify_capsule(capsule_path: Path, policy: Path, manifest_root: Path) -> tuple[int, dict]:
    """Verify a capsule and return (exit_code, result)."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_capsule.py"),
            str(capsule_path),
            "--policy", str(policy),
            "--manifest-root", str(manifest_root),
            "--required-level", "proof_only",
        ],
        capture_output=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), **{k: v for k, v in __import__("os").environ.items()}},
    )
    # Try stdout first, then stderr
    try:
        output = json.loads(result.stdout.decode())
    except json.JSONDecodeError:
        try:
            output = json.loads(result.stderr.decode())
        except json.JSONDecodeError:
            output = {"error": result.stderr.decode(), "stdout": result.stdout.decode()}
    return result.returncode, output


def tamper_proof(proof_path: Path) -> None:
    """Flip a byte in the proof's journal to create an invalid proof."""
    proof = json.loads(proof_path.read_text())
    journal = proof.get("journal", "")
    if journal:
        # Flip the first byte of the journal
        journal_bytes = bytes.fromhex(journal)
        tampered = bytes([journal_bytes[0] ^ 0xFF]) + journal_bytes[1:]
        proof["journal"] = tampered.hex()
        proof_path.write_text(json.dumps(proof, indent=2))


def main() -> int:
    print("=== RISC0 Tamper Test ===\n")

    receipt_dir = REPO_ROOT / "fixtures" / "risc0_golden"
    policy = receipt_dir / "policy.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        manifest_dir = Path(tmpdir) / "manifests"

        # Generate capsule
        print("[1/4] Generating RISC0 capsule...")
        if not run_pipeline(receipt_dir, output_dir, policy):
            print("FAIL: Pipeline failed")
            return 1
        print("      OK - Capsule generated")

        # Copy manifests
        golden_manifests = REPO_ROOT / "fixtures" / "golden_run" / "capsule" / "manifests"
        shutil.copytree(golden_manifests, manifest_dir)

        capsule_path = output_dir / "strategy_capsule.json"
        proof_path = output_dir / "adapter_proof.json"

        # Verify original capsule passes
        print("[2/4] Verifying original capsule...")
        exit_code, result = verify_capsule(capsule_path, policy, manifest_dir)
        if exit_code != 0 or result.get("proof_verified") is not True:
            print(f"FAIL: Original verification failed: {result}")
            return 1
        print("      OK - Original capsule verified")

        # Tamper with the proof
        print("[3/4] Tampering with proof...")
        tamper_proof(proof_path)
        print("      OK - Proof tampered (journal byte flipped)")

        # Verify tampered capsule is rejected
        print("[4/4] Verifying tampered capsule is rejected...")
        exit_code, result = verify_capsule(capsule_path, policy, manifest_dir)

        # Should fail with E054_PROOF_VERIFICATION_FAILED
        if exit_code == 0:
            print(f"FAIL: Tampered capsule was NOT rejected!")
            return 1

        error_code = result.get("error_code", "")
        if "E054" in error_code or "PROOF_VERIFICATION_FAILED" in str(result):
            print(f"      OK - Tampered capsule rejected with: {error_code}")
        else:
            print(f"      OK - Tampered capsule rejected (code: {error_code})")

    print("\n=== ALL TAMPER TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
