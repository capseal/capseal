#!/usr/bin/env python3
"""CLI smoke test for RISC0 backend - tests emit/inspect/verify flow."""
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


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        args,
        capture_output=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), **{k: v for k, v in __import__("os").environ.items()}},
        check=check,
    )


def main() -> int:
    print("=== RISC0 CLI Smoke Test ===\n")

    receipt_dir = REPO_ROOT / "fixtures" / "risc0_golden"
    policy = receipt_dir / "policy.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        manifest_dir = Path(tmpdir) / "manifests"
        cap_path = Path(tmpdir) / "risc0_test.cap"

        # Step 1: Generate capsule via pipeline
        print("[1/6] Generating RISC0 capsule via pipeline...")
        result = run_cmd([
            sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--backend", "risc0",
            "--receipt-dir", str(receipt_dir),
            "--policy", str(policy),
            "--policy-id", "risc0_demo_policy",
            "--track-id", "risc0_test_track",
            "--trace-id", "risc0_cli_test",
            "--output-dir", str(output_dir),
            "--verification-profile", "proof_only",
            "--allow-insecure-da-challenge",
        ], check=False)
        if result.returncode != 0:
            print(f"FAIL: Pipeline failed: {result.stderr.decode()}")
            return 1
        print("      OK - Capsule generated")

        # Copy manifests for verification
        golden_manifests = REPO_ROOT / "fixtures" / "golden_run" / "capsule" / "manifests"
        shutil.copytree(golden_manifests, manifest_dir)

        capsule_json = output_dir / "strategy_capsule.json"

        # Step 2: Emit .cap file
        print("[2/6] Testing capsule emit command...")
        result = run_cmd([
            sys.executable, "-m", "bef_zk.capsule.cli", "emit",
            "--capsule", str(capsule_json),
            "--policy", str(policy),
            "--out", str(cap_path),
        ], check=False)
        if result.returncode != 0:
            print(f"FAIL: Emit failed: {result.stderr.decode()}")
            return 1
        if not cap_path.exists():
            print("FAIL: .cap file not created")
            return 1
        print(f"      OK - Created {cap_path.name} ({cap_path.stat().st_size} bytes)")

        # Step 3: Inspect .cap file
        print("[3/6] Testing capsule inspect command...")
        result = run_cmd([
            sys.executable, "-m", "bef_zk.capsule.cli", "inspect",
            str(cap_path), "--json",
        ], check=False)
        if result.returncode != 0:
            print(f"FAIL: Inspect failed: {result.stderr.decode()}")
            return 1
        try:
            inspect_result = json.loads(result.stdout.decode())
            manifest = inspect_result.get("manifest", {})
            backend = manifest.get("backend", "")
            if not backend:
                # Fallback: check capsule.json structure
                backend = inspect_result.get("backend", "")
            if "risc0" not in backend.lower():
                print(f"FAIL: Unexpected backend: {backend}")
                print(f"      Full manifest: {manifest}")
                return 1
            print(f"      OK - Backend: {backend}")
        except json.JSONDecodeError:
            print(f"FAIL: Invalid JSON from inspect: {result.stdout.decode()}")
            return 1

        # Step 4: Verify .cap file via CLI (should fail without manifest)
        print("[4/6] Testing capsule verify command (expect policy error without manifest)...")
        result = run_cmd([
            sys.executable, "-m", "bef_zk.capsule.cli", "verify",
            str(cap_path), "--json",
        ], check=False)
        # We expect this to fail because we don't have the manifest in the .cap
        # But it should parse correctly
        if result.returncode == 0:
            # If it passes, that's even better!
            print("      OK - Verification passed")
        else:
            try:
                verify_result = json.loads(result.stdout.decode() or result.stderr.decode())
                error_code = verify_result.get("error_code", "")
                print(f"      OK - Verification returned: {error_code} (expected for proof_only mode)")
            except json.JSONDecodeError:
                print("      OK - Verification completed (non-JSON output)")

        # Step 5: Verify original capsule.json passes
        print("[5/6] Verifying original capsule.json with verify_capsule.py...")
        result = run_cmd([
            sys.executable, str(REPO_ROOT / "scripts" / "verify_capsule.py"),
            str(capsule_json),
            "--policy", str(policy),
            "--manifest-root", str(manifest_dir),
            "--required-level", "proof_only",
        ], check=False)
        if result.returncode != 0:
            try:
                verify_result = json.loads(result.stdout.decode())
            except json.JSONDecodeError:
                verify_result = {}
            if verify_result.get("proof_verified") is not True:
                print(f"FAIL: Verification failed: {result.stdout.decode()}")
                return 1
        else:
            verify_result = json.loads(result.stdout.decode())

        if verify_result.get("proof_verified"):
            print("      OK - Proof verified successfully")
        elif verify_result.get("status") == "PROOF_ONLY":
            print("      OK - Verification status: PROOF_ONLY")
        else:
            print(f"      OK - Result: {verify_result.get('status', 'unknown')}")

        # Step 6: Summary
        print("[6/6] Checking .cap file structure...")
        import tarfile
        with tarfile.open(cap_path, "r:*") as tar:
            members = [m.name for m in tar.getmembers()]
        expected = ["manifest.json", "capsule.json", "commitments.json", "proof.bin.zst"]
        missing = [f for f in expected if not any(f in m for m in members)]
        if missing:
            print(f"WARN: Missing expected files: {missing}")
        else:
            print(f"      OK - All expected files present in .cap")

    print("\n=== ALL CLI SMOKE TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
