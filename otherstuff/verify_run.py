#!/usr/bin/env python3
"""Verify receipts for agent evaluation runs.

This script validates the cryptographic receipt chain for completed eval runs.
It checks:
1. Each round's artifacts hash to their stored values
2. Each round's statement_hash is correctly computed
3. The run's chain_hash correctly chains all round statement_hashes

Usage:
    python verify_run.py <run_dir>
    python verify_run.py artifacts/agent_test
    python verify_run.py --round artifacts/agent_test/rounds/R0001_20250131_123456

Exit codes:
    0: All receipts verified successfully
    1: Verification failed (mismatches found)
    2: No receipts found or other error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "BEF-main"))

from bef_zk.shared.receipts import (
    verify_round_receipt,
    verify_run_receipt,
    collect_round_dirs,
)


def verify_single_round(round_dir: Path, verbose: bool = False) -> bool:
    """Verify a single round's receipt.

    Returns True if verified, False otherwise.
    """
    print(f"\nVerifying round: {round_dir.name}")

    result = verify_round_receipt(round_dir)

    if result["verified"]:
        print(f"  [PASS] {result['artifacts_checked']} artifacts verified")
        return True
    else:
        print(f"  [FAIL] Verification failed:")
        for m in result["mismatches"]:
            print(f"    - {m}")
        return False


def verify_full_run(run_dir: Path, verbose: bool = False) -> bool:
    """Verify an entire run's receipt chain.

    Returns True if all verified, False otherwise.
    """
    run_dir = Path(run_dir)
    print(f"Verifying run: {run_dir}")

    # Check for run_receipt.json
    run_receipt_path = run_dir / "run_receipt.json"
    if not run_receipt_path.exists():
        print(f"  [ERROR] run_receipt.json not found")
        print(f"  Hint: Run with --receipts flag to generate receipts")
        return False

    with open(run_receipt_path) as f:
        run_receipt = json.load(f)

    print(f"  Run ID: {run_receipt.get('run_id', 'unknown')}")
    print(f"  Rounds: {run_receipt.get('total_rounds', 0)}")
    print(f"  Chain hash: {run_receipt.get('chain_hash', 'none')[:16]}...")

    # Verify run receipt (checks chain_hash)
    run_result = verify_run_receipt(run_dir)

    if not run_result["verified"]:
        print(f"\n  [FAIL] Run receipt verification failed:")
        for m in run_result["mismatches"]:
            print(f"    - {m}")

    # Verify each round
    round_dirs = collect_round_dirs(run_dir)
    all_passed = True
    rounds_checked = 0

    for round_dir in round_dirs:
        if (round_dir / "round_receipt.json").exists():
            if not verify_single_round(round_dir, verbose):
                all_passed = False
            rounds_checked += 1

    print(f"\n{'='*60}")
    if all_passed and run_result["verified"]:
        print(f"[VERIFIED] All {rounds_checked} rounds and run receipt verified")
        return True
    else:
        print(f"[FAILED] Verification failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify agent evaluation run receipts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify entire run
    python verify_run.py artifacts/agent_test

    # Verify single round
    python verify_run.py --round artifacts/agent_test/rounds/R0001_20250131_123456

    # Verbose output
    python verify_run.py -v artifacts/agent_test
""",
    )

    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to run directory (or round directory with --round)",
    )
    parser.add_argument(
        "--round",
        action="store_true",
        help="Verify a single round directory instead of full run",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(2)

    if args.round:
        # Verify single round
        success = verify_single_round(run_dir, args.verbose)
    else:
        # Verify full run
        success = verify_full_run(run_dir, args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
