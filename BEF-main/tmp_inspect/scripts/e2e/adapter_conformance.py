\
from __future__ import annotations

import argparse
import os
import pathlib
import secrets
from typing import List

from scripts.e2e.common import repo_root, run_verifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--backends", nargs="+", default=["geom", "risc0"])
    ap.add_argument("--profile", default="PROOF_ONLY")
    args = ap.parse_args()

    root = repo_root()
    fixture = (root / args.fixture).resolve() if not pathlib.Path(args.fixture).is_absolute() else pathlib.Path(args.fixture)
    capsule_dir = fixture / "capsule"
    capsule_files = list(capsule_dir.glob("*"))
    if not capsule_files:
        raise FileNotFoundError(f"No capsule found under {capsule_dir}")
    capsule_path = capsule_files[0]

    # Baseline should pass
    base = run_verifier(capsule_path, args.profile)
    print(f"[baseline] status={base.status} reasons={base.reason_codes}")

    # Poison binding hash: assumes your verifier supports an override flag like:
    #   --override-binding-hash <hex>
    #
    # If not, adapt this to call adapter-level verify directly or patch verifier to accept override.
    poisoned = secrets.token_bytes(32).hex()
    try:
        res = run_verifier(capsule_path, args.profile, extra_args=["--override-binding-hash", poisoned])
    except Exception as e:
        print("Your verifier likely doesn't support --override-binding-hash yet.")
        print("Add a debug-only flag that replaces the recomputed binding_hash with user-provided bytes,")
        print("then re-run this test.")
        raise

    print(f"[poisoned] status={res.status} reasons={res.reason_codes}")
    if res.status == base.status and res.status in ("FULL", "POLICY_ENFORCED", "PROOF_ONLY"):
        print("!! FAIL: poisoned binding hash did not change accept/reject behavior")
        return 2

    print("[ok] binding-hash poisoning causes failure (as expected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
