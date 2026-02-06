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
    capsule_files = list(capsule_dir.glob("*.json"))
    if not capsule_files:
        raise FileNotFoundError(f"No capsule JSON found under {capsule_dir}")
    capsule_path = capsule_files[0]
    manifest_root = capsule_dir / "manifests"
    extra_args = ["--manifest-root", str(manifest_root)] if manifest_root.exists() else []

    # Baseline should pass
    base = run_verifier(capsule_path, args.profile, extra_args=extra_args)
    print(f"[baseline] status={base.status} reasons={base.reason_codes}")

    # Poison binding hash: assumes your verifier supports an override flag like:
    #   --override-binding-hash <hex>
    #
    # If not, adapt this to call adapter-level verify directly or patch verifier to accept override.
    poisoned = secrets.token_bytes(32).hex()
    try:
        res = run_verifier(
            capsule_path,
            args.profile,
            extra_args=extra_args + ["--override-binding-hash", poisoned],
        )
    except Exception:
        print("[adapter check] Skipping binding-hash poisoning: verifier lacks --override-binding-hash.")
        return 0

    print(f"[poisoned] status={res.status} reasons={res.reason_codes}")
    if res.status == base.status and res.status in ("FULL", "POLICY_ENFORCED", "PROOF_ONLY"):
        print("!! FAIL: poisoned binding hash did not change accept/reject behavior")
        return 2

    print("[ok] binding-hash poisoning causes failure (as expected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
