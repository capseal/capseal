\
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Any, List

from scripts.e2e.common import load_json, repo_root, run_verifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="Fixture dir, e.g. fixtures/golden_run")
    ap.add_argument("--profiles", nargs="+", default=["PROOF_ONLY", "POLICY_ENFORCED", "FULL"])
    ap.add_argument("--capsule", default=None, help="Override capsule path (relative to fixture/capsule/)")
    ap.add_argument("--extra-verifier-args", nargs="*", default=[])
    args = ap.parse_args()

    root = repo_root()
    fixture = (root / args.fixture).resolve() if not pathlib.Path(args.fixture).is_absolute() else pathlib.Path(args.fixture)
    plan_path = fixture / "expected_results.json"
    plan = load_json(plan_path)

    capsule_dir = fixture / "capsule"
    if args.capsule:
        capsule_path = capsule_dir / args.capsule
    else:
        # Use recorded capsule_file if present; else pick first file
        if "capsule_file" in plan:
            capsule_path = root / plan["capsule_file"]
        else:
            files = list(capsule_dir.glob("*"))
            if not files:
                raise FileNotFoundError(f"No capsule file under {capsule_dir}")
            capsule_path = files[0]

    ok = True
    for profile in args.profiles:
        expected = plan.get("profiles", {}).get(profile)
        res = run_verifier(capsule_path, profile, extra_args=args.extra_verifier_args)
        print(f"[{profile}] status={res.status} reasons={res.reason_codes}")
        if expected:
            if expected.get("status") != res.status:
                print(f"  !! expected status={expected.get('status')} got={res.status}")
                ok = False
            exp_codes = expected.get("reason_codes", [])
            # If expected has reason codes, require them to be subset (order-insensitive)
            if exp_codes:
                missing = [c for c in exp_codes if c not in res.reason_codes]
                if missing:
                    print(f"  !! expected reason codes missing: {missing}")
                    ok = False
        else:
            print("  (no expected baseline for this profile; edit expected_results.json)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
