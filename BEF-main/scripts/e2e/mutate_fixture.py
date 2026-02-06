\
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import random
from typing import Any, Dict, List, Tuple

from scripts.e2e.common import dump_json, load_json, repo_root, run_verifier


def flip_bit_in_hex(hex_str: str) -> str:
    b = bytearray(bytes.fromhex(hex_str))
    if not b:
        raise ValueError("empty hex")
    i = 0
    b[i] ^= 1
    return b.hex()


def apply_mutation(obj: Dict[str, Any], mutation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutation DSL (minimal):
      { "path": "capsule.header.statement_hash", "op": "flip_bit" }

    This assumes capsule is JSON (not BIN). For BIN fixtures, use "op":"corrupt_file" instead.
    """
    path = mutation["path"]
    op = mutation["op"]
    parts = path.split(".")
    cur: Any = obj
    for p in parts[:-1]:
        if p not in cur:
            raise KeyError(f"missing path component {p} in {path}")
        cur = cur[p]
    leaf = parts[-1]
    if leaf not in cur:
        raise KeyError(f"missing leaf {leaf} in {path}")

    if op == "flip_bit":
        val = cur[leaf]
        if not isinstance(val, str):
            raise TypeError("flip_bit expects hex string field")
        cur[leaf] = flip_bit_in_hex(val)
    elif op == "set":
        cur[leaf] = mutation["value"]
    else:
        raise ValueError(f"unknown op: {op}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--plan", required=True, help="Path to expected_results.json (mutation matrix)")
    ap.add_argument("--profiles", nargs="+", default=["PROOF_ONLY", "POLICY_ENFORCED", "FULL"])
    ap.add_argument("--keep", action="store_true", help="Keep mutated capsules on disk (debug)")
    args = ap.parse_args()

    root = repo_root()
    fixture = (root / args.fixture).resolve() if not pathlib.Path(args.fixture).is_absolute() else pathlib.Path(args.fixture)
    plan_path = (root / args.plan).resolve() if not pathlib.Path(args.plan).is_absolute() else pathlib.Path(args.plan)

    plan = load_json(plan_path)

    capsule_dir = fixture / "capsule"
    capsule_files = list(capsule_dir.glob("*.json"))
    if not capsule_files:
        raise RuntimeError("mutate_fixture currently supports JSON capsules only. Add BIN corruption ops if needed.")
    capsule_path = capsule_files[0]
    capsule_obj = json.loads(capsule_path.read_text(encoding="utf-8"))
    manifest_root = capsule_dir / "manifests"
    extra_args = ["--manifest-root", str(manifest_root)] if manifest_root.exists() else []

    mutations: List[Dict[str, Any]] = plan.get("mutations", [])
    if not mutations:
        print("No mutations in plan. Add plan['mutations'] entries and re-run.")
        return 0

    ok = True
    out_mut_dir = fixture / "_mutations"
    out_mut_dir.mkdir(parents=True, exist_ok=True)

    for m in mutations:
        mid = m.get("id", "mutation")
        apply = m.get("apply")
        expect = m.get("expect", {})

        # Apply mutation on a deep copy
        mutated = copy.deepcopy(capsule_obj)
        mutated = apply_mutation(mutated, apply)

        mut_path = out_mut_dir / f"{mid}.capsule.json"
        mut_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"\n[mutation {mid}] apply={apply}")
        for profile in args.profiles:
            exp = expect.get(profile)
            res = run_verifier(mut_path, profile, extra_args=extra_args)
            print(f"  [{profile}] status={res.status} reasons={res.reason_codes}")
            if exp:
                if exp.get("status") != res.status:
                    print(f"    !! expected status={exp.get('status')} got={res.status}")
                    ok = False
                exp_codes = exp.get("reason_codes", [])
                if exp_codes:
                    missing = [c for c in exp_codes if c not in res.reason_codes]
                    if missing:
                        print(f"    !! missing reason codes: {missing}")
                        ok = False
            else:
                print("    (no expectation provided for this profile)")

        if not args.keep:
            try:
                mut_path.unlink()
            except Exception:
                pass

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
