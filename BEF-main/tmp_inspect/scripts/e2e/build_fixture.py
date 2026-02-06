\
from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, Any, List, Optional

from scripts.e2e.common import dump_json, ensure_dir, repo_root, run_verifier


def find_capsule_file(run_dir: pathlib.Path) -> pathlib.Path:
    # Heuristic: prefer pipeline/strategy_capsule.{json,bin} if present; else first *capsule*.{json,bin}
    pipeline = run_dir / "pipeline"
    candidates: List[pathlib.Path] = []
    if pipeline.exists():
        for p in [pipeline / "strategy_capsule.json", pipeline / "strategy_capsule.bin"]:
            if p.exists():
                return p
        candidates.extend(sorted(pipeline.glob("*capsule*.json")))
        candidates.extend(sorted(pipeline.glob("*capsule*.bin")))
    candidates.extend(sorted(run_dir.glob("**/*capsule*.json")))
    candidates.extend(sorted(run_dir.glob("**/*capsule*.bin")))
    if not candidates:
        raise FileNotFoundError(f"No capsule file found under {run_dir}")
    return candidates[0]


def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src.exists():
        return False
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return True
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.rglob("*"):
        rel = child.relative_to(src)
        out = dst / rel
        if child.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(child.read_bytes())
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to out/capsule_runs/<run_id>/")
    ap.add_argument("--out", required=True, help="Fixture output dir, e.g. fixtures/golden_run")
    ap.add_argument("--profiles", nargs="*", default=["PROOF_ONLY", "POLICY_ENFORCED", "FULL"])
    args = ap.parse_args()

    root = repo_root()
    run_dir = (root / args.run_dir).resolve() if not pathlib.Path(args.run_dir).is_absolute() else pathlib.Path(args.run_dir)
    out_dir = (root / args.out).resolve() if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)

    ensure_dir(out_dir)

    capsule_src = find_capsule_file(run_dir)
    capsule_dst_dir = out_dir / "capsule"
    ensure_dir(capsule_dst_dir)
    capsule_dst = capsule_dst_dir / capsule_src.name
    capsule_dst.write_bytes(capsule_src.read_bytes())

    artifacts_dir = out_dir / "artifacts"
    ensure_dir(artifacts_dir)

    # Copy common subdirs if present
    copied = {}
    copied["row_archive"] = copy_tree(run_dir / "row_archive", artifacts_dir / "row_archive")
    copied["manifests"] = copy_tree(run_dir / "manifests", artifacts_dir / "manifests") or copy_tree(run_dir / "manifests", artifacts_dir / "manifests")
    copied["da"] = copy_tree(run_dir / "da", artifacts_dir / "da")
    copied["registries"] = copy_tree(run_dir / "registries", artifacts_dir / "registries")

    # Proof artifacts: attempt pipeline/proof* or pipeline/*.proof*
    proof_cands = list((run_dir / "pipeline").glob("proof*")) if (run_dir / "pipeline").exists() else []
    if proof_cands:
        copied["proof"] = copy_tree(proof_cands[0], artifacts_dir / proof_cands[0].name)
    else:
        copied["proof"] = False

    expected: Dict[str, Any] = {
        "fixture_version": 1,
        "run_dir": str(run_dir),
        "capsule_file": str(capsule_dst.relative_to(root)),
        "copied": copied,
        "profiles": {},
        "mutations": [],
    }

    # Try to run verifier and capture hashes/statuses
    for profile in args.profiles:
        try:
            res = run_verifier(capsule_dst, profile)
            expected["profiles"][profile] = {"status": res.status, "reason_codes": res.reason_codes}
            # Write hash fields if verifier provides them
            for k in ["capsule_hash", "payload_hash", "header_commit_hash", "instance_hash"]:
                if k in res.raw:
                    (capsule_dst_dir / f"{k}.txt").write_text(str(res.raw[k]) + "\n", encoding="utf-8")
                    expected[k] = res.raw[k]
        except Exception as e:
            expected["profiles"][profile] = {"status": "ERROR", "reason_codes": ["HARNESS_ERROR"], "error": str(e)}

    dump_json(out_dir / "expected_results.json", expected)
    print(f"[ok] fixture written to {out_dir}")
    print(f"[ok] edit {out_dir/'expected_results.json'} to add mutation expectations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
