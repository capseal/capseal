#!/usr/bin/env python3
"""Run a CapsuleBench demo, verify, then tamper and show REJECT.

Usage:
  PYTHONPATH=. python scripts/demo_run_and_verify.py \
    --backend geom \
    --policy policies/demo_policy_v1.json \
    --track-id demo_geom_fast \
    --manifest-signer-id test_manifest \
    --manifest-signer-key 3333333333333333333333333333333333333333333333333333333333333333

If the policy path is omitted or missing, the script writes a permissive demo
policy to a temp file. Results include key hashes and PASS/REJECT codes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _exists(p: str | Path) -> bool:
    return Path(p).expanduser().exists()


def _hash_capsule(capsule: dict[str, Any]) -> Tuple[str | None, str | None, str | None, str | None]:
    return (
        capsule.get("capsule_hash"),
        capsule.get("payload_hash"),
        capsule.get("header_hash") or capsule.get("header_commit_hash"),
        capsule.get("statement_hash"),
    )


def _write_demo_policy(path: Path, track_id: str) -> None:
    doc = {
        "schema": "bef_benchmark_policy_v1",
        "policy_id": "demo_policy_v1",
        "policy_version": "1.0",
        "tracks": [
            {
                "track_id": track_id,
                "description": "Demo track (GPU allowed, no deterministic build)",
                "rules": {
                    "forbid_gpu": False,
                    "require_deterministic_build": False,
                    "required_public_outputs": ["final_cnt"],
                },
            }
        ],
    }
    path.write_text(json.dumps(doc, indent=2))


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def run_capsule_bench(
    *,
    backend: str,
    policy_path: Path,
    policy_id: str,
    track_id: str,
    manifest_signer_id: str,
    manifest_signer_key: str,
) -> Path:
    run_cmd = [
        sys.executable,
        "-m",
        "capsule_bench.cli",
        "run",
        "--backend",
        backend,
        "--policy",
        str(policy_path),
        "--policy-id",
        policy_id,
        "--policy-version",
        "1.0",
        "--track-id",
        track_id,
        "--private-key",
        "demo_assets/demo_private_key.hex",
        "--verification-profile",
        "policy_enforced",
        "--manifest-signer-id",
        manifest_signer_id,
        "--manifest-signer-key",
        manifest_signer_key,
    ]
    subprocess.run(run_cmd, check=True, cwd=str(ROOT))
    # Find newest run dir
    out_root = ROOT / "out" / "capsule_runs"
    runs = sorted([p for p in out_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise RuntimeError("no capsule-bench run produced under out/capsule_runs/")
    run_dir = runs[-1]
    return run_dir


def verify_capsule(capsule_path: Path, *, policy_path: Path, manifest_root: Path, required_level: str = "policy_self_reported") -> dict:
    cmd = [
        sys.executable,
        "-m",
        "scripts.verify_capsule",
        str(capsule_path),
        "--policy",
        str(policy_path),
        "--manifest-root",
        str(manifest_root),
        "--required-level",
        required_level,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "verification failed")
    return json.loads(proc.stdout)


def tamper_capsule(src: Path) -> Path:
    dst = src.parent / (src.stem + ".tampered.json")
    capsule = json.loads(src.read_text())
    st = capsule.get("statement") or {}
    # Try to bump final_cnt if present, else flip policy track
    pis = st.get("public_inputs") or []
    changed = False
    for pi in pis:
        if pi.get("name") == "final_cnt":
            try:
                pi["value"] = int(pi.get("value") or 0) + 1
                changed = True
                break
            except Exception:
                continue
    if not changed:
        pol = capsule.get("policy") or {}
        pol["track_id"] = (pol.get("track_id") or "demo_geom_fast") + "_tamper"
        capsule["policy"] = pol
    # Note: do not recompute hashes; this is intentional to trigger REJECT
    dst.write_text(json.dumps(capsule, indent=2))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Run + Verify + Tamper demo for Capsules")
    ap.add_argument("--backend", default="geom")
    ap.add_argument("--policy", type=Path, help="Policy file path (if missing, a demo policy is generated)")
    ap.add_argument("--policy-id", default="demo_policy_v1")
    ap.add_argument("--track-id", default="demo_geom_fast")
    ap.add_argument("--manifest-signer-id", default="test_manifest")
    ap.add_argument("--manifest-signer-key", default="".join(["3"] * 64))
    args = ap.parse_args()

    # Policy selection / synthesis
    if not args.policy or not _exists(args.policy):
        tmp_dir = Path(tempfile.mkdtemp(prefix="capsule_demo_"))
        policy_path = tmp_dir / "demo_policy.json"
        _write_demo_policy(policy_path, args.track_id)
    else:
        policy_path = args.policy

    print("[1/4] Running capsule-bench run…")
    run_dir = run_capsule_bench(
        backend=args.backend,
        policy_path=policy_path,
        policy_id=args.policy_id,
        track_id=args.track_id,
        manifest_signer_id=args.manifest_signer_id,
        manifest_signer_key=args.manifest_signer_key,
    )
    pipeline_dir = run_dir / "pipeline"
    capsule_path = pipeline_dir / "strategy_capsule.json"
    manifests = run_dir / "manifests"
    # Ensure events log is present under pipeline base so verifier's confined
    # resolver can find it via rel_path (events/events.jsonl)
    try:
        src_events = run_dir / "events.jsonl"
        if src_events.exists():
            dst_events_dir = pipeline_dir / "events"
            dst_events_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_events, dst_events_dir / src_events.name)
    except Exception:
        pass
    capsule = json.loads(capsule_path.read_text())
    cap_hash, pay_hash, head_hash, stmt_hash = _hash_capsule(capsule)
    print("  run_dir:", run_dir)
    print("  capsule_path:", capsule_path)
    print("  capsule_hash:", cap_hash)
    print("  payload_hash:", pay_hash)
    print("  header_hash:", head_hash)
    print("  statement_hash:", stmt_hash)

    print("[2/4] Verifying capsule (expected PASS)…")
    result = verify_capsule(capsule_path, policy_path=policy_path, manifest_root=manifests)
    print(_pretty(result))

    print("[3/4] Tampering capsule and verifying (expected REJECT)…")
    tampered = tamper_capsule(capsule_path)
    try:
        _ = verify_capsule(tampered, policy_path=policy_path, manifest_root=manifests)
        print("ERROR: tampered capsule unexpectedly verified")
    except RuntimeError as exc:
        print("Tamper verification returned:", str(exc))

    print("[4/4] Done.")


if __name__ == "__main__":
    main()
