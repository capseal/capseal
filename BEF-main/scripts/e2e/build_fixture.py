\
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
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


def _resolve_source(
    entry: Any,
    *,
    base: pathlib.Path,
    fallback: Optional[List[pathlib.Path]] = None,
) -> Optional[pathlib.Path]:
    candidates: List[pathlib.Path] = []
    if isinstance(entry, dict):
        for key in ("abs_path", "path"):
            raw = entry.get(key)
            if raw:
                path = pathlib.Path(raw)
                if not path.is_absolute():
                    path = base / raw
                candidates.append(path)
    elif isinstance(entry, str):
        path = pathlib.Path(entry)
        if not path.is_absolute():
            path = base / entry
        candidates.append(path)
    if fallback:
        candidates.extend(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_rel(rel: str) -> pathlib.Path:
    rel = rel.lstrip("/\\")
    return pathlib.Path(rel)


def _copy_file(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_any(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src or not src.exists():
        return False
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    return _copy_file(src, dst)


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
    try:
        capsule_data = json.loads(capsule_dst.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid capsule
        raise RuntimeError(f"unable to parse capsule JSON: {capsule_dst}") from exc

    pipeline_dir = run_dir / "pipeline"

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

    # --- Build confined capsule tree so verifier can resolve artifacts ---
    artifacts_info = capsule_data.get("artifacts") or {}

    # Manifests bundle
    manif_src = run_dir / "manifests"
    manif_dst = capsule_dst_dir / "manifests"
    if manif_src.exists():
        copy_tree(manif_src, manif_dst)

    # Policy file
    policy_path = capsule_data.get("policy_path") or (run_dir / "policy.json")
    if policy_path:
        policy_src = pathlib.Path(policy_path) if isinstance(policy_path, str) else policy_path
        if not policy_src.is_absolute():
            policy_src = run_dir / policy_src
        _copy_file(policy_src, capsule_dst_dir / "policy.json")

    # Artifact manifest
    manifest_entry = artifacts_info.get("manifest")
    manifest_dest_rel = (manifest_entry or {}).get("rel_path") or "artifact_manifest.json"
    manifest_src = _resolve_source(
        manifest_entry,
        base=pipeline_dir,
        fallback=[pipeline_dir / "artifact_manifest.json"],
    )
    if manifest_src:
        canonical_manifest = capsule_dst_dir / "artifact_manifest.json"
        _copy_file(manifest_src, canonical_manifest)
        rel_path_norm = _normalize_rel(manifest_dest_rel)
        if rel_path_norm != pathlib.Path("artifact_manifest.json"):
            _copy_file(manifest_src, capsule_dst_dir / rel_path_norm)

    # Events log
    events_entry = artifacts_info.get("events_log")
    events_dest_rel = (events_entry or {}).get("rel_path") or "events/events.jsonl"
    events_src = _resolve_source(
        events_entry,
        base=run_dir,
        fallback=[run_dir / "events.jsonl", pipeline_dir / "events" / "events.jsonl"],
    )
    if events_src:
        _copy_file(events_src, capsule_dst_dir / _normalize_rel(events_dest_rel))

    # Proof artifacts (per rel_path)
    for proof_name, entry in (capsule_data.get("proofs") or {}).items():
        rel_path = entry.get("rel_path") or f"proofs/{proof_name}/{pathlib.Path(entry.get('path', '')).name}"
        fallback: List[pathlib.Path] = []
        for key in ("path", "json_path", "bin_path"):
            raw = entry.get(key)
            if raw:
                path = pathlib.Path(raw)
                if not path.is_absolute():
                    path = pipeline_dir / raw
                fallback.append(path)
        proof_art_entry = artifacts_info.get("proof") or {}
        raw_default = proof_art_entry.get("path")
        if raw_default:
            path = pathlib.Path(raw_default)
            if not path.is_absolute():
                path = pipeline_dir / raw_default
            fallback.append(path)
        src = _resolve_source(entry, base=pipeline_dir, fallback=fallback)
        if src:
            _copy_any(src, capsule_dst_dir / _normalize_rel(rel_path))

    # Row archive tree (respect rel_path if present)
    row_entry = capsule_data.get("row_archive") or artifacts_info.get("row_archive")
    if row_entry:
        dest_rel = row_entry.get("rel_path") or "row_archive"
        row_src = _resolve_source(
            row_entry,
            base=pipeline_dir,
            fallback=[pipeline_dir / "row_archive"],
        )
        if row_src:
            _copy_any(row_src, capsule_dst_dir / _normalize_rel(dest_rel))

    expected: Dict[str, Any] = {
        "fixture_version": 1,
        "run_dir": str(run_dir),
        "capsule_file": str(capsule_dst.relative_to(root)),
        "copied": copied,
        "profiles": {},
        "mutations": [],
    }

    # Default verifier args (policy + manifests if present)
    default_args: List[str] = []
    policy_file = capsule_dst_dir / "policy.json"
    if policy_file.exists():
        default_args += ["--policy", str(policy_file)]
    manifest_root = capsule_dst_dir / "manifests"
    if manifest_root.exists():
        default_args += ["--manifest-root", str(manifest_root)]

    # Try to run verifier and capture hashes/statuses
    for profile in args.profiles:
        try:
            res = run_verifier(capsule_dst, profile, extra_args=default_args)
            expected["profiles"][profile] = {"status": res.status, "reason_codes": res.reason_codes}
            # Write hash fields if verifier provides them
            for k in ["capsule_hash", "payload_hash", "header_commit_hash", "instance_hash"]:
                if k in res.raw:
                    (capsule_dst_dir / f"{k}.txt").write_text(str(res.raw[k]) + "\n", encoding="utf-8")
                    expected[k] = res.raw[k]
        except Exception as e:
            expected["profiles"][profile] = {
                "status": "ERROR",
                "reason_codes": ["HARNESS_ERROR"],
                "error": str(e),
            }

    dump_json(out_dir / "expected_results.json", expected)
    print(f"[ok] fixture written to {out_dir}")
    print(f"[ok] edit {out_dir/'expected_results.json'} to add mutation expectations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
