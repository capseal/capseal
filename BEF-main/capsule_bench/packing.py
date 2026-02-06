"""Utilities for assembling capsulepack artifacts."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict


def _hash_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_path(raw: str | None, base: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return path.resolve()


def _relative_target(entry: Any, default: str | None = None) -> str | None:
    if isinstance(entry, dict):
        return entry.get("rel_path") or default
    return default


def write_pack_meta(pack_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(p for p in pack_dir.rglob("*") if p.is_file()):
        entries.append({
            "path": path.relative_to(pack_dir).as_posix(),
            "sha256": _hash_file(path),
        })
    meta = {
        "schema": "capsulepack_meta_v1",
        "entries": entries,
    }
    (pack_dir / "pack_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _copy_policy(meta: dict[str, Any], pack_dir: Path) -> None:
    policy_src = Path(meta.get("policy_copy", meta.get("policy_path", "")))
    if not policy_src.exists():  # pragma: no cover - best effort
        return
    shutil.copy2(policy_src, pack_dir / "policy.json")


def _copy_manifests(meta: dict[str, Any], pack_dir: Path) -> None:
    root = Path(meta.get("manifests_root", ""))
    if not root.exists():
        return
    dest = pack_dir / "manifests"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(root, dest)


def _copy_events(meta: dict[str, Any], capsule: dict[str, Any], pack_dir: Path) -> None:
    events_path = Path(meta.get("events_path", ""))
    if events_path.exists():
        rel_entry = ((capsule.get("artifacts") or {}).get("events_log"))
        rel_path = _relative_target(rel_entry, f"events/{events_path.name}")
        dest = pack_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(events_path, dest)


def _copy_da_challenge(meta: dict[str, Any], pack_dir: Path) -> None:
    challenge_path = Path(meta.get("da_challenge_path", ""))
    if not challenge_path.exists():
        return
    dest = pack_dir / "da" / challenge_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(challenge_path, dest)


def _copy_row_archive(capsule: dict[str, Any], base: Path, pack_dir: Path) -> None:
    info = capsule.get("row_archive") or {}
    src = info.get("abs_path") or info.get("path")
    if not src:
        return
    src_path = _resolve_path(str(src), base)
    if not src_path or not src_path.exists():  # pragma: no cover - depends on run context
        return
    rel_dir = _relative_target(info, "row_archive")
    dest = pack_dir / rel_dir
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_path, dest)


def _copy_proofs(capsule: dict[str, Any], base: Path, pack_dir: Path) -> None:
    proofs_dir = pack_dir / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    for proof_name, entry in (capsule.get("proofs") or {}).items():
        target_dir = proofs_dir / proof_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for fmt in (entry.get("formats") or {}).values():
            src = _resolve_path(fmt.get("path"), base)
            if not src or not src.exists():
                continue
            rel_path = _relative_target(fmt)
            if rel_path:
                dest_path = pack_dir / rel_path
            else:
                dest_path = target_dir / src.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_path)


def _copy_capsule(capsule_path: Path, pack_dir: Path) -> dict[str, Any]:
    capsule = json.loads(capsule_path.read_text())
    (pack_dir / "capsule.json").write_text(json.dumps(capsule, indent=2))
    return capsule


def create_capsulepack(run_meta_path: Path, *, pack_name: str | None = None) -> tuple[Path, Path]:
    run_meta = json.loads(run_meta_path.read_text())
    run_dir = Path(run_meta.get("run_dir", run_meta_path.parent))
    pack_dir = run_dir / "capsulepack"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = Path(run_meta.get("pipeline_output", run_dir / "pipeline"))
    capsule_path = Path(run_meta.get("capsule_path", pipeline_dir / "strategy_capsule.json"))
    capsule = _copy_capsule(capsule_path, pack_dir)
    profile = (capsule.get("verification_profile") or "PROOF_ONLY").upper()
    if profile in {"FULL", "FULLY_VERIFIED"} and not capsule.get("da_challenge"):
        raise ValueError(
            "capsule declares verification_profile=FULL but no DA challenge is embedded"
        )
    _copy_policy(run_meta, pack_dir)
    _copy_manifests(run_meta, pack_dir)
    _copy_events(run_meta, capsule, pack_dir)
    _copy_da_challenge(run_meta, pack_dir)
    _copy_row_archive(capsule, pipeline_dir, pack_dir)
    _copy_proofs(capsule, pipeline_dir, pack_dir)
    write_pack_meta(pack_dir)

    tar_name = pack_name or run_meta.get("run_id") or "capsule_run"
    tar_path = run_dir / f"{tar_name}.capsulepack.tgz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(pack_dir, arcname="capsulepack")
    return pack_dir, tar_path
