#!/usr/bin/env python3
"""
Production emitter for project_trace_v1.

Traces a project directory into a canonical run directory:
    run_dir/
        manifest.json
        trace.jsonl
        commitments.json
        blobs/           (optional, content-addressed chunks)
        fold/
            repo_outline.json
            shards.json

Usage:
    python -m bef_zk.capsule.project_trace_emitter \
        --project <dir> --out <run_dir> [--store-blobs] [--num-shards N]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.project_trace import (
    canonical_json_bytes,
    sha256_bytes,
    make_manifest,
    walk_project,
    chain_rows,
    resolve_policy,
    DEFAULT_CHUNK_SIZE,
)


def _git_info(project_dir: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract git SHA and branch if available."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _load_prev_snapshot(prev_run: Path) -> dict:
    files: dict[str, dict] = {}
    chunks: dict[str, list[dict]] = {}
    trace_path = prev_run / "trace.jsonl"
    if not trace_path.exists():
        return {"files": files, "chunks": chunks}
    with open(trace_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") == "file_entry":
                files[row.get("path")] = row
            elif row.get("row_type") == "chunk_entry":
                chunks.setdefault(row.get("path"), []).append(row)
    return {"files": files, "chunks": chunks}


def emit_project_trace(
    project_dir: Path,
    run_dir: Path,
    store_blobs: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    opaque_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
    num_shards: int = 4,
    policy_id: str = "default_v1",
    *,
    incremental_from: Path | None = None,
    stats_out: dict | None = None,
) -> dict:
    """Emit a complete project_trace_v1 run directory.

    Returns the manifest dict.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = run_dir / "blobs" if store_blobs else None
    if blobs_dir:
        blobs_dir.mkdir(exist_ok=True)

    # Resolve policy preset (explicit patterns override preset)
    preset = resolve_policy(policy_id)
    if opaque_patterns is None:
        opaque_patterns = preset["opaque_patterns"]
    if include_patterns is None:
        include_patterns = preset["include_patterns"]
    large_file_threshold = preset.get("large_file_threshold", 0)
    policy_version = preset.get("policy_version", "0.0.0")
    review_rules = preset.get("review_rules", {})

    git_sha, git_branch = _git_info(project_dir)

    manifest = make_manifest(
        root_path=str(project_dir.resolve()),
        policy_id=policy_id,
        opaque_patterns=opaque_patterns,
        include_patterns=include_patterns,
        chunk_size=chunk_size,
        large_file_threshold=large_file_threshold,
        git_sha=git_sha,
        git_branch=git_branch,
        policy_version=policy_version,
        review_rules=review_rules,
    )
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))

    # Walk and emit rows
    prev_snapshot = None
    if incremental_from:
        prev_run = Path(incremental_from)
        with open(prev_run / "manifest.json") as f:
            prev_manifest = json.load(f)
        if prev_manifest.get("policy_id") != policy_id:
            raise ValueError("incremental source policy mismatch")
        if prev_manifest.get("chunk_size", DEFAULT_CHUNK_SIZE) != chunk_size:
            raise ValueError("incremental source chunk_size mismatch")
        if prev_manifest.get("opaque_patterns") != manifest["opaque_patterns"]:
            raise ValueError("incremental source opaque_patterns mismatch")
        if prev_manifest.get("include_patterns") != manifest["include_patterns"]:
            raise ValueError("incremental source include_patterns mismatch")
        prev_snapshot = _load_prev_snapshot(prev_run)

    walker_stats: dict[str, int] = {}
    rows = walk_project(
        project_dir,
        opaque_patterns=manifest["opaque_patterns"],
        include_patterns=manifest["include_patterns"],
        chunk_size=chunk_size,
        store_blobs=store_blobs,
        blobs_dir=blobs_dir,
        large_file_threshold=large_file_threshold,
        prev_snapshot=prev_snapshot,
        stats=walker_stats,
    )
    if stats_out is not None:
        stats_out.update(walker_stats)

    # Write manifest
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Write trace
    with open(run_dir / "trace.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Write commitments
    head, _ = chain_rows(rows, manifest_hash)
    commitments = {
        "manifest_hash": manifest_hash,
        "head_T": head,
        "total_rows": len(rows),
        "spec": "project_trace_v1",
    }
    with open(run_dir / "commitments.json", "w") as f:
        json.dump(commitments, f, indent=2, sort_keys=True)

    # Emit fold outputs
    _emit_fold(run_dir, rows, head, num_shards)

    return manifest


def _emit_fold(run_dir: Path, rows: list[dict], trace_head: str, num_shards: int):
    """Emit fold/ directory with repo_outline.json and shards.json."""
    fold_dir = run_dir / "fold"
    fold_dir.mkdir(exist_ok=True)

    # --- repo_outline.json ---
    dirs = []
    files = []
    for row in rows:
        rt = row.get("row_type")
        if rt == "dir_entry":
            dirs.append({
                "path": row["path"],
                "classification": row["classification"],
            })
        elif rt == "file_entry":
            entry = {
                "path": row["path"],
                "classification": row["classification"],
                "size": row["size"],
            }
            if "content_hash" in row:
                entry["content_hash"] = row["content_hash"]
            if "num_chunks" in row:
                entry["num_chunks"] = row["num_chunks"]
            # Detect language from extension
            ext = Path(row["path"]).suffix.lstrip(".")
            lang_map = {
                "py": "python", "js": "javascript", "ts": "typescript",
                "jsx": "javascript", "tsx": "typescript",
                "rs": "rust", "go": "go", "c": "c", "cpp": "cpp",
                "h": "c", "hpp": "cpp", "java": "java", "rb": "ruby",
                "sh": "shell", "json": "json", "yaml": "yaml",
                "yml": "yaml", "toml": "toml", "md": "markdown",
            }
            if ext in lang_map:
                entry["language"] = lang_map[ext]
            files.append(entry)

    outline = {
        "trace_root_hash": trace_head,
        "directories": dirs,
        "files": files,
        "summary": {
            "total_dirs": len(dirs),
            "total_files": len(files),
            "included_files": sum(1 for f in files if f["classification"] == "include"),
            "opaque_files": sum(1 for f in files if f["classification"] == "opaque"),
            "total_size": sum(f["size"] for f in files),
        },
    }
    with open(fold_dir / "repo_outline.json", "w") as f:
        json.dump(outline, f, indent=2, sort_keys=True)

    # --- shards.json ---
    included_files = [f for f in files if f["classification"] == "include"]

    # Partition by total bytes (greedy bin packing)
    shards: list[list[dict]] = [[] for _ in range(max(1, num_shards))]
    shard_sizes = [0] * len(shards)

    for finfo in sorted(included_files, key=lambda x: x["size"], reverse=True):
        smallest = min(range(len(shards)), key=lambda i: shard_sizes[i])
        shards[smallest].append({
            "path": finfo["path"],
            "content_hash": finfo.get("content_hash", ""),
            "size": finfo["size"],
        })
        shard_sizes[smallest] += finfo["size"]

    shards_output = {
        "trace_root_hash": trace_head,
        "num_shards": len(shards),
        "shards": [
            {
                "shard_id": i,
                "files": shard,
                "total_size": shard_sizes[i],
            }
            for i, shard in enumerate(shards)
            if shard  # skip empty shards
        ],
    }
    with open(fold_dir / "shards.json", "w") as f:
        json.dump(shards_output, f, indent=2, sort_keys=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emit project_trace_v1 run directory")
    parser.add_argument("--project", required=True, type=Path, help="Project directory to trace")
    parser.add_argument("--out", required=True, type=Path, help="Output run directory")
    parser.add_argument("--store-blobs", action="store_true", help="Store content-addressed chunks in blobs/")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of work shards for fold")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in bytes")
    parser.add_argument("--incremental-from", type=Path, help="Existing run directory to reuse")
    args = parser.parse_args()

    stats: dict[str, int] = {}
    manifest = emit_project_trace(
        args.project, args.out,
        store_blobs=args.store_blobs,
        num_shards=args.num_shards,
        chunk_size=args.chunk_size,
        incremental_from=args.incremental_from,
        stats_out=stats,
    )
    print(f"project_trace_v1 emitted at: {args.out}")
    print(f"  spec: {manifest['spec_id']}")
    print(f"  root: {manifest['root_path']}")
    if manifest.get("git_sha"):
        print(f"  git: {manifest['git_sha'][:12]} ({manifest.get('git_branch', '?')})")
    if stats.get("reused_files"):
        print(f"  Incremental reuse: {stats['reused_files']} files, {stats['reused_bytes']} bytes")


if __name__ == "__main__":
    main()
