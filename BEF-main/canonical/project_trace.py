#!/usr/bin/env python3
"""
Canonical contract for project_trace_v1.

Traces a project directory into a verifiable artifact set:
  trace.jsonl    — canonical rows (dir_entry, file_entry, chunk_entry)
  manifest.json  — policy + root info + version
  commitments.json — chain head + Merkle root
  blobs/         — content-addressed chunk storage (optional)

Row types:
  dir_entry   — directory exists (included or opaque)
  file_entry  — file exists + metadata + content_hash or opaque_fingerprint
  chunk_entry — byte range of an included file + chunk hash

Traversal order: sorted(os.listdir) at each level, depth-first.
Policy: include/exclude by glob pattern; excluded still emits opaque rows.

All logic here is the single source of truth — emitter and verifier both import.
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

# =============================================================================
# CANONICAL JSON (shared with bicep_state)
# =============================================================================

def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False,
    ).encode('utf-8')


# =============================================================================
# HASHING
# =============================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# POLICY
# =============================================================================

# Default opaque patterns — big/generated dirs we fingerprint but don't read
DEFAULT_OPAQUE_PATTERNS = [
    "node_modules", ".venv", "venv", "__pycache__", ".git",
    "dist", "build", "target", ".next", ".nuxt",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "*.pyc", "*.pyo", "*.o", "*.so", "*.dylib",
    "*.egg-info", "*.whl", "*.tar.gz", "*.zip",
]

# Default include patterns (source, config, CI, docs)
DEFAULT_INCLUDE_PATTERNS = [
    "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.rs", "*.go",
    "*.c", "*.cpp", "*.h", "*.hpp", "*.java", "*.rb", "*.sh",
    "*.json", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini",
    "*.csv", "*.tsv",
    "*.md", "*.rst", "*.txt",
    "Makefile", "Dockerfile", "docker-compose.yml",
    ".github/*", ".gitignore", "LICENSE*", "README*",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "tsconfig.json",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
    "requirements.txt", "Pipfile", "Gemfile",
]


# Review-oriented policy: binary/large-file opaque, code-only includes
REVIEW_OPAQUE_PATTERNS = [
    # Dependency / build / vendored dirs
    "node_modules", ".venv", ".venv_*", "venv", "venv_*",
    ".env", ".env_*", "__pycache__", ".git",
    "site-packages", "__pypackages__", "vendor",
    "dist", "build", "target", ".next", ".nuxt",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "*.egg-info", ".cache", ".cargo", "third_party",
    # Compiled / bytecode
    "*.pyc", "*.pyo", "*.o", "*.so", "*.dylib", "*.a", "*.lib",
    "*.class", "*.whl",
    # Archives
    "*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.gz", "*.bz2", "*.xz", "*.7z",
    "*.rar", "*.zst",
    # Binary documents / media
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg",
    "*.mp3", "*.mp4", "*.avi", "*.mov", "*.wav", "*.flac",
    "*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot",
    # ML / data blobs
    "*.onnx", "*.pt", "*.pth", "*.h5", "*.hdf5", "*.pkl", "*.pickle",
    "*.npy", "*.npz", "*.parquet", "*.arrow", "*.feather",
    "*.bin", "*.dat", "*.db", "*.sqlite", "*.sqlite3",
    # Executables
    "*.exe", "*.dll", "*.app", "*.dmg", "*.msi",
    "*.cap",
]

REVIEW_INCLUDE_PATTERNS = [
    # Source code
    "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.rs", "*.go",
    "*.c", "*.cpp", "*.h", "*.hpp", "*.java", "*.rb", "*.sh",
    "*.cs", "*.swift", "*.kt", "*.scala", "*.lua", "*.zig",
    # Config / CI
    "*.json", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini",
    "*.csv", "*.tsv",
    "*.env.example",
    "Makefile", "CMakeLists.txt", "Dockerfile", "docker-compose.yml",
    ".github/*", ".gitignore", "LICENSE*", "README*",
    # Package manifests
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "tsconfig.json",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
    "requirements.txt", "Pipfile", "Gemfile", "Gemfile.lock",
    # Docs (text-based only)
    "*.md", "*.rst", "*.txt", "*.adoc",
    # Web templates
    "*.html", "*.css", "*.scss", "*.less",
    # SQL / proto / graphql
    "*.sql", "*.proto", "*.graphql", "*.gql",
]

# Large file threshold for review policy (10 MB)
REVIEW_LARGE_FILE_THRESHOLD = 10 * 1024 * 1024

# Policy presets registry
DEFAULT_REVIEW_RULES = {
    "severity_order": ["error", "warning", "info"],
    "default_fail_on": "warning",
}

POLICY_PRESETS: dict[str, dict] = {
    "default_v1": {
        "policy_version": "1.0.0",
        "opaque_patterns": DEFAULT_OPAQUE_PATTERNS,
        "include_patterns": DEFAULT_INCLUDE_PATTERNS,
        "large_file_threshold": 0,  # no size gating
        "review_rules": DEFAULT_REVIEW_RULES,
    },
    "review_v1": {
        "policy_version": "1.0.0",
        "opaque_patterns": REVIEW_OPAQUE_PATTERNS,
        "include_patterns": REVIEW_INCLUDE_PATTERNS,
        "large_file_threshold": REVIEW_LARGE_FILE_THRESHOLD,
        "review_rules": DEFAULT_REVIEW_RULES,
    },
}


def resolve_policy(policy_id: str) -> dict:
    """Resolve a policy preset by name.

    Returns dict with opaque_patterns, include_patterns, large_file_threshold.
    """
    if policy_id in POLICY_PRESETS:
        return POLICY_PRESETS[policy_id]
    # Fallback to default
    return POLICY_PRESETS["default_v1"]


def _match_pattern(name: str, pattern: str) -> bool:
    """Simple glob matching (no recursive **)."""
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


def classify_path(
    rel_path: str,
    is_dir: bool,
    opaque_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
) -> str:
    """Classify a path as 'include', 'opaque', or 'exclude'.

    Returns: 'include' | 'opaque' | 'exclude'
    """
    if opaque_patterns is None:
        opaque_patterns = DEFAULT_OPAQUE_PATTERNS
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS

    name = os.path.basename(rel_path)

    # Check opaque first (directories and file patterns)
    for pat in opaque_patterns:
        if _match_pattern(name, pat):
            return "opaque"

    if is_dir:
        # Directories not in opaque list are included (traversed)
        return "include"

    # Check include patterns for files
    for pat in include_patterns:
        if _match_pattern(name, pat):
            return "include"

    # Files not matching any include pattern are excluded (still emitted as opaque)
    return "opaque"


# =============================================================================
# CHUNKING
# =============================================================================

DEFAULT_CHUNK_SIZE = 4096  # bytes


def chunk_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict]:
    """Split file into fixed-size chunks with hashes.

    Returns list of {offset, length, chunk_hash} dicts.
    """
    chunks = []
    with open(path, 'rb') as f:
        offset = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            chunks.append({
                "offset": offset,
                "length": len(data),
                "chunk_hash": sha256_bytes(data),
            })
            offset += len(data)
    return chunks


# =============================================================================
# OPAQUE FINGERPRINT
# =============================================================================

def opaque_dir_fingerprint(path: Path) -> dict:
    """Cheap fingerprint for an opaque directory.

    Doesn't read file contents — just listing + metadata.
    """
    entries = []
    total_size = 0
    file_count = 0
    dir_count = 0

    try:
        for name in sorted(os.listdir(path)):
            child = path / name
            try:
                st = child.stat()
                is_dir = stat.S_ISDIR(st.st_mode)
                entries.append(f"{name}:{st.st_size}:{int(is_dir)}")
                if is_dir:
                    dir_count += 1
                else:
                    file_count += 1
                    total_size += st.st_size
            except OSError:
                entries.append(f"{name}:?:?")
    except OSError:
        pass

    listing_hash = sha256_bytes("\n".join(entries).encode())

    return {
        "listing_hash": listing_hash,
        "file_count": file_count,
        "dir_count": dir_count,
        "total_size": total_size,
    }


def opaque_file_fingerprint(path: Path) -> dict:
    """Cheap fingerprint for an opaque file (metadata only, no content read)."""
    try:
        st = path.stat()
        return {
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    except OSError:
        return {"size": -1, "mtime_ns": 0}


# =============================================================================
# ROW SCHEMAS
# =============================================================================

def make_dir_entry(
    rel_path: str,
    classification: str,
    fingerprint: dict | None = None,
) -> dict:
    """Create a dir_entry row."""
    row = {
        "schema": "project_trace_v1",
        "row_type": "dir_entry",
        "path": rel_path,
        "classification": classification,
    }
    if fingerprint is not None:
        row["fingerprint"] = fingerprint
    return row


def make_file_entry(
    rel_path: str,
    classification: str,
    size: int,
    content_hash: str | None = None,
    fingerprint: dict | None = None,
    num_chunks: int = 0,
) -> dict:
    """Create a file_entry row."""
    row = {
        "schema": "project_trace_v1",
        "row_type": "file_entry",
        "path": rel_path,
        "classification": classification,
        "size": size,
    }
    if content_hash is not None:
        row["content_hash"] = content_hash
    if fingerprint is not None:
        row["fingerprint"] = fingerprint
    if num_chunks > 0:
        row["num_chunks"] = num_chunks
    return row


def make_chunk_entry(
    rel_path: str,
    chunk_index: int,
    offset: int,
    length: int,
    chunk_hash: str,
) -> dict:
    """Create a chunk_entry row."""
    return {
        "schema": "project_trace_v1",
        "row_type": "chunk_entry",
        "path": rel_path,
        "chunk_index": chunk_index,
        "offset": offset,
        "length": length,
        "chunk_hash": chunk_hash,
    }


# =============================================================================
# MANIFEST
# =============================================================================

def make_manifest(
    root_path: str,
    policy_id: str = "default_v1",
    opaque_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    large_file_threshold: int = 0,
    git_sha: str | None = None,
    git_branch: str | None = None,
    policy_version: str | None = None,
    review_rules: dict | None = None,
) -> dict:
    """Create project_trace_v1 manifest."""
    manifest = {
        "spec_id": "project_trace_v1",
        "version": "1.0.0",
        "root_path": root_path,
        "policy_id": policy_id,
        "opaque_patterns": opaque_patterns or DEFAULT_OPAQUE_PATTERNS,
        "include_patterns": include_patterns or DEFAULT_INCLUDE_PATTERNS,
        "chunk_size": chunk_size,
        "large_file_threshold": large_file_threshold,
    }
    if policy_version:
        manifest["policy_version"] = policy_version
    if review_rules:
        manifest["review_rules"] = review_rules
    if git_sha:
        manifest["git_sha"] = git_sha
    if git_branch:
        manifest["git_branch"] = git_branch
    return manifest


# =============================================================================
# TREE WALK (deterministic, depth-first, sorted)
# =============================================================================

def walk_project(
    root: Path,
    opaque_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    store_blobs: bool = False,
    blobs_dir: Path | None = None,
    large_file_threshold: int = 0,
    prev_snapshot: dict | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Walk project tree and produce canonical trace rows.

    Traversal: sorted entries at each level, depth-first.
    Returns list of rows in canonical order.
    """
    rows: list[dict] = []
    if stats is None:
        stats = {}
    stats.setdefault("reused_files", 0)
    stats.setdefault("reused_bytes", 0)
    prev_files = prev_snapshot.get("files") if prev_snapshot else {}
    prev_chunks = prev_snapshot.get("chunks") if prev_snapshot else {}
    reuse_enabled = bool(prev_snapshot)

    def _walk(dir_path: Path, rel_prefix: str):
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return

        for name in entries:
            full = dir_path / name
            rel = f"{rel_prefix}/{name}" if rel_prefix else name

            try:
                is_dir = full.is_dir()
                is_file = full.is_file()
            except OSError:
                continue

            classification = classify_path(rel, is_dir, opaque_patterns, include_patterns)

            if is_dir:
                if classification == "opaque":
                    fp = opaque_dir_fingerprint(full)
                    rows.append(make_dir_entry(rel, "opaque", fingerprint=fp))
                else:
                    rows.append(make_dir_entry(rel, "include"))
                    _walk(full, rel)

            elif is_file:
                st = full.stat()
                # Large file override: force opaque above threshold
                if (classification == "include"
                        and large_file_threshold > 0
                        and st.st_size > large_file_threshold):
                    classification = "opaque"
                if classification == "include":
                    content_hash = sha256_file(full)
                    reused = False
                    if reuse_enabled:
                        prev_file = prev_files.get(rel)
                        prev_chunk_rows = prev_chunks.get(rel)
                        if (
                            prev_file
                            and prev_chunk_rows
                            and prev_file.get("classification") == "include"
                            and prev_file.get("content_hash") == content_hash
                        ):
                            rows.append({**prev_file, "size": st.st_size, "content_hash": content_hash})
                            for chunk_row in prev_chunk_rows:
                                rows.append(dict(chunk_row))
                            stats["reused_files"] += 1
                            stats["reused_bytes"] += st.st_size
                            reused = True

                    if not reused:
                        chunks = chunk_file(full, chunk_size)

                        rows.append(make_file_entry(
                            rel, "include", st.st_size,
                            content_hash=content_hash,
                            num_chunks=len(chunks),
                        ))

                        for ci, ch in enumerate(chunks):
                            rows.append(make_chunk_entry(
                                rel, ci, ch["offset"], ch["length"], ch["chunk_hash"],
                            ))

                            if store_blobs and blobs_dir:
                                blob_path = blobs_dir / ch["chunk_hash"]
                                if not blob_path.exists():
                                    with open(full, 'rb') as f:
                                        f.seek(ch["offset"])
                                        blob_data = f.read(ch["length"])
                                    blob_path.write_bytes(blob_data)
                else:
                    fp = opaque_file_fingerprint(full)
                    rows.append(make_file_entry(
                        rel, "opaque", st.st_size, fingerprint=fp,
                    ))

    _walk(root, "")
    return rows


# =============================================================================
# HASH CHAIN (reuse from bicep but standalone)
# =============================================================================

def row_digest(row: dict) -> str:
    return sha256_bytes(canonical_json_bytes(row))


def chain_rows(rows: list[dict], manifest_hash: str) -> tuple[str, list[str]]:
    """Compute hash chain over rows.

    Returns (final_head, list_of_row_digests).
    """
    head = hashlib.sha256(f"genesis:{manifest_hash}".encode()).hexdigest()
    digests = []

    for row in rows:
        d = row_digest(row)
        digests.append(d)
        head = hashlib.sha256(f"{head}:{d}".encode()).hexdigest()

    return head, digests


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_project_trace(
    root: Path,
    trace_rows: list[dict],
    manifest: dict,
    manifest_hash: str,
    expected_head: str,
) -> tuple[bool, str]:
    """Verify a project trace against the actual filesystem.

    Re-walks the tree under the same policy and checks:
    1. Row-by-row equality (canonical JSON)
    2. Hash chain integrity
    3. Content hashes for included files
    4. Fingerprints for opaque entries
    """
    # Regenerate rows from filesystem
    regen_rows = walk_project(
        root,
        opaque_patterns=manifest.get("opaque_patterns"),
        include_patterns=manifest.get("include_patterns"),
        chunk_size=manifest.get("chunk_size", DEFAULT_CHUNK_SIZE),
        large_file_threshold=manifest.get("large_file_threshold", 0),
    )

    # Check row count
    if len(regen_rows) != len(trace_rows):
        return False, (
            f"Row count mismatch: expected {len(trace_rows)} "
            f"from trace, got {len(regen_rows)} from filesystem"
        )

    # Check each row
    for i, (orig, regen) in enumerate(zip(trace_rows, regen_rows)):
        orig_bytes = canonical_json_bytes(orig)
        regen_bytes = canonical_json_bytes(regen)
        if orig_bytes != regen_bytes:
            return False, (
                f"Row {i} mismatch (type={orig.get('row_type')}, "
                f"path={orig.get('path')})"
            )

    # Check hash chain
    head, _ = chain_rows(trace_rows, manifest_hash)
    if head != expected_head:
        return False, f"Hash chain head mismatch: expected {expected_head[:16]}..., got {head[:16]}..."

    return True, f"Verified {len(trace_rows)} rows, chain head matches"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "canonical_json_bytes", "sha256_bytes", "sha256_file",
    "classify_path", "chunk_file",
    "opaque_dir_fingerprint", "opaque_file_fingerprint",
    "make_dir_entry", "make_file_entry", "make_chunk_entry",
    "make_manifest", "walk_project",
    "row_digest", "chain_rows",
    "verify_project_trace",
    "DEFAULT_OPAQUE_PATTERNS", "DEFAULT_INCLUDE_PATTERNS",
    "DEFAULT_CHUNK_SIZE",
    "REVIEW_OPAQUE_PATTERNS", "REVIEW_INCLUDE_PATTERNS",
    "REVIEW_LARGE_FILE_THRESHOLD",
    "POLICY_PRESETS", "resolve_policy",
]
