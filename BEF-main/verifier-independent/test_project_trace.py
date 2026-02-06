#!/usr/bin/env python3
"""
Tests for project_trace_v1: emit, verify, fold, review packets.

Run: python verifier-independent/test_project_trace.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.project_trace import (
    canonical_json_bytes,
    sha256_bytes,
    walk_project,
    chain_rows,
    verify_project_trace,
    classify_path,
    make_manifest,
)
from bef_zk.capsule.project_trace_emitter import emit_project_trace

TOY_PROJECT = Path(__file__).parent / "fixtures" / "toy_project"


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# =============================================================================
# TESTS
# =============================================================================

def test_emit_and_verify() -> TestResult:
    """Emit a trace and verify it passes."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    # Load artifacts
    with open(tmp / "manifest.json") as f:
        manifest = json.load(f)
    with open(tmp / "commitments.json") as f:
        commitments = json.load(f)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    ok, msg = verify_project_trace(
        TOY_PROJECT, rows, manifest,
        commitments["manifest_hash"], commitments["head_T"],
    )
    return TestResult("emit_and_verify", ok, msg)


def test_node_modules_opaque() -> TestResult:
    """node_modules is classified as opaque, not traversed deeply."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    nm_rows = [r for r in rows if "node_modules" in r.get("path", "")]
    if not nm_rows:
        return TestResult("node_modules_opaque", False, "No node_modules rows found")

    # Should be exactly one opaque dir_entry, no children
    if len(nm_rows) != 1:
        return TestResult("node_modules_opaque", False,
                          f"Expected 1 opaque entry, got {len(nm_rows)}")

    if nm_rows[0].get("classification") != "opaque":
        return TestResult("node_modules_opaque", False,
                          f"Expected opaque, got {nm_rows[0].get('classification')}")

    if nm_rows[0].get("row_type") != "dir_entry":
        return TestResult("node_modules_opaque", False,
                          f"Expected dir_entry, got {nm_rows[0].get('row_type')}")

    return TestResult("node_modules_opaque", True,
                      "node_modules emitted as single opaque dir_entry with fingerprint")


def test_included_files_have_chunks() -> TestResult:
    """Included .py files have chunk_entry rows."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    py_files = [r for r in rows if r.get("row_type") == "file_entry"
                and r.get("path", "").endswith(".py")
                and r.get("classification") == "include"]
    chunk_rows = [r for r in rows if r.get("row_type") == "chunk_entry"
                  and r.get("path", "").endswith(".py")]

    if not py_files:
        return TestResult("included_files_have_chunks", False, "No included .py files found")
    if not chunk_rows:
        return TestResult("included_files_have_chunks", False, "No chunk rows for .py files")

    return TestResult("included_files_have_chunks", True,
                      f"{len(py_files)} .py files, {len(chunk_rows)} chunk rows")


def test_hash_chain_integrity() -> TestResult:
    """Hash chain is consistent."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    with open(tmp / "manifest.json") as f:
        manifest = json.load(f)
    with open(tmp / "commitments.json") as f:
        commitments = json.load(f)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    head, _ = chain_rows(rows, manifest_hash)

    if head != commitments["head_T"]:
        return TestResult("hash_chain_integrity", False,
                          f"Head mismatch: {head[:16]}... vs {commitments['head_T'][:16]}...")

    return TestResult("hash_chain_integrity", True,
                      f"Chain verified: {len(rows)} rows, head={head[:16]}...")


def test_tamper_detection() -> TestResult:
    """Modifying a row breaks verification."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    with open(tmp / "manifest.json") as f:
        manifest = json.load(f)
    with open(tmp / "commitments.json") as f:
        commitments = json.load(f)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Tamper: flip a content_hash
    for row in rows:
        if row.get("content_hash"):
            row["content_hash"] = "0" * 64
            break

    ok, msg = verify_project_trace(
        TOY_PROJECT, rows, manifest,
        commitments["manifest_hash"], commitments["head_T"],
    )
    return TestResult("tamper_detection", not ok,
                      msg if not ok else "Should have detected tamper")


def test_fold_repo_outline() -> TestResult:
    """fold/repo_outline.json is emitted with correct structure."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    outline_path = tmp / "fold" / "repo_outline.json"
    if not outline_path.exists():
        return TestResult("fold_repo_outline", False, "repo_outline.json not found")

    with open(outline_path) as f:
        outline = json.load(f)

    if "trace_root_hash" not in outline:
        return TestResult("fold_repo_outline", False, "Missing trace_root_hash")
    if not outline.get("files"):
        return TestResult("fold_repo_outline", False, "No files in outline")
    if not outline.get("summary"):
        return TestResult("fold_repo_outline", False, "No summary")

    s = outline["summary"]
    return TestResult("fold_repo_outline", True,
                      f"{s['total_files']} files ({s['included_files']} included, "
                      f"{s['opaque_files']} opaque), {s['total_dirs']} dirs")


def test_fold_shards() -> TestResult:
    """fold/shards.json partitions work correctly."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp, num_shards=2)

    shards_path = tmp / "fold" / "shards.json"
    if not shards_path.exists():
        return TestResult("fold_shards", False, "shards.json not found")

    with open(shards_path) as f:
        shards = json.load(f)

    if "trace_root_hash" not in shards:
        return TestResult("fold_shards", False, "Missing trace_root_hash")

    shard_list = shards.get("shards", [])
    if not shard_list:
        return TestResult("fold_shards", False, "No shards")

    # Every file in shards should have a content_hash
    all_files = []
    for s in shard_list:
        for f in s.get("files", []):
            all_files.append(f)
            if not f.get("content_hash"):
                return TestResult("fold_shards", False,
                                  f"File {f['path']} missing content_hash in shard")

    return TestResult("fold_shards", True,
                      f"{len(shard_list)} shards, {len(all_files)} files distributed")


def test_fold_trace_root_consistency() -> TestResult:
    """fold outputs reference the same trace_root_hash as commitments."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    with open(tmp / "commitments.json") as f:
        commitments = json.load(f)
    with open(tmp / "fold" / "repo_outline.json") as f:
        outline = json.load(f)
    with open(tmp / "fold" / "shards.json") as f:
        shards = json.load(f)

    head = commitments["head_T"]
    if outline["trace_root_hash"] != head:
        return TestResult("fold_trace_root_consistency", False,
                          "repo_outline trace_root_hash doesn't match commitments")
    if shards["trace_root_hash"] != head:
        return TestResult("fold_trace_root_consistency", False,
                          "shards trace_root_hash doesn't match commitments")

    return TestResult("fold_trace_root_consistency", True,
                      f"All fold artifacts reference {head[:16]}...")


def test_review_packet_references() -> TestResult:
    """A review packet can reference chunk hashes from the trace."""
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    emit_project_trace(TOY_PROJECT, tmp)

    with open(tmp / "commitments.json") as f:
        commitments = json.load(f)

    rows = []
    with open(tmp / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Find a file with chunks
    file_entry = None
    chunk_hashes = []
    for row in rows:
        if row.get("row_type") == "file_entry" and row.get("content_hash"):
            file_entry = row
            chunk_hashes = []
        if (file_entry and row.get("row_type") == "chunk_entry"
                and row.get("path") == file_entry["path"]):
            chunk_hashes.append(row["chunk_hash"])

    if not file_entry or not chunk_hashes:
        return TestResult("review_packet_references", False, "No file with chunks found")

    # Build a sample review packet
    review = {
        "schema": "review_packet_v1",
        "trace_root": commitments["head_T"],
        "findings": [
            {
                "file_path": file_entry["path"],
                "file_hash": file_entry["content_hash"],
                "chunk_hashes": chunk_hashes,
                "rule_id": "test_finding",
                "severity": "info",
                "message": "This is a test finding for pipeline validation.",
                "line_range": [1, 5],
            }
        ],
    }

    # Verify the review references valid trace data
    trace_hashes = {r["content_hash"] for r in rows
                    if r.get("row_type") == "file_entry" and r.get("content_hash")}
    trace_chunks = {r["chunk_hash"] for r in rows
                    if r.get("row_type") == "chunk_entry"}

    for finding in review["findings"]:
        if finding["file_hash"] not in trace_hashes:
            return TestResult("review_packet_references", False,
                              f"Finding references unknown file_hash")
        for ch in finding["chunk_hashes"]:
            if ch not in trace_chunks:
                return TestResult("review_packet_references", False,
                                  f"Finding references unknown chunk_hash")

    # Write the review packet for inspection
    review_dir = tmp / "reviews"
    review_dir.mkdir(exist_ok=True)
    with open(review_dir / "review_001.json", "w") as f:
        json.dump(review, f, indent=2, sort_keys=True)

    return TestResult("review_packet_references", True,
                      f"Review packet references {len(chunk_hashes)} chunks "
                      f"from {file_entry['path']}, all valid")


def test_deterministic_emission() -> TestResult:
    """Two emissions produce identical trace."""
    tmp1 = Path(tempfile.mkdtemp(prefix="pt1_"))
    tmp2 = Path(tempfile.mkdtemp(prefix="pt2_"))
    emit_project_trace(TOY_PROJECT, tmp1)
    emit_project_trace(TOY_PROJECT, tmp2)

    for fname in ["trace.jsonl", "commitments.json"]:
        b1 = (tmp1 / fname).read_bytes()
        b2 = (tmp2 / fname).read_bytes()
        if b1 != b2:
            return TestResult("deterministic_emission", False,
                              f"{fname} differs between runs")

    return TestResult("deterministic_emission", True,
                      "Two emissions are byte-for-byte identical")


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests() -> list[TestResult]:
    tests = [
        test_emit_and_verify,
        test_node_modules_opaque,
        test_included_files_have_chunks,
        test_hash_chain_integrity,
        test_tamper_detection,
        test_fold_repo_outline,
        test_fold_shards,
        test_fold_trace_root_consistency,
        test_review_packet_references,
        test_deterministic_emission,
    ]

    print("=" * 70)
    print("PROJECT_TRACE_V1 TESTS")
    print("=" * 70)
    print()

    results = []
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        print(result)

    print()
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")

    if passed < len(results):
        print("\nFAILED:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if all(r.passed for r in results) else 1)
