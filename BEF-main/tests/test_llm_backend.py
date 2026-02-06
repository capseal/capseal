from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bef_zk.capsule.finding_utils import (
    FINDING_NORM_VERSION,
    compute_finding_fingerprint,
)


def _run_cli(args: list[str], cwd: Path | None = None) -> None:
    cmd = [sys.executable, "-m", "bef_zk.capsule.cli", *args]
    subprocess.run(cmd, check=True, cwd=cwd)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(run_dir: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = run_dir / p
    return p


def _init_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / "main.py").write_text(
        "def add(a, b):\n"
        "    # TODO: replace stub\n"
        "    return a + b\n"
    )
    return repo


def _trace_and_review_stub(repo: Path, run_dir: Path) -> None:
    _run_cli([
        "trace",
        str(repo),
        "--out",
        str(run_dir),
        "--num-shards",
        "1",
    ])
    _run_cli([
        "review",
        "--run",
        str(run_dir),
        "--backend",
        "stub",
        "--project-dir",
        str(repo),
        "--agents",
        "1",
    ])


@pytest.mark.slow
def test_llm_replay_and_review_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def add(a, b):\n"
        "    # TODO: handle overflow\n"
        "    return a + b\n"
    )

    base_run = tmp_path / "base_run"
    head_run = tmp_path / "head_run"

    _run_cli([
        "trace",
        str(repo),
        "--out",
        str(base_run),
        "--num-shards",
        "1",
    ])
    _run_cli([
        "trace",
        str(repo),
        "--out",
        str(head_run),
        "--num-shards",
        "1",
    ])

    with open(base_run / "trace.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chunk_row = next(
        r for r in rows if r.get("row_type") == "chunk_entry" and r.get("path") == "main.py"
    )
    chunk_hash = chunk_row["chunk_hash"]

    raw_dir = tmp_path / "raw_replies"
    raw_dir.mkdir()
    raw_payload = {
        "findings": [
            {
                "rule_id": "test.todo",
                "severity": "warning",
                "file_path": "main.py",
                "primary_chunk_hash": chunk_hash,
                "supporting_chunk_hashes": [],
                "line_range": [1, 2],
                "message": "TODO found: resolve the placeholder logic.",
                "snippet": "# TODO: handle overflow",
            }
        ]
    }
    raw_file = raw_dir / "raw_shard_0.txt"
    raw_file.write_text(json.dumps(raw_payload, indent=2))

    _run_cli([
        "agent-review",
        "--run",
        str(base_run),
        "--shard",
        "0",
        "--backend",
        "llm",
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--llm-replay",
        str(raw_file),
        "--project-dir",
        str(repo),
    ])

    cache_dir = base_run / "reviews" / "packets" / "by_input_hash"
    assert any(cache_dir.glob("*.json")), "expected cached review packet"

    for run_dir in (base_run, head_run):
        _run_cli([
            "review",
            "--run",
            str(run_dir),
            "--backend",
            "llm",
            "--llm-provider",
            "mock",
            "--llm-model",
            "mock",
            "--llm-replay",
            str(raw_dir),
            "--project-dir",
            str(repo),
            "--agents",
            "1",
        ])

    head_review_packet = json.loads((head_run / "reviews" / "review_shard_0.json").read_text())
    head_fingerprint = head_review_packet["findings"][0]["finding_fingerprint"]

    _run_cli([
        "review-diff",
        "--base",
        str(base_run),
        "--head",
        str(head_run),
    ])
    diff_receipt = json.loads((head_run / "diff" / "receipt.json").read_text())
    assert diff_receipt["finding_norm_version"] == FINDING_NORM_VERSION
    assert diff_receipt["policy_version"] == "1.0.0"
    assert diff_receipt["fail_on"] == "warning"

    mock_explain = {
        "explanations": [
            {
                "fingerprint": head_fingerprint,
                "analysis": "Explain analysis",
                "recommendation": "Apply fix",
                "suggested_change": "do something",
            }
        ]
    }
    monkeypatch.setenv("CAPSEAL_MOCK_LLM_RAW", json.dumps(mock_explain))
    _run_cli([
        "explain-llm",
        "--run",
        str(head_run),
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--max-findings",
        "5",
        "--min-severity",
        "info",
    ])
    monkeypatch.delenv("CAPSEAL_MOCK_LLM_RAW", raising=False)

    _run_cli(["dag", "--run", str(head_run), "--project-dir", str(repo)])
    _run_cli([
        "verify-rollup",
        str(head_run / "workflow" / "rollup.json"),
        "--project-dir",
        str(repo),
    ])
    dag = json.loads((head_run / "workflow" / "dag.json").read_text())
    vertex_types = {v["type"] for v in dag["vertices"]}
    assert "review_diff_receipt_v1" in vertex_types
    assert "llm_explain_receipt_v1" in vertex_types

    prompt_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bef_zk.capsule.cli",
            "prompt-open",
            "--run",
            str(base_run),
            "--shard",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prompt bundle" in prompt_result.stdout.lower()
    review_packet = json.loads((base_run / "reviews" / "review_shard_0.json").read_text())
    primary = review_packet["findings"][0]["chunk_hashes"][0]
    assert primary == chunk_hash
    assert review_packet["findings"][0]["finding_norm_version"] == FINDING_NORM_VERSION
    assert review_packet["policy_version"] == "1.0.0"
    assert review_packet.get("prompt_bundle_hash")
    assert review_packet.get("llm_raw_hash")
    prompt_path = _resolve_path(base_run, review_packet["prompt_bundle_path"])
    raw_path = _resolve_path(base_run, review_packet["llm_raw_path"])
    assert review_packet["prompt_bundle_hash"] == _file_hash(prompt_path)
    assert review_packet["llm_raw_hash"] == _file_hash(raw_path)

    base_agg = json.loads((base_run / "reviews" / "aggregate.json").read_text())
    assert base_agg["backend_id"].startswith("llm")
    assert base_agg["finding_norm_version"] == FINDING_NORM_VERSION
    assert base_agg["policy_version"] == "1.0.0"


def test_explain_llm_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    review_dir = run_dir / "reviews"
    review_dir.mkdir(parents=True)
    (run_dir / "commitments.json").write_text(json.dumps({"head_T": "trace123", "total_rows": 1}))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "root_path": str(run_dir),
                "policy_id": "demo",
                "policy_version": "test-policy",
                "review_rules": {},
            }
        )
    )

    finding = {
        "severity": "warning",
        "rule_id": "demo.rule",
        "file_path": "main.py",
        "line_range": [1, 2],
        "message": "demo issue",
        "snippet": "print('hi')",
        "chunk_hashes": ["abc"],
        "finding_fingerprint": "fp_demo",
        "finding_norm_version": FINDING_NORM_VERSION,
    }
    review = {
        "schema": "review_packet_v1",
        "trace_root": "trace123",
        "backend": "llm:mock",
        "backend_id": "llm:mock",
        "policy_id": "demo",
        "findings": [finding],
    }
    (review_dir / "review_shard_0.json").write_text(json.dumps(review))

    mock_response = {
        "explanations": [
            {
                "fingerprint": "fp_demo",
                "analysis": "Summary",
                "recommendation": "Fix it",
                "suggested_change": "update code",
            }
        ]
    }
    monkeypatch.setenv("CAPSEAL_MOCK_LLM_RAW", json.dumps(mock_response))

    # Duplicate finding in another shard to ensure dedupe occurs.
    (review_dir / "review_shard_1.json").write_text(json.dumps(review))

    _run_cli([
        "explain-llm",
        "--run",
        str(run_dir),
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--max-findings",
        "5",
        "--min-severity",
        "info",
    ])

    explain_root = run_dir / "reviews" / "explain_llm"
    subdirs = [p for p in explain_root.iterdir() if p.is_dir()]
    assert subdirs, "no explain cache directory"
    cache_dir = subdirs[0]
    summary_path = cache_dir / "summary.json"
    assert summary_path.exists(), "summary.json not written"
    summary = json.loads(summary_path.read_text())
    assert summary["explanations"][0]["fingerprint"] == "fp_demo"
    assert summary["explanations"][0]["analysis"] == "Summary"
    assert "backend_id" in summary
    assert "finding_norm_version" in summary
    assert summary.get("policy_version") == "test-policy"
    assert summary.get("selected_findings") == 1
    assert "report_path" in summary
    report_path = _resolve_path(run_dir, summary["report_path"])
    assert report_path.exists()

    receipt_path = cache_dir / "receipt.json"
    assert receipt_path.exists(), "receipt.json missing"
    receipt_json = json.loads(receipt_path.read_text())
    assert receipt_json.get("finding_norm_version") == FINDING_NORM_VERSION
    assert receipt_json.get("policy_version") == "test-policy"
    assert receipt_json.get("report_hash") == _file_hash(report_path)
    assert len(receipt_json.get("selected_fingerprints", [])) == 1
    _run_cli([
        "verify-explain",
        "--run",
        str(run_dir),
        "--receipt",
        str(receipt_path),
    ])


def _snippet_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _base_finding(snippet: str, message: str, chunk_hash: str = "chunk123") -> dict:
    return {
        "file_path": "src/foo.py",
        "chunk_hashes": [chunk_hash],
        "snippet": snippet,
        "snippet_hash": _snippet_hash(snippet),
        "message": message,
        "rule_id": "demo.rule",
    }


def test_finding_norm_v2_message_normalization() -> None:
    backend = "llm:test"
    base = _base_finding("print('hi')", "Untrimmed   message")
    variant = _base_finding("print('hi')", "  untrimmed message   ")

    fp_base = compute_finding_fingerprint(base, backend, norm_version=FINDING_NORM_VERSION)
    fp_variant = compute_finding_fingerprint(variant, backend, norm_version=FINDING_NORM_VERSION)
    assert fp_base == fp_variant, "message whitespace should not change fingerprint"

    moved = _base_finding("print('hi')", "Untrimmed   message", chunk_hash="chunk999")
    fp_moved = compute_finding_fingerprint(moved, backend, norm_version=FINDING_NORM_VERSION)
    assert fp_moved != fp_base, "changing chunk context must change fingerprint"


def test_llm_review_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def add(a, b):\n" "    return a + b\n"
    )

    run_dir = tmp_path / "run"
    _run_cli([
        "trace",
        str(repo),
        "--out",
        str(run_dir),
        "--num-shards",
        "1",
    ])

    with open(run_dir / "trace.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chunk_hash = next(r["chunk_hash"] for r in rows if r.get("row_type") == "chunk_entry")

    raw_payload = {
        "findings": [
            {
                "rule_id": "cache.test",
                "severity": "warning",
                "file_path": "main.py",
                "primary_chunk_hash": chunk_hash,
                "supporting_chunk_hashes": [],
                "line_range": [1, 1],
                "message": "Cached finding",
                "snippet": "return a + b",
            }
        ]
    }

    monkeypatch.setenv("CAPSEAL_MOCK_LLM_RAW", json.dumps(raw_payload))
    _run_cli([
        "review",
        "--run",
        str(run_dir),
        "--backend",
        "llm",
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--project-dir",
        str(repo),
        "--agents",
        "1",
    ])

    packet_path = run_dir / "reviews" / "review_shard_0.json"
    first_packet = json.loads(packet_path.read_text())
    assert first_packet.get("llm_raw_hash")
    first_message = first_packet["findings"][0]["message"]

    monkeypatch.delenv("CAPSEAL_MOCK_LLM_RAW", raising=False)
    _run_cli([
        "review",
        "--run",
        str(run_dir),
        "--backend",
        "llm",
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--project-dir",
        str(repo),
        "--agents",
        "1",
    ])

    second_packet = json.loads(packet_path.read_text())
    assert second_packet["findings"][0]["message"] == first_message
    assert second_packet.get("llm_raw_hash")


def test_explain_llm_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def todo():\n" "    # TODO: fix\n" "    return 1\n"
    )

    run_dir = tmp_path / "run"
    _run_cli([
        "trace",
        str(repo),
        "--out",
        str(run_dir),
        "--num-shards",
        "1",
    ])

    _run_cli([
        "review",
        "--run",
        str(run_dir),
        "--backend",
        "stub",
        "--project-dir",
        str(repo),
        "--agents",
        "1",
    ])

    review_packet = json.loads((run_dir / "reviews" / "review_shard_0.json").read_text())
    fingerprint = review_packet["findings"][0]["finding_fingerprint"]
    explain_payload = {
        "explanations": [
            {
                "fingerprint": fingerprint,
                "analysis": "Explain cache",
                "recommendation": "Fix TODO",
            }
        ]
    }

    monkeypatch.setenv("CAPSEAL_MOCK_LLM_RAW", json.dumps(explain_payload))
    _run_cli([
        "explain-llm",
        "--run",
        str(run_dir),
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--max-findings",
        "5",
        "--min-severity",
        "info",
    ])

    explain_root = run_dir / "reviews" / "explain_llm"
    cache_dirs = [p for p in explain_root.iterdir() if p.is_dir()]
    assert cache_dirs, "explain cache missing"
    cache_dir = cache_dirs[0]
    summary_path = cache_dir / "summary.json"
    first_summary = json.loads(summary_path.read_text())

    monkeypatch.delenv("CAPSEAL_MOCK_LLM_RAW", raising=False)
    _run_cli([
        "explain-llm",
        "--run",
        str(run_dir),
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--max-findings",
        "5",
        "--min-severity",
        "info",
    ])

    second_summary = json.loads(summary_path.read_text())
    assert second_summary == first_summary


def test_dag_rejects_tampered_review_packet(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    run_dir = tmp_path / "run"
    _trace_and_review_stub(repo, run_dir)

    packet_path = run_dir / "reviews" / "review_shard_0.json"
    packet = json.loads(packet_path.read_text())
    packet["findings"][0]["file_hash"] = "0" * 64
    packet_path.write_text(json.dumps(packet, indent=2))

    with pytest.raises(subprocess.CalledProcessError):
        _run_cli(["dag", "--run", str(run_dir), "--project-dir", str(repo)])


def test_verify_rollup_rejects_tampered_diff_receipt(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    base_run = tmp_path / "base"
    head_run = tmp_path / "head"
    _trace_and_review_stub(repo, base_run)
    _trace_and_review_stub(repo, head_run)

    _run_cli([
        "review-diff",
        "--base",
        str(base_run),
        "--head",
        str(head_run),
    ])
    _run_cli(["dag", "--run", str(head_run), "--project-dir", str(repo)])

    diff_path = head_run / "diff" / "receipt.json"
    diff_data = json.loads(diff_path.read_text())
    diff_data["new_count"] = 999
    diff_path.write_text(json.dumps(diff_data, indent=2))

    with pytest.raises(subprocess.CalledProcessError):
        _run_cli([
            "verify-rollup",
            str(head_run / "workflow" / "rollup.json"),
            "--project-dir",
            str(repo),
        ])


def test_verify_explain_detects_summary_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    run_dir = tmp_path / "run"
    _trace_and_review_stub(repo, run_dir)

    fingerprint = json.loads((run_dir / "reviews" / "review_shard_0.json").read_text())["findings"][0]["finding_fingerprint"]
    mock_response = {
        "explanations": [
            {
                "fingerprint": fingerprint,
                "analysis": "Original",
                "recommendation": "Fix",
            }
        ]
    }
    monkeypatch.setenv("CAPSEAL_MOCK_LLM_RAW", json.dumps(mock_response))
    _run_cli([
        "explain-llm",
        "--run",
        str(run_dir),
        "--llm-provider",
        "mock",
        "--llm-model",
        "mock",
        "--max-findings",
        "5",
        "--min-severity",
        "info",
    ])
    monkeypatch.delenv("CAPSEAL_MOCK_LLM_RAW", raising=False)

    explain_root = run_dir / "reviews" / "explain_llm"
    cache_dir = next(p for p in explain_root.iterdir() if p.is_dir())
    summary_path = cache_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["explanations"][0]["analysis"] = "Tampered"
    summary_path.write_text(json.dumps(summary, indent=2))

    with pytest.raises(subprocess.CalledProcessError):
        _run_cli([
            "verify-explain",
            "--run",
            str(run_dir),
            "--hash",
            cache_dir.name,
        ])
