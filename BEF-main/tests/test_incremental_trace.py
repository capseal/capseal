from __future__ import annotations

import json
from pathlib import Path

import pytest

from bef_zk.capsule.project_trace_emitter import emit_project_trace


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    return 42\n")
    (repo / "util.py").write_text("def util(x):\n    return x * 2\n")
    return repo


def _read_trace(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with open(run_dir / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_incremental_trace_reuses_unchanged_files(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    run1 = tmp_path / "run1"
    emit_project_trace(repo, run1)

    # Touch only main.py
    (repo / "main.py").write_text("def main():\n    return 43\n")

    run2 = tmp_path / "run2"
    stats: dict[str, int] = {}
    emit_project_trace(repo, run2, incremental_from=run1, stats_out=stats)

    assert stats.get("reused_files") == 1

    rows1 = _read_trace(run1)
    rows2 = _read_trace(run2)
    util_chunks_1 = [r for r in rows1 if r.get("row_type") == "chunk_entry" and r.get("path") == "util.py"]
    util_chunks_2 = [r for r in rows2 if r.get("row_type") == "chunk_entry" and r.get("path") == "util.py"]
    assert util_chunks_1 == util_chunks_2


def test_incremental_trace_requires_compatible_source(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    base_run = tmp_path / "base"
    emit_project_trace(repo, base_run, policy_id="default_v1")

    with pytest.raises(ValueError):
        emit_project_trace(repo, tmp_path / "bad", policy_id="review_v1", incremental_from=base_run)
