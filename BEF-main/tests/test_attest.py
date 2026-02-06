from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str], *, cwd: Path | None = None, expect_fail: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "bef_zk.capsule.cli", *args]
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if expect_fail:
        assert proc.returncode != 0, proc.stdout + proc.stderr
    else:
        if proc.returncode != 0:
            raise AssertionError(proc.stdout + proc.stderr)
    return proc


def test_tabular_attest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "users.csv").write_text("age,price\n70,10\n30,15\n")

    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "profiles": {
                    "data_quality_v1": {
                        "tabular_v1": {
                            "include": ["data/**/*.csv"],
                            "max_missing_frac": 0.0,
                            "columns": {
                                "age": {"min": 0, "max": 65},
                                "price": {"min": 0}
                            },
                        }
                    }
                }
            },
            indent=2,
        )
    )

    base_run = tmp_path / "base_run"
    head_run = tmp_path / "head_run"

    _run_cli(["trace", str(repo), "--out", str(base_run)])
    _run_cli(["trace", str(repo), "--out", str(head_run)])

    _run_cli(
        [
            "attest",
            "--run",
            str(base_run),
            "--project-dir",
            str(repo),
            "--profile",
            str(profile),
            "--profile-id",
            "data_quality_v1",
            "--fail-on",
            "warning",
        ],
        expect_fail=True,
    )

    _run_cli(
        [
            "attest",
            "--run",
            str(base_run),
            "--project-dir",
            str(repo),
            "--profile",
            str(profile),
            "--profile-id",
            "data_quality_v1",
        ]
    )

    summary_path = base_run / "attestations" / "data_quality_v1" / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["violations"], "expected violations in base summary"

    # Update dataset to satisfy constraints
    (data_dir / "users.csv").write_text("age,price\n40,12\n30,15\n")
    _run_cli(["trace", str(repo), "--out", str(head_run)])
    _run_cli(
        [
            "attest",
            "--run",
            str(head_run),
            "--project-dir",
            str(repo),
            "--profile",
            str(profile),
            "--profile-id",
            "data_quality_v1",
        ]
    )

    _run_cli(
        [
            "attest-diff",
            "--base",
            str(base_run),
            "--head",
            str(head_run),
            "--profile-id",
            "data_quality_v1",
            "--fail-on",
            "warning",
        ]
    )
