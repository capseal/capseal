#!/usr/bin/env python3
"""Integration test: `capsule run` emits and verifies canonical bicep_v2 artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifier import verify_trace, verify_trace_correctness  # type: ignore

CLI = [sys.executable, "-m", "bef_zk.capsule.cli"]
POLICY = ROOT / "policies" / "demo_policy_v1.json"


def run(cmd, *, env=None) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                result.returncode, " ".join(cmd), result.stdout, result.stderr
            )
        )
    return result


def main() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    bicep_seed = "0123456789abcdeffedcba9876543210" * 2  # 64 hex bytes

    with tempfile.TemporaryDirectory(prefix="capsule_run_integration_") as td:
        td_path = Path(td)
        out_dir = td_path / "capsule_output"
        run(
            CLI
            + [
                "run",
                "--policy",
                str(POLICY),
                "--policy-id",
                "demo_policy_v1",
                "--output",
                str(out_dir),
                "--trace-id",
                "bicep_v2_integration",
                "--transition-spec",
                "bicep_v2",
                "--steps",
                "16",
                "--queries",
                "1",
                "--challenges",
                "1",
                "--bicep-seed",
                bicep_seed,
                "--bicep-steps",
                "5",
                "--bicep-audit-k",
                "4",
                "--no-sandbox",
            ],
            env=env,
        )

        run_dir = out_dir / "bicep_v2_run"
        trace = run_dir / "trace.jsonl"
        commitments = run_dir / "commitments.json"
        manifest = run_dir / "manifest.json"
        openings = run_dir / "audit_openings"

        assert trace.exists(), "trace.jsonl missing"
        assert commitments.exists(), "commitments.json missing"
        assert manifest.exists(), "manifest.json missing"
        assert openings.is_dir(), "audit_openings directory missing"

        ok, msg = verify_trace(trace, commitments)
        if not ok:
            raise RuntimeError(f"Trace verification failed: {msg}")
        manifest = json.load(open(manifest))
        ok, msg = verify_trace_correctness(
            trace,
            bicep_seed,
            transition_spec_id="bicep_v2",
            openings_path=openings,
            manifest=manifest,
        )
        if not ok:
            raise RuntimeError(f"Correctness verification failed: {msg}")

    print("PASS: capsule run emits canonical bicep_v2 artifacts")


if __name__ == "__main__":
    main()
