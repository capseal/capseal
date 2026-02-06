"""End-to-end CapsuleBench round-trip tests (conditionally enabled)."""
from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

RUN_ROUNDTRIP = os.environ.get("RUN_RISC0_ROUNDTRIP") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ROUNDTRIP,
    reason="Set RUN_RISC0_ROUNDTRIP=1 with a valid RISC0 toolchain to exercise the CLI round-trip.",
)


def test_risc0_cli_roundtrip_success(tmp_path: Path) -> None:
    """Run capsule-bench run/pack + verify_capsule.py for the RISC0 backend."""

    policies_dir = Path("policies")
    policy_path = policies_dir / "benchmark_policy_v1.json"
    if not policy_path.exists():
        pytest.skip("benchmark policy file not found")

    elf_path = os.environ.get("RISC0_GUEST_ELF")
    if not elf_path or not Path(elf_path).exists():
        pytest.skip("Provide RISC0_GUEST_ELF pointing to a guest ELF to run the round-trip test")

    run_dir = tmp_path / "cli_run"
    run_dir.mkdir()
    run_cmd = [
        os.environ.get("PYTHON", os.environ.get("PYTEST_PYTHON", "python")),
        "-m",
        "capsule_bench.cli",
        "run",
        "--backend",
        "risc0",
        "--policy",
        str(policy_path),
        "--policy-id",
        "benchmark_policy_v1",
        "--track-id",
        "baseline_no_accel",
        "--docker-image-digest",
        "sha256:integration-test",
        "--output-root",
        str(run_dir),
        "--run-config",
        json.dumps({"elf_path": str(elf_path)}),
    ]
    subprocess.run(run_cmd, check=True)

    run_subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if not run_subdirs:
        pytest.fail("capsule-bench run did not produce an output directory")
    actual_run_dir = run_subdirs[0]

    pack_cmd = [
        os.environ.get("PYTHON", os.environ.get("PYTEST_PYTHON", "python")),
        "-m",
        "capsule_bench.cli",
        "pack",
        "--run-dir",
        str(actual_run_dir),
    ]
    subprocess.run(pack_cmd, check=True)

    pack_files = list(run_dir.glob("**/*.capsulepack.tgz"))
    if not pack_files:
        pytest.fail("capsulepack archive missing")
    pack_file = pack_files[0]
    unpacked_dir = tmp_path / "unpacked"
    unpacked_dir.mkdir()
    with tarfile.open(pack_file, "r:gz") as tar:
        tar.extractall(path=unpacked_dir)

    capsule_path = unpacked_dir / "capsule.json"
    policy_copy = unpacked_dir / "policy.json"
    manifests_root = unpacked_dir / "manifests"
    events_path = unpacked_dir / "events" / "events.jsonl"

    capsule = json.loads(capsule_path.read_text())
    anchors = capsule.get("statement", {}).get("anchors", [])
    if anchors:
        assert anchors[0].get("event_chain_head"), "expected event_chain_head anchor"
    assert events_path.exists(), "events.jsonl missing from capsulepack"

    verify_cmd = [
        os.environ.get("PYTHON", os.environ.get("PYTEST_PYTHON", "python")),
        "-m",
        "scripts.verify_capsule",
        str(capsule_path),
        "--policy",
        str(policy_copy),
        "--manifest-root",
        str(manifests_root),
    ]
    subprocess.run(verify_cmd, check=True)
