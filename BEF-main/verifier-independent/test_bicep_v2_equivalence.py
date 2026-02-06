#!/usr/bin/env python3
"""
Equivalence test: production emitter vs generator (byte-for-byte).

Runs both:
  1) verifier-independent/verification_demo.emit_run_directory
  2) bef_zk/capsule/bicep_v2_executor.emit_bicep_v2_run

Using identical seed + params, then compares file-by-file contents.
Also includes a negative test to ensure mismatches are detected.

Run: python verifier-independent/test_bicep_v2_equivalence.py
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


# Make project modules importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verifier-independent"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "verification_demo",
    str(ROOT / "verifier-independent" / "verification_demo.py"),
)
mod = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore
gen_emit = mod.emit_run_directory
GEN_SEED = mod.SEED

# Import production emitter
from bef_zk.capsule.bicep_v2_executor import emit_bicep_v2_run


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _compare_files(a: Path, b: Path) -> tuple[bool, str]:
    if not a.exists() or not b.exists():
        return False, f"Missing file: {a if not a.exists() else b}"
    ba = _read_bytes(a)
    bb = _read_bytes(b)
    if ba == bb:
        return True, ""
    # Produce a tiny diff context: first mismatch offset + sha digests
    first_mismatch = next((i for i in range(min(len(ba), len(bb))) if ba[i] != bb[i]), -1)
    return (
        False,
        (
            f"Mismatch at byte {first_mismatch}; "
            f"{a.name} sha256={hashlib.sha256(ba).hexdigest()[:16]}..., "
            f"{b.name} sha256={hashlib.sha256(bb).hexdigest()[:16]}..."
        ),
    )


def _compare_runs(gen_dir: Path, exec_dir: Path) -> tuple[bool, str]:
    # Required files
    req = ["trace.jsonl", "manifest.json", "commitments.json"]
    for name in req:
        ok, msg = _compare_files(gen_dir / name, exec_dir / name)
        if not ok:
            return ok, f"{name}: {msg}"

    # Audit openings directory and files
    gen_open = gen_dir / "audit_openings"
    exec_open = exec_dir / "audit_openings"
    if not gen_open.is_dir() or not exec_open.is_dir():
        return False, "Missing audit_openings directory"

    gen_files = sorted([p.name for p in gen_open.iterdir() if p.is_file()])
    exec_files = sorted([p.name for p in exec_open.iterdir() if p.is_file()])
    if gen_files != exec_files:
        return False, f"Audit files differ: {gen_files[:3]}... vs {exec_files[:3]}..."

    for name in gen_files:
        ok, msg = _compare_files(gen_open / name, exec_open / name)
        if not ok:
            return ok, f"{name}: {msg}"

    return True, "All files identical"


class TestResult:
    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.message}"


def test_equivalence() -> TestResult:
    with tempfile.TemporaryDirectory(prefix="bicep_v2_eq_") as tmp:
        base = Path(tmp)
        gen_dir = base / "gen"
        exec_dir = base / "exec"
        gen_dir.mkdir()
        exec_dir.mkdir()

        # Use same parameters
        seed = GEN_SEED
        num_steps = 10
        num_paths = 4
        num_channels = 4
        audit_k = 8

        # Emit via generator path
        gen_emit(gen_dir, seed_hex=seed, num_steps=num_steps, num_paths=num_paths, num_channels=num_channels, audit_k=audit_k)

        # Emit via production executor
        emit_bicep_v2_run(exec_dir, seed_hex=seed, num_steps=num_steps, num_paths=num_paths, num_channels=num_channels, audit_k=audit_k)

        ok, msg = _compare_runs(gen_dir, exec_dir)
        return TestResult("equivalence_generator_vs_executor", ok, msg)


def test_negative_mutation_caught() -> TestResult:
    with tempfile.TemporaryDirectory(prefix="bicep_v2_eq_neg_") as tmp:
        base = Path(tmp)
        gen_dir = base / "gen"
        exec_dir = base / "exec"
        gen_dir.mkdir()
        exec_dir.mkdir()

        seed = GEN_SEED
        num_steps = 3
        num_paths = 2
        num_channels = 2
        audit_k = 2

        gen_emit(gen_dir, seed_hex=seed, num_steps=num_steps, num_paths=num_paths, num_channels=num_channels, audit_k=audit_k)
        emit_bicep_v2_run(exec_dir, seed_hex=seed, num_steps=num_steps, num_paths=num_paths, num_channels=num_channels, audit_k=audit_k)

        # Mutate one audit leaf in executor output
        target = exec_dir / "audit_openings" / "audit_step_0000.json"
        data = json.loads((target).read_text())
        if data["openings_pre"]:
            data["openings_pre"][0]["value_q"] += 1
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        ok, msg = _compare_runs(gen_dir, exec_dir)
        return TestResult("equivalence_catches_mutation", not ok, "Correctly detected mismatch" if not ok else msg)


def run_all_tests() -> list[TestResult]:
    print("=" * 70)
    print("BICEP_V2 EQUIVALENCE TESTS")
    print("=" * 70)
    print()

    results = [test_equivalence(), test_negative_mutation_caught()]
    for r in results:
        print(r)

    print()
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_all_tests()
    sys.exit(0 if all(r.passed for r in res) else 1)
