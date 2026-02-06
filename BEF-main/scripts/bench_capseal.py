#!/usr/bin/env python3
"""Simple benchmark harness for CapSeal run/verify.

Runs run_pipeline (scalable sizes) and capseal verify, collecting wall time
and peak RSS. Writes JSON results to out/bench_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies" / "demo_policy_v1.json"


def _peak_rss_bytes(pid: int) -> int:
    """Return peak RSS for a live process by sampling /proc (Linux-only)."""
    peak = 0
    status_path = Path("/proc") / str(pid) / "status"
    while True:
        if not status_path.exists():
            break
        try:
            txt = status_path.read_text()
        except Exception:
            break
        rss_kb = 0
        hwm_kb = 0
        for line in txt.splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    hwm_kb = int(parts[1])
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    rss_kb = int(parts[1])
        if hwm_kb:
            peak = max(peak, hwm_kb * 1024)
        elif rss_kb:
            peak = max(peak, rss_kb * 1024)
        # Sleep a little to reduce overhead
        time.sleep(0.02)
    return peak


def _run_timed(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> Tuple[float, int, int]:
    """Run a command, returning (elapsed_sec, peak_rss_bytes, returncode)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
    peak = _peak_rss_bytes(proc.pid)
    rc = proc.wait()
    elapsed = time.perf_counter() - t0
    return elapsed, peak, rc


def bench_once(size_label: str, steps: int, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Run pipeline with scalable steps
    run_cmd = [
        "python", str(ROOT / "scripts" / "run_pipeline.py"),
        "--backend", "geom",
        "--output-dir", str(out_dir),
        "--trace-id", f"bench_{size_label}",
        "--policy", str(POLICY),
        "--policy-id", "demo_policy_v1",
        "--policy-version", "1.0",
        "--allow-insecure-da-challenge",
        "--verification-profile", "proof_only",
        "--steps", str(steps),
        "--num-queries", "8",
        "--num-challenges", "2",
    ]
    run_time, run_peak, run_rc = _run_timed(run_cmd, cwd=ROOT)
    capsule_path = out_dir / "strategy_capsule.json"
    verify_stats = {}
    verify_time = 0.0
    verify_peak = 0
    if capsule_path.exists():
        ver_cmd = [str(ROOT / "capseal"), "verify", str(capsule_path), "--json"]
        t0 = time.perf_counter()
        proc = subprocess.Popen(ver_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        verify_peak = _peak_rss_bytes(proc.pid)
        out = proc.communicate()[0]
        verify_time = time.perf_counter() - t0
        try:
            payload = json.loads(out)
            verify_stats = payload.get("verify_stats", {})
        except Exception:
            verify_stats = {}
    return {
        "size": size_label,
        "steps": steps,
        "run": {"time_sec": run_time, "peak_rss_bytes": run_peak, "return": run_rc},
        "verify": {"time_sec": verify_time, "peak_rss_bytes": verify_peak, "stats": verify_stats},
        "capsule": str(capsule_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CapSeal run/verify")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "bench_results.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    sizes = [
        ("small", 64),
        ("medium", 512),
        ("large", 2048),
    ]
    results: List[Dict] = []

    # Warmup pass for cold vs warm cache separation
    for label, steps in sizes:
        out_dir = ROOT / "out" / f"bench_{label}"
        results.append({"label": label, "pass": "warmup", **bench_once(label, steps, out_dir)})
    # Measured passes
    for label, steps in sizes:
        out_dir = ROOT / "out" / f"bench_{label}"
        for i in range(args.repeats):
            rec = bench_once(label, steps, out_dir)
            rec["pass"] = f"run_{i+1}"
            results.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()

