#!/usr/bin/env python3
"""
Capsule Demo - concise, investor-friendly output.

Usage:
  python demo.py emit   ran_computations/trading/momentum_strategy_v2
  python demo.py verify demo.cap
  python demo.py verify ran_computations/trading/momentum_strategy_v2/tampered/tampered.cap

Optional:
  python demo.py verify demo.cap --full   # prints raw JSON too
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict

DEFAULT_PYTHON = ".venv/bin/python"


def _pick_python() -> str:
    return DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON) else sys.executable


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=check,
    )


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def emit(computation_path: str, out_path: str = "demo.cap") -> None:
    py = _pick_python()

    cmd = [
        py, "-m", "bef_zk.capsule.cli", "emit",
        "--source", computation_path,
        "--policy", os.path.join(computation_path, "policy.json"),
        "--out", out_path,
    ]

    p = _run(cmd, check=True)

    stdout = p.stdout or ""
    capsule_id = None
    backend = None
    m = re.search(r"Capsule ID:\s*([0-9a-fA-F]+)", stdout)
    if m:
        capsule_id = m.group(1)
    m = re.search(r"Backend:\s*([A-Za-z0-9_\-]+)", stdout)
    if m:
        backend = m.group(1)

    size = _fmt_bytes(os.path.getsize(out_path)) if os.path.exists(out_path) else "?"

    print(f"EMIT: {out_path} ({size})")
    if capsule_id:
        print(f"  capsule_id={capsule_id}")
    if backend:
        print(f"  backend={backend}")


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def verify(cap_file: str, full: bool = False) -> None:
    py = _pick_python()

    cmd = [py, "-m", "bef_zk.capsule.cli", "verify", cap_file, "--json"]

    p = _run(cmd, check=False)
    raw = (p.stdout or "").strip()

    json_text = raw
    if not raw.startswith("{"):
        start = raw.find("{")
        if start != -1:
            json_text = raw[start:].strip()

    try:
        d = json.loads(json_text)
    except Exception:
        print("VERIFY: ERROR (could not parse JSON)")
        if p.stderr:
            print(p.stderr.strip())
        if raw:
            print(raw)
        sys.exit(2)

    status = d.get("status", "UNKNOWN")
    if status == "REJECT":
        code = d.get("error_code", "E???")
        print(f"REJECT: {code}")
        if full:
            print(json.dumps(d, indent=2))
        return

    backend = d.get("backend_id") or d.get("backend") or "unknown"
    proof_verified = bool(d.get("proof_verified", d.get("policy_verified", False)))

    verify_sec = _safe_get(d, "verify_stats.time_verify_sec", 0.0) or 0.0
    verify_ms = verify_sec * 1000.0

    print(f"STATUS: {status}")
    print(f"PROOF:  {'verified' if proof_verified else 'not-verified'} (backend={backend})")
    print(f"TIME:   {verify_ms:.1f} ms")

    if full:
        print(json.dumps(d, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_emit = sub.add_parser("emit")
    ap_emit.add_argument("computation_path")
    ap_emit.add_argument("--out", default="demo.cap")

    ap_ver = sub.add_parser("verify")
    ap_ver.add_argument("cap_file")
    ap_ver.add_argument("--full", action="store_true", help="also print raw JSON")

    args = ap.parse_args()

    if args.cmd == "emit":
        emit(args.computation_path, out_path=args.out)
    elif args.cmd == "verify":
        verify(args.cap_file, full=args.full)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
