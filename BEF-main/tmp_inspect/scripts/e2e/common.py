\
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VerifyResult:
    status: str
    reason_codes: List[str]
    raw: Dict[str, Any]


def repo_root() -> pathlib.Path:
    # scripts/e2e/common.py -> repo root is two parents up
    return pathlib.Path(__file__).resolve().parents[2]


def run_cmd(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err


def run_verifier(
    capsule_path: pathlib.Path,
    profile: str,
    extra_args: Optional[List[str]] = None,
) -> VerifyResult:
    """
    Calls your verifier as a subprocess.

    Expected command (edit here if needed):
      python scripts/verify_capsule.py <capsule> --profile <PROFILE> --json

    Expected output:
      a single JSON object on stdout with keys:
        - status
        - reason_codes (optional list)
        - optionally capsule_hash/payload_hash/header_commit_hash/instance_hash
    """
    extra_args = extra_args or []
    root = repo_root()
    verifier = root / "scripts" / "verify_capsule.py"
    if not verifier.exists():
        raise FileNotFoundError(f"Expected verifier at {verifier}. Edit run_verifier() if different.")

    cmd = [sys.executable, str(verifier), str(capsule_path), "--profile", profile, "--json", *extra_args]
    rc, out, err = run_cmd(cmd, cwd=root)
    if rc != 0 and not out.strip():
        raise RuntimeError(f"Verifier exited {rc} with no JSON output.\nSTDERR:\n{err}")

    # Some verifiers print logs + a JSON line; try last JSON object
    obj = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                break
            except Exception:
                continue
    if obj is None:
        # fallback: try whole stdout as JSON
        obj = json.loads(out)

    status = str(obj.get("status", obj.get("profile_status", "UNKNOWN")))
    reasons = obj.get("reason_codes") or obj.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    reasons = [str(x) for x in reasons]
    return VerifyResult(status=status, reason_codes=reasons, raw=obj)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        # rsync-like copy
        ensure_dir(dst)
        for child in src.rglob("*"):
            rel = child.relative_to(src)
            out = dst / rel
            if child.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(child.read_bytes())
        return True
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return True
