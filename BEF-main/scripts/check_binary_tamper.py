#!/usr/bin/env python3
"""Generate binary artifacts and ensure tampered copies fail verification."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import json

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.codec import canonical_decode
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.serialization import proof_from_bytes
from bef_zk.zk_geom.verifier import zk_verify_geom
from scripts.geom_programs import GEOM_PROGRAM
from scripts.verify_capsule import verify_capsule

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    repo = str(REPO_ROOT)
    env["PYTHONPATH"] = repo if not existing else f"{repo}:{existing}"
    return env


def _run_pipeline(out_dir: Path, steps: int, num_queries: int) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_geom_pipeline.py"),
        "--steps",
        str(steps),
        "--num-queries",
        str(num_queries),
        "--output-dir",
        str(out_dir),
        "--skip-nova",
    ]
    subprocess.run(cmd, check=True, env=_python_env(), cwd=REPO_ROOT)


def _load_capsule(path: Path) -> dict:
    if path.suffix == ".bin":
        return canonical_decode(path.read_bytes())  # type: ignore[return-value]
    return json.loads(path.read_text())


def _geom_params(capsule: dict) -> GeomAIRParams:
    params = capsule["params"]
    return GeomAIRParams(
        steps=int(params["steps"]),
        num_challenges=int(params["num_challenges"]),
        r_challenges=[int(x) for x in params["r_challenges"]],
        matrix=[[2, 1], [1, 1]],
    )


def _verify_bytes(capsule: dict, proof_bytes: bytes) -> bool:
    params = _geom_params(capsule)
    init = GeomInitialState()
    row_width = int(capsule["params"]["row_width"])
    vc = STCVectorCommitment(chunk_len=row_width)
    proof = proof_from_bytes(proof_bytes)
    ok, _ = zk_verify_geom(GEOM_PROGRAM, params, init, vc, proof)
    return ok


def _flip_offsets(length: int, count: int = 4) -> Sequence[int]:
    if length == 0:
        return []
    step = max(1, length // count)
    return [(i * step) % length for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary tamper regression check")
    parser.add_argument("--output-dir", type=Path, default=Path("out/tamper_check"))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=16)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_pipeline(out_dir, args.steps, args.num_queries)

    capsule_json = out_dir / "strategy_capsule.json"
    capsule_data = _load_capsule(capsule_json)
    proof_bin = out_dir / "geom_proof.bin"
    proof_bytes = proof_bin.read_bytes()

    # Baseline verification via CLI helper
    verify_capsule(capsule_json)

    print("[tamper] Baseline proof verified. Beginning tamper checks…")
    offsets = _flip_offsets(len(proof_bytes))
    if not offsets:
        raise SystemExit("proof bin empty; cannot tamper")

    for idx, offset in enumerate(offsets):
        mutated = bytearray(proof_bytes)
        mutated[offset] ^= 0xFF
        try:
            ok = _verify_bytes(capsule_data, bytes(mutated))
        except Exception:
            ok = False
        if ok:
            raise RuntimeError(f"tampered proof #{idx} unexpectedly verified")
        print(f"  - Tamper #{idx} rejected (offset {offset})")

    capsule_bin = out_dir / "strategy_capsule.bin"
    if capsule_bin.exists():
        data = bytearray(capsule_bin.read_bytes())
        data[0] ^= 0x01
        tampered_capsule = out_dir / "strategy_capsule.tampered.bin"
        tampered_capsule.write_bytes(bytes(data))
        try:
            verify_capsule(tampered_capsule)
        except Exception:
            print("  - Tampered capsule rejected as expected")
        else:
            raise RuntimeError("tampered capsule unexpectedly verified")
        finally:
            tampered_capsule.unlink(missing_ok=True)

    print("[tamper] All tamper checks passed.")


if __name__ == "__main__":
    main()
