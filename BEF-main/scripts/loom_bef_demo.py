#!/usr/bin/env python3
"""Run the BEF EEG pipeline and wrap the results in a Capseal demo run."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _first_scalar(value: Any) -> float | int | str | bool | None:
    current = value
    while isinstance(current, list) and current:
        current = current[0]
    if isinstance(current, (int, float, str, bool)):
        return current
    return None


def _flatten_scalars(value: Any, limit: int = 256) -> list[float]:
    out: list[float] = []

    def _visit(node: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, (int, float)):
            out.append(float(node))
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                _visit(child)
                if len(out) >= limit:
                    break

    _visit(value)
    return out


def _build_public_inputs(payload: dict[str, Any], *, source: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    entries: list[dict[str, Any]] = []
    summary: dict[str, float] = {}

    def _add(name: str, raw: Any) -> None:
        val = _first_scalar(raw)
        if val is None:
            return
        entries.append({"name": name, "value": val})
        if isinstance(val, (int, float)):
            summary[name] = float(val)

    _add("bef:prediction", payload.get("prediction"))
    _add("bef:rt_prediction", payload.get("rt_prediction"))
    _add("bef:aleatoric_uncertainty", payload.get("aleatoric_uncertainty"))
    _add("bef:epistemic_uncertainty", payload.get("epistemic_uncertainty"))
    _add("bef:total_uncertainty", payload.get("total_uncertainty"))

    attention = payload.get("attention_weights")
    weights = _flatten_scalars(attention, limit=512)
    if weights:
        mean = sum(weights) / len(weights)
        max_val = max(weights)
        max_idx = weights.index(max_val)
        entropy = 0.0
        for w in weights:
            if w > 0:
                entropy -= w * math.log(w)
        entries.append({"name": "bef:attention_mean", "value": mean})
        entries.append({"name": "bef:attention_max", "value": max_val})
        entries.append({"name": "bef:attention_argmax", "value": max_idx})
        entries.append({"name": "bef:attention_entropy", "value": entropy})
        summary["bef:attention_mean"] = mean
        summary["bef:attention_max"] = max_val
        summary["bef:attention_argmax"] = max_idx
    entries.append({"name": "bef:eeg_source", "value": source.name})
    return entries, summary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_bef_pipeline(args: argparse.Namespace, tmpdir: Path) -> Path:
    if args.bef_output and Path(args.bef_output).exists():
        return Path(args.bef_output)
    script = args.bef_scripts_root / "run_bef_eeg.py"
    if not script.exists():
        raise FileNotFoundError(f"BEF EEG runner not found at {script}")
    output = tmpdir / "bef_outputs.json"
    cmd = [
        args.python,
        str(script),
        "--input",
        str(args.bef_input),
        "--output",
        str(output),
    ]
    if args.bef_checkpoint:
        cmd.extend(["--checkpoint", str(args.bef_checkpoint)])
    if args.bef_output_dim:
        cmd.extend(["--output-dim", str(args.bef_output_dim)])
    if args.bef_n_paths:
        cmd.extend(["--n-paths", str(args.bef_n_paths)])
    if args.bef_mc_samples:
        cmd.extend(["--mc-samples", str(args.bef_mc_samples)])
    if args.bef_use_multiscale:
        cmd.append("--use-multiscale")
    if args.bef_return_intermediates:
        cmd.append("--return-intermediates")
    if args.bef_cpu:
        cmd.append("--cpu")
    _run(cmd, cwd=args.bef_scripts_root.parent)
    return output


def run_capsule_bench(args: argparse.Namespace, public_inputs: Path) -> Path:
    run_id = args.run_id or time.strftime("bef_loom_%Y%m%d_%H%M%S")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trace_id = args.trace_id or run_id
    cmd = [
        str(args.capsule_bench),
        "run",
        "--backend",
        args.backend,
        "--policy",
        str(args.policy),
        "--policy-id",
        args.policy_id,
        "--policy-version",
        args.policy_version,
        "--track-id",
        args.track_id,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--trace-id",
        trace_id,
        "--private-key",
        str(args.private_key),
    ]
    if args.manifest_signer_id:
        cmd.extend(["--manifest-signer-id", args.manifest_signer_id])
    if args.manifest_signer_key:
        cmd.extend(["--manifest-signer-key", str(args.manifest_signer_key)])
    if args.docker_image_digest:
        cmd.extend(["--docker-image-digest", args.docker_image_digest])
    if args.allow_insecure_da:
        cmd.append("--allow-insecure-da-challenge")
    cmd.append("--")
    cmd.extend(["--extra-public-inputs", str(public_inputs)])
    if args.pipeline_steps:
        cmd.extend(["--steps", str(args.pipeline_steps)])
    if args.pipeline_num_queries:
        cmd.extend(["--num-queries", str(args.pipeline_num_queries)])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return output_root / run_id


def verify_capsule(run_dir: Path, policy: Path) -> str:
    capsule_path = run_dir / "pipeline" / "strategy_capsule.json"
    manifest_root = run_dir / "manifests"
    cmd = [
        sys.executable,
        "-m",
        "scripts.verify_capsule",
        str(capsule_path),
        "--policy",
        str(policy),
        "--manifest-root",
        str(manifest_root),
    ]
    result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Loom-style demo that fuses BEF EEG outputs into a Capseal run")
    ap.add_argument("--bef-input", type=Path, required=True, help="Path to EEG tensor consumed by run_bef_eeg.py")
    ap.add_argument("--bef-checkpoint", type=Path, help="Optional BEF checkpoint (.pt)")
    ap.add_argument("--bef-output", type=Path, help="Reuse existing BEF outputs JSON instead of rerunning the model")
    ap.add_argument("--bef-output-dim", type=int, default=1)
    ap.add_argument("--bef-n-paths", type=int, default=32)
    ap.add_argument("--bef-mc-samples", type=int, default=1)
    ap.add_argument("--bef-use-multiscale", action="store_true")
    ap.add_argument("--bef-return-intermediates", action="store_true")
    ap.add_argument("--bef-cpu", action="store_true")
    default_bef_root = Path.home() / "scratch" / "CapsuleTech"
    ap.add_argument("--bef-repo", type=Path, default=default_bef_root)
    ap.add_argument(
        "--bef-scripts-root",
        type=Path,
        default=default_bef_root / "scripts",
        help="Directory that contains run_bef_eeg.py",
    )
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "out" / "capsule_runs")
    ap.add_argument("--run-id", type=str)
    ap.add_argument("--trace-id", type=str)
    ap.add_argument("--policy", type=Path, default=REPO_ROOT / "policies" / "demo_policy_v1.json")
    ap.add_argument("--policy-id", default="demo_policy_v1")
    ap.add_argument("--policy-version", default="1.0")
    ap.add_argument("--track-id", default="bef_eeg_demo")
    ap.add_argument("--backend", default="geom")
    ap.add_argument("--docker-image-digest")
    ap.add_argument("--allow-insecure-da", action="store_true")
    ap.add_argument("--pipeline-steps", type=int)
    ap.add_argument("--pipeline-num-queries", type=int)
    ap.add_argument("--private-key", type=Path, default=REPO_ROOT / "demo_assets" / "demo_private_key.hex")
    ap.add_argument("--manifest-signer-id", default="test_manifest")
    ap.add_argument("--manifest-signer-key", default="3" * 64)
    ap.add_argument("--capsule-bench", type=Path, default=REPO_ROOT / ".venv" / "bin" / "capsule-bench")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-verify", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.bef_scripts_root = args.bef_scripts_root.expanduser().resolve()
    args.bef_input = args.bef_input.expanduser().resolve()
    args.policy = args.policy.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.private_key = args.private_key.expanduser().resolve()
    args.capsule_bench = args.capsule_bench.expanduser().resolve()
    if not args.capsule_bench.exists():
        raise FileNotFoundError(f"capsule-bench not found at {args.capsule_bench}")

    with tempfile.TemporaryDirectory(prefix="bef_capsule_") as tmp:
        tmpdir = Path(tmp)
        bef_output = run_bef_pipeline(args, tmpdir)
        payload = _load_json(bef_output)
        public_inputs, summary = _build_public_inputs(payload, source=args.bef_input)
        public_inputs_path = tmpdir / "bef_public_inputs.json"
        public_inputs_path.write_text(json.dumps(public_inputs, indent=2))

        print("BEF EEG summary (embedded as public inputs):")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        run_dir = run_capsule_bench(args, public_inputs_path)
        pipeline_dir = run_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bef_output, pipeline_dir / "bef_outputs.json")
        print(f"CapsuleBench run directory: {run_dir}")

        if not args.skip_verify:
            verification = verify_capsule(run_dir, args.policy)
            print("Verifier output:")
            print(verification)

        print("Done. You can now run `capsule inspect` or `capsule verify` on the generated capsule.")


if __name__ == "__main__":
    main()
