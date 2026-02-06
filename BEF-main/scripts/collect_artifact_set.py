#!/usr/bin/env python3
"""Collect artifact hashes and emit fusion_pipeline_artifacts_v1 JSON."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict


def blake2s_file(path: Path) -> str:
    h = hashlib.blake2s()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def blake2s_str(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8")).hexdigest()


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        return out
    except Exception:
        return "unknown"


def git_dirty(repo: Path) -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).decode().strip()
        return bool(status)
    except Exception:
        return False


def command_version(cmd: list) -> str:
    try:
        out = subprocess.check_output(cmd).decode().strip()
        return out
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect artifact hashes for BICEP/ENN/FusionAlpha")
    parser.add_argument("--bicep-root", default="BICEPsrc/BICEPrust/bicep", help="Path to BICEP workspace")
    parser.add_argument("--enn-root", default="enn-cpp", help="Path to ENN workspace")
    parser.add_argument("--fusion-root", default="FusionAlpha", help="Path to FusionAlpha workspace")
    parser.add_argument("--telemetry", required=True, help="Telemetry CSV produced by ENN")
    parser.add_argument("--calibrator", required=True, help="Calibrator JSON file")
    parser.add_argument("--graph-spec", required=False, help="Graph spec JSON file")
    parser.add_argument("--graph-json", required=False, help="Serialized graph JSON file")
    parser.add_argument("--prop-config", required=False, help="Prop config JSON file")
    parser.add_argument("--seed-spec", help="Seed spec path", default="BICEPsrc/BICEPrust/bicep/crates/bicep-core/src/seed.rs")
    parser.add_argument("--output", default="artifact_set.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    telemetry_path = Path(args.telemetry)
    calibrator_path = Path(args.calibrator)
    if not telemetry_path.exists():
        raise FileNotFoundError(telemetry_path)
    if not calibrator_path.exists():
        raise FileNotFoundError(calibrator_path)

    artifact = {
        "schema": "fusion_pipeline_artifacts_v1",
        "bicep": {
            "bicep_version": git_commit(repo_root / args.bicep_root),
            "dirty": git_dirty(repo_root / args.bicep_root),
            "integrator": "EulerMaruyama",
            "model_params_hash": blake2s_str("default"),
            "seed_spec_id": "seed_spec_v1",
            "seed_spec_hash": blake2s_file((repo_root / args.seed_spec) if not Path(args.seed_spec).is_absolute() else Path(args.seed_spec)),
            "trajectory_schema_hash": blake2s_str("bicep_trajectory_v1"),
            "cargo_lock_hash": blake2s_file(repo_root / args.bicep_root / "Cargo.lock") if (repo_root / args.bicep_root / "Cargo.lock").exists() else "missing",
        },
        "enn": {
            "model_id": "demo",
            "weights_hash": blake2s_str("sample_weights"),
            "config_hash": blake2s_str("sample_config"),
            "compile_flags_hash": blake2s_str(os.environ.get("CXXFLAGS", "default")),
            "telemetry_schema_hash": blake2s_str("margin,target,obs_reliability,alpha_entropy,collapse_temperature"),
            "calibrator_id": "unknown",
            "calibrator_hash": blake2s_file(calibrator_path),
        },
        "fusion": {
            "fusion_version": git_commit(repo_root / args.fusion_root),
            "dirty": git_dirty(repo_root / args.fusion_root),
            "cargo_lock_hash": blake2s_file(repo_root / args.fusion_root / "Cargo.lock") if (repo_root / args.fusion_root / "Cargo.lock").exists() else "missing",
            "graph_spec_id": "graph_spec_v1" if args.graph_spec else "unknown",
            "graph_spec_hash": blake2s_file(Path(args.graph_spec)) if args.graph_spec else "unknown",
            "graph_hash": blake2s_file(Path(args.graph_json)) if args.graph_json else "unknown",
            "prop_config_hash": blake2s_file(Path(args.prop_config)) if args.prop_config else blake2s_str("default"),
        },
        "features": {
            "feature_schema_hash": blake2s_str("state_mean,state_std,state_q10,state_q90,aleatoric_unc,epistemic_unc"),
        },
        "toolchain": {
            "compiler": os.environ.get("CXX", "unknown"),
            "cxxflags": os.environ.get("CXXFLAGS", "unknown"),
            "ldflags": os.environ.get("LDFLAGS", "unknown"),
            "os_fingerprint": os.uname().sysname,
            "threads": {
                "omp": int(os.environ.get("OMP_NUM_THREADS", "0")),
                "eigen": int(os.environ.get("EIGEN_NUM_THREADS", "0")) if os.environ.get("EIGEN_NUM_THREADS") else 0,
                "mkl": int(os.environ.get("MKL_NUM_THREADS", "0")) if os.environ.get("MKL_NUM_THREADS") else 0,
            },
            "versions": {
                "python": command_version(["python", "--version"]),
                "gcc": command_version(["g++", "--version"]),
                "rustc": command_version(["rustc", "--version"]),
                "cargo": command_version(["cargo", "--version"]),
            },
        },
        "binaries": {},
        "python": {},
    }

    # Read calibrator ID if available
    try:
        with calibrator_path.open("r", encoding="utf-8") as fh:
            calib_data = json.load(fh)
        artifact["enn"]["calibrator_id"] = calib_data.get("calibrator_id", "unknown")
    except Exception:
        pass

    # Binary hashes if present
    enn_binary = repo_root / args.enn_root / "apps" / "bicep_to_enn"
    if enn_binary.exists():
        artifact["binaries"]["bicep_to_enn"] = blake2s_file(enn_binary)
    fusion_lib = repo_root / args.fusion_root / "target" / "release" / "libfusion_alpha.so"
    if fusion_lib.exists():
        artifact["binaries"]["libfusion_alpha"] = blake2s_file(fusion_lib)
    reqs = repo_root / "requirements.txt"
    if reqs.exists():
        artifact.setdefault("python", {})["requirements_hash"] = blake2s_file(reqs)

    # Compute artifact set hash
    def flatten(d: Dict, prefix: str = "") -> list:
        items = []
        for key in sorted(d.keys()):
            value = d[key]
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(flatten(value, name))
            else:
                items.append(f"{name}={value}")
        return items

    concat = "|".join(flatten(artifact))
    artifact["artifact_set_hash"] = blake2s_str(concat)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"Wrote artifact set to {output_path}")


if __name__ == "__main__":
    main()
