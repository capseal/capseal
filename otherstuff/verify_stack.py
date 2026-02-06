#!/usr/bin/env python3
"""Stack verification harness - proves which implementations are actually running.

Produces evidence that the FULL versions (not stubs) are being executed:
- ENN: Full entangled network (not simple MLP)
- BICEP: Real Rust SDE simulator (not synthetic binomial sampler)
- FusionAlpha: Rust backend (not Python fallback)

Usage:
    python -m verify_stack --repo-root . --run-dir artifacts/latest_bicep
    python -m verify_stack --dry-run  # Check without running
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fingerprinting utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def get_git_info(repo_path: Path) -> Dict[str, Any]:
    """Get git commit hash and dirty status."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        dirty_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

        return {
            "commit": commit,
            "is_dirty": len(dirty_files) > 0,
            "dirty_files": dirty_files[:10],  # Limit to first 10
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Evidence collection dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ComponentEvidence:
    """Evidence for a single component."""
    component: str
    verdict: str  # PASS, FAIL, WARN
    implementation: str  # "full", "stub", "unknown"
    reason: str

    entrypoint_path: Optional[str] = None
    module_name: Optional[str] = None
    binary_path: Optional[str] = None

    code_fingerprint: Optional[str] = None
    fingerprint_sources: List[str] = field(default_factory=list)

    git_info: Dict[str, Any] = field(default_factory=dict)

    feature_witness: Dict[str, Any] = field(default_factory=dict)
    architecture_dump: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceReport:
    """Complete provenance report."""
    timestamp: str
    repo_root: str
    run_dir: str

    enn: ComponentEvidence = field(default_factory=lambda: ComponentEvidence("enn", "UNKNOWN", "unknown", "Not checked"))
    bicep: ComponentEvidence = field(default_factory=lambda: ComponentEvidence("bicep", "UNKNOWN", "unknown", "Not checked"))
    fusion: ComponentEvidence = field(default_factory=lambda: ComponentEvidence("fusion", "UNKNOWN", "unknown", "Not checked"))

    integration_checks: Dict[str, Any] = field(default_factory=dict)
    overall_verdict: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "repo_root": self.repo_root,
            "run_dir": self.run_dir,
            "enn": asdict(self.enn),
            "bicep": asdict(self.bicep),
            "fusion": asdict(self.fusion),
            "integration_checks": self.integration_checks,
            "overall_verdict": self.overall_verdict,
        }


# ---------------------------------------------------------------------------
# ENN Verification
# ---------------------------------------------------------------------------


def verify_enn(repo_root: Path, run_dir: Path, dry_run: bool = False) -> ComponentEvidence:
    """Verify which ENN implementation would be/is used."""
    evidence = ComponentEvidence(
        component="enn",
        verdict="UNKNOWN",
        implementation="unknown",
        reason="Not yet checked",
    )

    # Check which training script exists and would be selected
    train_enn_path = repo_root / "train_enn.py"
    train_simple_path = repo_root / "train_simple_enn.py"

    # Determine which script would be used (same logic as loop_runner.py)
    if train_enn_path.exists():
        selected_script = train_enn_path
        evidence.extra["selection_reason"] = "train_enn.py exists (preferred)"
    elif train_simple_path.exists():
        selected_script = train_simple_path
        evidence.extra["selection_reason"] = "train_enn.py not found, using train_simple_enn.py"
    else:
        evidence.verdict = "FAIL"
        evidence.reason = "No training script found!"
        return evidence

    evidence.entrypoint_path = str(selected_script.resolve())
    evidence.code_fingerprint = sha256_file(selected_script)
    evidence.fingerprint_sources = [str(selected_script)]
    evidence.git_info = get_git_info(repo_root)

    # Load and inspect the module to find the model class
    try:
        spec = importlib.util.spec_from_file_location("train_module", selected_script)
        module = importlib.util.module_from_spec(spec)

        # We need torch to be importable
        import torch
        import torch.nn as nn

        spec.loader.exec_module(module)

        # Find the model class
        model_class = None
        model_class_name = None

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                # Prefer ENN over SimpleENN
                if name == "ENN":
                    model_class = obj
                    model_class_name = name
                    break
                elif name == "SimpleENN":
                    model_class = obj
                    model_class_name = name
                # Keep looking for ENN

        if model_class is None:
            evidence.verdict = "FAIL"
            evidence.reason = "No nn.Module subclass found in training script"
            return evidence

        evidence.module_name = f"{module.__name__}.{model_class_name}"
        evidence.extra["model_class_name"] = model_class_name

        # Feature witness: Check for ENN-specific components
        feature_witness = {
            "model_class": model_class_name,
            "has_entangled_cell": False,
            "has_collapse": False,
            "has_psd_matrix": False,
            "has_lambda_param": False,
            "has_mc_dropout": False,
            "is_simple_mlp": False,
        }

        # Check source code for ENN-specific patterns
        # Read the file directly instead of using inspect.getsource (more reliable)
        try:
            source = selected_script.read_text()
        except Exception:
            source = ""

        # EntangledCell presence
        if "EntangledCell" in source or "entangle" in source.lower():
            feature_witness["has_entangled_cell"] = True

        # Collapse mechanism
        if "Collapse" in source or "attention" in source.lower():
            feature_witness["has_collapse"] = True

        # PSD matrix (E = L @ L.T)
        if "L @ L.T" in source or "L.T" in source:
            feature_witness["has_psd_matrix"] = True

        # Lambda parameter
        if "lambda" in source.lower() or "log_lambda" in source:
            feature_witness["has_lambda_param"] = True

        # MC dropout
        if "mc_dropout" in source.lower() or "n_samples" in source:
            feature_witness["has_mc_dropout"] = True

        # Simple MLP check (only Linear/ReLU/etc)
        if model_class_name == "SimpleENN":
            feature_witness["is_simple_mlp"] = True

        evidence.feature_witness = feature_witness

        # Try to instantiate and get architecture
        try:
            model = model_class(input_dim=2)
            evidence.architecture_dump = repr(model)

            # Get parameter breakdown
            param_breakdown = {}
            total_params = 0
            for name, param in model.named_parameters():
                param_breakdown[name] = {
                    "shape": list(param.shape),
                    "numel": param.numel(),
                }
                total_params += param.numel()
            evidence.extra["param_breakdown"] = param_breakdown
            evidence.extra["total_params"] = total_params

            # Check for ENN-specific attributes (runtime verification)
            if hasattr(model, "cells"):
                evidence.extra["num_entangled_layers"] = len(model.cells)
                feature_witness["has_entangled_cell"] = True
            if hasattr(model, "k"):
                evidence.extra["entanglement_dim_k"] = model.k
            if hasattr(model, "collapse"):
                feature_witness["has_collapse"] = True
            if hasattr(model, "get_lambda_values"):
                try:
                    lambdas = model.get_lambda_values()
                    evidence.extra["lambda_values"] = lambdas
                    feature_witness["has_lambda_param"] = True
                except:
                    pass
            if hasattr(model, "get_entanglement_matrices"):
                feature_witness["has_psd_matrix"] = True
            if hasattr(model, "mc_dropout_inference"):
                feature_witness["has_mc_dropout"] = True

            # Check if it's a simple MLP by absence of ENN features
            if model_class_name == "SimpleENN":
                feature_witness["is_simple_mlp"] = True
            elif not any([
                feature_witness["has_entangled_cell"],
                feature_witness["has_collapse"],
                feature_witness["has_psd_matrix"],
            ]):
                # Has none of the key ENN features
                feature_witness["is_simple_mlp"] = True

        except Exception as e:
            evidence.extra["instantiation_error"] = str(e)

        # Determine verdict
        is_full_enn = (
            feature_witness["has_entangled_cell"] or
            feature_witness["has_collapse"] or
            feature_witness["has_psd_matrix"]
        )

        if is_full_enn and not feature_witness["is_simple_mlp"]:
            evidence.verdict = "PASS"
            evidence.implementation = "full"
            evidence.reason = f"Full ENN detected: {model_class_name} with entanglement/collapse"
        elif feature_witness["is_simple_mlp"]:
            evidence.verdict = "FAIL"
            evidence.implementation = "stub"
            evidence.reason = f"Simple MLP detected (not full ENN): {model_class_name}"
        else:
            evidence.verdict = "WARN"
            evidence.implementation = "unknown"
            evidence.reason = f"Could not confirm ENN features in {model_class_name}"

    except ImportError as e:
        evidence.verdict = "FAIL"
        evidence.reason = f"Import error: {e}"
    except Exception as e:
        evidence.verdict = "FAIL"
        evidence.reason = f"Error inspecting ENN: {e}"

    return evidence


# ---------------------------------------------------------------------------
# BICEP Verification
# ---------------------------------------------------------------------------


def verify_bicep(repo_root: Path, run_dir: Path, dry_run: bool = False) -> ComponentEvidence:
    """Verify which BICEP implementation would be/is used."""
    evidence = ComponentEvidence(
        component="bicep",
        verdict="UNKNOWN",
        implementation="unknown",
        reason="Not yet checked",
    )

    # Check for Rust binary
    binary_path = repo_root / "BICEPsrc/BICEPrust/bicep/target/release/double_well_point"
    run_bicep_path = repo_root / "run_bicep.py"
    loop_runner_path = repo_root / "loop_runner.py"

    evidence.fingerprint_sources = []

    # Check binary existence
    if binary_path.exists():
        evidence.binary_path = str(binary_path.resolve())
        evidence.code_fingerprint = sha256_file(binary_path)
        evidence.fingerprint_sources.append(str(binary_path))

        # Get binary info
        stat = binary_path.stat()
        evidence.extra["binary_size"] = stat.st_size
        evidence.extra["binary_mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Check for Cargo.lock
        cargo_lock = repo_root / "BICEPsrc/BICEPrust/bicep/Cargo.lock"
        if cargo_lock.exists():
            evidence.extra["cargo_lock_fingerprint"] = sha256_file(cargo_lock)[:16]
            evidence.fingerprint_sources.append(str(cargo_lock))
    else:
        evidence.extra["binary_exists"] = False
        evidence.extra["expected_binary_path"] = str(binary_path)

    # Check run_bicep.py
    if run_bicep_path.exists():
        evidence.entrypoint_path = str(run_bicep_path.resolve())
        rb_fingerprint = sha256_file(run_bicep_path)
        evidence.fingerprint_sources.append(str(run_bicep_path))
        evidence.extra["run_bicep_fingerprint"] = rb_fingerprint[:16]

    evidence.git_info = get_git_info(repo_root / "BICEPsrc/BICEPrust/bicep")

    # Feature witness
    feature_witness = {
        "rust_binary_exists": binary_path.exists(),
        "binary_size_bytes": binary_path.stat().st_size if binary_path.exists() else 0,
        "has_sde_integrator": False,
        "integrator_params": {},
        "uses_real_dynamics": False,
    }

    # Analyze run_bicep.py for integrator details
    if run_bicep_path.exists():
        source = run_bicep_path.read_text()

        # Check for SDE parameters
        if "steps" in source and "dt" in source and "temperature" in source:
            feature_witness["has_sde_integrator"] = True

        # Extract default parameters
        import re
        for param in ["steps", "dt", "temperature", "left_threshold", "right_threshold"]:
            match = re.search(rf'{param}[:\s]*=?\s*([0-9.e\-]+)', source)
            if match:
                try:
                    feature_witness["integrator_params"][param] = float(match.group(1))
                except:
                    pass

        # Check if it uses subprocess to call Rust binary
        if "subprocess" in source and "BICEP_POINT_BINARY" in source:
            feature_witness["uses_real_dynamics"] = True

    evidence.feature_witness = feature_witness

    # Check loop_runner.py for synthetic fallback
    if loop_runner_path.exists():
        lr_source = loop_runner_path.read_text()
        if "_run_synthetic_bicep" in lr_source:
            evidence.extra["has_synthetic_fallback"] = True
            # Check the default use_synthetic flag
            if "use_synthetic: bool = False" in lr_source:
                evidence.extra["default_is_real"] = True
            elif "use_synthetic: bool = True" in lr_source:
                evidence.extra["default_is_synthetic"] = True

    # Determine verdict
    if binary_path.exists() and feature_witness["uses_real_dynamics"]:
        evidence.verdict = "PASS"
        evidence.implementation = "full"
        evidence.reason = f"Rust BICEP binary found ({evidence.extra.get('binary_size', 0):,} bytes) with SDE integrator"
    elif binary_path.exists():
        evidence.verdict = "WARN"
        evidence.implementation = "unknown"
        evidence.reason = "Rust binary exists but could not confirm it's being used"
    else:
        evidence.verdict = "FAIL"
        evidence.implementation = "stub"
        evidence.reason = f"Rust binary not found at {binary_path}"

    # If we can run a quick test
    if not dry_run and binary_path.exists():
        try:
            result = subprocess.run(
                [str(binary_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 or "Usage" in result.stdout or "double_well" in result.stdout.lower():
                evidence.extra["binary_runs"] = True
                evidence.extra["help_output_preview"] = result.stdout[:200]
            else:
                evidence.extra["binary_runs"] = False
                evidence.extra["binary_stderr"] = result.stderr[:200]
        except Exception as e:
            evidence.extra["binary_test_error"] = str(e)

    return evidence


# ---------------------------------------------------------------------------
# FusionAlpha Verification
# ---------------------------------------------------------------------------


def verify_fusion(repo_root: Path, run_dir: Path, dry_run: bool = False) -> ComponentEvidence:
    """Verify which FusionAlpha implementation would be/is used."""
    evidence = ComponentEvidence(
        component="fusion",
        verdict="UNKNOWN",
        implementation="unknown",
        reason="Not yet checked",
    )

    fuse_field_path = repo_root / "fuse_field.py"
    fusion_lib_path = repo_root / "FusionAlpha/target/release"
    fusion_utils_path = repo_root / "fusion_utils.py"

    evidence.fingerprint_sources = []

    # Check fuse_field.py
    if fuse_field_path.exists():
        evidence.entrypoint_path = str(fuse_field_path.resolve())
        evidence.code_fingerprint = sha256_file(fuse_field_path)
        evidence.fingerprint_sources.append(str(fuse_field_path))

    # Check for Rust library
    rust_so = fusion_lib_path / "libfusion_alpha.so"
    rust_pyd = fusion_lib_path / "fusion_alpha.pyd"

    rust_lib = None
    if rust_so.exists():
        rust_lib = rust_so
    elif rust_pyd.exists():
        rust_lib = rust_pyd

    feature_witness = {
        "rust_library_exists": rust_lib is not None,
        "rust_library_path": str(rust_lib) if rust_lib else None,
        "python_fallback_exists": fusion_utils_path.exists(),
        "runtime_backend": None,
        "backend_id": None,
    }

    if rust_lib:
        evidence.binary_path = str(rust_lib.resolve())
        rust_fingerprint = sha256_file(rust_lib)
        evidence.fingerprint_sources.append(str(rust_lib))
        evidence.extra["rust_lib_fingerprint"] = rust_fingerprint[:16]
        evidence.extra["rust_lib_size"] = rust_lib.stat().st_size

    evidence.git_info = get_git_info(repo_root / "FusionAlpha")

    # Try to actually import and check runtime
    if not dry_run:
        try:
            # First try importing from existing Python path (e.g., installed in venv)
            import fusion_alpha as fa
            feature_witness["runtime_backend"] = "rust"
            evidence.module_name = fa.__name__

            # Try to get backend ID or version
            if hasattr(fa, "__version__"):
                feature_witness["backend_id"] = fa.__version__
            if hasattr(fa, "backend_id"):
                feature_witness["backend_id"] = fa.backend_id()
            if hasattr(fa, "__file__"):
                evidence.extra["rust_module_file"] = fa.__file__

            # Check available functions
            available_funcs = [name for name in dir(fa) if not name.startswith("_")]
            feature_witness["available_functions"] = available_funcs

            # Verify propagate_field exists
            if hasattr(fa, "propagate_field"):
                feature_witness["has_propagate_field"] = True
                sig = inspect.signature(fa.propagate_field) if callable(fa.propagate_field) else None
                if sig:
                    feature_witness["propagate_field_params"] = list(sig.parameters.keys())

            # Update binary path to the actual loaded module location
            if hasattr(fa, "__file__") and fa.__file__:
                module_dir = Path(fa.__file__).parent
                # Look for the actual .so file
                for so_file in module_dir.glob("*.so"):
                    evidence.binary_path = str(so_file.resolve())
                    evidence.extra["rust_lib_fingerprint"] = sha256_file(so_file)[:16]
                    evidence.extra["rust_lib_size"] = so_file.stat().st_size
                    if str(so_file) not in evidence.fingerprint_sources:
                        evidence.fingerprint_sources.append(str(so_file))
                    break

        except ImportError as e:
            feature_witness["runtime_backend"] = "python_fallback"
            feature_witness["import_error"] = str(e)

            # Check if Python fallback would be used
            if fusion_utils_path.exists():
                evidence.extra["python_fallback_fingerprint"] = sha256_file(fusion_utils_path)[:16]
        except Exception as e:
            feature_witness["runtime_check_error"] = str(e)

    evidence.feature_witness = feature_witness

    # Determine verdict based on runtime check
    if feature_witness["runtime_backend"] == "rust":
        evidence.verdict = "PASS"
        evidence.implementation = "full"
        evidence.reason = "Rust FusionAlpha backend loaded successfully"
    elif feature_witness["rust_library_exists"] and not dry_run:
        evidence.verdict = "FAIL"
        evidence.implementation = "stub"
        evidence.reason = f"Rust library exists but import failed: {feature_witness.get('import_error', 'unknown')}"
    elif feature_witness["rust_library_exists"]:
        evidence.verdict = "WARN"
        evidence.implementation = "unknown"
        evidence.reason = "Rust library exists (dry-run mode, not verified)"
    else:
        evidence.verdict = "FAIL"
        evidence.implementation = "stub"
        evidence.reason = "No Rust library found, would use Python fallback"

    return evidence


# ---------------------------------------------------------------------------
# Integration checks
# ---------------------------------------------------------------------------


def check_integration(repo_root: Path, run_dir: Path) -> Dict[str, Any]:
    """Check end-to-end integration between components."""
    checks = {
        "artifacts_exist": {},
        "dataflow_valid": False,
        "artifact_fingerprints": {},
    }

    # Check which artifacts exist
    artifacts = [
        "grid.npz",
        "beta_posteriors.npz",
        "enn.npz",
        "enn.pt",
        "fusion.npz",
        "active_sampling_plan.json",
    ]

    for artifact in artifacts:
        path = run_dir / artifact
        checks["artifacts_exist"][artifact] = path.exists()
        if path.exists():
            checks["artifact_fingerprints"][artifact] = sha256_file(path)[:16]

    # Check dataflow validity
    required_for_flow = ["grid.npz", "enn.npz"]
    checks["dataflow_valid"] = all(
        checks["artifacts_exist"].get(a, False) for a in required_for_flow
    )

    # Check fusion.npz for backend indicator
    fusion_path = run_dir / "fusion.npz"
    if fusion_path.exists():
        try:
            import numpy as np
            data = np.load(fusion_path, allow_pickle=True)
            if "backend" in data.files:
                checks["fusion_recorded_backend"] = str(data["backend"])
        except Exception as e:
            checks["fusion_read_error"] = str(e)

    # Check enn_config.json for training info
    config_path = run_dir / "enn_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            checks["enn_config"] = {
                "k": config.get("k"),
                "n_layers": config.get("n_layers"),
                "seed": config.get("seed"),
            }
        except Exception as e:
            checks["enn_config_error"] = str(e)

    return checks


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_report(report: ProvenanceReport) -> str:
    """Generate a detailed markdown report."""
    lines = []

    lines.append("# Runtime Provenance Report")
    lines.append(f"\n**Generated:** {report.timestamp}")
    lines.append(f"**Repo Root:** `{report.repo_root}`")
    lines.append(f"**Run Directory:** `{report.run_dir}`")
    lines.append(f"\n## Overall Verdict: **{report.overall_verdict}**\n")

    # Summary table
    lines.append("## Component Summary\n")
    lines.append("| Component | Verdict | Implementation | Key Evidence |")
    lines.append("|-----------|---------|----------------|--------------|")

    for comp in [report.enn, report.bicep, report.fusion]:
        verdict_emoji = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}.get(comp.verdict, "?")
        evidence = comp.code_fingerprint[:8] if comp.code_fingerprint else "N/A"
        lines.append(f"| {comp.component.upper()} | {verdict_emoji} | {comp.implementation} | `{evidence}...` |")

    lines.append("\n---\n")

    # Detailed sections
    for comp in [report.enn, report.bicep, report.fusion]:
        lines.append(f"## {comp.component.upper()} Details\n")
        lines.append(f"**Verdict:** {comp.verdict}")
        lines.append(f"**Implementation:** {comp.implementation}")
        lines.append(f"**Reason:** {comp.reason}\n")

        if comp.entrypoint_path:
            lines.append(f"**Entrypoint:** `{comp.entrypoint_path}`")
        if comp.binary_path:
            lines.append(f"**Binary:** `{comp.binary_path}`")
        if comp.module_name:
            lines.append(f"**Module:** `{comp.module_name}`")
        if comp.code_fingerprint:
            lines.append(f"**Fingerprint:** `{comp.code_fingerprint}`")

        lines.append("\n### Fingerprint Sources\n")
        for src in comp.fingerprint_sources:
            lines.append(f"- `{src}`")

        if comp.git_info:
            lines.append("\n### Git Info\n")
            lines.append(f"- Commit: `{comp.git_info.get('commit', 'N/A')}`")
            lines.append(f"- Dirty: {comp.git_info.get('is_dirty', 'N/A')}")

        if comp.feature_witness:
            lines.append("\n### Feature Witness\n")
            lines.append("```json")
            lines.append(json.dumps(comp.feature_witness, indent=2, default=str))
            lines.append("```")

        if comp.architecture_dump:
            lines.append("\n### Architecture Dump\n")
            lines.append("```")
            lines.append(comp.architecture_dump[:2000])
            if len(comp.architecture_dump) > 2000:
                lines.append("... (truncated)")
            lines.append("```")

        lines.append("\n---\n")

    # Integration checks
    lines.append("## Integration Checks\n")
    lines.append("```json")
    lines.append(json.dumps(report.integration_checks, indent=2, default=str))
    lines.append("```\n")

    # How to reproduce
    lines.append("## How to Reproduce\n")
    lines.append("```bash")
    lines.append(f"python -m verify_stack --repo-root {report.repo_root} --run-dir {report.run_dir}")
    lines.append("```\n")

    # If FAIL, how to fix
    failures = []
    if report.enn.verdict == "FAIL":
        failures.append(("ENN", report.enn.reason, "Ensure train_enn.py exists and contains the full ENN class"))
    if report.bicep.verdict == "FAIL":
        failures.append(("BICEP", report.bicep.reason, "Build BICEP: cd BICEPsrc/BICEPrust/bicep && cargo build --release -p bicep-examples"))
    if report.fusion.verdict == "FAIL":
        failures.append(("FusionAlpha", report.fusion.reason, "Build FusionAlpha: cd FusionAlpha && cargo build --release -p fusion-bindings"))

    if failures:
        lines.append("## How to Fix Failures\n")
        for name, reason, fix in failures:
            lines.append(f"### {name}\n")
            lines.append(f"**Problem:** {reason}\n")
            lines.append(f"**Fix:** {fix}\n")

    return "\n".join(lines)


def print_summary_table(report: ProvenanceReport) -> None:
    """Print a concise summary table to console."""
    print()
    print("=" * 90)
    print(" STACK VERIFICATION RESULTS ".center(90))
    print("=" * 90)
    print()
    print(f"{'Component':<12} | {'Verdict':<6} | {'Impl':<8} | {'Fingerprint':<18} | {'Reason':<30}")
    print("-" * 90)

    for comp in [report.enn, report.bicep, report.fusion]:
        fp = comp.code_fingerprint[:16] if comp.code_fingerprint else "N/A"
        reason = comp.reason[:28] + ".." if len(comp.reason) > 30 else comp.reason
        print(f"{comp.component.upper():<12} | {comp.verdict:<6} | {comp.implementation:<8} | {fp:<18} | {reason:<30}")

    print("-" * 90)
    print(f"\nOVERALL: {report.overall_verdict}")
    print()


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------


def run_verification(
    repo_root: Path,
    run_dir: Path,
    dry_run: bool = False,
) -> ProvenanceReport:
    """Run full stack verification."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = ProvenanceReport(
        timestamp=timestamp,
        repo_root=str(repo_root.resolve()),
        run_dir=str(run_dir.resolve()),
    )

    print("[Verify] Checking ENN implementation...")
    report.enn = verify_enn(repo_root, run_dir, dry_run)

    print("[Verify] Checking BICEP implementation...")
    report.bicep = verify_bicep(repo_root, run_dir, dry_run)

    print("[Verify] Checking FusionAlpha implementation...")
    report.fusion = verify_fusion(repo_root, run_dir, dry_run)

    print("[Verify] Running integration checks...")
    report.integration_checks = check_integration(repo_root, run_dir)

    # Determine overall verdict
    verdicts = [report.enn.verdict, report.bicep.verdict, report.fusion.verdict]
    if all(v == "PASS" for v in verdicts):
        report.overall_verdict = "PASS - All components verified as FULL implementations"
    elif any(v == "FAIL" for v in verdicts):
        fails = [c for c, v in zip(["ENN", "BICEP", "FUSION"], verdicts) if v == "FAIL"]
        report.overall_verdict = f"FAIL - Stub/fallback detected in: {', '.join(fails)}"
    else:
        report.overall_verdict = "WARN - Some components could not be fully verified"

    return report


def save_provenance(report: ProvenanceReport, run_dir: Path) -> Path:
    """Save provenance report and return the directory."""
    provenance_dir = run_dir / "provenance" / report.timestamp
    provenance_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = provenance_dir / "provenance_report.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    # Save markdown
    md_path = provenance_dir / "provenance_report.md"
    md_path.write_text(generate_markdown_report(report))

    # Save fingerprints separately
    fingerprints = {
        "enn": {
            "fingerprint": report.enn.code_fingerprint,
            "sources": report.enn.fingerprint_sources,
        },
        "bicep": {
            "fingerprint": report.bicep.code_fingerprint,
            "sources": report.bicep.fingerprint_sources,
        },
        "fusion": {
            "fingerprint": report.fusion.code_fingerprint,
            "sources": report.fusion.fingerprint_sources,
        },
    }
    fp_path = provenance_dir / "fingerprints.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprints, f, indent=2)

    # Save architecture.json for ENN
    if report.enn.extra.get("param_breakdown"):
        arch_path = provenance_dir / "architecture.json"
        with open(arch_path, "w") as f:
            json.dump({
                "model_class": report.enn.extra.get("model_class_name"),
                "total_params": report.enn.extra.get("total_params"),
                "param_breakdown": report.enn.extra.get("param_breakdown"),
                "feature_witness": report.enn.feature_witness,
            }, f, indent=2)

    # Save BICEP manifest
    bicep_manifest = {
        "binary_path": report.bicep.binary_path,
        "fingerprint": report.bicep.code_fingerprint,
        "integrator_params": report.bicep.feature_witness.get("integrator_params", {}),
        "uses_real_dynamics": report.bicep.feature_witness.get("uses_real_dynamics"),
    }
    bicep_path = provenance_dir / "bicep_run_manifest.json"
    with open(bicep_path, "w") as f:
        json.dump(bicep_manifest, f, indent=2)

    # Save Fusion manifest
    fusion_manifest = {
        "backend": report.fusion.feature_witness.get("runtime_backend"),
        "library_path": report.fusion.binary_path,
        "fingerprint": report.fusion.code_fingerprint,
        "backend_id": report.fusion.feature_witness.get("backend_id"),
    }
    fusion_path = provenance_dir / "fusion_manifest.json"
    with open(fusion_path, "w") as f:
        json.dump(fusion_manifest, f, indent=2)

    return provenance_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that full implementations (not stubs) are being used"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root directory",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="artifacts/latest_bicep",
        help="Run directory with artifacts",
    )
    parser.add_argument(
        "--plan",
        type=str,
        help="Path to sampling plan (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check without running actual imports/binaries",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON report to stdout instead of saving",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = (repo_root / args.run_dir).resolve()

    if not repo_root.exists():
        print(f"Error: repo-root does not exist: {repo_root}")
        sys.exit(1)

    if not run_dir.exists():
        print(f"Warning: run-dir does not exist: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

    # Run verification
    report = run_verification(repo_root, run_dir, dry_run=args.dry_run)

    if args.json_only:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        # Print summary table
        print_summary_table(report)

        # Save reports
        provenance_dir = save_provenance(report, run_dir)

        print(f"Provenance report saved to: {provenance_dir}")
        print(f"  - provenance_report.md")
        print(f"  - provenance_report.json")
        print(f"  - fingerprints.json")
        print(f"  - architecture.json")
        print(f"  - bicep_run_manifest.json")
        print(f"  - fusion_manifest.json")

    # Exit with appropriate code
    if "FAIL" in report.overall_verdict:
        sys.exit(1)
    elif "WARN" in report.overall_verdict:
        sys.exit(0)  # Warnings don't fail
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
