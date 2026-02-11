"""Canonical risk evaluation for CLI and MCP gate paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from capseal.risk_labels import generate_label
from capseal.shared.features import (
    discretize_features,
    extract_patch_features,
    features_to_grid_idx,
)
from capseal.shared.scoring import lookup_posterior_at_idx

THRESHOLD_APPROVE = 0.3
THRESHOLD_FLAG = 0.6
THRESHOLD_DENY = 0.6


@dataclass(frozen=True)
class RiskResult:
    p_fail: float
    decision: str  # approve | flag | deny
    features: dict[str, Any]
    grid_cell: int
    confidence: float
    uncertainty: float
    observations: int
    label: str
    reason: str
    model_loaded: bool


def _resolve_model_path(workspace: str | Path, model_path: str | Path | None) -> Path:
    if model_path is not None:
        return Path(model_path).expanduser().resolve()
    ws = Path(workspace).expanduser().resolve()
    return ws / ".capseal" / "models" / "beta_posteriors.npz"


def _load_posteriors(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    alpha_key = "alpha" if "alpha" in data.files else "alphas" if "alphas" in data.files else None
    beta_key = "beta" if "beta" in data.files else "betas" if "betas" in data.files else None
    if alpha_key is None or beta_key is None:
        return None

    alpha = data[alpha_key]
    beta = data[beta_key]
    if len(alpha) != len(beta):
        return None
    return alpha, beta


def _guess_change_type(diff_text: str, description: str | None, action_type: str | None) -> str:
    text = f"{description or ''}\n{action_type or ''}\n{diff_text}".lower()
    if any(tok in text for tok in ("format", "whitespace", "lint", "pep8")):
        return "format"
    if any(tok in text for tok in ("refactor", "rename", "reorgan", "cleanup")):
        return "refactor"
    if any(tok in text for tok in ("fix", "bug", "vuln", "security", "patch")):
        return "fix"
    if any(tok in text for tok in ("feature", "add ", "implement", "new endpoint")):
        return "feature"
    return "feature"


def _security_signal(diff_text: str, findings: list[dict[str, Any]], description: str | None) -> float:
    highest_finding = 0.0
    for finding in findings:
        sev = str(
            finding.get("severity")
            or finding.get("extra", {}).get("severity")
            or "info"
        ).lower()
        if sev in {"error", "high"}:
            highest_finding = max(highest_finding, 1.0)
        elif sev in {"warning", "medium"}:
            highest_finding = max(highest_finding, 0.66)
        elif sev in {"low"}:
            highest_finding = max(highest_finding, 0.33)

    text = f"{description or ''}\n{diff_text}".lower()
    keyword_hits = sum(
        1
        for kw in (
            "auth",
            "token",
            "password",
            "secret",
            "crypto",
            "security",
            "subprocess",
            "shell=true",
            "os.system",
            "exec(",
        )
        if kw in text
    )
    keyword_signal = min(1.0, keyword_hits / 3.0)
    return max(highest_finding, keyword_signal)


def _build_feature_map(
    raw_features: dict[str, Any],
    levels: list[int],
    diff_text: str,
    findings: list[dict[str, Any]],
    description: str | None,
    action_type: str | None,
) -> dict[str, Any]:
    files_touched = int(raw_features.get("files_touched", 0) or 0)
    lines_changed = int(raw_features.get("lines_changed", 0) or 0)
    security = _security_signal(diff_text, findings, description)
    change_type = _guess_change_type(diff_text, description, action_type)

    has_security_imports = any(
        kw in diff_text.lower() for kw in ("import subprocess", "import os", "crypto", "jwt", "secrets")
    )

    return {
        "lines_changed": lines_changed,
        "files_touched": files_touched,
        "modules_crossed": max(0, files_touched - 1),
        "cyclomatic_complexity": int(raw_features.get("cyclomatic_complexity", 0) or 0),
        "finding_severity": int(raw_features.get("finding_severity", 0) or 0),
        "test_coverage_delta": int(raw_features.get("test_coverage_delta", 0) or 0),
        "change_type": change_type,
        "security": round(float(security), 4),
        "has_security_imports": has_security_imports,
        "levels": levels,
    }


def _decision_from_p_fail(p_fail: float, approve_threshold: float, deny_threshold: float) -> str:
    if p_fail >= deny_threshold:
        return "deny"
    if p_fail >= approve_threshold:
        return "flag"
    return "approve"


def to_internal_decision(decision: str) -> str:
    return {
        "approve": "pass",
        "deny": "skip",
        "flag": "human_review",
    }.get(decision, "pass")


def synthesize_diff_for_files(files_affected: list[str], description: str = "") -> str:
    if not files_affected:
        return "diff --git a/unknown b/unknown\n+++ b/unknown\n@@ -1,1 +1,2 @@\n+# no diff provided\n"

    parts: list[str] = []
    for file_path in files_affected:
        parts.append(f"diff --git a/{file_path} b/{file_path}")
        parts.append(f"--- a/{file_path}")
        parts.append(f"+++ b/{file_path}")
        parts.append("@@ -1,1 +1,2 @@")
        parts.append(f"+# {description[:80] if description else 'planned change'}")
    return "\n".join(parts) + "\n"


def evaluate_risk(
    diff: str,
    model_path: str | Path | None = None,
    workspace: str | Path = ".",
    *,
    findings: list[dict[str, Any]] | None = None,
    action_type: str | None = None,
    description: str | None = None,
    approve_threshold: float = THRESHOLD_APPROVE,
    deny_threshold: float = THRESHOLD_DENY,
) -> RiskResult:
    """Canonical p_fail computation used by CLI and MCP gate paths."""
    findings = findings or []
    model = _resolve_model_path(workspace, model_path)
    loaded = _load_posteriors(model)

    raw_features = extract_patch_features(diff, findings)
    levels = discretize_features(raw_features)
    grid_idx = features_to_grid_idx(levels)

    if loaded is None:
        p_fail = 0.5
        uncertainty = 0.5
        observations = 0
        confidence = 0.0
        decision = "approve"
        reason = "risk model not trained; approving with caution"
    else:
        alpha, beta = loaded
        posterior = lookup_posterior_at_idx(alpha, beta, grid_idx)
        p_fail = float(posterior["q"])
        uncertainty = float(posterior["uncertainty"])
        observations = int(max(0.0, float(posterior["alpha"] + posterior["beta"] - 2)))
        confidence = float(min(1.0, observations / (observations + 5.0))) if observations > 0 else 0.0
        decision = _decision_from_p_fail(p_fail, approve_threshold, deny_threshold)
        reason = (
            f"p_fail={p_fail:.3f} "
            f"(approve<{approve_threshold:.2f}, deny>={deny_threshold:.2f})"
        )

    features = _build_feature_map(raw_features, levels, diff, findings, description, action_type)
    label = generate_label(features)

    return RiskResult(
        p_fail=p_fail,
        decision=decision,
        features=features,
        grid_cell=grid_idx,
        confidence=confidence,
        uncertainty=uncertainty,
        observations=observations,
        label=label,
        reason=reason,
        model_loaded=loaded is not None,
    )


def evaluate_risk_for_finding(
    finding: dict[str, Any],
    workspace: str | Path = ".",
    model_path: str | Path | None = None,
    *,
    approve_threshold: float = THRESHOLD_APPROVE,
    deny_threshold: float = THRESHOLD_DENY,
) -> RiskResult:
    """Evaluate risk for a Semgrep finding by building a synthetic diff preview."""
    file_path = str(finding.get("path", "unknown"))
    severity = str(finding.get("extra", {}).get("severity", "warning"))
    start_line = int(finding.get("start", {}).get("line", 1) or 1)
    end_line = int(finding.get("end", {}).get("line", start_line + 5) or start_line + 5)
    lines_changed = max(5, end_line - start_line + 1)

    diff_preview = (
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        f"@@ -{start_line},{lines_changed} +{start_line},{lines_changed} @@\n"
        f"+# finding: {finding.get('check_id', 'unknown')}\n"
    )

    finding_stub = [{"severity": severity}]
    description = finding.get("extra", {}).get("message") or finding.get("check_id") or "semgrep finding"
    return evaluate_risk(
        diff_preview,
        model_path=model_path,
        workspace=workspace,
        findings=finding_stub,
        action_type="scan_finding",
        description=str(description),
        approve_threshold=approve_threshold,
        deny_threshold=deny_threshold,
    )


def evaluate_action_risk(
    action_type: str,
    description: str,
    files_affected: list[str] | None = None,
    diff_text: str | None = None,
    *,
    workspace: str | Path = ".",
    model_path: str | Path | None = None,
) -> RiskResult:
    """Evaluate risk for an MCP action proposal."""
    files_affected = files_affected or []
    if diff_text:
        diff = diff_text
        findings: list[dict[str, Any]] = []
    else:
        diff = synthesize_diff_for_files(files_affected, description=description)
        findings = [{"severity": "warning", "path": f} for f in files_affected]
    return evaluate_risk(
        diff,
        model_path=model_path,
        workspace=workspace,
        findings=findings,
        action_type=action_type,
        description=description,
    )


__all__ = [
    "RiskResult",
    "THRESHOLD_APPROVE",
    "THRESHOLD_FLAG",
    "THRESHOLD_DENY",
    "evaluate_action_risk",
    "evaluate_risk",
    "evaluate_risk_for_finding",
    "synthesize_diff_for_files",
    "to_internal_decision",
]
