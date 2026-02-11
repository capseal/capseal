"""Human-readable risk labels derived from extracted patch features."""

from __future__ import annotations

from typing import Any


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value <= low:
        return labels[0]
    if value <= high:
        return labels[1]
    return labels[2]


def _normalize_change_type(change_type: str | None) -> str | None:
    if not change_type:
        return None
    mapping = {
        "format": "formatting",
        "refactor": "refactor",
        "feature": "behavioral",
        "fix": "bugfix",
        "behavioral": "behavioral",
        "bugfix": "bugfix",
    }
    return mapping.get(change_type.lower(), change_type.lower())


def generate_label(features: dict[str, Any]) -> str:
    """Create a short English label from risk features."""
    if not features:
        return "unclassified"

    parts: list[str] = []

    lines_changed = float(features.get("lines_changed", 0) or 0)
    files_touched = float(features.get("files_touched", 0) or 0)
    modules_crossed = float(features.get("modules_crossed", 0) or 0)
    security_score = float(features.get("security", 0.0) or 0.0)
    test_changes = float(features.get("test_coverage_delta", 0) or 0)
    change_type = _normalize_change_type(features.get("change_type"))

    complexity_label = _bucket(lines_changed, 10, 50, ("simple", "moderate", "complex"))
    scope_label = _bucket(files_touched, 1, 3, ("single-file", "multi-file", "cross-cutting"))
    parts.append(complexity_label)
    parts.append(scope_label)

    if security_score >= 0.5:
        parts.append("security-sensitive")
    if modules_crossed >= 2:
        parts.append("cross-module")
    elif modules_crossed >= 1:
        parts.append("adjacent-module")

    if change_type:
        parts.append(change_type)

    if test_changes <= 0:
        parts.append("untested")

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    uniq: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            uniq.append(part)

    return " + ".join(uniq) if uniq else "unclassified"


__all__ = ["generate_label"]
