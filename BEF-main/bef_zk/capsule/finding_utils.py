from __future__ import annotations

from typing import Any

DEFAULT_POLICY_SEVERITY_ORDER = ["error", "warning", "info"]
FINDING_NORM_VERSION = "finding_norm_v2"
LEGACY_FINDING_NORM_VERSION = "finding_norm_v1"


def policy_severity_order(rules: dict | None) -> list[str]:
    if isinstance(rules, dict):
        order = rules.get("severity_order")
        if isinstance(order, list) and order:
            return order
    return list(DEFAULT_POLICY_SEVERITY_ORDER)


def severity_rank(order: list[str]) -> dict[str, int]:
    return {level: idx for idx, level in enumerate(order)}


def _normalize_path_component(path: str) -> str:
    return path.replace("\\", "/").lower()


def _normalized_message(text: str) -> str:
    return " ".join(text.split()).lower()


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _normalize_snippet(text: str) -> str:
    return " ".join(text.split()).lower()


def snippet_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(_normalize_snippet(text).encode()).hexdigest()


def compute_finding_fingerprint(
    finding: dict,
    backend_id: str = "",
    norm_version: str | None = None,
) -> str:
    import hashlib

    if norm_version is None:
        norm_version = finding.get("finding_norm_version")
    norm_version = norm_version or FINDING_NORM_VERSION

    chunk_hashes = finding.get("chunk_hashes")
    if isinstance(chunk_hashes, list) and chunk_hashes:
        primary_chunk = chunk_hashes[0]
    else:
        primary_chunk = ""
    snippet_hash_value = finding.get("snippet_hash", "")

    if norm_version == LEGACY_FINDING_NORM_VERSION:
        parts = [
            backend_id,
            finding.get("rule_id", ""),
            finding.get("file_path", ""),
            primary_chunk,
            snippet_hash_value,
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    normalized_path = _normalize_path_component(finding.get("file_path", ""))
    normalized_message = _normalized_message(finding.get("message", ""))
    message_hash = _hash_text(normalized_message) if normalized_message else ""
    symbol_hint = finding.get("symbol") or finding.get("function") or ""

    parts = [
        norm_version,
        backend_id,
        finding.get("rule_id", ""),
        normalized_path,
        primary_chunk,
        snippet_hash_value,
        message_hash,
        symbol_hint,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()
