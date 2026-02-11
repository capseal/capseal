"""Structured MCP response builders."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from capseal.risk_engine import THRESHOLD_APPROVE, THRESHOLD_DENY, RiskResult

SCHEMA_VERSION = "1.0"


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def format_gate_response(
    result: RiskResult,
    *,
    session_id: str,
    action_type: str,
    description: str,
    files_affected: list[str],
    override: str | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    decision = override or result.decision
    base_reason = override_reason or result.reason
    receipt_id = f"{_utc_stamp()}-gate-{result.grid_cell}"
    files_str = ", ".join(files_affected) if files_affected else "none"
    decision_text = {
        "approve": "APPROVED",
        "deny": "DENIED",
        "flag": "FLAGGED",
    }.get(decision, decision.upper())
    summary = (
        f"CAPSEAL GATE: {decision_text} | p_fail={result.p_fail:.2f} | "
        f"{result.label} | files={files_str}"
    )

    return {
        "decision": decision,
        "predicted_failure": round(result.p_fail, 6),
        "p_fail": round(result.p_fail, 6),
        "confidence": round(result.confidence, 6),
        "uncertainty": round(result.uncertainty, 6),
        "observations": result.observations,
        "grid_cell": result.grid_cell,
        "grid_idx": result.grid_cell,
        "label": result.label,
        "features": result.features,
        "thresholds": {
            "approve_below": THRESHOLD_APPROVE,
            "deny_above": THRESHOLD_DENY,
        },
        "receipt_id": receipt_id,
        "session_id": session_id,
        "action_type": action_type,
        "description": description,
        "files_affected": files_affected,
        "model_loaded": result.model_loaded,
        "reason": base_reason,
        "human_summary": summary,
        "schema_version": SCHEMA_VERSION,
    }


def format_record_response(
    *,
    recorded: bool,
    session_id: str,
    action_type: str,
    files_affected: list[str],
    action_id: str | None = None,
    receipt_hash: str | None = None,
    receipt_chain_length: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    if recorded:
        summary = (
            f"CAPSEAL RECORD: recorded {action_type} "
            f"({receipt_chain_length} total actions)"
        )
    else:
        summary = f"CAPSEAL RECORD: failed ({error or 'unknown error'})"
    return {
        "recorded": recorded,
        "action_id": action_id,
        "receipt_hash": receipt_hash,
        "session_id": session_id,
        "action_type": action_type,
        "files_affected": files_affected,
        "receipt_chain_length": receipt_chain_length,
        "error": error,
        "human_summary": summary,
        "schema_version": SCHEMA_VERSION,
    }


def format_seal_response(
    *,
    sealed: bool,
    session_id: str | None,
    session_name: str,
    cap_file: str | None = None,
    chain_hash: str | None = None,
    actions_count: int = 0,
    proof_generated: bool = False,
    proof_verified: bool = False,
    model_updated: bool = False,
    model_episodes: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if sealed:
        summary = f"CAPSEAL SEALED: {actions_count} actions -> {cap_file}"
    else:
        summary = f"CAPSEAL SEALED: failed ({error or 'unknown error'})"
    return {
        "sealed": sealed,
        "session_id": session_id,
        "session_name": session_name,
        "cap_file": cap_file,
        "hash": chain_hash,
        "actions_count": actions_count,
        "verification": "passed" if sealed else "failed",
        "proof_generated": proof_generated,
        "proof_verified": proof_verified,
        "model_updated": model_updated,
        "model_episodes": model_episodes,
        "error": error,
        "human_summary": summary,
        "schema_version": SCHEMA_VERSION,
    }


def format_status_response(
    *,
    session_id: str | None,
    session_active: bool,
    workspace: str,
    actions_count: int,
    gates_count: int,
    denials_count: int,
    model_loaded: bool,
    uptime_seconds: int,
    recent_sessions: list[dict[str, Any]],
    project_stats: dict[str, Any],
) -> dict[str, Any]:
    state = "active" if session_active else "inactive"
    summary = (
        f"CAPSEAL STATUS | Session: {state} | Workspace: {workspace} | "
        f"Actions: {actions_count} | Denied: {denials_count}"
    )
    return {
        "session_id": session_id,
        "session_active": session_active,
        "workspace": workspace,
        "actions_count": actions_count,
        "gates_count": gates_count,
        "denials_count": denials_count,
        "model_loaded": model_loaded,
        "uptime_seconds": uptime_seconds,
        "recent_sessions": recent_sessions,
        "project_stats": project_stats,
        "human_summary": summary,
        "schema_version": SCHEMA_VERSION,
    }


def format_context_response(
    *,
    file_path: str,
    total_changes: int,
    total_sessions: int,
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = (
        f"CAPSEAL CONTEXT: {file_path} has {total_changes} change(s) "
        f"across {total_sessions} session(s)"
    )
    return {
        "file": file_path,
        "total_changes": total_changes,
        "total_sessions": total_sessions,
        "sessions": sessions,
        "human_summary": summary,
        "schema_version": SCHEMA_VERSION,
    }


__all__ = [
    "SCHEMA_VERSION",
    "format_context_response",
    "format_gate_response",
    "format_record_response",
    "format_seal_response",
    "format_status_response",
]
