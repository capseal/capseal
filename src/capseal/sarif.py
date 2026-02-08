"""SARIF 2.1.0 export for CapSeal sessions.

Converts CapSeal session data (actions + manifest) into a SARIF log
suitable for GitHub Code Scanning, VS Code SARIF Viewer, or any
SARIF-compatible tool.

Usage:
    from capseal.sarif import build_sarif_log

    sarif = build_sarif_log(actions, manifest, cap_path, workspace=ws)
    print(json.dumps(sarif, indent=2))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"
CAPSEAL_VERSION = "0.4.0"


def build_sarif_log(
    actions: list[dict],
    manifest: dict,
    cap_path: Path | str,
    workspace: Path | str | None = None,
) -> dict:
    """Build a complete SARIF 2.1.0 log from session data.

    Args:
        actions: List of action dicts from actions.jsonl
        manifest: Manifest dict from the .cap file
        cap_path: Path to the .cap file
        workspace: Optional workspace root for relativizing paths

    Returns:
        SARIF 2.1.0 log as a dict
    """
    results = []
    for i, action in enumerate(actions):
        result = _action_to_sarif_result(action, i)
        results.append(result)

    run = {
        "tool": _build_tool_component(),
        "results": results,
        "properties": {
            "capseal.capFile": str(cap_path),
            "capseal.sessionName": manifest.get("session_name", ""),
            "capseal.actionsCount": len(actions),
        },
    }

    if workspace:
        run["originalUriBaseIds"] = {
            "%SRCROOT%": {
                "uri": f"file:///{str(Path(workspace).resolve()).lstrip('/')}/"
            }
        }

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [run],
    }


def _action_to_sarif_result(action: dict, index: int) -> dict:
    """Map one action to a SARIF result with ruleId, level, location, properties."""
    metadata = action.get("metadata") or {}
    gate_decision = action.get("gate_decision")
    description = metadata.get("description", action.get("action_type", "unknown"))
    files_affected = metadata.get("files_affected", [])

    rule_id = _decision_to_rule_id(gate_decision)
    level = _map_gate_to_level(gate_decision)

    result: dict[str, Any] = {
        "ruleId": rule_id,
        "level": level,
        "message": {
            "text": description,
        },
        "properties": {
            "capseal.actionId": action.get("action_id", f"act_{index:04d}"),
            "capseal.chainPosition": index,
        },
    }

    # Add gate score if available
    gate_score = action.get("gate_score")
    if gate_score is not None:
        result["properties"]["capseal.gateScore"] = gate_score

    # Add receipt hash if available
    receipt_hash = action.get("receipt_hash")
    if receipt_hash:
        result["properties"]["capseal.receiptHash"] = receipt_hash

    # Add locations from files_affected
    if files_affected:
        locations = []
        for f in files_affected:
            locations.append({
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": f,
                        "uriBaseId": "%SRCROOT%",
                    }
                }
            })
        result["locations"] = locations

    return result


def _map_gate_to_level(gate_decision: str | None) -> str:
    """Map gate decision to SARIF level.

    skip/deny → error
    human_review/flag → warning
    pass/approve/None → note
    """
    normalized = _normalize_decision(gate_decision)
    if normalized == "deny":
        return "error"
    elif normalized == "flag":
        return "warning"
    return "note"


def _normalize_decision(gate_decision: str | None) -> str:
    """Normalize gate decision to approve/deny/flag."""
    return {
        "pass": "approve",
        "skip": "deny",
        "human_review": "flag",
        "approve": "approve",
        "deny": "deny",
        "flag": "flag",
    }.get(gate_decision or "", "approve")


def _decision_to_rule_id(gate_decision: str | None) -> str:
    """Map gate decision to SARIF rule ID."""
    normalized = _normalize_decision(gate_decision)
    if normalized == "deny":
        return "capseal/gate-denied"
    elif normalized == "flag":
        return "capseal/gate-flagged"
    return "capseal/gate-approved"


def _build_tool_component() -> dict:
    """Build the SARIF tool component with CapSeal rule definitions."""
    return {
        "driver": {
            "name": "CapSeal",
            "version": CAPSEAL_VERSION,
            "informationUri": "https://github.com/capseal/capseal",
            "rules": [
                {
                    "id": "capseal/gate-denied",
                    "name": "GateDenied",
                    "shortDescription": {
                        "text": "Action was denied by the CapSeal risk gate"
                    },
                    "fullDescription": {
                        "text": "The CapSeal risk model predicted a high failure probability for this action and denied it."
                    },
                    "defaultConfiguration": {
                        "level": "error"
                    },
                },
                {
                    "id": "capseal/gate-flagged",
                    "name": "GateFlagged",
                    "shortDescription": {
                        "text": "Action was flagged for human review by the CapSeal risk gate"
                    },
                    "fullDescription": {
                        "text": "The CapSeal risk model predicted moderate uncertainty for this action and flagged it for human review."
                    },
                    "defaultConfiguration": {
                        "level": "warning"
                    },
                },
                {
                    "id": "capseal/gate-approved",
                    "name": "GateApproved",
                    "shortDescription": {
                        "text": "Action was approved by the CapSeal risk gate"
                    },
                    "fullDescription": {
                        "text": "The CapSeal risk model predicted a low failure probability for this action and approved it."
                    },
                    "defaultConfiguration": {
                        "level": "note"
                    },
                },
            ],
        }
    }


__all__ = [
    "build_sarif_log",
    "_map_gate_to_level",
    "_normalize_decision",
    "SARIF_SCHEMA",
    "SARIF_VERSION",
]
