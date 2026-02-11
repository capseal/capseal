from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from capseal import mcp_server
from capseal.mcp_responses import format_gate_response
from capseal.risk_engine import RiskResult


def _reset_mcp_globals(workspace: Path) -> None:
    mcp_server._workspace = str(workspace)
    mcp_server._runtime = None
    mcp_server._runtime_path = None
    mcp_server._session_cache = None
    mcp_server._session_cache_count = 0
    mcp_server._gate_decisions.clear()


def _seed_model(workspace: Path) -> None:
    model_dir = workspace / ".capseal" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.ones(1024, dtype=np.int64)
    beta = np.ones(1024, dtype=np.int64)
    np.savez(model_dir / "beta_posteriors.npz", alpha=alpha, beta=beta, n_episodes=0)


def test_gate_response_schema_fields() -> None:
    risk = RiskResult(
        p_fail=0.81,
        decision="deny",
        features={"lines_changed": 47, "files_touched": 4},
        grid_cell=847,
        confidence=0.73,
        uncertainty=0.11,
        observations=12,
        label="complex + cross-cutting + security-sensitive",
        reason="test",
        model_loaded=True,
    )
    payload = format_gate_response(
        risk,
        session_id="ses_test",
        action_type="code_edit",
        description="test",
        files_affected=["auth.py"],
    )
    required = [
        "decision",
        "p_fail",
        "confidence",
        "grid_cell",
        "label",
        "features",
        "thresholds",
        "receipt_id",
        "session_id",
        "human_summary",
        "schema_version",
    ]
    for field in required:
        assert field in payload
    assert payload["schema_version"] == "1.0"
    assert payload["decision"] in ("approve", "flag", "deny")


def test_all_mcp_tools_return_json_payloads(tmp_path: Path) -> None:
    workspace = tmp_path
    (workspace / ".capseal" / "runs").mkdir(parents=True, exist_ok=True)
    _seed_model(workspace)
    _reset_mcp_globals(workspace)

    # Status
    status = asyncio.run(mcp_server._handle_status({}))
    status_payload = json.loads(status[0].text)
    assert status_payload["schema_version"] == "1.0"

    # Gate
    gate = asyncio.run(
        mcp_server._handle_gate(
            {
                "action_type": "code_edit",
                "description": "test gate",
                "files_affected": ["app.py"],
                "diff_text": "--- a/app.py\n+++ b/app.py\n@@ -1 +1,2 @@\n+import os\n",
            }
        )
    )
    gate_payload = json.loads(gate[0].text)
    assert gate_payload["schema_version"] == "1.0"
    assert "human_summary" in gate_payload

    # Record
    record = asyncio.run(
        mcp_server._handle_record(
            {
                "action_type": "code_edit",
                "description": "record test",
                "tool_name": "edit",
                "success": True,
                "files_affected": ["app.py"],
            }
        )
    )
    record_payload = json.loads(record[0].text)
    assert record_payload["schema_version"] == "1.0"
    assert record_payload["recorded"] is True

    # Context
    context = asyncio.run(mcp_server._handle_context({"file": "app.py"}))
    context_payload = json.loads(context[0].text)
    assert context_payload["schema_version"] == "1.0"

    # Seal
    seal = asyncio.run(mcp_server._handle_seal({"session_name": "test"}))
    seal_payload = json.loads(seal[0].text)
    assert seal_payload["schema_version"] == "1.0"
    assert "human_summary" in seal_payload
