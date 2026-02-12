from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from capseal.cli.ops_cmd import ops_group


def test_ops_status_outputs_summary(monkeypatch, tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    cfg_path = ws / ".capseal" / "operator.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"runpod_pod_id": "pod-1"}))

    monkeypatch.setattr(
        "capseal.cli.ops_cmd._runtime",
        lambda workspace: (cfg_path, {"runpod_pod_id": "pod-1"}, "pod-1", "key-1"),
    )
    monkeypatch.setattr(
        "capseal.cli.ops_cmd.get_pod_status",
        lambda api_key, pod_id: {"pod_id": pod_id, "status": "RUNNING", "network_ready": True},
    )

    runner = CliRunner()
    result = runner.invoke(ops_group, ["-w", str(ws), "status"])
    assert result.exit_code == 0
    assert "pod_id: pod-1" in result.output
    assert "status: RUNNING" in result.output


def test_ops_teardown_clears_saved_pod_id(monkeypatch, tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    cfg_path = ws / ".capseal" / "operator.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"runpod_pod_id": "pod-1"}))

    called = {"terminated": False}
    monkeypatch.setattr(
        "capseal.cli.ops_cmd._runtime",
        lambda workspace: (cfg_path, {"runpod_pod_id": "pod-1"}, "pod-1", "key-1"),
    )
    monkeypatch.setattr(
        "capseal.cli.ops_cmd.terminate_pod",
        lambda api_key, pod_id: called.__setitem__("terminated", True),
    )

    runner = CliRunner()
    result = runner.invoke(ops_group, ["-w", str(ws), "teardown", "--yes"])
    assert result.exit_code == 0
    assert called["terminated"] is True
    data = json.loads(cfg_path.read_text())
    assert "runpod_pod_id" not in data
