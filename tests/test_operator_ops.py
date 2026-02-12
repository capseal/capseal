from __future__ import annotations

import copy
import json
from pathlib import Path

from capseal.operator.config import DEFAULT_CONFIG
from capseal.operator.ops import (
    provision_operator_config,
    render_verify_report,
    verify_operator_setup,
)


def test_provision_operator_config_writes_voice_gate_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    cfg_path = provision_operator_config(
        workspace,
        enable_voice=True,
        voice_provider="openai",
        live_call=False,
        telegram_chat_id="12345",
        speak_gate_decisions=["deny", "flag"],
        speak_min_score=0.61,
    )

    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text())
    assert data["voice"]["enabled"] is True
    assert data["voice"]["provider"] == "openai"
    assert data["voice"]["speak_gate_events"] is True
    assert data["voice"]["speak_gate_decisions"] == ["deny", "flag"]
    assert data["voice"]["speak_min_score"] == 0.61
    assert data["channels"]["telegram"]["chat_id"] == "12345"


def test_verify_operator_setup_fails_without_channels(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    ok, checks = verify_operator_setup(workspace, cfg)

    assert ok is False
    assert any(c.name == "channels" and c.status == "fail" for c in checks)


def test_verify_operator_setup_passes_with_channel_and_voice_env(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / ".capseal").mkdir()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["channels"]["telegram"]["chat_id"] = "12345"
    cfg["channels"]["telegram"]["bot_token_env"] = "CAPSEAL_TELEGRAM_BOT_TOKEN"
    cfg["voice"]["enabled"] = True
    cfg["voice"]["provider"] = "openai"
    cfg["voice"]["live_call"] = False

    monkeypatch.setenv("CAPSEAL_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    ok, checks = verify_operator_setup(workspace, cfg)

    assert ok is True
    assert not any(c.status == "fail" for c in checks)
    report = render_verify_report(checks)
    assert "telegram" in report
    assert "voice_provider" in report
