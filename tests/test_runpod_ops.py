from __future__ import annotations

import json
from pathlib import Path

from capseal.operator.runpod_ops import (
    _redact_payload,
    clear_pod_id_in_config,
    resolve_runpod_api_key,
    resolve_runpod_pod_id,
)


def test_resolve_runpod_pod_id_prefers_top_level() -> None:
    cfg = {"runpod_pod_id": "pod-top", "voice": {"runpod_pod_id": "pod-voice"}}
    assert resolve_runpod_pod_id(cfg) == "pod-top"


def test_resolve_runpod_pod_id_falls_back_to_voice() -> None:
    cfg = {"voice": {"runpod_pod_id": "pod-voice"}}
    assert resolve_runpod_pod_id(cfg) == "pod-voice"


def test_resolve_runpod_api_key_from_config(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert resolve_runpod_api_key({"runpod_api_key": "abc"}) == "abc"


def test_clear_pod_id_in_config(tmp_path: Path) -> None:
    config_path = tmp_path / "operator.json"
    config_path.write_text(
        json.dumps({"runpod_pod_id": "pod-1", "voice": {"runpod_pod_id": "pod-1"}})
    )
    clear_pod_id_in_config(config_path)
    data = json.loads(config_path.read_text())
    assert "runpod_pod_id" not in data
    assert "runpod_pod_id" not in data.get("voice", {})


def test_redact_payload_masks_sensitive_env_values() -> None:
    raw = {
        "env": [
            "HF_TOKEN=hf_secret_value_1234567890",
            "PUBLIC_KEY=ssh-ed25519 " + ("A" * 120),
        ],
        "runtime": {"ports": [{"privatePort": 22}]},
        "apiKey": "abc",
    }
    redacted = _redact_payload(raw)
    assert redacted["env"][0] == "HF_TOKEN=***"
    assert "..." in redacted["env"][1]
    assert redacted["apiKey"] == "***"
