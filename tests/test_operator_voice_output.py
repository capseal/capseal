from __future__ import annotations

import asyncio
import copy
from pathlib import Path

from capseal.operator.composer import Message
from capseal.operator.config import DEFAULT_CONFIG
from capseal.operator.daemon import OperatorDaemon


class _DummyVoiceCall:
    def __init__(self) -> None:
        self.connected = True
        self.spoken: list[str] = []
        self.connect_calls = 0
        self.disconnect_calls = 0

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def listen_loop(self) -> None:
        return


class _DummyNarrator:
    def __init__(self) -> None:
        self.available = True
        self.spoken: list[str] = []

    async def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return True


def _base_config() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["notify_threshold"] = 0.95  # higher than denied score to test force path
    cfg["voice"]["enabled"] = True
    cfg["voice"]["provider"] = "openai"
    cfg["voice"]["live_call"] = False
    cfg["voice"]["speak_gate_events"] = True
    cfg["voice"]["speak_gate_decisions"] = ["deny", "flag"]
    cfg["voice"]["speak_min_score"] = 0.55
    return cfg


def test_should_force_gate_voice_for_denied_event(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    daemon = OperatorDaemon(workspace, _base_config())
    event = {"type": "gate", "data": {"decision": "denied"}}
    assert daemon._should_force_gate_voice(event, score=0.80) is True
    assert daemon._should_force_gate_voice(event, score=0.40) is False

    approved = {"type": "gate", "data": {"decision": "approve"}}
    assert daemon._should_force_gate_voice(approved, score=0.90) is False


def test_process_event_forces_broadcast_even_above_notify_threshold(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    daemon = OperatorDaemon(workspace, _base_config())
    seen: list[float] = []

    async def _fake_broadcast(message: Message, score: float, event=None):
        seen.append(score)

    daemon._broadcast = _fake_broadcast  # type: ignore[assignment]

    event = {
        "type": "gate",
        "data": {
            "decision": "denied",
            "files": ["auth.py"],
            "p_fail": 0.8,
        },
    }
    asyncio.run(daemon._process_event(event))
    assert seen, "forced voice gate events should still broadcast"


def test_broadcast_speaks_live_call_for_forced_gate_voice(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    daemon = OperatorDaemon(workspace, _base_config())
    daemon._voice_active = True  # Voice toggle is off by default; tests should opt in.
    dummy_call = _DummyVoiceCall()
    daemon.voice_call = dummy_call  # type: ignore[assignment]
    dummy_narrator = _DummyNarrator()
    daemon.narrator = dummy_narrator  # type: ignore[assignment]
    daemon.voice = None

    msg = Message(
        short_text="denied",
        full_text="denied full",
        voice_text="i blocked a risky edit",
    )
    event = {"type": "gate", "data": {"decision": "deny"}}
    asyncio.run(daemon._broadcast(msg, score=0.80, event=event))

    assert dummy_narrator.spoken == ["i blocked a risky edit"]


def test_auto_stop_sets_voice_pod_stopped(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    daemon = OperatorDaemon(workspace, _base_config())
    dummy = _DummyVoiceCall()
    daemon.voice_call = dummy  # type: ignore[assignment]
    daemon._voice_idle_seconds = 1
    daemon._voice_pod_stopped = False

    asyncio.run(daemon._auto_stop_voice())

    assert dummy.disconnect_calls == 1
    assert daemon._voice_pod_stopped is True


def test_resume_voice_reconnects_when_pod_not_managed(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    daemon = OperatorDaemon(workspace, _base_config())
    dummy = _DummyVoiceCall()
    dummy.connected = False
    daemon.voice_call = dummy  # type: ignore[assignment]
    daemon._voice_pod_stopped = True
    daemon._runpod_pod_id = None
    daemon._runpod_api_key = None

    asyncio.run(daemon._resume_voice_on_session_start())

    assert dummy.connect_calls == 1
    assert dummy.connected is True
