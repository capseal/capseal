from __future__ import annotations

from capseal.operator.voice_call import VoiceCallManager, _normalize_moshi_ws_url


def test_normalize_moshi_ws_url_adds_chat_path() -> None:
    assert _normalize_moshi_ws_url("wss://abc-8898.proxy.runpod.net") == "wss://abc-8898.proxy.runpod.net/api/chat"
    assert _normalize_moshi_ws_url("abc-8898.proxy.runpod.net") == "wss://abc-8898.proxy.runpod.net/api/chat"
    assert _normalize_moshi_ws_url("wss://x/api/chat") == "wss://x/api/chat"


def test_protocol_auto_detects_json_stream() -> None:
    mgr = VoiceCallManager({"personaplex_ws_url": "wss://api.personaplex.io/v1/stream"})
    assert mgr.protocol == "json_stream"


def test_protocol_auto_detects_moshi_binary() -> None:
    mgr = VoiceCallManager({"personaplex_ws_url": "wss://abc-8898.proxy.runpod.net"})
    assert mgr.protocol == "moshi_binary"
