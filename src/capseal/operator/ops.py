"""Operator verification/provisioning helpers."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG

_VALID_VOICE_PROVIDERS = {"openai", "personaplex"}
_VALID_GATE_DECISIONS = {"approve", "flag", "deny"}


@dataclass(frozen=True)
class VerifyCheck:
    name: str
    status: str  # pass | warn | fail
    detail: str
    hint: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_decision(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"approved", "approve", "pass"}:
        return "approve"
    if raw in {"denied", "deny", "skip"}:
        return "deny"
    if raw in {"flagged", "flag", "review", "human_review"}:
        return "flag"
    return raw


def verify_operator_setup(workspace: Path, config: dict[str, Any]) -> tuple[bool, list[VerifyCheck]]:
    """Validate operator runtime readiness."""
    checks: list[VerifyCheck] = []

    ws = workspace.resolve()
    capseal_dir = ws / ".capseal"
    events_path = capseal_dir / "events.jsonl"

    if ws.exists() and ws.is_dir():
        checks.append(VerifyCheck("workspace", "pass", f"workspace found: {ws}"))
    else:
        checks.append(VerifyCheck("workspace", "fail", f"workspace missing: {ws}"))
        return False, checks

    if capseal_dir.exists():
        checks.append(VerifyCheck("workspace_state", "pass", f"state dir found: {capseal_dir}"))
    else:
        checks.append(
            VerifyCheck(
                "workspace_state",
                "warn",
                f"state dir missing: {capseal_dir}",
                hint="run `capseal init` once to create workspace metadata",
            )
        )

    threshold = config.get("notify_threshold", 0.5)
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = -1.0
    if 0.0 <= threshold_value <= 1.0:
        checks.append(VerifyCheck("notify_threshold", "pass", f"notify_threshold={threshold_value:.2f}"))
    else:
        checks.append(
            VerifyCheck(
                "notify_threshold",
                "fail",
                f"invalid notify_threshold={threshold!r}",
                hint="set notify_threshold between 0.0 and 1.0",
            )
        )

    channels = config.get("channels", {}) if isinstance(config, dict) else {}
    telegram = channels.get("telegram", {}) if isinstance(channels, dict) else {}
    has_channel = False

    tg_token = telegram.get("bot_token")
    tg_env_name = telegram.get("bot_token_env")
    if tg_env_name and os.environ.get(tg_env_name):
        tg_token = os.environ.get(tg_env_name)
    if not tg_token:
        tg_token = os.environ.get("CAPSEAL_TELEGRAM_BOT_TOKEN") or os.environ.get("CAPSEAL_TELEGRAM_TOKEN")
    tg_chat_id = telegram.get("chat_id") or os.environ.get("CAPSEAL_TELEGRAM_CHAT_ID")

    if tg_token and tg_chat_id:
        has_channel = True
        checks.append(VerifyCheck("telegram", "pass", "telegram configured"))
    elif tg_token or tg_chat_id:
        checks.append(
            VerifyCheck(
                "telegram",
                "warn",
                "telegram partially configured",
                hint="set both bot token and chat_id",
            )
        )
    else:
        checks.append(
            VerifyCheck(
                "telegram",
                "warn",
                "telegram not configured",
                hint="run `capseal operator --provision --telegram-chat-id <id>`",
            )
        )

    whatsapp = channels.get("whatsapp", {}) if isinstance(channels, dict) else {}
    if isinstance(whatsapp, dict) and whatsapp.get("phone_number_id") and (whatsapp.get("access_token") or os.environ.get("CAPSEAL_WHATSAPP_ACCESS_TOKEN")) and whatsapp.get("recipient"):
        has_channel = True
        checks.append(VerifyCheck("whatsapp", "pass", "whatsapp configured"))

    imessage = channels.get("imessage", {}) if isinstance(channels, dict) else {}
    if isinstance(imessage, dict) and imessage.get("recipient"):
        has_channel = True
        checks.append(VerifyCheck("imessage", "pass", "imessage configured"))

    if has_channel:
        checks.append(VerifyCheck("channels", "pass", "at least one notification channel configured"))
    else:
        checks.append(
            VerifyCheck(
                "channels",
                "fail",
                "no notification channels configured",
                hint="configure telegram (or whatsapp/imessage) before running operator",
            )
        )

    voice = config.get("voice", {}) if isinstance(config, dict) else {}
    if not isinstance(voice, dict):
        voice = {}
    voice_enabled = bool(voice.get("enabled"))
    if not voice_enabled:
        checks.append(
            VerifyCheck(
                "voice",
                "warn",
                "voice output disabled",
                hint="enable with `capseal operator --provision --voice`",
            )
        )
    else:
        provider = str(voice.get("provider", "openai")).strip().lower()
        if provider not in _VALID_VOICE_PROVIDERS:
            checks.append(
                VerifyCheck(
                    "voice_provider",
                    "fail",
                    f"invalid voice provider: {provider}",
                    hint="set voice.provider to openai or personaplex",
                )
            )
        else:
            checks.append(VerifyCheck("voice_provider", "pass", f"voice provider={provider}"))

        if provider == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                checks.append(VerifyCheck("voice_auth", "pass", "OPENAI_API_KEY set"))
            else:
                checks.append(
                    VerifyCheck(
                        "voice_auth",
                        "fail",
                        "OPENAI_API_KEY missing for voice synthesis",
                        hint="export OPENAI_API_KEY before running operator",
                    )
                )
        elif provider == "personaplex":
            checks.append(
                VerifyCheck(
                    "voice_auth",
                    "warn",
                    "HF_TOKEN not set (continuing unauthenticated)",
                    hint="set HF_TOKEN if your PersonaPlex deployment requires auth",
                )
                if not os.environ.get("HF_TOKEN")
                else VerifyCheck("voice_auth", "pass", "HF_TOKEN set")
            )

        decisions = voice.get("speak_gate_decisions", ["deny", "flag"])
        if not isinstance(decisions, list):
            checks.append(
                VerifyCheck(
                    "voice_gate_decisions",
                    "fail",
                    "voice.speak_gate_decisions must be a list",
                    hint="use values like ['deny', 'flag']",
                )
            )
        else:
            bad = [d for d in decisions if _normalize_decision(str(d)) not in _VALID_GATE_DECISIONS]
            if bad:
                checks.append(
                    VerifyCheck(
                        "voice_gate_decisions",
                        "fail",
                        f"invalid gate decisions: {bad}",
                        hint="allowed: approve, flag, deny",
                    )
                )
            else:
                checks.append(VerifyCheck("voice_gate_decisions", "pass", f"decisions={decisions}"))

        if bool(voice.get("live_call")):
            try:
                from .voice_call import HAS_WEBSOCKETS

                if HAS_WEBSOCKETS:
                    checks.append(VerifyCheck("voice_live_call", "pass", "live call dependency available"))
                else:
                    checks.append(
                        VerifyCheck(
                            "voice_live_call",
                            "fail",
                            "websockets not installed",
                            hint="install websockets to enable live call mode",
                        )
                    )
            except Exception:
                checks.append(
                    VerifyCheck(
                        "voice_live_call",
                        "fail",
                        "failed to import voice_call module",
                        hint="check capseal.operator.voice_call import path",
                    )
                )
        else:
            checks.append(VerifyCheck("voice_live_call", "warn", "live_call disabled"))

    if events_path.exists():
        checks.append(VerifyCheck("events_stream", "pass", f"events stream exists: {events_path}"))
    else:
        checks.append(
            VerifyCheck(
                "events_stream",
                "warn",
                f"events stream missing: {events_path}",
                hint="start a capseal session to generate events.jsonl",
            )
        )

    hard_fail = any(c.status == "fail" for c in checks)
    return (not hard_fail), checks


def provision_operator_config(
    workspace: Path,
    *,
    config_path: Path | None = None,
    enable_voice: bool = False,
    voice_provider: str = "openai",
    live_call: bool = False,
    notify_threshold: float | None = None,
    telegram_chat_id: str | None = None,
    telegram_bot_token: str | None = None,
    use_token_env: bool = True,
    speak_gate_events: bool = True,
    speak_gate_decisions: list[str] | None = None,
    speak_min_score: float = 0.55,
) -> Path:
    """Create/update operator config with safe defaults."""
    ws = workspace.resolve()
    target = (config_path.resolve() if config_path else ws / ".capseal" / "operator.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(DEFAULT_CONFIG)
    existing = _read_json(target)
    if existing:
        config = _deep_merge(config, existing)

    if notify_threshold is not None:
        config["notify_threshold"] = float(notify_threshold)

    voice = config.setdefault("voice", {})
    if enable_voice or live_call:
        voice["enabled"] = True
    voice["provider"] = str(voice_provider).strip().lower()
    voice["live_call"] = bool(live_call)
    voice["speak_gate_events"] = bool(speak_gate_events)
    voice["speak_min_score"] = float(speak_min_score)

    if speak_gate_decisions:
        normalized = [_normalize_decision(v) for v in speak_gate_decisions]
        voice["speak_gate_decisions"] = [v for v in normalized if v in _VALID_GATE_DECISIONS] or ["deny", "flag"]

    channels = config.setdefault("channels", {})
    telegram = channels.setdefault("telegram", {})

    if telegram_chat_id:
        telegram["chat_id"] = telegram_chat_id
    if telegram_bot_token:
        telegram["bot_token"] = telegram_bot_token
        telegram["bot_token_env"] = None
    elif use_token_env:
        telegram.setdefault("bot_token_env", "CAPSEAL_TELEGRAM_BOT_TOKEN")
        if not telegram.get("bot_token_env"):
            telegram["bot_token_env"] = "CAPSEAL_TELEGRAM_BOT_TOKEN"

    tmp_path = target.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    tmp_path.replace(target)
    return target


def render_verify_report(checks: list[VerifyCheck]) -> str:
    """Render a compact multi-line verification report."""
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    lines = []
    for item in checks:
        line = f"[{icon.get(item.status, item.status.upper())}] {item.name}: {item.detail}"
        if item.hint:
            line += f" | hint: {item.hint}"
        lines.append(line)
    return "\n".join(lines)

