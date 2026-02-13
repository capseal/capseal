"""
CapSeal Operator — Configuration

Loads config from:
  1. Defaults
  2. Global operator config (CLI --config or ~/.capseal/operator.json)
  3. Workspace override (<workspace>/.capseal/operator.json)
  4. Environment variables

Rationale:
  - Global config is user-scoped (voice server URL, RunPod creds, Telegram channel).
  - Workspace config is repo-scoped (optional overrides).
  - Avoid using the current working directory as an implicit config source when a
    workspace is explicitly provided; that's a footgun when running the operator
    from one repo while monitoring another.
"""

import json
import copy
import os
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG = {
    "enabled": True,
    "notify_threshold": 0.5,
    "voice": {
        "enabled": False,
        "provider": "openai",  # "openai" or "personaplex"
        "openai_voice": "onyx",  # alloy, echo, fable, onyx, nova, shimmer
        "personaplex_url": "https://api.personaplex.io/v1",
        "voice_preset": "NATM1",
        "live_call": False,
        "personaplex_ws_url": "wss://api.personaplex.io/v1/stream",
        "protocol": "auto",  # auto-detect (moshi_binary or json_stream)
        "speak_gate_events": True,
        "speak_gate_decisions": ["deny", "flag"],
        "speak_min_score": 0.55,
        # Keep narration on by default, but require explicit opt-in for live command parsing.
        "listen_commands": False,
        # Full-duplex conversation mode (mic uplink -> PersonaPlex).
        "freeform": False,
        "mic_enabled": False,
        "mic_backend": "auto",  # auto, pulse, alsa
        "mic_input": "default",
        # Bluetooth UX: avoid opening the BT headset mic when output is BT (keeps A2DP quality).
        "bt_avoid_headset_mic": True,
        # Best-effort: temporarily disable WirePlumber's autoswitch-to-headset-profile while voice is active.
        "bt_disable_autoswitch": True,
        "reconnect_interval_seconds": 30,
        "auto_stop_idle_seconds": 1800,
        "resume_wait_seconds": 45,
    },
    "channels": {
        "telegram": {
            "bot_token": None,
            "bot_token_env": None,
            "chat_id": None,
            "voice_notes": False,
            "decision_buttons": True,
        },
        "whatsapp": {
            "phone_number_id": None,
            "access_token": None,
            "access_token_env": None,
            "recipient": None,
        },
        "imessage": {
            "recipient": None,
        },
    },
    "nlp": {
        "enabled": True,
        "llm_fallback": True,
    },
    "quiet_hours": {
        "enabled": False,
        "start": "22:00",
        "end": "08:00",
        "only_critical": True,
    },
    "budget": {
        "cap_per_session": None,
        "alert_at_percent": 80,
    },
}


def load_config(config_path: Optional[Path] = None, workspace: Optional[Path] = None) -> dict:
    """Load operator config.

    `config_path` (CLI --config) is treated as the global user config layer.
    If absent, ~/.capseal/operator.json is used as the global layer.

    If `workspace` is provided, <workspace>/.capseal/operator.json is loaded
    as a workspace-specific override on top of the global layer.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    loaded_files: list[tuple[Path, dict]] = []

    # Global user config layer
    # NOTE: During pytest runs we intentionally avoid reading real user config from disk.
    # Tests should be hermetic and must not depend on ~/.capseal/operator.json existing.
    is_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    capseal_home = Path(os.environ.get("CAPSEAL_HOME")) if os.environ.get("CAPSEAL_HOME") else None
    default_global_path = (capseal_home or (Path.home() / ".capseal")) / "operator.json"

    global_path = config_path if config_path else default_global_path
    if is_pytest and config_path is None:
        global_path = None
    if global_path is not None and global_path.exists():
        global_cfg = _read_json(global_path)
        if global_cfg:
            config = _merge(config, global_cfg)
            loaded_files.append((global_path, global_cfg))

    # Workspace override layer
    if workspace:
        ws_config_path = Path(workspace) / ".capseal" / "operator.json"
        if ws_config_path.exists():
            ws_cfg = _read_json(ws_config_path)
            if ws_cfg:
                config = _merge(config, ws_cfg)
                loaded_files.append((ws_config_path, ws_cfg))

    # Legacy fallback: if workspace is not provided, allow CWD .capseal/operator.json.
    # When a workspace is provided, never implicitly use CWD as a config source.
    if workspace is None:
        cwd_config = Path(".capseal") / "operator.json"
        if cwd_config.exists():
            cwd_cfg = _read_json(cwd_config)
            if cwd_cfg:
                config = _merge(config, cwd_cfg)
                loaded_files.append((cwd_config, cwd_cfg))

    if loaded_files:
        for path, _cfg in loaded_files:
            print(f"[config] Loaded from {path}")
    else:
        print("[config] Using defaults (no config file found)")
        print("[config] Set up channels: capseal operator --setup telegram")
        print("[config] Or set env vars: CAPSEAL_TELEGRAM_BOT_TOKEN, CAPSEAL_TELEGRAM_CHAT_ID")

    # Resolve secret indirections from config, then allow explicit env overrides.
    _resolve_secret_env_refs(config)
    _apply_env_overrides(config)
    for path, cfg in loaded_files:
        _warn_on_plaintext_secrets(cfg, path)
    return config


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] Warning: Could not read {path}: {e}")
        return {}


def _merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_secret_env_refs(config: dict) -> None:
    """Resolve *_env config fields to actual secret values from process env."""
    channels = config.setdefault("channels", {})

    tg = channels.setdefault("telegram", {})
    token_env = tg.get("bot_token_env")
    if token_env and os.environ.get(token_env):
        tg["bot_token"] = os.environ[token_env]

    wa = channels.setdefault("whatsapp", {})
    access_env = wa.get("access_token_env")
    if access_env and os.environ.get(access_env):
        wa["access_token"] = os.environ[access_env]


def _apply_env_overrides(config: dict) -> None:
    """Apply explicit env var overrides after file/default loading."""
    channels = config.setdefault("channels", {})
    tg = channels.setdefault("telegram", {})
    wa = channels.setdefault("whatsapp", {})

    telegram_token = _first_env("CAPSEAL_TELEGRAM_BOT_TOKEN", "CAPSEAL_TELEGRAM_TOKEN")
    if telegram_token:
        tg["bot_token"] = telegram_token

    telegram_chat_id = os.environ.get("CAPSEAL_TELEGRAM_CHAT_ID")
    if telegram_chat_id:
        tg["chat_id"] = telegram_chat_id

    wa_access_token = os.environ.get("CAPSEAL_WHATSAPP_ACCESS_TOKEN")
    if wa_access_token:
        wa["access_token"] = wa_access_token
    wa_phone = os.environ.get("CAPSEAL_WHATSAPP_PHONE_NUMBER_ID")
    if wa_phone:
        wa["phone_number_id"] = wa_phone
    wa_recipient = os.environ.get("CAPSEAL_WHATSAPP_RECIPIENT")
    if wa_recipient:
        wa["recipient"] = wa_recipient

    notify_threshold = os.environ.get("CAPSEAL_NOTIFY_THRESHOLD")
    if notify_threshold:
        try:
            config["notify_threshold"] = float(notify_threshold)
        except ValueError:
            print(f"[config] Warning: invalid CAPSEAL_NOTIFY_THRESHOLD={notify_threshold!r}")

    voice_enabled = os.environ.get("CAPSEAL_VOICE_ENABLED")
    if voice_enabled is not None:
        config.setdefault("voice", {})["enabled"] = _to_bool(voice_enabled)

    voice_provider = os.environ.get("CAPSEAL_VOICE_PROVIDER")
    if voice_provider:
        config.setdefault("voice", {})["provider"] = voice_provider

    voice_live_call = os.environ.get("CAPSEAL_VOICE_LIVE_CALL")
    if voice_live_call is not None:
        config.setdefault("voice", {})["live_call"] = _to_bool(voice_live_call)

    voice_speak_gates = os.environ.get("CAPSEAL_VOICE_SPEAK_GATES")
    if voice_speak_gates is not None:
        config.setdefault("voice", {})["speak_gate_events"] = _to_bool(voice_speak_gates)

    voice_speak_decisions = os.environ.get("CAPSEAL_VOICE_SPEAK_DECISIONS")
    if voice_speak_decisions:
        values = [x.strip().lower() for x in voice_speak_decisions.split(",") if x.strip()]
        if values:
            config.setdefault("voice", {})["speak_gate_decisions"] = values

    voice_speak_min_score = os.environ.get("CAPSEAL_VOICE_SPEAK_MIN_SCORE")
    if voice_speak_min_score:
        try:
            config.setdefault("voice", {})["speak_min_score"] = float(voice_speak_min_score)
        except ValueError:
            print(f"[config] Warning: invalid CAPSEAL_VOICE_SPEAK_MIN_SCORE={voice_speak_min_score!r}")

    voice_listen_commands = os.environ.get("CAPSEAL_VOICE_LISTEN_COMMANDS")
    if voice_listen_commands is not None:
        config.setdefault("voice", {})["listen_commands"] = _to_bool(voice_listen_commands)

    voice_freeform = os.environ.get("CAPSEAL_VOICE_FREEFORM")
    if voice_freeform is not None:
        config.setdefault("voice", {})["freeform"] = _to_bool(voice_freeform)

    voice_mic_enabled = os.environ.get("CAPSEAL_VOICE_MIC_ENABLED")
    if voice_mic_enabled is not None:
        config.setdefault("voice", {})["mic_enabled"] = _to_bool(voice_mic_enabled)

    voice_mic_backend = os.environ.get("CAPSEAL_VOICE_MIC_BACKEND")
    if voice_mic_backend:
        config.setdefault("voice", {})["mic_backend"] = voice_mic_backend.strip().lower()

    voice_mic_input = os.environ.get("CAPSEAL_VOICE_MIC_INPUT")
    if voice_mic_input:
        config.setdefault("voice", {})["mic_input"] = voice_mic_input.strip()


def _warn_on_plaintext_secrets(loaded_config: Optional[dict], source_path: Optional[Path]) -> None:
    """Warn when secrets are stored directly in config JSON."""
    if not loaded_config or not source_path:
        return

    channels = loaded_config.get("channels", {})
    tg = channels.get("telegram", {})
    if tg.get("bot_token"):
        print(
            "[config] Warning: Telegram bot token is stored in plaintext in "
            f"{source_path}. Prefer CAPSEAL_TELEGRAM_BOT_TOKEN."
        )

    wa = channels.get("whatsapp", {})
    if wa.get("access_token"):
        print(
            "[config] Warning: WhatsApp access token is stored in plaintext in "
            f"{source_path}. Prefer CAPSEAL_WHATSAPP_ACCESS_TOKEN."
        )


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _to_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}
