"""
CapSeal Operator — Configuration

Loads config from:
  1. CLI --config flag
  2. .capseal/operator.json in workspace
  3. ~/.capseal/operator.json
  4. Environment variables
  5. Defaults
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
    """Load config from file, falling back to defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    source_path: Optional[Path] = None
    loaded_config: Optional[dict] = None

    # Try explicit path first
    if config_path and config_path.exists():
        source_path = config_path
    else:
        # Try workspace .capseal/operator.json (use actual workspace, not CWD)
        if workspace:
            workspace_config = Path(workspace) / ".capseal" / "operator.json"
            if workspace_config.exists():
                source_path = workspace_config

        # Fallback: try CWD .capseal/operator.json
        if source_path is None:
            cwd_config = Path(".capseal") / "operator.json"
            if cwd_config.exists():
                source_path = cwd_config

        # Try home directory
        if source_path is None:
            home_config = Path.home() / ".capseal" / "operator.json"
            if home_config.exists():
                source_path = home_config

    if source_path:
        loaded_config = _read_json(source_path)
        config = _merge(config, loaded_config)
        print(f"[config] Loaded from {source_path}")
    else:
        print("[config] Using defaults (no config file found)")
        print("[config] Set up channels: capseal operator --setup telegram")
        print("[config] Or set env vars: CAPSEAL_TELEGRAM_BOT_TOKEN, CAPSEAL_TELEGRAM_CHAT_ID")

    # Resolve secret indirections from config, then allow explicit env overrides.
    _resolve_secret_env_refs(config)
    _apply_env_overrides(config)
    _warn_on_plaintext_secrets(loaded_config, source_path)
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
