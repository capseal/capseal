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
            "chat_id": None,
            "voice_notes": False,
            "decision_buttons": True,
        },
        "whatsapp": {
            "phone_number_id": None,
            "access_token": None,
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
    config = DEFAULT_CONFIG.copy()

    # Try explicit path first
    if config_path and config_path.exists():
        config = _merge(config, _read_json(config_path))
        print(f"[config] Loaded from {config_path}")
        return config

    # Try workspace .capseal/operator.json (use actual workspace, not CWD)
    if workspace:
        workspace_config = Path(workspace) / ".capseal" / "operator.json"
        if workspace_config.exists():
            config = _merge(config, _read_json(workspace_config))
            print(f"[config] Loaded from {workspace_config}")
            return config

    # Fallback: try CWD .capseal/operator.json
    cwd_config = Path(".capseal") / "operator.json"
    if cwd_config.exists():
        config = _merge(config, _read_json(cwd_config))
        print(f"[config] Loaded from {cwd_config}")
        return config

    # Try home directory
    home_config = Path.home() / ".capseal" / "operator.json"
    if home_config.exists():
        config = _merge(config, _read_json(home_config))
        print(f"[config] Loaded from {home_config}")
        return config

    # Environment variable overrides
    import os
    if os.environ.get("CAPSEAL_TELEGRAM_TOKEN"):
        config.setdefault("channels", {}).setdefault("telegram", {})
        config["channels"]["telegram"]["bot_token"] = os.environ["CAPSEAL_TELEGRAM_TOKEN"]
    if os.environ.get("CAPSEAL_TELEGRAM_CHAT_ID"):
        config.setdefault("channels", {}).setdefault("telegram", {})
        config["channels"]["telegram"]["chat_id"] = os.environ["CAPSEAL_TELEGRAM_CHAT_ID"]
    if os.environ.get("CAPSEAL_NOTIFY_THRESHOLD"):
        config["notify_threshold"] = float(os.environ["CAPSEAL_NOTIFY_THRESHOLD"])
    if os.environ.get("CAPSEAL_VOICE_ENABLED"):
        config.setdefault("voice", {})["enabled"] = True
    if os.environ.get("CAPSEAL_VOICE_PROVIDER"):
        config.setdefault("voice", {})["provider"] = os.environ["CAPSEAL_VOICE_PROVIDER"]

    print("[config] Using defaults (no config file found)")
    print("[config] Set up channels: capseal operator --setup telegram")
    print("[config] Or set env vars: CAPSEAL_TELEGRAM_TOKEN, CAPSEAL_TELEGRAM_CHAT_ID")
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
