"""RunPod helpers for CapSeal operator voice lifecycle."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def resolve_runpod_pod_id(config: dict[str, Any]) -> str | None:
    """Resolve pod id from loaded operator config."""
    raw = (
        config.get("runpod_pod_id")
        or config.get("voice", {}).get("runpod_pod_id")
        or ""
    )
    value = str(raw).strip()
    return value or None


def resolve_runpod_api_key(config: dict[str, Any]) -> str | None:
    """Resolve API key from config or environment."""
    raw = config.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY") or ""
    value = str(raw).strip()
    return value or None


def _import_runpod(api_key: str):
    try:
        import runpod  # type: ignore
    except ImportError as exc:
        raise RuntimeError("runpod package is not installed") from exc
    runpod.api_key = api_key
    return runpod


def get_pod_status(api_key: str, pod_id: str) -> dict[str, Any]:
    runpod = _import_runpod(api_key)
    payload = runpod.get_pod(pod_id)
    data = payload if isinstance(payload, dict) else {}
    runtime = data.get("runtime") or {}
    ports = runtime.get("ports") or []
    return {
        "pod_id": pod_id,
        "status": str(data.get("desiredStatus") or data.get("status") or "UNKNOWN").upper(),
        "network_ready": bool(ports),
        "raw": _redact_payload(data),
    }


_SENSITIVE_KEY_RE = re.compile(r"(token|secret|password|api[_-]?key|private[_-]?key|auth)", re.I)


def _redact_env_entry(entry: str) -> str:
    if "=" not in entry:
        return entry
    key, value = entry.split("=", 1)
    if _SENSITIVE_KEY_RE.search(key):
        return f"{key}=***"
    # Truncate very long env values to avoid exposing large key blobs.
    if len(value) > 64:
        return f"{key}={value[:12]}...{value[-8:]}"
    return entry


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                out[key] = "***"
                continue
            if str(key).lower() == "env" and isinstance(inner, list):
                out[key] = [_redact_env_entry(str(x)) for x in inner]
                continue
            out[key] = _redact_payload(inner)
        return out
    if isinstance(value, list):
        return [_redact_payload(x) for x in value]
    return value


def stop_pod(api_key: str, pod_id: str) -> None:
    runpod = _import_runpod(api_key)
    runpod.stop_pod(pod_id)


def resume_pod(api_key: str, pod_id: str) -> None:
    runpod = _import_runpod(api_key)
    resume_fn = getattr(runpod, "resume_pod", None)
    if callable(resume_fn):
        try:
            # Newer SDKs expect resume_pod(pod_id, gpu_count)
            resume_fn(pod_id)
        except TypeError:
            resume_fn(pod_id, 1)
        return
    start_fn = getattr(runpod, "start_pod", None)
    if callable(start_fn):
        start_fn(pod_id)
        return
    raise RuntimeError("runpod SDK has no resume_pod/start_pod function")


def terminate_pod(api_key: str, pod_id: str) -> None:
    runpod = _import_runpod(api_key)
    runpod.terminate_pod(pod_id)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def operator_config_paths(workspace: Path) -> list[Path]:
    """Preferred operator config search order."""
    return [workspace / ".capseal" / "operator.json", Path.home() / ".capseal" / "operator.json"]


def load_operator_config(workspace: Path) -> tuple[Path, dict[str, Any]]:
    """Load the first existing operator config, fallback to workspace path."""
    for path in operator_config_paths(workspace):
        if path.exists():
            return path, _read_json(path)
    target = workspace / ".capseal" / "operator.json"
    return target, _read_json(target)


def clear_pod_id_in_config(path: Path) -> None:
    """Remove persisted pod id after teardown."""
    cfg = _read_json(path)
    if not cfg:
        return
    cfg.pop("runpod_pod_id", None)
    voice = cfg.get("voice")
    if isinstance(voice, dict):
        voice.pop("runpod_pod_id", None)
    _write_json(path, cfg)
