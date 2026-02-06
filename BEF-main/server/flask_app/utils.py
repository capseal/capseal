"""Utility helpers for the CapSeal Flask layer."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import current_app, has_app_context

from .services import CLIResult

DEFAULT_EVENT_SUBDIR = Path("server_data/events")


ERROR_CODE_HTTP = {
    "E001": 400,
    "E002": 400,
    "E003": 422,
    "E004": 422,
    "E010": 409,
    "E011": 409,
    "E012": 409,
    "E013": 409,
    "E020": 403,
    "E021": 403,
    "E022": 403,
    "E030": 412,
    "E031": 412,
    "E032": 412,
    "E033": 412,
    "E034": 412,
    "E036": 412,
    "E040": 412,
    "E041": 412,
    "E050": 422,
    "E051": 422,
    "E052": 422,
    "E053": 422,
    "E054": 422,
    "E055": 422,
    "E060": 422,
    "E061": 422,
    "E062": 422,
    "E063": 422,
    "E064": 422,
    "E065": 422,
    "E066": 422,
    "E070": 424,
    "E071": 424,
    "E072": 424,
    "E073": 424,
    "E074": 424,
    "E075": 424,
    "E076": 424,
    "E077": 424,
    "E101": 412,
    "E102": 412,
    "E103": 412,
    "E104": 412,
    "E105": 412,
    "E106": 412,
    "E107": 412,
    "E108": 412,
    "E109": 412,
    "E120": 412,
    "E201": 409,
    "E202": 409,
    "E204": 409,
    "E205": 409,
    "E301": 412,
    "E302": 412,
    "E303": 412,
}


def make_request_id() -> str:
    return uuid.uuid4().hex


def resolve_project_root() -> Path:
    if has_app_context():
        path = current_app.config.get("CAPSEAL_PROJECT_ROOT")
        if path:
            return Path(path)
    return Path(__file__).resolve().parents[2]


def resolve_event_root() -> Path:
    if has_app_context():
        root = current_app.config.get("CAPSEAL_EVENT_ROOT")
        if root:
            return Path(root)
    return resolve_project_root() / DEFAULT_EVENT_SUBDIR


def map_cli_to_http(result: CLIResult) -> int:
    if result.returncode == 0:
        return 200
    payload = result.payload or {}
    error_code = payload.get("error_code") if isinstance(payload, dict) else None
    if error_code:
        for prefix, status in ERROR_CODE_HTTP.items():
            if error_code.startswith(prefix):
                return status
    return 500


def build_command_response(result: CLIResult, request_id: str) -> Dict[str, Any]:
    payload = result.payload if isinstance(result.payload, dict) else None
    response = {
        "requestId": request_id,
        "status": "SUCCESS" if result.returncode == 0 else "ERROR",
        "exitCode": result.returncode,
        "errorCode": payload.get("error_code") if payload else None,
        "command": result.command,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if payload:
        response["result"] = payload
    return response


def ready_checks() -> dict[str, str]:
    root = resolve_project_root()
    checks: dict[str, str] = {}
    fetch_script = root / "scripts" / "fetch_dataset.py"
    checks["fetch_script"] = "ok" if fetch_script.exists() else "missing"
    event_root = resolve_event_root()
    try:
        event_root.mkdir(parents=True, exist_ok=True)
        checks["event_store"] = "ok"
    except OSError:
        checks["event_store"] = "unwritable"
    cli_entry = root / "bef_zk" / "capsule" / "cli" / "__init__.py"
    checks["cli"] = "ok" if cli_entry.exists() else "missing"
    return checks


def readiness_status() -> tuple[str, dict[str, str]]:
    checks = ready_checks()
    status = "ready" if all(value == "ok" for value in checks.values()) else "not_ready"
    return status, checks


def redact_path(path: str | None) -> str | None:
    if not path:
        return None
    root = resolve_project_root()
    try:
        rel = Path(path).resolve().relative_to(root)
        return str(rel)
    except ValueError:
        return path


def ensure_api_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def ingest_events_from_response(response: dict) -> None:
    result = response.get("result")
    if not isinstance(result, dict):
        return
    trace_id = result.get("trace_id") or result.get("traceId")
    output_dir = result.get("output_dir") or result.get("outputDir")
    if not trace_id or not output_dir:
        return
    project_root = resolve_project_root()
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = project_root / output_dir
    events_path = out_path / "events.jsonl"
    if not events_path.exists():
        return
    event_root = resolve_event_root()
    event_root.mkdir(parents=True, exist_ok=True)
    dest = event_root / f"{trace_id}.jsonl"
    shutil.copy2(events_path, dest)
    store = None
    if has_app_context():
        store = current_app.extensions.get("event_store")
    if store:
        store.load_existing()
