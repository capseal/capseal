"""Blueprint exposing the CapSeal CLI over HTTP with validation and standard responses."""
from __future__ import annotations

import inspect

from flask import Blueprint, current_app, jsonify, request

from .errors import APIError
from .models import (
    AuditRequest,
    EmitRequest,
    FetchRequest,
    ReplayRequest,
    RunRequest,
    SandboxTestRequest,
    VerifyRequest,
)
from .services import (
    CLIResult,
    capseal_audit,
    capseal_emit,
    capseal_fetch,
    capseal_replay,
    capseal_run,
    capseal_sandbox_status,
    capseal_sandbox_test,
    capseal_verify,
    get_run_events,
)
from .storage import attach_artifacts
from .utils import (
    build_command_response,
    ingest_events_from_response,
    make_request_id,
    map_cli_to_http,
    resolve_event_root,
)

api_bp = Blueprint("capseal_api", __name__)


def _execute(fn, payload) -> tuple[dict, int]:
    request_id = make_request_id()
    result: CLIResult = fn(payload) if payload is not None else fn(None)  # type: ignore[arg-type]
    response = build_command_response(result, request_id)
    store = current_app.extensions.get("artifact_store")
    if store is not None:
        attach_artifacts(response, store)
    ingest_events_from_response(response)
    status = map_cli_to_http(result)
    return response, status


def _json_request() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise APIError(400, "JSON body required")
    return data


def _async_requested() -> bool:
    return request.args.get("async", "false").lower() in {"1", "true", "yes"}


def _maybe_async(description: str, handler, payload):
    if _async_requested():
        manager = current_app.extensions["job_manager"]
        job_payload = payload.model_dump() if hasattr(payload, "model_dump") else payload
        job = manager.submit(description, _handler_path(handler), job_payload)
        job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
        if not job_id:
            raise APIError(500, "job manager failed to return id")
        return jsonify({
            "status": "SUBMITTED",
            "jobId": job_id,
            "location": f"/api/jobs/{job_id}",
        }), 202
    body, status = _execute(handler, payload)
    return jsonify(body), status


def _handler_path(fn) -> str:
    module = inspect.getmodule(fn)
    if not module:
        raise RuntimeError("Handler missing module")
    return f"{module.__name__}.{fn.__name__}"


@api_bp.post("/fetch")
def api_fetch():
    req = FetchRequest.model_validate(_json_request())
    return _maybe_async("fetch", capseal_fetch, req)


@api_bp.post("/run")
def api_run():
    req = RunRequest.model_validate(_json_request())
    return _maybe_async("run", capseal_run, req)


@api_bp.post("/emit")
def api_emit():
    req = EmitRequest.model_validate(_json_request())
    return _maybe_async("emit", capseal_emit, req)


@api_bp.post("/verify")
def api_verify():
    req = VerifyRequest.model_validate(_json_request())
    return _maybe_async("verify", capseal_verify, req)


@api_bp.post("/replay")
def api_replay():
    req = ReplayRequest.model_validate(_json_request())
    return _maybe_async("replay", capseal_replay, req)


@api_bp.post("/audit")
def api_audit():
    req = AuditRequest.model_validate(_json_request())
    return _maybe_async("audit", capseal_audit, req)


@api_bp.get("/runs/<run_id>/events")
def api_run_events(run_id: str):
    event_root = resolve_event_root()
    events = get_run_events(run_id, str(event_root))
    return jsonify({"runId": run_id, "events": events}), 200


@api_bp.get("/sandbox/status")
def api_sandbox_status():
    payload, status = _execute(lambda _: capseal_sandbox_status(), None)
    return jsonify(payload), status


@api_bp.post("/sandbox/test")
def api_sandbox_test():
    req = SandboxTestRequest.model_validate(_json_request())
    payload, status = _execute(lambda _: capseal_sandbox_test(req), None)
    return jsonify(payload), status


@api_bp.get("/jobs/<job_id>")
def api_job_status(job_id: str):
    manager = current_app.extensions["job_manager"]
    job = manager.get(job_id)
    if not job:
        raise APIError(404, "job not found")
    return jsonify(job.to_dict()), 200


@api_bp.get("/jobs")
def api_job_list():
    manager = current_app.extensions["job_manager"]
    limit_raw = request.args.get("limit")
    try:
        limit = int(limit_raw) if limit_raw else 50
    except ValueError:
        raise APIError(400, "limit must be an integer")
    jobs = [job.to_dict() for job in manager.list(limit=limit)]
    return jsonify({"jobs": jobs}), 200


@api_bp.post("/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    manager = current_app.extensions["job_manager"]
    if not manager.cancel(job_id):
        raise APIError(404, "job not found or already finished")
    return jsonify({"jobId": job_id, "status": "canceled"}), 200
