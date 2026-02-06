"""SSE streaming routes for real-time event delivery.

Provides Server-Sent Events (SSE) streaming for:
- Live event log during run execution
- Contract-compliant response formats matching ui/src/contracts/contracts.ts
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

from flask import Blueprint, Response, current_app, jsonify, request, send_file

from .errors import APIError
from .services import (
    capseal_audit,
    capseal_emit,
    capseal_row,
    capseal_verify,
    get_run_events,
)
from .models import AuditRequest, VerifyRequest
from .utils import resolve_event_root

# Import token governance modules
try:
    from bef_zk.capsule.event_log import EventLog
    from bef_zk.capsule.context_pack import BudgetLedger
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False

sse_bp = Blueprint("sse_api", __name__)

_run_dir_cache: dict[str, Path] = {}


def _project_root() -> Path:
    return Path(current_app.config.get("CAPSEAL_PROJECT_ROOT", "."))


def _verification_root() -> Path:
    root = Path(
        current_app.config.get(
            "CAPSEAL_VERIFICATION_ROOT",
            _project_root() / "server_data" / "verifications",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _find_capsule_path(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "strategy_capsule.json",
        run_dir / "capsule.json",
        run_dir / "capsule" / "strategy_capsule.json",
        run_dir / "capsule" / "capsule.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _is_run_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return _find_capsule_path(path) is not None


def _run_events_file(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "events.jsonl",
        run_dir / "events" / "events.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_capsule(run_dir: Path) -> dict:
    capsule_path = _find_capsule_path(run_dir)
    if not capsule_path:
        return {}
    try:
        return json.loads(capsule_path.read_text())
    except json.JSONDecodeError:
        return {}


def _event_metadata(run_dir: Path) -> tuple[str | None, int]:
    events_file = _run_events_file(run_dir)
    if not events_file:
        return None, 0
    last_type = None
    count = 0
    try:
        with open(events_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_type = event.get("type") or event.get("event_type")
    except IOError:
        return None, 0
    return last_type, count


def _load_events(run_dir: Path, limit: int | None = None) -> list[dict]:
    events_file = _run_events_file(run_dir)
    if not events_file:
        return []
    events: list[dict] = []
    try:
        with open(events_file, "r") as handle:
            lines = handle.readlines()
        if limit is not None and limit > 0:
            lines = lines[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except IOError:
        return []
    return events


def _collect_artifacts(run_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    artifact_types = {
        "strategy_capsule.json": "receipt",
        "capsule.json": "receipt",
        "adapter_proof.bin": "proof",
        "adapter_proof.json": "proof",
        "events.jsonl": "events",
        "stc_trace.json": "trace",
        "artifact_manifest.json": "manifest",
    }
    for file in run_dir.iterdir():
        if file.is_file():
            artifact_type = artifact_types.get(file.name, "other")
            artifacts.append(
                {
                    "name": file.name,
                    "path": str(file),
                    "hash": "",
                    "size_bytes": file.stat().st_size,
                    "artifact_type": artifact_type,
                }
            )
    # Surface row archive as directory entry if available
    row_archive = run_dir / "row_archive"
    if row_archive.exists() and row_archive.is_dir():
        artifacts.append(
            {
                "name": "row_archive",
                "path": str(row_archive),
                "hash": "",
                "size_bytes": 0,
                "artifact_type": "trace",
            }
        )
    return artifacts


def _scan_run_directories() -> list[Path]:
    project_root = _project_root()
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for base in [project_root / "out", project_root / "ran_computations", project_root / "capsules"]:
        if not base.exists():
            continue
        for capsule_path in base.rglob("strategy_capsule.json"):
            run_dir = capsule_path.parent
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(run_dir)
            _run_dir_cache.setdefault(run_dir.name, run_dir)
    return run_dirs


def _find_run_directory(run_id: str) -> Path | None:
    cached = _run_dir_cache.get(run_id)
    if cached and cached.exists():
        return cached
    project_root = _project_root()
    candidates = [
        project_root / "out" / run_id,
        project_root / "out" / "capsule_runs" / run_id,
        project_root / "ran_computations" / run_id,
    ]
    for candidate in candidates:
        if _is_run_dir(candidate):
            _run_dir_cache[run_id] = candidate
            return candidate
    for base in [project_root / "out", project_root / "ran_computations", project_root / "capsules"]:
        if not base.exists():
            continue
        for capsule_path in base.rglob("strategy_capsule.json"):
            run_dir = capsule_path.parent
            if run_dir.name == run_id and _is_run_dir(run_dir):
                _run_dir_cache[run_id] = run_dir
                return run_dir
    return None


def _load_verification_state(run_id: str) -> dict | None:
    path = _verification_root() / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _store_verification_state(run_id: str, report: dict) -> None:
    state = {
        "status": (report.get("status") or "").upper(),
        "recorded_at": datetime.utcnow().isoformat() + "Z",
        "report": report,
    }
    path = _verification_root() / f"{run_id}.json"
    path.write_text(json.dumps(state, indent=2))


def _verification_status(run_id: str) -> str:
    state = _load_verification_state(run_id)
    if not state:
        return "unverified"
    status = (state.get("status") or state.get("report", {}).get("status") or "").lower()
    if status == "verified":
        return "verified"
    if status in {"rejected", "failed", "error"}:
        return "failed"
    if status:
        return status
    return "unverified"


def _build_run_summary(run_dir: Path, capsule: dict | None = None) -> dict:
    capsule_data = capsule or _load_capsule(run_dir)
    policy = capsule_data.get("policy", {}) if isinstance(capsule_data, dict) else {}
    proof_system = capsule_data.get("proof_system", {}) if isinstance(capsule_data, dict) else {}
    project_id = policy.get("track_id") or capsule_data.get("track_id") or "default"
    created_at = capsule_data.get("created_at")
    if isinstance(created_at, (int, float)):
        created_at = datetime.fromtimestamp(created_at).isoformat() + "Z"
    if not created_at:
        created_at = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat() + "Z"
    last_event, event_count = _event_metadata(run_dir)
    return {
        "run_id": run_dir.name,
        "project_id": project_id,
        "track_id": project_id,
        "policy_id": policy.get("policy_id"),
        "policy_hash": policy.get("policy_hash"),
        "backend": proof_system.get("backend_id", "unknown"),
        "capsule_hash": capsule_data.get("capsule_hash"),
        "verification_status": _verification_status(run_dir.name),
        "created_at": created_at,
        "last_event": last_event,
        "event_count": event_count,
    }


def _safe_run_path(run_dir: Path, relative: str) -> Path:
    candidate = (run_dir / relative).resolve()
    run_root = run_dir.resolve()
    if not str(candidate).startswith(str(run_root)):
        raise APIError(403, "invalid artifact path")
    if not candidate.exists():
        raise APIError(404, "artifact not found")
    return candidate


@sse_bp.get("/health")
def api_health():
    capabilities = {
        "run": True,
        "verify": True,
        "audit": True,
        "evidence": True,
        "export": True,
        "sse": True,
    }
    return jsonify({
        "ok": True,
        "version": current_app.config.get("CAPSEAL_VERSION", "dev"),
        "capabilities": capabilities,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


def _stream_events(events_file: Path, run_id: str, last_event_id: str | None = None) -> Generator[str, None, None]:
    """Generator that tails events.jsonl and yields SSE-formatted events.

    Handles:
    - File not yet created (waits with heartbeat)
    - New lines appended during streaming
    - Clean termination on run_completed event
    - Reconnection via Last-Event-ID header
    """
    last_seq = 0

    # Check for Last-Event-ID header for reconnection (passed in from route)
    if last_event_id:
        try:
            last_seq = int(last_event_id)
        except ValueError:
            pass

    # Send initial connection event
    yield f"event: connected\ndata: {json.dumps({'run_id': run_id, 'resume_from': last_seq})}\n\n"

    wait_count = 0
    max_wait = 60  # Max 60 seconds waiting for file

    while wait_count < max_wait:
        if not events_file.exists():
            # Send heartbeat while waiting for file
            yield f": heartbeat waiting for events file\n\n"
            time.sleep(1)
            wait_count += 1
            continue

        try:
            with open(events_file, "r") as f:
                # Skip to where we left off
                seen = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    seq = event.get("seq", 0)
                    if seq <= last_seq:
                        continue

                    last_seq = seq
                    seen += 1

                    # Format as SSE
                    event_type = event.get("event_type", "event")
                    yield f"id: {seq}\nevent: {event_type}\ndata: {json.dumps(event)}\n\n"

                    # Check for terminal events
                    if event_type in ("run_completed", "run_failed", "run_cancelled"):
                        yield f"event: stream_end\ndata: {json.dumps({'reason': event_type})}\n\n"
                        return

                # If we read events, continue tailing
                if seen > 0:
                    wait_count = 0  # Reset wait counter
                else:
                    # No new events, send heartbeat
                    yield f": heartbeat seq={last_seq}\n\n"
                    time.sleep(0.5)
                    wait_count += 0.5

        except IOError:
            yield f": heartbeat io-retry\n\n"
            time.sleep(1)
            wait_count += 1

    # Timeout - send final event
    yield f"event: stream_timeout\ndata: {json.dumps({'last_seq': last_seq})}\n\n"


@sse_bp.get("/runs/<run_id>/events/stream")
def api_events_stream(run_id: str):
    """SSE endpoint for streaming run events in real-time.

    Supports:
    - Last-Event-ID header for reconnection
    - Heartbeats while waiting for events
    - Automatic termination on run completion
    """
    # Capture request context values before entering generator
    last_event_id = request.headers.get("Last-Event-ID")

    event_root = Path(resolve_event_root())
    events_file = event_root / run_id / "events.jsonl"
    flat_events = event_root / f"{run_id}.jsonl"
    if flat_events.exists():
        events_file = flat_events

    run_dir = _find_run_directory(run_id)
    if run_dir:
        candidate = _run_events_file(run_dir)
        if candidate:
            events_file = candidate

    return Response(
        _stream_events(events_file, run_id, last_event_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@sse_bp.get("/runs/<run_id>")
def api_get_run(run_id: str):
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")

    capsule = _load_capsule(run_dir)
    summary = _build_run_summary(run_dir, capsule)
    event_limit = request.args.get("event_limit", type=int)
    events = _load_events(run_dir, limit=event_limit or 500)
    artifacts = _collect_artifacts(run_dir)
    verification = _load_verification_state(run_id)

    response = {
        **summary,
        "capsule": capsule,
        "events": events,
        "artifacts": artifacts,
        "verification": verification,
    }
    return jsonify(response), 200


@sse_bp.post("/runs/<run_id>/verify")
def api_verify_run(run_id: str):
    """Verify a run and return VerifyReport matching contracts.ts.

    Returns:
        VerifyReport with layers, errors, timings matching the contract schema.
    """
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    capsule_path = _find_capsule_path(run_dir)
    if not capsule_path:
        raise APIError(404, f"Capsule not found for run {run_id}")

    # Get optional params from request body
    body = request.get_json(silent=True) or {}

    req = VerifyRequest(
        capsulePath=str(capsule_path),
        mode=body.get("mode", "proof-only"),
        policyPath=body.get("policy_path"),
    )

    start_time = time.time()
    result = capseal_verify(req)
    total_ms = (time.time() - start_time) * 1000

    # Transform to contract-compliant VerifyReport
    payload = result.payload or {}

    # Determine layer statuses
    def layer_status(key: str, fallback_keys: list[str] = None) -> dict:
        value = payload.get(key)
        if value is None and fallback_keys:
            for fk in fallback_keys:
                value = payload.get(fk)
                if value is not None:
                    break

        if value is True:
            return {"status": "pass"}
        elif value is False:
            return {"status": "fail", "message": payload.get(f"{key}_error", "")}
        else:
            return {"status": "unknown"}

    verify_report = {
        "run_id": run_id,
        "status": "verified" if result.returncode == 0 else "rejected",
        "exit_code": result.returncode,
        "layers": {
            "l0_hash": layer_status("hash_ok", ["capsule_hash_ok"]),
            "l1_commitment": layer_status("row_index_commitment_ok", ["merkle_ok"]),
            "l2_constraint": layer_status("policy_verified", ["policy_ok"]),
            "l3_proximity": layer_status("proof_verified", ["fri_ok"]),
            "l4_receipt": layer_status("capsule_valid", ["schema_ok"]),
        },
        "errors": [],
        "timings": {
            "total_ms": round(total_ms, 2),
            "parse_ms": payload.get("parse_ms"),
            "proof_verify_ms": payload.get("proof_verify_ms"),
            "merkle_verify_ms": payload.get("merkle_verify_ms"),
        },
        "proof_size_bytes": payload.get("proof_size_bytes"),
        "backend_id": payload.get("backend_id", "unknown"),
        "proof_system_id": payload.get("proof_system_id", "stark_fri"),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }

    # Extract errors from result
    if result.returncode != 0:
        error_code = payload.get("error_code", f"E{result.returncode:03d}_UNKNOWN")
        verify_report["errors"].append({
            "code": error_code,
            "message": payload.get("error_message", result.stderr.strip()[:200]),
            "hint": payload.get("error_hint"),
        })

    status = 200 if result.returncode == 0 else 400
    try:
        _store_verification_state(run_id, verify_report)
    except Exception:
        # Persistence failure should not break the API response
        pass
    return jsonify(verify_report), status


@sse_bp.get("/runs/<run_id>/artifacts")
def api_list_run_artifacts(run_id: str):
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    return jsonify({"artifacts": _collect_artifacts(run_dir)}), 200


@sse_bp.get("/runs/<run_id>/artifacts/<path:artifact_name>")
def api_download_artifact(run_id: str, artifact_name: str):
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    path = _safe_run_path(run_dir, artifact_name)
    return send_file(path, as_attachment=True, download_name=path.name)


@sse_bp.get("/runs/<run_id>/capsule")
def api_download_capsule(run_id: str):
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    capsule_path = _find_capsule_path(run_dir)
    if not capsule_path or not capsule_path.exists():
        raise APIError(404, f"Capsule not found for run {run_id}")
    return send_file(capsule_path, as_attachment=True, download_name=capsule_path.name)


@sse_bp.get("/runs/<run_id>/rows/<int:row_number>")
def api_open_row(run_id: str, row_number: int):
    if row_number < 0:
        raise APIError(400, "row index must be non-negative")
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    capsule_path = _find_capsule_path(run_dir)
    if not capsule_path:
        raise APIError(404, f"Capsule not found for run {run_id}")
    schema_id = request.args.get("schema_id")
    result = capseal_row(
        {
            "capsulePath": str(capsule_path),
            "row": row_number,
            "schemaId": schema_id,
        }
    )
    if result.returncode != 0 or not isinstance(result.payload, dict):
        raise APIError(500, result.stderr.strip() or "row opening failed")
    return jsonify(result.payload), 200


@sse_bp.post("/runs/<run_id>/export")
def api_export_run(run_id: str):
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")
    capsule_path = _find_capsule_path(run_dir)
    if not capsule_path:
        raise APIError(404, f"Capsule not found for run {run_id}")

    body = request.get_json(silent=True) or {}
    export_format = (body.get("format") or "capsule").lower()
    if export_format not in {"capsule", "receipt", "proof", "full"}:
        raise APIError(400, "invalid export format")

    timestamp = time.strftime("%Y%m%d%H%M%S")
    export_name = f"{run_id}-{export_format}-{timestamp}.cap"
    out_path = run_dir / export_name
    archive_dir = run_dir / "row_archive"
    policy_path = run_dir / "policy.json"
    manifests_dir = run_dir / "manifests"
    artifacts_dir = run_dir / "proofs"
    if not artifacts_dir.exists():
        artifacts_dir = run_dir

    emit_req = {
        "outPath": str(out_path),
        "capsulePath": str(capsule_path),
        "source": str(run_dir),
        "artifactsDir": str(artifacts_dir) if artifacts_dir.exists() else None,
        "archiveDir": str(archive_dir) if archive_dir.exists() else None,
        "policyPath": str(policy_path) if policy_path.exists() else None,
        "manifestsDir": str(manifests_dir) if manifests_dir.exists() else None,
        "profile": body.get("profile", "proof-only"),
    }
    result = capseal_emit(emit_req)
    if result.returncode != 0:
        raise APIError(500, result.stderr.strip() or "export failed")

    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    return jsonify(
        {
            "run_id": run_id,
            "format": export_format,
            "artifact": export_name,
            "size_bytes": size_bytes,
            "download_url": f"/api/runs/{run_id}/artifacts/{export_name}",
        }
    ), 200


@sse_bp.get("/runs/<run_id>/audit")
def api_audit_run(run_id: str):
    """Audit a run and return AuditReport matching contracts.ts.

    Returns:
        AuditReport with chain validity, event counts, timeline.
    """
    # Find the capsule file
    project_root = Path(current_app.config.get("CAPSEAL_PROJECT_ROOT", "."))
    capsule_paths = [
        project_root / "out" / run_id / "strategy_capsule.json",
        project_root / "ran_computations" / run_id / "strategy_capsule.json",
    ]

    capsule_path = None
    for path in capsule_paths:
        if path.exists():
            capsule_path = path
            break

    if not capsule_path:
        raise APIError(404, f"Capsule not found for run {run_id}")

    req = AuditRequest(capsulePath=str(capsule_path), format="json")
    result = capseal_audit(req)

    payload = result.payload or {}

    # Transform to contract-compliant AuditReport
    events = payload.get("events", [])
    event_counts = {}
    timeline = []

    for event in events:
        event_type = event.get("type") or event.get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        timeline.append({
            "seq": event.get("seq", 0),
            "event_type": event_type,
            "event_hash": event.get("event_hash", event.get("hash", ""))[:16],
            "prev_hash": event.get("prev_event_hash", event.get("prev_hash", ""))[:16],
            "timestamp": event.get("timestamp"),
        })

    audit_report = {
        "run_id": run_id,
        "chain_valid": payload.get("chain_valid", payload.get("hash_chain_valid", True)),
        "chain_length": len(events),
        "genesis_hash": payload.get("genesis_hash", "0" * 64),
        "head_hash": events[-1].get("event_hash", "")[:16] if events else "",
        "event_counts": event_counts,
        "timeline": timeline,
        "first_event": timeline[0] if timeline else None,
        "last_event": timeline[-1] if timeline else None,
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }

    return jsonify(audit_report), 200


@sse_bp.get("/runs/<run_id>/evidence")
def api_evidence_run(run_id: str):
    """Get evidence index for a run matching contracts.ts EvidenceIndex.

    Returns:
        EvidenceIndex with openable rows, artifacts, evidence status.
    """
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")

    # Load capsule metadata
    capsule_path = _find_capsule_path(run_dir)
    capsule_data = {}
    if capsule_path and capsule_path.exists():
        try:
            capsule_data = json.loads(capsule_path.read_text())
        except json.JSONDecodeError:
            pass

    artifacts = _collect_artifacts(run_dir)

    # Scan row archive
    row_archive = run_dir / "row_archive"
    openable_rows = []
    row_count = 0

    if row_archive.exists():
        for chunk_file in sorted(row_archive.glob("chunk_*.json")):
            try:
                chunk_data = json.loads(chunk_file.read_text())
                # Handle both list format and dict format
                if isinstance(chunk_data, list):
                    rows_in_chunk = len(chunk_data)
                elif isinstance(chunk_data, dict):
                    rows_in_chunk = len(chunk_data.get("rows", []))
                else:
                    rows_in_chunk = 1
                for i in range(rows_in_chunk):
                    openable_rows.append({
                        "row_index": row_count + i,
                        "chunk_file": str(chunk_file.relative_to(run_dir)),
                        "has_proof": True,
                    })
                row_count += rows_in_chunk
            except (json.JSONDecodeError, IOError):
                pass

    # Build evidence index
    evidence_index = {
        "run_id": run_id,
        "capsule_hash": capsule_data.get("capsule_hash", ""),
        "row_root": capsule_data.get("row_index_ref", {}).get("commitment", ""),
        "events_root": None,
        "openable_rows": openable_rows[:100],  # Limit for response size
        "row_count": row_count,
        "tree_arity": capsule_data.get("row_index_ref", {}).get("tree_arity", 16),
        "artifacts": artifacts,
        "evidence_status": {
            "binding": "unknown",
            "availability": "pass" if artifacts else "unknown",
            "enforcement": "unknown",
            "determinism": "unknown",
            "replayable": "unknown",
        },
    }

    return jsonify(evidence_index), 200


@sse_bp.get("/runs")
def api_list_runs():
    """List all runs as RunSummary objects matching contracts.ts."""
    project_filter = request.args.get("project_id")
    if project_filter:
        project_filter = project_filter.lower()
        if project_filter in {"", "default"}:
            project_filter = None
    status_filter = request.args.get("status")
    if status_filter:
        status_filter = status_filter.lower()

    runs: list[dict] = []
    for run_dir in _scan_run_directories():
        capsule = _load_capsule(run_dir)
        summary = _build_run_summary(run_dir, capsule)
        project_id = (summary.get("project_id") or "").lower()
        if project_filter and project_id != project_filter:
            continue
        status = (summary.get("verification_status") or "").lower()
        if status_filter and status != status_filter:
            continue
        runs.append(summary)

    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", type=int) or 0
    if limit is not None and limit >= 0:
        runs = runs[offset : offset + limit]
    elif offset:
        runs = runs[offset:]

    return jsonify({"runs": runs}), 200


@sse_bp.get("/runs/<run_id>/budget")
def api_budget_run(run_id: str):
    """Get budget/token governance summary matching contracts.ts BudgetSummary.

    Returns token spend, oracle calls, and budget status from event log.
    """
    run_dir = _find_run_directory(run_id)
    if not run_dir:
        raise APIError(404, f"Run directory not found for {run_id}")

    # Try to load event log for token governance data
    events_file = _run_events_file(run_dir) or (run_dir / "events.jsonl")
    oracle_calls = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost_usd = 0.0

    if events_file.exists() and HAS_GOVERNANCE:
        try:
            event_log = EventLog(run_dir)
            spend = event_log.get_total_token_spend()
            total_tokens_in = spend.get("tokens_in", 0)
            total_tokens_out = spend.get("tokens_out", 0)
            total_cost_usd = spend.get("cost_usd", 0.0)

            # Get oracle call details
            for event in event_log.get_oracle_calls():
                oracle_calls.append({
                    "seq": event.seq,
                    "oracle_id": event.data.get("oracle_id", "unknown"),
                    "call_id": event.data.get("call_id", ""),
                    "context_root": event.data.get("context_root", "")[:16],
                    "tokens_in": event.data.get("tokens_in", 0),
                    "tokens_out": event.data.get("tokens_out", 0),
                    "model": event.data.get("model", ""),
                    "cost_usd": event.data.get("cost_usd", 0.0),
                    "latency_ms": event.data.get("latency_ms"),
                    "success": event.data.get("success", True),
                    "timestamp": event.timestamp,
                })
        except Exception:
            pass  # Fall through to manual parsing
    elif events_file.exists():
        # Manual parsing fallback
        try:
            with open(events_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("event_type") == "oracle_call" or event.get("type") == "oracle_call":
                            data = event.get("data", event)
                            total_tokens_in += data.get("tokens_in", 0)
                            total_tokens_out += data.get("tokens_out", 0)
                            total_cost_usd += data.get("cost_usd", 0.0)
                            oracle_calls.append({
                                "seq": event.get("seq", 0),
                                "oracle_id": data.get("oracle_id", "unknown"),
                                "call_id": data.get("call_id", ""),
                                "context_root": data.get("context_root", "")[:16] if data.get("context_root") else "",
                                "tokens_in": data.get("tokens_in", 0),
                                "tokens_out": data.get("tokens_out", 0),
                                "model": data.get("model", ""),
                                "cost_usd": data.get("cost_usd", 0.0),
                                "latency_ms": data.get("latency_ms"),
                                "success": data.get("success", True),
                                "timestamp": event.get("timestamp"),
                            })
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

    # Load budget constraints from capsule if available
    capsule_path = run_dir / "strategy_capsule.json"
    budget_tokens = 1_000_000  # Default
    budget_calls = 100
    budget_usd = 50.0

    if capsule_path.exists():
        try:
            capsule = json.loads(capsule_path.read_text())
            budget_config = capsule.get("budget", {})
            budget_tokens = budget_config.get("tokens", budget_tokens)
            budget_calls = budget_config.get("oracle_calls", budget_calls)
            budget_usd = budget_config.get("usd", budget_usd)
        except (json.JSONDecodeError, IOError):
            pass

    budget_summary = {
        "run_id": run_id,
        "budget": {
            "tokens": budget_tokens,
            "oracle_calls": budget_calls,
            "usd": budget_usd,
        },
        "spent": {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "tokens_total": total_tokens_in + total_tokens_out,
            "oracle_calls": len(oracle_calls),
            "usd": round(total_cost_usd, 4),
        },
        "remaining": {
            "tokens": max(0, budget_tokens - (total_tokens_in + total_tokens_out)),
            "oracle_calls": max(0, budget_calls - len(oracle_calls)),
            "usd": round(max(0.0, budget_usd - total_cost_usd), 4),
        },
        "utilization": {
            "tokens_pct": round((total_tokens_in + total_tokens_out) / budget_tokens * 100, 2) if budget_tokens > 0 else 0,
            "calls_pct": round(len(oracle_calls) / budget_calls * 100, 2) if budget_calls > 0 else 0,
            "usd_pct": round(total_cost_usd / budget_usd * 100, 2) if budget_usd > 0 else 0,
        },
        "oracle_calls": oracle_calls,
        "governance_enabled": HAS_GOVERNANCE,
    }

    return jsonify(budget_summary), 200
