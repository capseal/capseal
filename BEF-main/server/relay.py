"""FastAPI relay for CapsuleBench live events."""
from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import subprocess
import tarfile
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Deque, Dict, List, Set

from fastapi import BackgroundTasks, Body, Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:  # Optional dependency for R2 downloads
    import boto3
except ImportError:  # pragma: no cover - optional
    boto3 = None

try:  # Optional for DA relay signing
    from coincurve import PrivateKey
except ImportError:  # pragma: no cover - optional
    PrivateKey = None

from bef_zk.capsule.da import build_da_challenge, challenge_signature_payload
from .event_store import EventStore
try:
    from .event_store_pg import PostgresEventStore
except ImportError:  # pragma: no cover - psycopg optional locally
    PostgresEventStore = None


def _build_store() -> EventStore:
    dsn = os.environ.get("DATABASE_URL")
    if dsn and PostgresEventStore is not None:
        try:
            print("[relay] using PostgresEventStore", flush=True)
            return PostgresEventStore(dsn)
        except Exception as exc:  # pragma: no cover - log fallback
            print(f"[relay] PostgresEventStore init failed: {exc}; falling back", flush=True)
    path_store = EventStore(Path("server_data/events"))
    path_store.load_existing()
    return path_store


app = FastAPI(title="CapsuleBench Relay")
store = _build_store()
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", "server_data/artifacts")).resolve()
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

RELAY_ADMIN_TOKEN = os.environ.get("RELAY_ADMIN_TOKEN")
INGEST_TOKEN_TTL = int(os.environ.get("INGEST_TOKEN_TTL", "3600"))
RELAY_WS_BASE = os.environ.get("RELAY_WS_BASE", "ws://localhost:8000")
RATE_LIMIT_MAX = int(os.environ.get("EVENT_RATE_LIMIT_MAX", "120"))
RATE_LIMIT_WINDOW = int(os.environ.get("EVENT_RATE_LIMIT_WINDOW", "10"))
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_PREFIX = os.environ.get("R2_PREFIX", "")
REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_ROOT = Path(os.environ.get("VERIFICATION_ROOT", "server_data/verifications")).resolve()
VERIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
_r2_client = None
CAPSULEPACK_MAX_BYTES = int(os.environ.get("CAPSULEPACK_MAX_BYTES", str(512 * 1024 * 1024)))
DA_RELAY_ID = os.environ.get("DA_RELAY_ID", "relay_main_v1")
_DA_RELAY_PRIV_HEX = os.environ.get("DA_CHALLENGE_PRIVATE_KEY")
_DA_RELAY_KEY = (
    PrivateKey(bytes.fromhex(_DA_RELAY_PRIV_HEX))
    if _DA_RELAY_PRIV_HEX and PrivateKey is not None
    else None
)


@dataclass
class IngestToken:
    token: str
    expires_at: float


_ingest_tokens: Dict[str, IngestToken] = {}
_token_lock = asyncio.Lock()
_rate_counters: Dict[str, Deque[float]] = defaultdict(deque)
_rate_lock = asyncio.Lock()

class TokenRequest(BaseModel):
    ttl_seconds: int | None = None


class TokenResponse(BaseModel):
    ingest_url: str
    token: str
    expires_at: float


class ArtifactEntry(BaseModel):
    name: str
    size_bytes: int
    content_type: str
    storage: str = "local"
    object_key: str | None = None


class ArtifactManifest(BaseModel):
    artifacts: List[ArtifactEntry]


class DACommitRequest(BaseModel):
    capsule_commit_hash: str
    payload_hash: str
    chunk_handles_root: str | None = None
    num_chunks: int | None = None


class DAChallengeRequest(BaseModel):
    capsule_commit_hash: str


_subscriptions: Dict[str, Set[WebSocket]] = {}
_sub_lock = asyncio.Lock()
_DA_COMMITS: Dict[str, dict] = {}


async def _issue_ingest_token(run_id: str, ttl: int | None = None) -> IngestToken:
    token_value = secrets.token_urlsafe(32)
    expires = time.time() + (ttl or INGEST_TOKEN_TTL)
    record = IngestToken(token=token_value, expires_at=expires)
    async with _token_lock:
        _ingest_tokens[run_id] = record
    return record


async def _validate_ingest_token(run_id: str, presented: str | None) -> bool:
    if not presented:
        return False
    async with _token_lock:
        record = _ingest_tokens.get(run_id)
    if not record:
        return False
    if record.token != presented:
        return False
    if record.expires_at < time.time():
        return False
    return True


def _require_admin(token: str | None) -> None:
    if RELAY_ADMIN_TOKEN and token != f"Bearer {RELAY_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


async def _rate_limit(identifier: str) -> None:
    now = time.time()
    async with _rate_lock:
        window = _rate_counters[identifier]
        while window and now - window[0] > RATE_LIMIT_WINDOW:
            window.popleft()
        if len(window) >= RATE_LIMIT_MAX:
            raise HTTPException(status_code=429, detail="rate limited")
        window.append(now)


async def rate_limit_dependency(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    identifier = f"{client}:{request.url.path}"
    await _rate_limit(identifier)


async def _broadcast(run_id: str, message: str) -> None:
    async with _sub_lock:
        clients = list(_subscriptions.get(run_id, set()))
    for ws in clients:
        try:
            await ws.send_text(message)
        except RuntimeError:
            pass


@app.websocket("/ws/ingest/{run_id}")
async def ingest_socket(ws: WebSocket, run_id: str) -> None:
    token = ws.query_params.get("token") or ws.headers.get("x-ingest-token")
    if not await _validate_ingest_token(run_id, token):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        while True:
            message = await ws.receive_text()
            await asyncio.to_thread(store.append, run_id, message)
            await _broadcast(run_id, message)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/subscribe/{run_id}")
async def subscribe_socket(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    async with _sub_lock:
        _subscriptions.setdefault(run_id, set()).add(ws)
    try:
        history = store.history(run_id)
        for entry in history:
            await ws.send_text(entry)
        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        pass
    finally:
        async with _sub_lock:
            _subscriptions.get(run_id, set()).discard(ws)


@app.get("/runs/{run_id}/events")
async def get_events(
    run_id: str,
    after_seq: int | None = Query(default=None),
    _: None = Depends(rate_limit_dependency),
) -> JSONResponse:
    history = store.history(run_id, after_seq)
    payload = []
    for line in history:
        try:
            payload.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse(payload)


@app.post("/v1/da/commit")
async def register_da_commit(request: DACommitRequest) -> dict:
    _DA_COMMITS[request.capsule_commit_hash] = {
        "payload_hash": request.payload_hash,
        "chunk_handles_root": request.chunk_handles_root,
        "num_chunks": request.num_chunks,
        "committed_at": time.time(),
    }
    return {"status": "committed"}


@app.post("/v1/da/challenge")
async def issue_da_challenge(request: DAChallengeRequest) -> dict:
    record = _DA_COMMITS.get(request.capsule_commit_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="commit not found")
    if _DA_RELAY_KEY is None:
        raise HTTPException(status_code=503, detail="relay signing key unavailable")
    issued_ms = int(time.time() * 1000)
    challenge = build_da_challenge(
        capsule_commit_hash=request.capsule_commit_hash,
        relay_pubkey_id=DA_RELAY_ID,
        issued_at_ms=issued_ms,
        expires_at_ms=issued_ms + 10 * 60 * 1000,
    )
    challenge["payload_hash"] = record.get("payload_hash")
    challenge["chunk_handles_root"] = record.get("chunk_handles_root")
    challenge["num_chunks"] = record.get("num_chunks")
    payload_bytes = challenge_signature_payload(challenge)
    digest = hashlib.sha256(payload_bytes).digest()
    challenge["relay_signature"] = _DA_RELAY_KEY.sign(digest, hasher=None).hex()
    return {"challenge": challenge}


@app.get("/runs/{run_id}/snapshot")
async def get_snapshot(run_id: str, _: None = Depends(rate_limit_dependency)) -> JSONResponse:
    history = store.history(run_id)
    latest = json.loads(history[-1]) if history else None
    response = {
        "run_id": run_id,
        "latest_seq": latest.get("seq") if latest else 0,
        "latest_event": latest,
    }
    status = 200 if latest else 404
    return JSONResponse(response, status_code=status)


def _format_ts(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _run_metadata(run_id: str) -> dict | None:
    history = store.history(run_id)
    if not history:
        return None
    created_at_ms: int | None = None
    backend = policy_id = track_id = trace_id = None
    for line in history:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if created_at_ms is None:
            created_at_ms = event.get("ts_ms")
        if event.get("type") == "run_started":
            data = event.get("data") or {}
            backend = data.get("backend") or backend
            policy_id = data.get("policy_id") or policy_id
            track_id = data.get("track_id") or track_id
            trace_id = data.get("trace_id") or trace_id
    return {
        "run_id": run_id,
        "trace_id": trace_id or run_id,
        "backend": backend or "unknown",
        "policy_id": policy_id or "unknown",
        "track_id": track_id or "unknown",
        "created_at": _format_ts(created_at_ms),
        "verification_status": (_load_verification_state(run_id) or {}).get("status"),
    }


def _artifact_dir(run_id: str) -> Path:
    return (ARTIFACTS_ROOT / run_id).resolve()


def _verification_path(run_id: str) -> Path:
    return VERIFICATION_ROOT / f"{run_id}.json"


def _load_verification_state(run_id: str) -> dict | None:
    path = _verification_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _store_verification_state(run_id: str, data: dict) -> None:
    path = _verification_path(run_id)
    path.write_text(json.dumps(data, indent=2))


def _artifact_index_path(run_id: str) -> Path:
    return _artifact_dir(run_id) / "artifacts.json"


def _load_artifact_manifest(run_id: str) -> List[dict] | None:
    index_path = _artifact_index_path(run_id)
    if not index_path.exists():
        return None
    try:
        return json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None


def _store_artifact_manifest(run_id: str, artifacts: List[dict]) -> None:
    index_path = _artifact_index_path(run_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(artifacts, indent=2))


def _list_artifacts(run_id: str) -> List[dict]:
    manifest = _load_artifact_manifest(run_id)
    if manifest is not None:
        return manifest
    root = _artifact_dir(run_id)
    if not root.exists() or not root.is_dir():
        return []
    result: List[dict] = []
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_name = child.relative_to(root).as_posix()
        stat = child.stat()
        content_type = mimetypes.guess_type(child.name)[0] or "application/octet-stream"
        result.append(
            {
                "name": rel_name,
                "size_bytes": stat.st_size,
                "content_type": content_type,
                "storage": "local",
                "object_key": child.as_posix(),
            }
        )
    return result


def _artifact_path(run_id: str, artifact_name: str) -> Path:
    root = _artifact_dir(run_id)
    safe_name = Path(artifact_name)
    path = (root / safe_name).resolve()
    if not str(path).startswith(str(root)):
        raise HTTPException(status_code=403, detail="forbidden")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


def _find_artifact(run_id: str, artifact_name: str) -> dict | None:
    for entry in _list_artifacts(run_id):
        if entry.get("name") == artifact_name:
            return entry
    return None


def _ensure_r2_client():  # pragma: no cover - requires boto3 + network
    global _r2_client
    if _r2_client is not None:
        return _r2_client
    if not all([R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise RuntimeError("R2 configuration missing")
    if boto3 is None:
        raise RuntimeError("boto3 is not installed; install capsulebench[r2]")
    _r2_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )
    return _r2_client


def _materialize_artifact(entry: dict) -> tuple[Path, bool]:
    storage = entry.get("storage") or "local"
    object_key = entry.get("object_key")
    if storage == "local" and object_key:
        path = Path(object_key)
        if not path.exists():
            raise FileNotFoundError(f"artifact missing: {object_key}")
        return path, False
    if storage == "r2" and object_key:
        client = _ensure_r2_client()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        client.download_file(R2_BUCKET_NAME, object_key, str(tmp_path))
        return tmp_path, True
    raise RuntimeError("unsupported artifact storage")


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_pack_meta(pack_root: Path) -> None:
    meta_path = pack_root / "pack_meta.json"
    if not meta_path.exists():
        raise RuntimeError("capsulepack missing pack_meta.json")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("pack_meta.json is not valid JSON") from exc
    entries = meta.get("entries") or []
    root_resolved = pack_root.resolve()
    for entry in entries:
        rel = entry.get("path")
        digest = entry.get("sha256")
        if not rel or not digest:
            raise RuntimeError("invalid pack_meta entry")
        target = (pack_root / rel).resolve()
        if not str(target).startswith(str(root_resolved)):
            raise RuntimeError("pack_meta entry escapes extraction root")
        if not target.is_file():
            raise RuntimeError(f"capsulepack entry missing: {rel}")
        if _hash_file(target) != digest:
            raise RuntimeError(f"capsulepack entry hash mismatch: {rel}")


def _extract_capsulepack(tar_path: Path) -> Path:
    tmp_dir = tempfile.mkdtemp(prefix="capsulepack_")
    tmp_path = Path(tmp_dir)
    total_size = 0
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError("capsulepack contains unsafe paths")
            if member.issym() or member.islnk():
                raise RuntimeError("capsulepack contains symbolic links, which are not allowed")
            total_size += int(member.size or 0)
            if total_size > CAPSULEPACK_MAX_BYTES:
                raise RuntimeError("capsulepack exceeds maximum allowed size")
            archive.extract(member, tmp_path)
    root = tmp_path / "capsulepack"
    if not root.exists():
        raise RuntimeError("capsulepack root missing after extraction")
    _verify_pack_meta(root)
    return root


def _run_verification_sync(run_id: str) -> dict:
    manifest = _list_artifacts(run_id)
    archive_entry = next((entry for entry in manifest if entry.get("name", "").endswith(".capsulepack.tgz")), None)
    if not archive_entry:
        raise RuntimeError("capsulepack archive not registered")
    tar_path, cleanup = _materialize_artifact(archive_entry)
    try:
        pack_root = _extract_capsulepack(tar_path)
        capsule = pack_root / "capsule.json"
        policy = pack_root / "policy.json"
        manifests_dir = pack_root / "manifests"
        if not capsule.exists() or not policy.exists():
            raise RuntimeError("capsule or policy missing in archive")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_capsule.py"),
            str(capsule),
            "--policy",
            str(policy),
            "--manifest-root",
            str(manifests_dir),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(REPO_ROOT))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        timestamp = time.time()
        if result.returncode == 0:
            try:
                report = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                report = {"detail": result.stdout}
            state = {"status": "VERIFIED", "completed_at": timestamp, "report": report}
        else:
            try:
                report = json.loads(result.stderr or "{}")
            except json.JSONDecodeError:
                report = {"error": result.stderr}
            state = {"status": "FAILED", "completed_at": timestamp, "report": report}
        _store_verification_state(run_id, state)
        return state
    finally:
        if cleanup and tar_path.exists():
            tar_path.unlink()


@app.get("/runs")
async def list_runs(_: None = Depends(rate_limit_dependency)) -> JSONResponse:
    runs = []
    for run_id in store.run_ids():
        meta = _run_metadata(run_id)
        if meta:
            runs.append(meta)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return JSONResponse(runs)


@app.get("/runs/{run_id}")
async def run_detail(run_id: str, _: None = Depends(rate_limit_dependency)) -> dict:
    history = store.history(run_id)
    if not history:
        raise HTTPException(status_code=404, detail="run not found")
    metadata = _run_metadata(run_id) or {"run_id": run_id}
    # include latest events (e.g., last 100)
    latest_events = []
    for line in history[-100:]:
        try:
            latest_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    last_seq = 0
    if latest_events:
        last_seq = max(int(evt.get("seq", 0)) for evt in latest_events)
    return {
        "run": metadata,
        "events": latest_events,
        "last_seq": last_seq,
        "artifacts": _list_artifacts(run_id),
        "verification": _load_verification_state(run_id),
    }


@app.get("/runs/{run_id}/artifacts")
async def list_run_artifacts(run_id: str, _: None = Depends(rate_limit_dependency)) -> JSONResponse:
    if not _run_metadata(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(_list_artifacts(run_id))


@app.get("/runs/{run_id}/artifacts/{artifact_path:path}")
async def download_artifact(run_id: str, artifact_path: str, _: None = Depends(rate_limit_dependency)):
    path = _artifact_path(run_id, artifact_path)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/runs/{run_id}/artifact_meta/{artifact_path:path}")
async def artifact_meta(run_id: str, artifact_path: str) -> dict:
    entry = _find_artifact(run_id, artifact_path)
    if not entry:
        raise HTTPException(status_code=404, detail="artifact not found")
    return entry


@app.put("/runs/{run_id}/artifacts")
async def register_artifacts(
    run_id: str,
    manifest: ArtifactManifest = Body(...),
    authorization: str | None = Header(default=None),
):
    if not _run_metadata(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    _require_admin(authorization)
    entries = [entry.dict() for entry in manifest.artifacts]
    _store_artifact_manifest(run_id, entries)
    return {"count": len(entries)}


def _verification_task(run_id: str) -> None:
    _store_verification_state(run_id, {"status": "VERIFYING", "started_at": time.time()})
    try:
        _run_verification_sync(run_id)
    except Exception as exc:  # pragma: no cover - dependent on verifier
        state = {
            "status": "FAILED",
            "completed_at": time.time(),
            "report": {"error": str(exc)},
        }
        _store_verification_state(run_id, state)


@app.post("/runs/{run_id}/verify")
async def trigger_verification(run_id: str, background: BackgroundTasks):
    if not _run_metadata(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    background.add_task(_verification_task, run_id)
    return {"status": "VERIFYING"}


@app.post("/runs/{run_id}/ingest_token", response_model=TokenResponse)
async def issue_token_endpoint(
    run_id: str,
    request: TokenRequest,
    authorization: str | None = Header(default=None),
):
    _require_admin(authorization)
    ttl = request.ttl_seconds if request.ttl_seconds and request.ttl_seconds > 0 else None
    record = await _issue_ingest_token(run_id, ttl)
    ingest_url = f"{RELAY_WS_BASE.rstrip('/')}/ws/ingest/{run_id}"
    return TokenResponse(ingest_url=ingest_url, token=record.token, expires_at=record.expires_at)
@app.get("/")
async def root() -> dict:
    return {"service": "capsule relay", "ok": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
