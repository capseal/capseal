"""Flask application factory for the CapSeal API wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from flask import Flask, jsonify, make_response, request
from pydantic import ValidationError

from server.event_store import EventStore

from .auth import API_KEY_HEADER, require_api_key
from .errors import APIError
from .jobs import JobManagerRedis, LocalJobManager
from .routes import api_bp
from .sse_routes import sse_bp
from .storage import ArtifactStore
from .utils import ensure_api_keys, readiness_status, resolve_project_root


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    project_root = resolve_project_root()
    app.config.setdefault("CAPSEAL_PROJECT_ROOT", str(project_root))
    app.config.setdefault("CAPSEAL_EVENT_ROOT", str(project_root / "server_data" / "events"))
    app.config.setdefault("CAPSEAL_ARTIFACT_ROOT", str(project_root / "server_data" / "artifacts"))
    app.config.setdefault("CAPSEAL_VERIFICATION_ROOT", str(project_root / "server_data" / "verifications"))
    app.config.setdefault("CAPSEAL_API_KEYS", ensure_api_keys(os.environ.get("CAPSEAL_API_KEYS")))
    app.config.setdefault("CORS_ALLOW_ORIGINS", os.environ.get("CORS_ALLOW_ORIGINS", "*"))
    app.config.setdefault("ARTIFACT_STORE_MODE", os.environ.get("ARTIFACT_STORE", "local"))
    app.config.setdefault("CAPSEAL_ARTIFACT_MAX_BYTES", _int_env("CAPSEAL_ARTIFACT_MAX_BYTES", 20_000_000_000))
    app.config.setdefault("CAPSEAL_ARTIFACT_MAX_FILES", _int_env("CAPSEAL_ARTIFACT_MAX_FILES", 5000))
    app.config.setdefault("CAPSEAL_ARTIFACT_MAX_AGE", _int_env("CAPSEAL_ARTIFACT_MAX_AGE", 7 * 24 * 3600))
    app.config.setdefault("CAPSEAL_ARTIFACT_GC_INTERVAL", _int_env("CAPSEAL_ARTIFACT_GC_INTERVAL", 900))
    app.config.setdefault("ARTIFACT_SIGNED_URL_TTL", _int_env("ARTIFACT_SIGNED_URL_TTL", 3600))
    app.config.setdefault("ARTIFACT_S3_BUCKET", os.environ.get("S3_BUCKET"))
    app.config.setdefault("ARTIFACT_S3_PREFIX", os.environ.get("S3_PREFIX", "capseal"))
    app.config.setdefault("ARTIFACT_S3_REGION", os.environ.get("AWS_REGION", "us-east-1"))
    app.config.setdefault("ARTIFACT_S3_ENDPOINT", os.environ.get("S3_ENDPOINT_URL"))
    app.config.setdefault("API_RATE_LIMITS", os.environ.get("API_RATE_LIMITS", "120 per minute"))

    if config:
        app.config.update(config)

    # Shared extensions
    artifact_store = _build_artifact_store(app)
    app.extensions["artifact_store"] = artifact_store

    event_store = EventStore(Path(app.config["CAPSEAL_EVENT_ROOT"]))
    event_store.load_existing()
    app.extensions["event_store"] = event_store

    app.config.setdefault("REDIS_URL", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    app.config.setdefault("JOB_QUEUE_NAME", os.environ.get("JOB_QUEUE_NAME", "capseal"))
    app.config.setdefault("JOB_RESULT_TTL", int(os.environ.get("JOB_RESULT_TTL", "86400")))
    app.config.setdefault("JOB_TIMEOUT", int(os.environ.get("JOB_TIMEOUT", "3600")))
    app.config.setdefault("JOB_HISTORY_SIZE", int(os.environ.get("JOB_HISTORY_SIZE", "200")))

    try:
        app.extensions["job_manager"] = JobManagerRedis(app)
    except RuntimeError:
        app.extensions["job_manager"] = LocalJobManager(app)

    # Require API key only if keys configured
    if app.config.get("CAPSEAL_API_KEYS") and not getattr(api_bp, "_auth_registered", False):
        api_bp.before_request(require_api_key)
        setattr(api_bp, "_auth_registered", True)

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(sse_bp, url_prefix="/api")

    _configure_cors(app)
    _configure_rate_limits(app)

    @app.get("/health")
    def _health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.get("/ready")
    def _ready() -> tuple[dict[str, Any], int]:
        status, checks = readiness_status()
        return {"status": status, "checks": checks}, 200

    @app.get("/openapi.json")
    def _openapi() -> tuple[Any, int]:
        spec_path = project_root / "server" / "flask_app" / "openapi.json"
        try:
            spec = json.loads(spec_path.read_text())
        except FileNotFoundError:
            spec = {"openapi": "3.1.0", "info": {"title": "CapSeal API", "version": "1.0.0"}}
        return jsonify(spec), 200

    @app.errorhandler(APIError)
    def _handle_api_error(err: APIError):  # type: ignore[override]
        return err.to_response()

    @app.errorhandler(ValidationError)
    def _handle_validation(err: ValidationError):  # type: ignore[override]
        return jsonify({
            "status": "ERROR",
            "message": "validation error",
            "details": err.errors(),
        }), 422

    return app


__all__ = ["create_app"]


def _build_artifact_store(app: Flask):
    root = Path(app.config["CAPSEAL_ARTIFACT_ROOT"])
    max_bytes = app.config.get("CAPSEAL_ARTIFACT_MAX_BYTES")
    max_files = app.config.get("CAPSEAL_ARTIFACT_MAX_FILES")
    max_age = app.config.get("CAPSEAL_ARTIFACT_MAX_AGE")
    gc_interval = app.config.get("CAPSEAL_ARTIFACT_GC_INTERVAL", 900)
    mode = (app.config.get("ARTIFACT_STORE_MODE") or "local").lower()
    if mode == "s3":
        from .storage_s3 import S3ArtifactStore

        cache = ArtifactStore(
            root,
            max_bytes=max_bytes,
            max_files=max_files,
            max_age_seconds=max_age,
            gc_interval=gc_interval,
        )
        bucket = app.config.get("ARTIFACT_S3_BUCKET")
        if not bucket:
            raise RuntimeError("S3_BUCKET must be set for ARTIFACT_STORE=s3")
        return S3ArtifactStore(
            bucket=bucket,
            region=app.config.get("ARTIFACT_S3_REGION", "us-east-1"),
            prefix=app.config.get("ARTIFACT_S3_PREFIX", "capseal"),
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=app.config.get("ARTIFACT_S3_ENDPOINT"),
            cache=cache,
            url_ttl=int(app.config.get("ARTIFACT_SIGNED_URL_TTL", 3600)),
        )
    return ArtifactStore(
        root,
        max_bytes=max_bytes,
        max_files=max_files,
        max_age_seconds=max_age,
        gc_interval=gc_interval,
    )


def _configure_cors(app: Flask) -> None:
    origins = _parse_origins(app.config.get("CORS_ALLOW_ORIGINS"))
    if not origins:
        return
    allow_headers = app.config.get("CORS_ALLOW_HEADERS", "Content-Type,X-API-Key")
    allow_methods = app.config.get("CORS_ALLOW_METHODS", "GET,POST,OPTIONS")

    def _allowed(origin: str | None) -> bool:
        if not origin:
            return False
        return "*" in origins or origin in origins

    @app.after_request
    def _add_headers(response):  # type: ignore[override]
        origin = request.headers.get("Origin")
        if _allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        elif "*" in origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = allow_headers
        response.headers["Access-Control-Allow-Methods"] = allow_methods
        return response

    @app.before_request
    def _handle_preflight():  # type: ignore[override]
        if request.method != "OPTIONS":
            return None
        origin = request.headers.get("Origin")
        resp = make_response("", 204)
        if _allowed(origin):
            resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = allow_headers
        resp.headers["Access-Control-Allow-Methods"] = allow_methods
        return resp


def _configure_rate_limits(app: Flask) -> None:
    limits = _parse_limits(app.config.get("API_RATE_LIMITS"))
    if not limits:
        return
    limiter = SimpleRateLimiter(app.config.get("REDIS_URL", "redis://localhost:6379/0"), limits)
    if not limiter.available:
        return

    @app.before_request
    def _check_rate():  # type: ignore[override]
        if request.method == "OPTIONS":
            return None
        key = _rate_limit_key()
        allowed, retry_after = limiter.hit(key)
        if not allowed:
            extra = {"retryAfter": retry_after} if retry_after is not None else None
            raise APIError(429, "rate limit exceeded", extra=extra)
        return None

    app.extensions["limiter"] = limiter


def _rate_limit_key() -> str:
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return api_key
    return request.remote_addr or "anonymous"


def _parse_limits(value: Any) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        return tokens or None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return None


def _parse_origins(value: Any) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or ["*"]
    if isinstance(value, Iterable):  # type: ignore[arg-type]
        parts = [str(item).strip() for item in value if str(item).strip()]
        return parts or ["*"]
    return ["*"]


class SimpleRateLimiter:
    def __init__(self, redis_url: str, limits: list[str]) -> None:
        self.redis_url = redis_url
        self.limits = [_parse_limit_tuple(limit) for limit in limits]
        self.limits = [entry for entry in self.limits if entry]
        try:
            from redis import Redis  # type: ignore

            self.redis = Redis.from_url(redis_url)
            _ = self.redis.ping()
            self.available = True
        except Exception:  # pragma: no cover
            self.redis = None
            self.available = False

    def hit(self, key: str) -> tuple[bool, int | None]:
        if not self.available or not self.limits:
            return True, None
        assert self.redis is not None
        for count, window in self.limits:
            bucket = f"rl:{key}:{window}"
            try:
                pipe = self.redis.pipeline()
                pipe.incr(bucket)
                pipe.expire(bucket, window, nx=True)
                current, _ = pipe.execute()
            except Exception:  # pragma: no cover
                return True, None
            if current > count:
                ttl = self.redis.ttl(bucket)
                return False, max(ttl, 0)
        return True, None


def _parse_limit_tuple(limit: str) -> tuple[int, int] | None:
    tokens = [token for token in limit.lower().replace("/", " ").split() if token and token != "per"]
    if len(tokens) < 2:
        return None
    try:
        count = int(tokens[0])
    except ValueError:
        return None
    unit = tokens[1]
    window = {
        "second": 1,
        "sec": 1,
        "s": 1,
        "minute": 60,
        "min": 60,
        "m": 60,
        "hour": 3600,
        "h": 3600,
        "day": 86400,
        "d": 86400,
    }.get(unit)
    if window is None:
        return None
    return count, window


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
