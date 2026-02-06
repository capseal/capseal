"""Job managers for async command execution."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue as ThreadQueue
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from redis import Redis
    from rq import Queue
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    HAVE_RQ = True
except Exception:  # pragma: no cover
    Redis = None  # type: ignore[assignment]
    Queue = None  # type: ignore[assignment]
    Job = Any  # type: ignore[assignment]
    HAVE_RQ = False



@dataclass
class JobMetadata:
    job_id: str
    description: str
    status: str
    enqueued_at: float | None
    started_at: float | None
    finished_at: float | None
    http_status: int | None
    result: dict[str, Any] | None
    error: str | None
    location: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "description": self.description,
            "status": self.status,
            "enqueuedAt": self.enqueued_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "result": self.result,
            "statusCode": self.http_status,
            "error": self.error,
            "location": self.location,
        }


class JobManagerRedis:
    """Small helper that encapsulates the Redis queue used by async endpoints."""

    def __init__(self, app) -> None:
        if not HAVE_RQ:
            raise RuntimeError("redis/rq packages not available")
        self.app = app
        redis_url = app.config.get("REDIS_URL", "redis://localhost:6379/0")
        self.connection = Redis.from_url(redis_url)
        queue_name = app.config.get("JOB_QUEUE_NAME", "capseal")
        default_timeout = int(app.config.get("JOB_TIMEOUT", 60 * 60))
        self.result_ttl = int(app.config.get("JOB_RESULT_TTL", 24 * 60 * 60))
        self.queue = Queue(
            queue_name,
            connection=self.connection,
            default_timeout=default_timeout,
        )
        self.history_key = f"{queue_name}:history"
        self.history_size = int(app.config.get("JOB_HISTORY_SIZE", 200))

    def submit(self, description: str, handler_path: str, payload: dict[str, Any] | None) -> Job:
        job_id = uuid.uuid4().hex
        job = self.queue.enqueue(
            "server.flask_app.worker.perform_task",
            kwargs={
                "handler": handler_path,
                "payload": payload,
            },
            job_id=job_id,
            result_ttl=self.result_ttl,
        )
        job.meta["description"] = description
        job.meta["handler"] = handler_path
        job.save_meta()
        self.connection.lpush(self.history_key, job.id)
        self.connection.ltrim(self.history_key, 0, self.history_size - 1)
        return job

    def get(self, job_id: str) -> Optional[JobMetadata]:
        job = self._fetch(job_id)
        if not job:
            return None
        return self._serialize(job)

    def list(self, limit: int = 50) -> list[JobMetadata]:
        limit = max(1, min(limit, self.history_size))
        job_ids = [job_id.decode("utf-8") for job_id in self.connection.lrange(self.history_key, 0, limit - 1)]
        items: list[JobMetadata] = []
        for job_id in job_ids:
            job = self._fetch(job_id)
            if job:
                items.append(self._serialize(job))
        return items

    def cancel(self, job_id: str) -> bool:
        job = self._fetch(job_id)
        if not job:
            return False
        job.cancel()
        return True

    def _fetch(self, job_id: str) -> Job | None:
        try:
            return Job.fetch(job_id, connection=self.connection)
        except NoSuchJobError:
            return None

    def _serialize(self, job: Job) -> JobMetadata:
        status = job.get_status()
        result_payload = job.result if job.is_finished else None
        http_status = None
        if isinstance(result_payload, dict) and "response" in result_payload:
            result_dict = result_payload.get("response")
            http_status = result_payload.get("status")
        else:
            result_dict = result_payload if isinstance(result_payload, dict) else None
        error = None
        if status == "failed" and job.exc_info:
            error = job.exc_info
        elif isinstance(result_payload, dict):
            error = result_payload.get("error")
        return JobMetadata(
            job_id=job.id,
            description=job.meta.get("description", ""),
            status=status,
            enqueued_at=_ts(job.enqueued_at),
            started_at=_ts(job.started_at),
            finished_at=_ts(job.ended_at),
            http_status=http_status,
            result=result_dict,
            error=error,
            location=f"/api/jobs/{job.id}",
        )


def _ts(value: datetime | None) -> float | None:
    return value.timestamp() if isinstance(value, datetime) else None


class LocalJobManager:
    """In-memory fallback queue used when RQ/Redis is unavailable."""

    def __init__(self, app) -> None:
        self.app = app
        self._jobs: Dict[str, JobMetadata] = {}
        self._history = deque(maxlen=int(app.config.get("JOB_HISTORY_SIZE", 200)))
        self._queue: ThreadQueue[tuple[str, str, callable]] = ThreadQueue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, description: str, handler_path: str, payload: dict[str, Any] | None):
        job_id = uuid.uuid4().hex
        record = JobMetadata(
            job_id=job_id,
            description=description,
            status="queued",
            enqueued_at=time.time(),
            started_at=None,
            finished_at=None,
            http_status=None,
            result=None,
            error=None,
            location=f"/api/jobs/{job_id}"
        )
        with self._lock:
            self._jobs[job_id] = record
            self._history.appendleft(job_id)
        self._queue.put((job_id, handler_path, payload))
        return record

    def get(self, job_id: str) -> Optional[JobMetadata]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[JobMetadata]:
        with self._lock:
            ids = list(self._history)[:limit]
            return [self._jobs[jid] for jid in ids if jid in self._jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            job.status = "canceled"
            job.finished_at = time.time()
            job.error = "canceled before start"
            return True
        return False

    def _loop(self) -> None:
        while True:
            try:
                job_id, handler_path, payload = self._queue.get(timeout=0.5)
            except Empty:
                continue
            job = self.get(job_id)
            if not job or job.status == "canceled":
                continue
            job.status = "running"
            job.started_at = time.time()
            try:
                from server.flask_app.worker import _resolve_handler
                from server.flask_app.routes import _execute

                fn = _resolve_handler(handler_path)
                with self.app.app_context():
                    body, status = _execute(fn, payload)
                job.result = body
                job.http_status = status
                job.status = "succeeded" if 200 <= status < 400 else "failed"
            except Exception as exc:  # pragma: no cover
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()
                self._queue.task_done()


__all__ = ["JobManagerRedis", "JobMetadata", "LocalJobManager"]
