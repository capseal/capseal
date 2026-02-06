"""RQ worker entry point for CapSeal async jobs."""
from __future__ import annotations

import importlib

from redis import Redis
from rq import Worker

from . import create_app
from .routes import _execute

app = create_app()


def perform_task(handler: str, payload: dict | None):
    """Execute a CLI handler inside an application context."""
    fn = _resolve_handler(handler)
    with app.app_context():
        body, status = _execute(fn, payload)
        return {"response": body, "status": status}


def run_worker() -> None:
    redis_url = app.config.get("REDIS_URL", "redis://localhost:6379/0")
    queue_name = app.config.get("JOB_QUEUE_NAME", "capseal")
    connection = Redis.from_url(redis_url)
    worker = Worker([queue_name], connection=connection)
    worker.work(with_scheduler=True)


def _resolve_handler(path: str):
    module_name, _, func_name = path.rpartition(".")
    if not module_name:
        raise RuntimeError(f"Invalid handler path: {path}")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise RuntimeError(f"Handler not found: {path}")
    return fn


if __name__ == "__main__":
    run_worker()
