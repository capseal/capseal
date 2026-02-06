"""Error helpers for the CapSeal Flask API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from flask import jsonify


@dataclass
class APIError(Exception):
    """Structured error type that carries an HTTP status code."""

    status: int
    message: str
    error_code: str | None = None
    extra: Dict[str, Any] | None = None

    def to_response(self):
        payload = {
            "status": "ERROR",
            "message": self.message,
        }
        if self.error_code:
            payload["errorCode"] = self.error_code
        if self.extra:
            payload.update(self.extra)
        return jsonify(payload), self.status


__all__ = ["APIError"]
