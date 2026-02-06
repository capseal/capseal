"""Simple API key authentication for the CapSeal Flask service."""
from __future__ import annotations

from typing import Iterable

from flask import current_app, request

API_KEY_HEADER = "X-API-Key"


def require_api_key() -> None:
    """Enforce API key auth when CAPSEAL_API_KEYS is set.

    Multiple API keys can be provided via configuration. Keys are stored as a set
    to allow constant time membership checks.
    """

    keys: Iterable[str] | None = current_app.config.get("CAPSEAL_API_KEYS")
    if not keys:
        return

    provided = request.headers.get(API_KEY_HEADER, "").strip()
    if not provided or provided not in keys:
        from werkzeug.exceptions import Unauthorized

        raise Unauthorized(description="Missing or invalid API key")


__all__ = ["require_api_key", "API_KEY_HEADER"]
