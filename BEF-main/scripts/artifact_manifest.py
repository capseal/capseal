"""Helpers for artifact manifest detection/encoding."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_FILENAME = "artifact_manifest.json"


def load_manifest(base: Path) -> Optional[Dict[str, Any]]:
    path = base / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def encoding_for_path(manifest: Dict[str, Any], base: Path, target: Path) -> Optional[str]:
    idx = manifest.get("path_index", {})
    if not idx:
        return None
    rel = os.path.relpath(target.resolve(), base.resolve())
    entry = idx.get(rel)
    if entry:
        return entry.get("encoding_id")
    return None


def resolve_manifest_path(base: Path, rel_path: str) -> Path:
    return (base / rel_path).resolve()
