"""Content-addressed artifact storage backends."""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class StoredArtifact:
    hash: str
    rel_path: str
    size: int
    url: str | None = None


class ArtifactStore:
    """Local content-addressed artifact store with optional GC quotas."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int | None = None,
        max_files: int | None = None,
        max_age_seconds: int | None = None,
        gc_interval: int = 900,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.max_age = max_age_seconds
        self.gc_interval = max(gc_interval, 60) if gc_interval else 0
        self._lock = threading.Lock()
        self._gc_thread: threading.Thread | None = None
        if self.gc_interval and self._needs_gc():
            self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
            self._gc_thread.start()

    def store(self, path: Path) -> StoredArtifact:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = self._hash_file(path)
        dest_dir = self.root / digest[:2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / digest
        if not dest.exists():
            shutil.copy2(path, dest)
        rel = dest.relative_to(self.root)
        artifact = StoredArtifact(
            hash=digest,
            rel_path=str(rel),
            size=path.stat().st_size,
        )
        if self._needs_gc():
            self.garbage_collect()
        return artifact

    def garbage_collect(self) -> None:
        """Enforce disk/age/count quotas."""

        with self._lock:
            entries = list(self._scan_files())
            if not entries:
                return
            now = time.time()
            removed = set()
            if self.max_age is not None:
                for entry in entries:
                    if now - entry.mtime > self.max_age:
                        self._remove_path(entry.path)
                        removed.add(entry.path)
                entries = [e for e in entries if e.path not in removed]
            total_bytes = sum(entry.size for entry in entries)
            if self.max_bytes is not None and total_bytes > self.max_bytes:
                for entry in sorted(entries, key=lambda e: e.mtime):
                    if total_bytes <= self.max_bytes:
                        break
                    self._remove_path(entry.path)
                    total_bytes -= entry.size
            if self.max_files is not None and len(entries) > self.max_files:
                over = len(entries) - self.max_files
                for entry in sorted(entries, key=lambda e: e.mtime)[:over]:
                    self._remove_path(entry.path)

    def _needs_gc(self) -> bool:
        return any(value is not None for value in (self.max_age, self.max_bytes, self.max_files))

    def _gc_loop(self) -> None:  # pragma: no cover - background maintenance
        while True:
            time.sleep(self.gc_interval)
            try:
                self.garbage_collect()
            except Exception:
                continue

    def _scan_files(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                yield _CacheEntry(path=path, size=stat.st_size, mtime=stat.st_mtime)

    def _remove_path(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        # remove empty parents two levels deep
        parent = path.parent
        try:
            if not any(parent.iterdir()):
                parent.rmdir()
                grand = parent.parent
                if grand != self.root and not any(grand.iterdir()):
                    grand.rmdir()
        except OSError:
            pass

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


@dataclass
class _CacheEntry:
    path: Path
    size: int
    mtime: float


def attach_artifacts(response: dict, store: ArtifactStore) -> None:
    result = response.get("result")
    if not isinstance(result, dict):
        return
    keys = [
        "capPath",
        "capsulePath",
        "capsule_path",
        "capsule_json",
    ]
    artifacts: list[Dict[str, str | int]] = []
    for key in keys:
        value = result.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.exists():
            continue
        stored = store.store(path)
        artifacts.append({
            "type": key,
            "hash": stored.hash,
            "path": stored.rel_path,
            "size": stored.size,
            "url": stored.url,
        })
    if artifacts:
        response.setdefault("artifacts", []).extend(artifacts)


__all__ = ["ArtifactStore", "StoredArtifact", "attach_artifacts"]
