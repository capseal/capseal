"""Artifact publishing helpers (local + R2)."""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


@dataclass
class ArtifactRecord:
    name: str
    size_bytes: int
    content_type: str
    storage: str
    object_key: str | None = None


class ArtifactPublisher:
    """Publishes capsulepack files to durable storage (R2 or local fallback)."""

    def __init__(self) -> None:
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        self.endpoint = os.environ.get("R2_ENDPOINT_URL")
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.prefix = os.environ.get("R2_PREFIX", "")
        self._r2_client = None
        if self.bucket and self.endpoint and self.access_key and self.secret_key:
            if boto3 is None:
                raise RuntimeError("boto3 is required for R2 uploads; install boto3 or disable R2 uploads")
            self._r2_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
        self.local_root = Path(os.environ.get("ARTIFACTS_OUTPUT_ROOT", "artifacts_out")).resolve()

    def _r2_prefix(self, run_id: str, rel_name: str) -> str:
        rel = rel_name.lstrip("/")
        base = f"{run_id}/{rel}"
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{base}"
        return base

    def _publish_r2(self, run_id: str, path: Path, rel_name: str) -> ArtifactRecord:
        assert self._r2_client is not None
        object_key = self._r2_prefix(run_id, rel_name)
        self._r2_client.upload_file(str(path), self.bucket, object_key)
        return ArtifactRecord(
            name=rel_name,
            size_bytes=path.stat().st_size,
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            storage="r2",
            object_key=object_key,
        )

    def _publish_local(self, run_id: str, path: Path, rel_name: str) -> ArtifactRecord:
        dest = self.local_root / run_id / rel_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return ArtifactRecord(
            name=rel_name,
            size_bytes=path.stat().st_size,
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            storage="local",
            object_key=str(dest),
        )

    def publish(self, run_id: str, pack_dir: Path, tar_path: Path) -> List[ArtifactRecord]:
        if self._r2_client is None and not self.local_root:
            raise RuntimeError("No artifact output configured")
        records: List[ArtifactRecord] = []
        for file_path in sorted(p for p in pack_dir.rglob("*") if p.is_file()):
            rel_name = Path("capsulepack") / file_path.relative_to(pack_dir)
            rel_name_str = rel_name.as_posix()
            if self._r2_client is not None:
                records.append(self._publish_r2(run_id, file_path, rel_name_str))
            else:
                records.append(self._publish_local(run_id, file_path, rel_name_str))
        tar_rel_name = f"{run_id}.capsulepack.tgz"
        if self._r2_client is not None:
            records.append(self._publish_r2(run_id, tar_path, tar_rel_name))
        else:
            records.append(self._publish_local(run_id, tar_path, tar_rel_name))
        return records


def records_to_json(records: Iterable[ArtifactRecord]) -> list[dict]:
    result = []
    for record in records:
        result.append(
            {
                "name": record.name,
                "size_bytes": record.size_bytes,
                "content_type": record.content_type,
                "storage": record.storage,
                "object_key": record.object_key,
            }
        )
    return result


def save_manifest(records: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2))
