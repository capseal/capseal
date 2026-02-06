"""S3-backed artifact store that mirrors uploads into object storage."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # pragma: no cover - boto3 optional
    boto3 = None
    BotoConfig = None

from .storage import ArtifactStore, StoredArtifact


class S3ArtifactStore:
    """Store artifacts in S3-compatible backends with signed URLs."""

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        prefix: str = "",
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        cache: ArtifactStore | None = None,
        url_ttl: int = 3600,
    ) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 artifact storage")
        session = boto3.session.Session()
        config = BotoConfig(signature_version="s3v4") if BotoConfig else None
        self.client = session.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=config,
        )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache = cache
        self.url_ttl = max(300, url_ttl)

    def store(self, path: Path) -> StoredArtifact:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = self._hash_file(path)
        key = self._object_key(digest)
        extra_args = {"ACL": "private"}
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.url_ttl,
        )
        if self.cache:
            try:
                self.cache.store(path)
            except Exception:
                pass
        return StoredArtifact(hash=digest, rel_path=key, size=path.stat().st_size, url=url)

    def _object_key(self, digest: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{digest[:2]}/{digest}"
        return f"{digest[:2]}/{digest}"

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


__all__ = ["S3ArtifactStore"]
