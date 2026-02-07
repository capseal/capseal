"""Dataset specification and access log helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from bef_zk.codec import ENCODING_ID, canonical_encode

HASH_PREFIX_DATASET_SPEC = b"DATASET_SPEC_V1::"
HASH_PREFIX_ACCESS_LOG = b"ACCESS_LOG_V1::"


@dataclass
class DatasetSpecV1:
    dataset_id: str
    root: str
    chunk_arity: int
    schema: str = "dataset_spec_v1"
    chunk_size_mode: str = "file"
    codec: str = "binary"
    ordering: str = "lexicographic_path"
    num_chunks: int = 0
    manifest_rel_path: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_obj(self) -> Dict[str, Any]:
        payload = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "root": self.root,
            "chunk_arity": int(self.chunk_arity),
            "chunk_size_mode": self.chunk_size_mode,
            "codec": self.codec,
            "ordering": self.ordering,
            "num_chunks": int(self.num_chunks),
        }
        if self.manifest_rel_path:
            payload["manifest_rel_path"] = self.manifest_rel_path
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "DatasetSpecV1":
        return cls(
            dataset_id=str(obj.get("dataset_id", "")),
            root=str(obj.get("root", "")),
            chunk_arity=int(obj.get("chunk_arity", 2)),
            chunk_size_mode=str(obj.get("chunk_size_mode", "file")),
            codec=str(obj.get("codec", "binary")),
            ordering=str(obj.get("ordering", "lexicographic_path")),
            num_chunks=int(obj.get("num_chunks", 0)),
            manifest_rel_path=obj.get("manifest_rel_path"),
            extra=dict(obj.get("extra", {})),
        )


def compute_dataset_spec_hash(spec: DatasetSpecV1) -> str:
    encoded = canonical_encode(spec.to_obj(), encoding_id=ENCODING_ID)
    return hashlib.sha256(HASH_PREFIX_DATASET_SPEC + encoded).hexdigest()


@dataclass
class AccessLogEntry:
    dataset_id: str
    chunk_id: int
    chunk_hash: str
    row_indices: List[int] | None = None
    extra: Dict[str, Any] | None = None

    def to_obj(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "chunk_id": int(self.chunk_id),
            "chunk_hash": self.chunk_hash,
        }
        if self.row_indices:
            payload["row_indices"] = list(self.row_indices)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def compute_access_log_root(entries: List[AccessLogEntry]) -> str:
    acc = b"\x00" * 32
    for entry in entries:
        encoded = canonical_encode(entry.to_obj(), encoding_id=ENCODING_ID)
        acc = hashlib.sha256(HASH_PREFIX_ACCESS_LOG + acc + encoded).digest()
    return acc.hex()


__all__ = [
    "DatasetSpecV1",
    "AccessLogEntry",
    "compute_dataset_spec_hash",
    "compute_access_log_root",
]
