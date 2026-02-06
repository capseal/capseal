"""Scaffolding for the bef_compact_v1 encoder/decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAGIC = 0xBEF0C0DE
VERSION_MAJOR = 1
VERSION_MINOR = 0


@dataclass
class BinaryEncoder:
    field_type: str = "goldilocks"

    def __post_init__(self) -> None:
        self.buffer = bytearray()

    def write_varint(self, value: int) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 encoder not implemented yet")

    def write_bytes(self, data: bytes) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 encoder not implemented yet")

    def write_fp(self, value: int) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 encoder not implemented yet")

    def encode_proof(self, proof: Any) -> bytes:  # pragma: no cover
        raise NotImplementedError("bef_compact_v1 encoder not implemented yet")


@dataclass
class BinaryDecoder:
    buffer: bytes
    field_type: str = "goldilocks"

    def __post_init__(self) -> None:
        self.offset = 0

    def read_varint(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 decoder not implemented yet")

    def read_bytes(self, length: int) -> bytes:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 decoder not implemented yet")

    def read_fp(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError("bef_compact_v1 decoder not implemented yet")

    def decode_proof(self) -> Any:  # pragma: no cover
        raise NotImplementedError("bef_compact_v1 decoder not implemented yet")


__all__ = [
    "MAGIC",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "BinaryEncoder",
    "BinaryDecoder",
]
