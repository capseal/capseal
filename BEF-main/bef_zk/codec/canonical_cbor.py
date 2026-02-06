"""Deterministic DAG-CBOR encoder/decoder used for canonical artifacts."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ENCODING_ID = "dag_cbor_compact_fields_v1"
LEGACY_ENCODING_ID = "dag_cbor_canonical_v1"
ENCODING_ID = DEFAULT_ENCODING_ID

HASH_PREFIX_CAPSULE = b"BEF_CAPSULE_V1"
HASH_PREFIX_SEED = b"BEF_AUDIT_SEED_V1"


@dataclass(frozen=True)
class FieldElement:
    value: int
    bits: int = 64


class CanonicalCBORError(ValueError):
    """Raised when encoding or decoding fails."""


def _encode_uint(major: int, value: int) -> bytes:
    if value < 24:
        return bytes([(major << 5) | value])
    if value < 1 << 8:
        return bytes([(major << 5) | 24, value])
    if value < 1 << 16:
        return bytes([(major << 5) | 25]) + value.to_bytes(2, "big")
    if value < 1 << 32:
        return bytes([(major << 5) | 26]) + value.to_bytes(4, "big")
    if value < 1 << 64:
        return bytes([(major << 5) | 27]) + value.to_bytes(8, "big")
    raise CanonicalCBORError("integer too large for canonical encoder")


def _encode_int(value: int) -> bytes:
    if value >= 0:
        return _encode_uint(0, value)
    abs_val = -1 - value
    if abs_val < 0:
        raise CanonicalCBORError("negative integer outside CBOR range")
    return _encode_uint(1, abs_val)


def _encode_bytes(buf: bytes) -> bytes:
    return _encode_uint(2, len(buf)) + buf


def _encode_text(text: str) -> bytes:
    data = text.encode("utf-8")
    return _encode_uint(3, len(data)) + data


def _encode_array(values: Iterable[Any], encoding_id: str) -> bytes:
    values = list(values)
    out = bytearray(_encode_uint(4, len(values)))
    for value in values:
        out.extend(_encode_value(value, encoding_id))
    return bytes(out)


def _encode_map(obj: dict[str, Any], encoding_id: str) -> bytes:
    entries = []
    for key, value in obj.items():
        if not isinstance(key, (str, bytes)):
            key = str(key)
        encoded_key = _encode_value(key, encoding_id)
        entries.append((encoded_key, key, value))
    entries.sort(key=lambda item: item[0])
    out = bytearray(_encode_uint(5, len(entries)))
    for encoded_key, _key, value in entries:
        out.extend(encoded_key)
        out.extend(_encode_value(value, encoding_id))
    return bytes(out)


def _encode_value(value: Any, encoding_id: str) -> bytes:
    if value is None:
        return b"\xf6"
    if value is True:
        return b"\xf5"
    if value is False:
        return b"\xf4"
    if isinstance(value, FieldElement):
        if encoding_id == DEFAULT_ENCODING_ID:
            byte_len = max(1, (value.bits + 7) // 8)
            data = value.value.to_bytes(byte_len, "little", signed=False)
            return _encode_bytes(data)
        value = value.value
    if isinstance(value, int):
        return _encode_int(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _encode_bytes(bytes(value))
    if isinstance(value, str):
        return _encode_text(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise CanonicalCBORError("NaN not supported in canonical encoding")
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, (list, tuple)):
        return _encode_array(value, encoding_id)
    if isinstance(value, dict):
        return _encode_map(value, encoding_id)
    if isinstance(value, Path):
        return _encode_text(str(value))
    raise CanonicalCBORError(f"unsupported type {type(value)!r}")


def canonical_encode(obj: Any, encoding_id: str = DEFAULT_ENCODING_ID) -> bytes:
    return _encode_value(_sanitize(obj), encoding_id)


def _read_uint(ai: int, data: memoryview, pos: int) -> tuple[int, int]:
    if ai < 24:
        return ai, pos
    if ai == 24:
        return int(data[pos]), pos + 1
    if ai == 25:
        return int.from_bytes(data[pos : pos + 2], "big"), pos + 2
    if ai == 26:
        return int.from_bytes(data[pos : pos + 4], "big"), pos + 4
    if ai == 27:
        return int.from_bytes(data[pos : pos + 8], "big"), pos + 8
    raise CanonicalCBORError("indefinite-length values not supported")


def _decode_value(data: memoryview, pos: int) -> tuple[Any, int]:
    if pos >= len(data):
        raise CanonicalCBORError("unexpected end of data")
    initial = data[pos]
    major = initial >> 5
    ai = initial & 0x1F
    pos += 1

    if major in (0, 1):
        value, pos = _read_uint(ai, data, pos)
        if major == 0:
            return value, pos
        return -1 - value, pos
    if major == 2:
        length, pos = _read_uint(ai, data, pos)
        end = pos + length
        if end > len(data):
            raise CanonicalCBORError("byte string truncated")
        return bytes(data[pos:end]), end
    if major == 3:
        length, pos = _read_uint(ai, data, pos)
        end = pos + length
        if end > len(data):
            raise CanonicalCBORError("text string truncated")
        return data[pos:end].tobytes().decode("utf-8"), end
    if major == 4:
        length, pos = _read_uint(ai, data, pos)
        items = []
        for _ in range(length):
            value, pos = _decode_value(data, pos)
            items.append(value)
        return items, pos
    if major == 5:
        length, pos = _read_uint(ai, data, pos)
        result: dict[str, Any] = {}
        last_encoded_key = None
        for _ in range(length):
            key_pos = pos
            key, pos = _decode_value(data, pos)
            encoded_key = canonical_encode(key, DEFAULT_ENCODING_ID)
            if last_encoded_key is not None and encoded_key < last_encoded_key:
                raise CanonicalCBORError("map keys out of order")
            value, pos = _decode_value(data, pos)
            if not isinstance(key, (str, bytes)):
                key = str(key)
            result[key] = value
            last_encoded_key = encoded_key
        return result, pos
    if major == 7:
        if ai == 20:
            return False, pos
        if ai == 21:
            return True, pos
        if ai == 22:
            return None, pos
        if ai == 26:
            if pos + 4 > len(data):
                raise CanonicalCBORError("float32 truncated")
            return struct.unpack(">f", data[pos : pos + 4])[0], pos + 4
        if ai == 27:
            if pos + 8 > len(data):
                raise CanonicalCBORError("float64 truncated")
            return struct.unpack(">d", data[pos : pos + 8])[0], pos + 8
        raise CanonicalCBORError(f"unsupported simple value ai={ai}")
    raise CanonicalCBORError(f"unsupported major type {major}")


def canonical_decode(blob: bytes | bytearray | memoryview) -> Any:
    data = memoryview(blob)
    value, pos = _decode_value(data, 0)
    if pos != len(data):
        raise CanonicalCBORError("trailing data in CBOR blob")
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return value


__all__ = [
    "ENCODING_ID",
    "HASH_PREFIX_CAPSULE",
    "HASH_PREFIX_SEED",
    "FieldElement",
    "canonical_encode",
    "canonical_decode",
    "CanonicalCBORError",
]
