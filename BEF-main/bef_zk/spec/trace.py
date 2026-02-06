from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import hashlib

from bef_zk.codec import canonical_encode, ENCODING_ID

HASH_PREFIX_TRACE_SPEC_V1 = b"TRACE_SPEC_V1::"


@dataclass
class TraceSpecV1:
    spec_version: str
    trace_format_id: str
    record_schema_ref: str
    encoding_id: str
    field_modulus_id: str

    def to_obj(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "trace_format_id": self.trace_format_id,
            "record_schema_ref": self.record_schema_ref,
            "encoding_id": self.encoding_id,
            "field_modulus_id": self.field_modulus_id,
        }

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "TraceSpecV1":
        return cls(
            spec_version=obj.get("spec_version", ""),
            trace_format_id=obj.get("trace_format_id", ""),
            record_schema_ref=obj.get("record_schema_ref", ""),
            encoding_id=obj.get("encoding_id", ENCODING_ID),
            field_modulus_id=obj.get("field_modulus_id", ""),
        )


def compute_trace_spec_hash(spec: TraceSpecV1) -> str:
    payload = canonical_encode(spec.to_obj(), encoding_id=ENCODING_ID)
    digest = hashlib.sha256(HASH_PREFIX_TRACE_SPEC_V1 + payload).hexdigest()
    return digest
