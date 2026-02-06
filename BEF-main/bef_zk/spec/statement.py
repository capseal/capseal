from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import hashlib

from bef_zk.codec import canonical_encode, ENCODING_ID

HASH_PREFIX_STATEMENT_V1 = b"STATEMENT_V1::"


@dataclass
class StatementV1:
    statement_version: str
    trace_spec_hash: str
    policy_hash: str
    trace_root: str
    public_inputs: List[Dict[str, Any]] = field(default_factory=list)
    anchors: List[Dict[str, Any]] = field(default_factory=list)

    def to_obj(self) -> Dict[str, Any]:
        return {
            "statement_version": self.statement_version,
            "trace_spec_hash": self.trace_spec_hash,
            "policy_hash": self.policy_hash,
            "trace_root": self.trace_root,
            "public_inputs": list(self.public_inputs),
            "anchors": list(self.anchors),
        }

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "StatementV1":
        return cls(
            statement_version=obj.get("statement_version", ""),
            trace_spec_hash=obj.get("trace_spec_hash", ""),
            policy_hash=obj.get("policy_hash", ""),
            trace_root=obj.get("trace_root", ""),
            public_inputs=list(obj.get("public_inputs", [])),
            anchors=list(obj.get("anchors", [])),
        )


def compute_statement_hash(statement: StatementV1) -> str:
    payload = canonical_encode(statement.to_obj(), encoding_id=ENCODING_ID)
    return hashlib.sha256(HASH_PREFIX_STATEMENT_V1 + payload).hexdigest()
