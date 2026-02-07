"""Helpers for per-sample evaluation trace commitments."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from bef_zk.codec import ENCODING_ID, canonical_encode

HASH_PREFIX_EVAL_TRACE = b"EVAL_TRACE_V1::"


@dataclass
class EvalTraceRow:
    sample_id: str
    row_index: int
    output_hash: str
    predicted: int
    label: int
    correct: int

    def to_obj(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "row_index": int(self.row_index),
            "output_hash": self.output_hash,
            "predicted": int(self.predicted),
            "label": int(self.label),
            "correct": int(self.correct),
        }

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "EvalTraceRow":
        return cls(
            sample_id=str(obj.get("sample_id", "")),
            row_index=int(obj.get("row_index", 0)),
            output_hash=str(obj.get("output_hash", "")),
            predicted=int(obj.get("predicted", 0)),
            label=int(obj.get("label", 0)),
            correct=int(obj.get("correct", 0)),
        )


def compute_eval_trace_root(rows: Iterable[EvalTraceRow]) -> str:
    acc = b"\x00" * 32
    for row in rows:
        encoded = canonical_encode(row.to_obj(), encoding_id=ENCODING_ID)
        acc = hashlib.sha256(HASH_PREFIX_EVAL_TRACE + acc + encoded).digest()
    return acc.hex()


__all__ = [
    "EvalTraceRow",
    "compute_eval_trace_root",
]
