"""Utilities to export geometry traces into bef_trace_v1 chunks."""
from __future__ import annotations

from typing import Dict, List

from .geom_air import GeomTrace

GEOM_ROW_BASE_FIELDS = [
    "pc",
    "opcode",
    "gas",
    "acc",
    "x1",
    "x2",
    "cnt",
    "m11",
    "m12",
    "m22",
]


def flatten_geom_row(row) -> List[int]:
    """Return the canonical STC row vector for a GeomAirRow."""
    values: List[int] = [
        int(row.pc),
        int(row.opcode),
        int(row.gas),
        int(row.acc),
        int(row.x1),
        int(row.x2),
        int(row.cnt),
        int(row.m11),
        int(row.m12),
        int(row.m22),
    ]
    values.extend(int(x) for x in row.s)
    values.extend(int(x) for x in row.pow)
    return values


def geom_trace_to_bef_trace(trace: GeomTrace, trace_id: str) -> Dict:
    if not trace.rows:
        raise ValueError("trace has no rows")
    row_width = len(flatten_geom_row(trace.rows[0]))
    chunks = []
    offset = 0
    for chunk_index, row in enumerate(trace.rows):
        values = flatten_geom_row(row)
        if len(values) != row_width:
            raise ValueError("row width mismatch in trace")
        chunks.append(
            {
                "chunk_index": chunk_index,
                "offset": offset,
                "values": values,
            }
        )
        offset += row_width
    return {
        "schema": "bef_trace_v1",
        "trace_id": trace_id,
        "field_modulus": trace.params.modulus,
        "num_steps": trace.params.steps,
        "vector_length": offset,
        "chunk_length": row_width,
        "row_fields": GEOM_ROW_BASE_FIELDS
        + [f"s[{i}]" for i in range(trace.params.num_challenges)]
        + [f"pow[{i}]" for i in range(trace.params.num_challenges)],
        "chunks": chunks,
    }
