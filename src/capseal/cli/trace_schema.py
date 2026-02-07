"""Trace schema definitions for semantic row bindings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FieldDef:
    """Definition of a single trace field."""
    name: str
    field_type: str  # "felt", "u64", "i64", "f64_scaled", "hash"
    meaning: str
    unit: str | None = None
    scale: int | None = None  # For f64_scaled: value = raw / scale


@dataclass
class TraceSchema:
    """Schema for trace row semantics."""
    schema_id: str
    schema_version: str
    description: str
    fields: list[FieldDef] = field(default_factory=list)
    row_width: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "description": self.description,
            "row_width": self.row_width,
            "fields": [
                {
                    "name": f.name,
                    "type": f.field_type,
                    "meaning": f.meaning,
                    "unit": f.unit,
                    "scale": f.scale,
                }
                for f in self.fields
            ],
        }

    def hash(self) -> str:
        """Compute canonical hash of schema."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def decode_row(self, row_values: list[int]) -> dict[str, Any]:
        """Decode raw row values into semantic dict."""
        result = {}
        for i, f in enumerate(self.fields):
            if i >= len(row_values):
                result[f.name] = None
                continue

            raw = row_values[i]
            if f.field_type == "f64_scaled" and f.scale:
                result[f.name] = raw / f.scale
            elif f.field_type == "hash":
                result[f.name] = hex(raw)
            else:
                result[f.name] = raw
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceSchema":
        fields = [
            FieldDef(
                name=f["name"],
                field_type=f["type"],
                meaning=f["meaning"],
                unit=f.get("unit"),
                scale=f.get("scale"),
            )
            for f in d.get("fields", [])
        ]
        return cls(
            schema_id=d["schema_id"],
            schema_version=d["schema_version"],
            description=d["description"],
            fields=fields,
            row_width=d.get("row_width", len(fields)),
        )

    @classmethod
    def load(cls, path: Path) -> "TraceSchema":
        return cls.from_dict(json.loads(path.read_text()))


# Built-in schemas
SCHEMAS: dict[str, TraceSchema] = {}


def register_schema(schema: TraceSchema) -> None:
    SCHEMAS[schema.schema_id] = schema


def get_schema(schema_id: str) -> TraceSchema | None:
    return SCHEMAS.get(schema_id)


# Register built-in schemas
register_schema(TraceSchema(
    schema_id="momentum_v2_row_v1",
    schema_version="1.0",
    description="Momentum strategy trace row with price/signal/position",
    row_width=14,
    fields=[
        FieldDef("step", "u64", "Trace step index"),
        FieldDef("timestamp", "u64", "Unix timestamp (seconds)"),
        FieldDef("price", "f64_scaled", "Asset price", unit="USD", scale=10**8),
        FieldDef("returns", "f64_scaled", "Log returns", scale=10**12),
        FieldDef("volatility", "f64_scaled", "Realized volatility (annualized)", scale=10**8),
        FieldDef("signal_raw", "f64_scaled", "Raw signal value", scale=10**8),
        FieldDef("signal_clipped", "f64_scaled", "Signal after clipping", scale=10**8),
        FieldDef("position", "f64_scaled", "Target position size", scale=10**8),
        FieldDef("pnl_step", "f64_scaled", "Step P&L", unit="USD", scale=10**8),
        FieldDef("pnl_cumulative", "f64_scaled", "Cumulative P&L", unit="USD", scale=10**8),
        FieldDef("state_hash", "hash", "State commitment at this step"),
        FieldDef("input_hash", "hash", "Input data hash"),
        FieldDef("output_hash", "hash", "Output data hash"),
        FieldDef("aux", "felt", "Auxiliary field element"),
    ],
))

register_schema(TraceSchema(
    schema_id="signal_extraction_v1",
    schema_version="1.0",
    description="Signal extraction pipeline trace (BICEP/ENN/FusionAlpha)",
    row_width=8,
    fields=[
        FieldDef("step", "u64", "Pipeline stage index"),
        FieldDef("stage_id", "hash", "Stage identifier hash"),
        FieldDef("input_hash", "hash", "Stage input commitment"),
        FieldDef("output_hash", "hash", "Stage output commitment"),
        FieldDef("duration_ms", "u64", "Stage duration in milliseconds"),
        FieldDef("status", "u64", "Status code (0=ok, 1=warn, 2=error)"),
        FieldDef("metric_1", "f64_scaled", "Primary metric (context-dependent)", scale=10**8),
        FieldDef("metric_2", "f64_scaled", "Secondary metric", scale=10**8),
    ],
))


__all__ = [
    "FieldDef",
    "TraceSchema",
    "register_schema",
    "get_schema",
    "SCHEMAS",
]
