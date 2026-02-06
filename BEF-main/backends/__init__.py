"""Trace adapter backends and registry."""

from .geom_adapter import GeomTraceAdapter
from .risc0_adapter import Risc0TraceAdapter

ADAPTERS = {
    GeomTraceAdapter.name: GeomTraceAdapter,
    Risc0TraceAdapter.name: Risc0TraceAdapter,
}

__all__ = ["ADAPTERS", "GeomTraceAdapter", "Risc0TraceAdapter"]
