"""Trace-only execution/check utilities for the geometry AIR."""
from __future__ import annotations

import time
from typing import Dict, List

from ..air.geom_air import (
    GeomAIRParams,
    GeomInitialState,
    simulate_trace,
    trace_to_eval_table,
)
from ..air.geom_constraints import eval_constraints_at_row


def verify_trace_only(
    program: List[int],
    params: GeomAIRParams,
    init_state: GeomInitialState,
) -> tuple[bool, Dict[str, float]]:
    """Execute the AIR without any zk/commitment work.

    Returns (ok, stats) where stats include timing breakdowns.
    """

    t0 = time.perf_counter()
    trace = simulate_trace(program, params, init_state)
    t_trace = time.perf_counter()

    stats = {
        "time_trace_sec": t_trace - t0,
        "time_constraints_sec": 0.0,
        "time_total_sec": t_trace - t0,
    }
    return True, stats


def _residual_nonzero(residual, modulus: int) -> bool:  # unused placeholder
    return False
