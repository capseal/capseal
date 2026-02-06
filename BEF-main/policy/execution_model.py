#!/usr/bin/env python3
"""
Execution model v1: simple deterministic fill assumptions for backtests.

- Mode 'passive': partial fill probability; unfilled portion ignored (no cross).
- Mode 'twap': assume half-spread + impact paid deterministically; full fill.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecConfig:
    mode: str = 'passive'  # 'passive' | 'twap'
    passive_fill_prob: float = 0.5
    tod_multiplier: float = 1.0  # time-of-day cost multiplier


def apply_execution(side: str,
                    size: float,
                    unit_return: float,
                    cost_all_in: float,
                    cfg: ExecConfig) -> float:
    """Return realized PnL in return units (not dollars)."""
    if side == 'FLAT' or size <= 0.0:
        return 0.0
    sign = 1.0 if side == 'LONG' else -1.0

    if cfg.mode == 'twap':
        pnl = size * (sign * unit_return - cfg.tod_multiplier * cost_all_in)
        return pnl

    # passive (probabilistic fill modeled deterministically by expectation)
    fill_prob = max(0.0, min(1.0, cfg.passive_fill_prob))
    expected_fill = size * fill_prob
    pnl = expected_fill * (sign * unit_return - cfg.tod_multiplier * cost_all_in)
    return pnl


__all__ = ['ExecConfig', 'apply_execution']

