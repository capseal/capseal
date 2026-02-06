#!/usr/bin/env python3
"""
Policy v1: deterministic gating, sizing, and action selection from ENN + FusionAlpha.

Semantics:
- p_trade: probability used for edge/cost break-even; v1 uses ENN q_pred.
- q_fused: optional filter/confirmatory score (can be calibrated separately).
- m_hat: magnitude proxy (E[|R|]); if unknown, default to 1.0 in backtest.

Outputs a simple Action tuple for backtest/execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Costs:
    half_spread_bps: float = 1.0
    fee_bps: float = 0.0
    impact_bps: float = 0.0

    @property
    def all_in(self) -> float:
        return (self.half_spread_bps + self.fee_bps + self.impact_bps) * 1e-4


@dataclass(frozen=True)
class Filters:
    pass_: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    # thresholds
    t_pred_long: float = 0.55
    t_fused_long: float = 0.55
    t_exit: float = 0.52
    buffer_mult: float = 1.0
    # sizing
    kelly_cap: float = 0.25
    shrink: float = 0.5
    var_floor: float = 1e-4
    # reliability haircut
    min_reliability: float = 0.1
    max_reliability: float = 1.0


@dataclass(frozen=True)
class Action:
    side: str  # 'FLAT' | 'LONG' | 'SHORT'
    size: float  # fraction of notional
    edge: float  # expected edge after costs
    reason: str


def decide(q_pred: float,
           q_fused: Optional[float],
           m_hat: float,
           obs_reliability: float,
           costs: Costs,
           filters: Filters,
           cfg: PolicyConfig) -> Action:
    # Choose p_trade semantics (v1: ENN)
    p_trade = float(q_pred)
    m_hat = float(m_hat)
    obs_rel = float(max(cfg.min_reliability, min(cfg.max_reliability, obs_reliability)))

    # Break-even gate
    edge_before_cost = (2.0 * p_trade - 1.0) * m_hat
    edge = edge_before_cost - costs.all_in

    if not filters.pass_:
        return Action('FLAT', 0.0, edge, 'filter_blocked')
    if edge < costs.all_in * cfg.buffer_mult:
        return Action('FLAT', 0.0, edge, 'below_break_even')

    long_ok = (p_trade >= cfg.t_pred_long)
    short_ok = (p_trade <= (1.0 - cfg.t_pred_long))
    fused_ok_long = True
    fused_ok_short = True
    if q_fused is not None:
        fused_ok_long = (q_fused >= cfg.t_fused_long)
        fused_ok_short = (q_fused <= (1.0 - cfg.t_fused_long))

    var_eff = max(cfg.var_floor, m_hat * m_hat)
    kelly = edge / var_eff
    size = max(0.0, min(cfg.kelly_cap, cfg.shrink * obs_rel * kelly))

    if long_ok and fused_ok_long:
        return Action('LONG', size, edge, 'enter_long')
    if short_ok and fused_ok_short:
        return Action('SHORT', size, edge, 'enter_short')
    return Action('FLAT', 0.0, edge, 'no_threshold')


__all__ = ['Costs', 'Filters', 'PolicyConfig', 'Action', 'decide']

