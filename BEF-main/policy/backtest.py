#!/usr/bin/env python3
"""
Deterministic backtest harness for ENN + FusionAlpha policy v1.

Inputs:
- ENN telemetry CSV from enn-cpp/apps/bicep_to_enn (sequence_id, step, margin, q_pred, obs_reliability, target, ...)
- Optional fused CSV with node-level propagated scores (fusion_alpha_results.csv) to supply q_fused per node.
  If provided, maps each sample to a node via a simple binning of state_mean using provided bounds metadata
  in a sidecar graph JSON (graph_v1.json) or uses a constant q_fused if mapping unavailable.

Outputs:
- Summary metrics: accuracy, Brier, ECE, turnover, cost-adjusted expected PnL (unit returns), decision counts.
- Deterministic CSV log of decisions and realized unit PnL.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from policy_v1 import Costs, Filters, PolicyConfig, decide
from execution_model import ExecConfig, apply_execution


@dataclass
class Sample:
    sequence_id: int
    step: int
    q_pred: float
    obs_rel: float
    target: float
    m_hat: float
    q_fused: Optional[float]
    state_mean: Optional[float] = None


def load_enn_telemetry(path: Path) -> List[Sample]:
    rows: List[Sample] = []
    with path.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'sequence_id', 'step', 'q_pred', 'obs_reliability', 'target'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Missing columns in ENN telemetry: need {required}')
        for row in reader:
            seq = int(float(row['sequence_id']))
            step = int(float(row['step']))
            q_pred = float(row['q_pred'])
            obs_rel = float(row['obs_reliability'])
            target = float(row['target'])
            m_hat = float(row.get('m_hat', 1.0))
            state_mean = None
            if 'state_mean' in (reader.fieldnames or []):
                try:
                    state_mean = float(row['state_mean'])
                except Exception:
                    state_mean = None
            rows.append(Sample(seq, step, q_pred, obs_rel, target, m_hat, None, state_mean))
    return rows


def load_fused_scores(path: Path) -> Dict[int, float]:
    score_by_node: Dict[int, float] = {}
    with path.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        if 'node_id' not in reader.fieldnames or 'propagated_q' not in reader.fieldnames:
            raise ValueError('Fused CSV must have node_id, propagated_q columns')
        for row in reader:
            score_by_node[int(float(row['node_id']))] = float(row['propagated_q'])
    return score_by_node


def maybe_map_fused(samples: List[Sample], fused: Dict[int, float], graph_json: Optional[Path]) -> None:
    if not fused:
        return
    bounds = None
    if graph_json and graph_json.exists():
        with graph_json.open('r', encoding='utf-8') as fh:
            g = json.load(fh)
        stats = g.get('stats') or {}
        bounds = stats.get('state_bounds')
        n_bins = int(stats.get('n_bins', 0)) if stats else 0
    # If we had per-sample state_mean we could bin it; the ENN telemetry has it at final step only.
    # Implement mapping if bounds + n_bins present and state_mean available.
    if bounds is not None and n_bins > 0:
        lo = float(bounds[0][0])
        hi = float(bounds[1][0])
        span = max(hi - lo, 1e-8)
        for s in samples:
            if s.state_mean is None:
                continue
            x = float(s.state_mean)
            z = (x - lo) / span
            z = max(0.0, min(0.999999, z))
            idx = int(z * n_bins)
            if idx in fused:
                s.q_fused = fused[idx]
    elif len(fused) == 1:
        only_score = next(iter(fused.values()))
        for s in samples:
            s.q_fused = only_score


def ece_brier(probs: np.ndarray, targets: np.ndarray, bins: int = 10) -> Tuple[float, float]:
    cuts = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = cuts[i], cuts[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(targets[mask].mean())
        w = float(mask.sum()) / len(probs)
        ece += w * abs(acc - conf)
    brier = float(np.mean((probs - targets) ** 2))
    return ece, brier


def main() -> None:
    p = argparse.ArgumentParser(description='Backtest policy v1 on ENN/FusionAlpha outputs')
    p.add_argument('--enn', required=True, help='ENN telemetry CSV (enn_predictions.csv)')
    p.add_argument('--fused', help='Fused CSV (fusion_alpha_results.csv)')
    p.add_argument('--graph', help='Graph JSON (graph_v1.json) for mapping fused scores')
    p.add_argument('--out', default='policy_backtest_log.csv')
    p.add_argument('--costs', nargs='*', help='half=1.0 fee=0.0 impact=0.0 (bps)')
    p.add_argument('--mode', choices=['passive', 'twap'], default='passive')
    args = p.parse_args()

    samples = load_enn_telemetry(Path(args.enn))
    fused: Dict[int, float] = {}
    if args.fused:
        fused = load_fused_scores(Path(args.fused))
        maybe_map_fused(samples, fused, Path(args.graph) if args.graph else None)

    # Costs
    half = 1.0
    fee = 0.0
    impact = 0.0
    if args.costs:
        for kv in args.costs:
            k, v = kv.split('=', 1)
            if k == 'half':
                half = float(v)
            elif k == 'fee':
                fee = float(v)
            elif k == 'impact':
                impact = float(v)
    costs = Costs(half_spread_bps=half, fee_bps=fee, impact_bps=impact)
    exec_cfg = ExecConfig(mode=args.mode)
    cfg = PolicyConfig()

    # Deterministic pass
    probs = []
    targets = []
    actions = []
    pnl_sum = 0.0
    trades = 0
    with open(args.out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['sequence_id', 'step', 'q_pred', 'q_fused', 'obs_reliability', 'm_hat', 'action', 'size', 'edge', 'unit_pnl', 'target'])
        for s in samples:
            a = decide(
                q_pred=s.q_pred,
                q_fused=s.q_fused,
                m_hat=s.m_hat,
                obs_reliability=s.obs_rel,
                costs=costs,
                filters=Filters(pass_=True),
                cfg=cfg,
            )
            probs.append(s.q_pred)
            targets.append(s.target)
            unit_return = (1.0 if s.target >= 0.5 else -1.0) * s.m_hat
            unit_pnl = apply_execution(a.side, a.size, unit_return, costs.all_in, exec_cfg)
            pnl_sum += unit_pnl
            if a.side != 'FLAT' and a.size > 0:
                trades += 1
            writer.writerow([s.sequence_id, s.step, s.q_pred, s.q_fused if s.q_fused is not None else '', s.obs_rel, s.m_hat, a.side, f'{a.size:.6f}', f'{a.edge:.6f}', f'{unit_pnl:.6f}', s.target])

    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    ece, brier = ece_brier(probs_arr, targets_arr)
    acc = float(((probs_arr >= 0.5).astype(np.int32) == targets_arr.astype(np.int32)).mean())
    print(f'[Backtest] N={len(samples)} acc={acc:.3f} ece={ece:.3f} brier={brier:.3f} trades={trades} unit_pnl={pnl_sum:.6f}')
    print(f'[Backtest] Log written to {args.out}')


if __name__ == '__main__':
    main()
