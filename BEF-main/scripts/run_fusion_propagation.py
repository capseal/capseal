#!/usr/bin/env python3
"""
Fuse ENN predictions with the generated FusionAlpha graph and run a
severity-scaled propagation in pure Python (no Rust bindings required).

Outputs fusion_alpha_results.csv storing node-level priors, confidences,
and propagated committor scores so you can inspect the signal.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ATT_DIM_DEFAULT = int(os.getenv('FUSION_ATT_DIM', '32'))
MAX_ATTENTION_ENTROPY = math.log(max(ATT_DIM_DEFAULT, 2))
TEMP_REFERENCE = float(os.getenv('FUSION_ATT_TEMP_REF', '1.0'))
TEMP_SPAN = float(os.getenv('FUSION_ATT_TEMP_SPAN', '3.0'))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _coerce_state(value) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        return float(value[0])
    if isinstance(value, str):
        text = value.strip()
        try:
            if text.startswith('[') and text.endswith(']'):
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return float(parsed[0]) if parsed else 0.0
                return float(parsed)
            return float(text)
        except Exception:
            return 0.0
    return 0.0


def discretize_state(val: float, bounds: np.ndarray, n_bins: int) -> int:
    state = np.array([val], dtype=np.float32)
    normalized = (state - bounds[0, 0]) / (bounds[1, 0] - bounds[0, 0] + 1e-8)
    normalized = np.clip(normalized, 0.0, 0.999)
    idx = int(normalized[0] * n_bins)
    return idx


@dataclass
class StateFeatures:
    mean: float
    std: float
    q10: float
    q90: float
    aleatoric: float
    epistemic: float


def load_final_states(sequence_csv: str) -> Dict[int, StateFeatures]:
    df = pd.read_csv(sequence_csv)
    if 'state_0' in df.columns:
        df['state_value'] = df['state_0'].apply(_coerce_state)
    elif 'state' in df.columns:
        df['state_value'] = df['state'].apply(_coerce_state)
    else:
        df['state_value'] = df['input'].apply(_coerce_state)
    df['state_std_val'] = df.get('state_std', 0.0)
    df['state_q10_val'] = df.get('state_q10', df['state_value'])
    df['state_q90_val'] = df.get('state_q90', df['state_value'])
    df['aleatoric_val'] = df.get('aleatoric_unc', 0.0)
    df['epistemic_val'] = df.get('epistemic_unc', 0.0)
    idx = df.groupby('sequence_id')['step'].idxmax()
    finals = df.loc[idx, ['sequence_id', 'state_value', 'state_std_val', 'state_q10_val', 'state_q90_val', 'aleatoric_val', 'epistemic_val']]
    result: Dict[int, StateFeatures] = {}
    for row in finals.itertuples():
        result[int(row.sequence_id)] = StateFeatures(
            mean=float(row.state_value),
            std=float(row.state_std_val),
            q10=float(row.state_q10_val),
            q90=float(row.state_q90_val),
            aleatoric=float(row.aleatoric_val),
            epistemic=float(row.epistemic_val),
        )
    return result


def confidence_from_features(feat: StateFeatures) -> float:
    spread = max(1e-6, feat.q90 - feat.q10)
    noise = feat.std + feat.aleatoric + feat.epistemic
    conf = 1.0 / (1.0 + spread + noise)
    return float(np.clip(conf, 0.01, 0.99))


def aggregate_node_priors(graph_data: dict, preds_csv: str, seq_states: Dict[int, StateFeatures]):
    nodes = np.array(graph_data['nodes'], dtype=np.float32)
    edges = np.array(graph_data['edges'], dtype=np.float32)
    stats = graph_data['stats']
    bounds = np.array(stats['state_bounds'], dtype=np.float32)
    n_bins = stats['n_bins']

    agg_preds: Dict[int, Dict[str, float]] = {}
    preds_df = pd.read_csv(preds_csv)
    for row in preds_df.itertuples():
        seq_id = int(row.sequence_id)
        state_feat = seq_states.get(seq_id)
        if state_feat is None:
            continue
        node_idx = discretize_state(state_feat.mean, bounds, n_bins)
        bucket = agg_preds.setdefault(node_idx, {
            'sum_pred': 0.0,
            'sum_conf': 0.0,
            'sum_target': 0.0,
            'count': 0,
            'sum_entropy': 0.0,
            'cnt_entropy': 0,
            'sum_temp': 0.0,
            'cnt_temp': 0,
            'sum_attn_max': 0.0,
            'cnt_attn_max': 0,
        })
        bucket['sum_pred'] += float(row.final_prediction)

        sharp_factor = 1.0
        entropy_val = getattr(row, 'attention_entropy', float('nan'))
        if isinstance(entropy_val, str):
            try:
                entropy_val = float(entropy_val)
            except Exception:
                entropy_val = float('nan')
        if math.isfinite(entropy_val):
            bucket['sum_entropy'] += float(entropy_val)
            bucket['cnt_entropy'] += 1
            norm_entropy = float(np.clip(entropy_val / max(MAX_ATTENTION_ENTROPY, 1e-6), 0.0, 1.0))
            sharp_factor *= (1.0 - 0.5 * norm_entropy)

        temp_val = getattr(row, 'collapse_temperature', float('nan'))
        if isinstance(temp_val, str):
            try:
                temp_val = float(temp_val)
            except Exception:
                temp_val = float('nan')
        if math.isfinite(temp_val):
            bucket['sum_temp'] += float(temp_val)
            bucket['cnt_temp'] += 1
            temp_norm = (float(temp_val) - TEMP_REFERENCE) / max(TEMP_SPAN, 1e-6)
            temp_norm = float(np.clip(temp_norm, -0.5, 1.0))
            sharp_factor *= (1.0 - 0.4 * max(0.0, temp_norm))

        attn_max_val = getattr(row, 'attention_max', float('nan'))
        if isinstance(attn_max_val, str):
            try:
                attn_max_val = float(attn_max_val)
            except Exception:
                attn_max_val = float('nan')
        if math.isfinite(attn_max_val):
            bucket['sum_attn_max'] += float(attn_max_val)
            bucket['cnt_attn_max'] += 1
            sharp_factor *= (0.5 + 0.5 * float(np.clip(attn_max_val, 0.0, 1.0)))

        combined_conf = 0.5 * float(row.confidence) + 0.5 * confidence_from_features(state_feat)
        combined_conf *= sharp_factor
        bucket['sum_conf'] += combined_conf
        bucket['sum_target'] += float(row.target)
        bucket['count'] += 1

    n_nodes = nodes.shape[0]
    q0 = np.full(n_nodes, np.nan, dtype=np.float32)
    conf = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.int32)
    entropy = np.full(n_nodes, np.nan, dtype=np.float32)
    temperature = np.full(n_nodes, np.nan, dtype=np.float32)
    attn_max = np.full(n_nodes, np.nan, dtype=np.float32)
    goal_nodes: List[int] = []

    for idx, stats in agg_preds.items():
        counts[idx] = stats['count']
        q0[idx] = stats['sum_pred'] / stats['count']
        conf[idx] = stats['sum_conf'] / stats['count']
        frac_pos = stats['sum_target'] / stats['count']
        if stats['cnt_entropy'] > 0:
            entropy[idx] = stats['sum_entropy'] / stats['cnt_entropy']
        if stats['cnt_temp'] > 0:
            temperature[idx] = stats['sum_temp'] / stats['cnt_temp']
        if stats['cnt_attn_max'] > 0:
            attn_max[idx] = stats['sum_attn_max'] / stats['cnt_attn_max']
        if frac_pos >= 0.5:
            goal_nodes.append(idx)

    for idx in goal_nodes:
        q0[idx] = 1.0
        conf[idx] = 1.0

    return nodes, edges, q0, conf, counts, goal_nodes, entropy, temperature, attn_max


def build_adjacency(edges: np.ndarray, n_nodes: int) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for u, v, w in edges:
        u_i = int(u)
        v_i = int(v)
        adj[u_i].append((v_i, float(w)))
    return adj


def propagate(q0: np.ndarray, conf: np.ndarray, adj: List[List[Tuple[int, float]]], goal_nodes: List[int],
              entropy: np.ndarray | None = None, temperature: np.ndarray | None = None,
              attn_max: np.ndarray | None = None):
    n = len(q0)
    q = np.where(np.isfinite(q0), q0, 0.5).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    for idx in goal_nodes:
        fixed[idx] = True
        q[idx] = 1.0

    finite_mask = np.isfinite(q0)
    avg_conf = float(np.mean(conf[finite_mask])) if finite_mask.any() else 0.5

    entropy_component = 0.0
    if entropy is not None:
        ent_vals = entropy[np.isfinite(entropy)]
        if ent_vals.size > 0:
            norm = np.clip(ent_vals / max(MAX_ATTENTION_ENTROPY, 1e-6), 0.0, 1.0)
            entropy_component = float(norm.mean())

    temp_component = 0.0
    if temperature is not None:
        temp_vals = temperature[np.isfinite(temperature)]
        if temp_vals.size > 0:
            temp_norm = (temp_vals - TEMP_REFERENCE) / max(TEMP_SPAN, 1e-6)
            temp_norm = np.clip(temp_norm, 0.0, 1.0)
            temp_component = float(temp_norm.mean())

    attn_component = 0.0
    if attn_max is not None:
        attn_vals = attn_max[np.isfinite(attn_max)]
        if attn_vals.size > 0:
            attn_component = float(1.0 - np.clip(attn_vals, 0.0, 1.0).mean())

    severity_raw = (0.5 * (1.0 - avg_conf)
                     + 0.25 * entropy_component
                     + 0.15 * temp_component
                     + 0.10 * attn_component)
    severity = float(np.clip(severity_raw, 0.05, 0.95))
    t_steps = max(10, int(20 + 60 * severity))

    for step in range(t_steps):
        new_q = q.copy()
        delta = 0.0
        for i in range(n):
            if fixed[i]:
                continue
            neighbor_sum = 0.0
            weight_sum = 0.0
            for j, w in adj[i]:
                weight_sum += w
                neighbor_sum += w * q[j]
            if weight_sum == 0:
                continue
            neighbor_avg = neighbor_sum / weight_sum
            blended = severity * neighbor_avg + (1.0 - severity) * q[i]
            if np.isfinite(q0[i]):
                prior_weight = max(1e-3, conf[i])
                blended = (prior_weight * q0[i] + weight_sum * neighbor_avg) / (prior_weight + weight_sum)
            delta = max(delta, abs(new_q[i] - blended))
            new_q[i] = blended
        q = new_q
        if delta < 1e-4:
            break

    return q, severity, step + 1, avg_conf


def save_results(path: str, nodes: np.ndarray, q0: np.ndarray, conf: np.ndarray,
                 q_prop: np.ndarray, counts: np.ndarray, goal_nodes: List[int],
                 entropy: np.ndarray, temperature: np.ndarray, attn_max: np.ndarray):
    rows = []
    goal_set = set(goal_nodes)
    for idx in range(len(nodes)):
        rows.append({
            'node_id': idx,
            'x': float(nodes[idx][0]),
            'y': float(nodes[idx][1]) if nodes.shape[1] > 1 else 0.0,
            'prior_q': float(q0[idx]) if np.isfinite(q0[idx]) else '',
            'confidence': float(conf[idx]) if conf[idx] > 0 else '',
            'attention_entropy': float(entropy[idx]) if np.isfinite(entropy[idx]) else '',
            'collapse_temperature': float(temperature[idx]) if np.isfinite(temperature[idx]) else '',
            'attention_max': float(attn_max[idx]) if np.isfinite(attn_max[idx]) else '',
            'propagated_q': float(q_prop[idx]),
            'samples': int(counts[idx]),
            'is_goal': 1 if idx in goal_set else 0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run committor propagation using ENN priors")
    parser.add_argument('--graph', default=os.path.join(REPO_ROOT, 'fusion_graph.json'))
    parser.add_argument('--enn-preds', default=os.path.join(REPO_ROOT, 'enn-cpp', 'enn_predictions.csv'))
    parser.add_argument('--sequence-csv', default=os.path.join(REPO_ROOT, 'enn-cpp', 'parity_data.csv'))
    parser.add_argument('--out', default=os.path.join(REPO_ROOT, 'fusion_alpha_results.csv'))
    args = parser.parse_args()

    if not (os.path.exists(args.graph) and os.path.exists(args.enn_preds) and os.path.exists(args.sequence_csv)):
        print('[WARN] Missing inputs for FusionAlpha propagation. Skipping.')
        return

    with open(args.graph) as f:
        graph_data = json.load(f)

    seq_states = load_final_states(args.sequence_csv)
    nodes, edges, q0, conf, counts, goal_nodes, entropy, temperature, attn_max = aggregate_node_priors(
        graph_data, args.enn_preds, seq_states)

    if not np.isfinite(q0).any():
        print('[WARN] No ENN priors mapped to graph nodes. Skipping propagation.')
        return

    adj = build_adjacency(edges, len(nodes))
    q_prop, severity, t_steps, avg_conf = propagate(q0, conf, adj, goal_nodes, entropy, temperature, attn_max)
    save_results(args.out, nodes, q0, conf, q_prop, counts, goal_nodes, entropy, temperature, attn_max)

    top_idx = np.argsort(q_prop)[::-1][:5]
    print(f"[FusionAlpha] severity={severity:.3f}, avg_conf={avg_conf:.3f}, iterations={t_steps}")
    print("Top propagated nodes:")
    for idx in top_idx:
        prior = q0[idx] if np.isfinite(q0[idx]) else float('nan')
        print(f"  node={idx:<4d} prior={prior:.3f} propagated={q_prop[idx]:.3f} samples={counts[idx]}")
    print(f"Saved propagation results to {args.out}")


if __name__ == '__main__':
    main()
