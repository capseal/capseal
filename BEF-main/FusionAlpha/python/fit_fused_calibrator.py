#!/usr/bin/env python3
"""
Fit a calibrator for FusionAlpha propagated scores (q_fused).

Inputs: CSV with either columns (propagated_q,target) or (score,target).
Outputs: JSON with Platt parameters and calibration metrics (ECE/Brier/curve).
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import List, Tuple

import numpy as np


def load_score_target(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    scores: List[float] = []
    targets: List[float] = []
    with open(csv_path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        score_key = 'propagated_q' if 'propagated_q' in fields else ('score' if 'score' in fields else None)
        if score_key is None or 'target' not in fields:
            raise ValueError("CSV must include ('propagated_q' or 'score') and 'target'")
        for row in reader:
            try:
                scores.append(float(row[score_key]))
                targets.append(float(row['target']))
            except Exception as exc:
                raise ValueError(f'Invalid row: {row}') from exc
    if not scores:
        raise ValueError('No rows loaded from CSV')
    return np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def platt_fit(scores: np.ndarray, targets: np.ndarray, max_iter: int = 500, lr: float = 5e-3,
              tol: float = 1e-6) -> Tuple[float, float]:
    A = 0.0
    B = 0.0
    for _ in range(max_iter):
        z = np.clip(A * scores + B, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        error = p - targets
        grad_A = float(np.dot(error, scores) / len(scores))
        grad_B = float(np.sum(error) / len(scores))
        if max(abs(grad_A), abs(grad_B)) < tol:
            break
        A -= lr * grad_A
        B -= lr * grad_B
    return float(A), float(B)


def reliability_curve(probs: np.ndarray, targets: np.ndarray, bins: int):
    cuts = np.linspace(0.0, 1.0, bins + 1)
    buckets = []
    ece = 0.0
    for i in range(bins):
        lo = cuts[i]
        hi = cuts[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        bucket_conf = float(probs[mask].mean())
        bucket_acc = float(targets[mask].mean())
        weight = float(mask.sum()) / len(probs)
        ece += weight * abs(bucket_acc - bucket_conf)
        buckets.append({
            'lower': float(lo),
            'upper': float(hi),
            'confidence': bucket_conf,
            'accuracy': bucket_acc,
            'weight': weight,
        })
    return buckets, float(ece)


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def main() -> None:
    p = argparse.ArgumentParser(description='Fit calibrator for FusionAlpha propagated scores')
    p.add_argument('csv', help='CSV with propagated_q/score and target columns')
    p.add_argument('out', help='Output JSON calibrator path')
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--calibrator-id', default='fused_platt')
    args = p.parse_args()

    scores, targets = load_score_target(args.csv)
    A, B = platt_fit(scores, targets)
    z = np.clip(A * scores + B, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-z))

    curve, ece = reliability_curve(probs, targets, args.bins)
    brier = brier_score(probs, targets)

    payload = {
        'schema': 'fusion_fused_calibrator_v1',
        'method': 'platt',
        'calibrator_id': args.calibrator_id,
        'params': {'A': A, 'B': B},
        'metrics': {'ece': ece, 'brier': brier, 'bins': curve},
        'fit_on': {'path': args.csv, 'n': int(len(scores))},
    }
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f'[OK] Wrote fused calibrator to {args.out}')


if __name__ == '__main__':
    main()

