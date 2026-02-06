#!/usr/bin/env python3
"""Fit a simple calibrator (Platt scaling) for ENN telemetry."""

import argparse
import csv
import json
from typing import List, Tuple

import numpy as np


def load_margin_target(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    margins: List[float] = []
    targets: List[float] = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if "margin" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("CSV must contain 'margin' and 'target' columns")
        for row in reader:
            try:
                margins.append(float(row["margin"]))
                targets.append(float(row["target"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid row: {row}") from exc
    if not margins:
        raise ValueError("No rows loaded from telemetry CSV")
    return np.asarray(margins, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def platt_fit(margins: np.ndarray, targets: np.ndarray, max_iter: int = 500, lr: float = 1e-2,
              tol: float = 1e-6) -> Tuple[float, float]:
    A = 0.0
    B = 0.0
    for _ in range(max_iter):
        z = np.clip(A * margins + B, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(z))
        error = p - targets
        grad_A = np.dot(error, margins) / len(margins)
        grad_B = np.sum(error) / len(margins)
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
            "lower": float(lo),
            "upper": float(hi),
            "confidence": bucket_conf,
            "accuracy": bucket_acc,
            "weight": weight,
        })
    return buckets, float(ece)


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def monotonicity_check(margins: np.ndarray, probs: np.ndarray) -> bool:
    order = np.argsort(margins)
    diffs = np.diff(probs[order])
    return bool(np.all(diffs >= -1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit calibrator for ENN telemetry")
    parser.add_argument("telemetry", help="CSV file with margin,target columns")
    parser.add_argument("output", help="Path to calibrator JSON")
    parser.add_argument("--method", choices=["platt"], default="platt")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--model-id", default="unknown_model")
    parser.add_argument("--calibrator-id", default="calibrator_platt")
    args = parser.parse_args()

    margins, targets = load_margin_target(args.telemetry)
    A, B = platt_fit(margins, targets)
    z = np.clip(A * margins + B, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(z))

    curve, ece = reliability_curve(probs, targets, args.bins)
    brier = brier_score(probs, targets)
    monotonic = monotonicity_check(margins, probs)

    calibrator = {
        "schema": "enn_calibrator_v1",
        "method": "platt",
        "calibrator_id": args.calibrator_id,
        "fit_on": {
            "model_id": args.model_id,
            "telemetry_path": args.telemetry,
            "num_samples": int(len(margins)),
        },
        "params": {
            "A": A,
            "B": B,
        },
        "metrics": {
            "ece": ece,
            "brier": brier,
            "monotonic": monotonic,
            "bins": curve,
        },
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(calibrator, fh, indent=2)
    print(f"Wrote calibrator JSON to {args.output}")


if __name__ == "__main__":
    main()
