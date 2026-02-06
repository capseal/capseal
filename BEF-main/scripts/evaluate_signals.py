#!/usr/bin/env python3
"""Compute basic signal metrics for ENN telemetry (raw vs calibrated)."""

import argparse
import csv
import json
from typing import List, Tuple

import numpy as np


def load_columns(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    margins: List[float] = []
    calibrated: List[float] = []
    targets: List[float] = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"margin", "target"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Telemetry must include {required}")
        for row in reader:
            margins.append(float(row["margin"]))
            calibrated.append(float(row.get("obs_reliability", row.get("q_pred", 0.0))))
            targets.append(float(row["target"]))
    if not margins:
        raise ValueError("Empty telemetry file")
    return np.asarray(margins), np.asarray(calibrated), np.asarray(targets)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred > 0.5) == (target > 0.5)))


def reliability_curve(pred: np.ndarray, target: np.ndarray, bins: int = 10) -> Tuple[List[dict], float]:
    cuts = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    buckets = []
    for i in range(bins):
        lo, hi = cuts[i], cuts[i + 1]
        mask = (pred >= lo) & (pred <= hi if i == bins - 1 else pred < hi)
        if not mask.any():
            continue
        conf = float(pred[mask].mean())
        acc = float(target[mask].mean())
        weight = float(mask.sum()) / len(pred)
        ece += weight * abs(conf - acc)
        buckets.append({
            "lower": float(lo),
            "upper": float(hi),
            "confidence": conf,
            "accuracy": acc,
            "weight": weight,
        })
    return buckets, float(ece)


def summarize(name: str, pred: np.ndarray, target: np.ndarray) -> dict:
    curve, ece = reliability_curve(pred, target)
    return {
        "name": name,
        "accuracy": accuracy(pred, target),
        "brier": brier(pred, target),
        "ece": ece,
        "reliability_curve": curve,
    }


def apply_calibrator(margins: np.ndarray, calibrator_path: str) -> np.ndarray:
    with open(calibrator_path, "r", encoding="utf-8") as fh:
        calibrator = json.load(fh)
    params = calibrator.get("params", {})
    A = float(params.get("A", 0.0))
    B = float(params.get("B", 0.0))
    z = np.clip(A * margins + B, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(z))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ENN signal quality")
    parser.add_argument("telemetry", help="Telemetry CSV from apps/bicep_to_enn")
    parser.add_argument("--calibrator", help="Optional calibrator JSON to recompute reliability")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    margins, stored_reliability, targets = load_columns(args.telemetry)
    raw_probs = sigmoid(margins)
    calibrated = stored_reliability
    if args.calibrator:
        calibrated = apply_calibrator(margins, args.calibrator)

    results = {
        "samples": int(len(margins)),
        "raw": summarize("raw", raw_probs, targets),
        "calibrated": summarize("calibrated", calibrated, targets),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote evaluation report to {args.output}")
    else:
        for key, val in results.items():
            if key == "samples":
                print(f"Samples: {val}")
            else:
                print(f"[{key}] accuracy={val['accuracy']:.3f} brier={val['brier']:.3f} ece={val['ece']:.3f}")


if __name__ == "__main__":
    main()
