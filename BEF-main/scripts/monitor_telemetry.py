#!/usr/bin/env python3
"""Telemetry drift monitor with PSI/KS warnings."""

import argparse
import csv
import math
from typing import Dict, List

import numpy as np

def load_column(path: str, column: str) -> np.ndarray:
    values: List[float] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' missing in {path}")
        for row in reader:
            try:
                values.append(float(row[column]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No rows found in {path}")
    return np.asarray(values)


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    cuts = np.linspace(0.0, 1.0, bins + 1)
    # If values not normalized to 0-1, map via ranks
    if (baseline.min() < 0.0) or (baseline.max() > 1.0):
        baseline_norm = (baseline - baseline.min()) / (baseline.max() - baseline.min() + 1e-9)
        current_norm = (current - baseline.min()) / (baseline.max() - baseline.min() + 1e-9)
    else:
        baseline_norm = baseline
        current_norm = current
    psi_total = 0.0
    for i in range(bins):
        lo, hi = cuts[i], cuts[i + 1]
        base_mask = (baseline_norm >= lo) & (baseline_norm < hi if i < bins - 1 else baseline_norm <= hi)
        curr_mask = (current_norm >= lo) & (current_norm < hi if i < bins - 1 else current_norm <= hi)
        base_ratio = base_mask.sum() / len(baseline_norm)
        curr_ratio = curr_mask.sum() / len(current_norm)
        base_ratio = max(base_ratio, 1e-4)
        curr_ratio = max(curr_ratio, 1e-4)
        psi_total += (curr_ratio - base_ratio) * math.log(curr_ratio / base_ratio)
    return psi_total


def ks_stat(baseline: np.ndarray, current: np.ndarray) -> float:
    data = np.concatenate([baseline, current])
    data.sort()
    cdf_base = np.searchsorted(np.sort(baseline), data, side="right") / len(baseline)
    cdf_curr = np.searchsorted(np.sort(current), data, side="right") / len(current)
    return float(np.max(np.abs(cdf_curr - cdf_base)))


def monitor_column(name: str, base: np.ndarray, curr: np.ndarray, psi_threshold: float, ks_threshold: float) -> Dict:
    column_psi = psi(base, curr)
    column_ks = ks_stat(base, curr)
    alert = column_psi > psi_threshold or column_ks > ks_threshold
    recommendation = None
    if alert:
        recommendation = "increase_diffusion" if column_psi > psi_threshold else "reduce_allocation"
    return {
        "column": name,
        "psi": column_psi,
        "ks": column_ks,
        "alert": alert,
        "action": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor telemetry drift vs baseline")
    parser.add_argument("baseline", help="Baseline telemetry CSV")
    parser.add_argument("current", help="Current telemetry CSV")
    parser.add_argument("--columns", nargs="*", default=["alpha_entropy", "collapse_temperature", "margin"],
                        help="Columns to monitor")
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--ks-threshold", type=float, default=0.2)
    args = parser.parse_args()

    report = []
    for column in args.columns:
        base_values = load_column(args.baseline, column)
        curr_values = load_column(args.current, column)
        report.append(monitor_column(column, base_values, curr_values, args.psi_threshold, args.ks_threshold))

    alerts = [r for r in report if r["alert"]]
    for r in report:
        print(f"[{r['column']}] PSI={r['psi']:.3f} KS={r['ks']:.3f} alert={r['alert']} action={r['action']}")
    if alerts:
        raise SystemExit("Drift monitor detected alerts")


if __name__ == "__main__":
    main()
