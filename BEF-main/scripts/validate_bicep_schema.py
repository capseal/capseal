#!/usr/bin/env python3
"""Validate BICEP CSV/Parquet files against bicep_schema_v1."""

import argparse
import csv
import json
from pathlib import Path
from typing import List

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

REQUIRED_COLUMNS = [
    "sequence_id",
    "step",
    "input",
    "target",
    "state_mean",
    "state_std",
    "state_q10",
    "state_q90",
    "aleatoric_unc",
    "epistemic_unc",
]


def load_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for col in REQUIRED_COLUMNS:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing column '{col}'")
        return list(reader)


def load_parquet(path: Path) -> List[dict]:
    if pd is None:
        raise RuntimeError("pandas/pyarrow required for Parquet validation")
    df = pd.read_parquet(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")
    return df.to_dict(orient="records")


def check_order(rows: List[dict]) -> None:
    last_seq = None
    last_step = -1
    for row in rows:
        seq = int(float(row["sequence_id"]))
        step = int(float(row["step"]))
        if last_seq is None or seq != last_seq:
            last_seq = seq
            last_step = -1
        if step != last_step + 1:
            raise ValueError(f"Non-consecutive step for sequence {seq}: got {step}, expected {last_step + 1}")
        last_step = step


def check_values(rows: List[dict]) -> None:
    for row in rows:
        mean = float(row["state_mean"])
        std = float(row["state_std"])
        q10 = float(row["state_q10"])
        q90 = float(row["state_q90"])
        alea = float(row["aleatoric_unc"])
        epis = float(row["epistemic_unc"]) 
        target = float(row["target"])
        if std < 0 or alea < 0 or epis < 0:
            raise ValueError("Negative variance/uncertainty encountered")
        if q10 > q90:
            raise ValueError("q10 > q90")
        if not (q10 - 1e-6 <= mean <= q90 + 1e-6):
            raise ValueError("state_mean outside [q10,q90]")
        if not (0.0 - 1e-6 <= target <= 1.0 + 1e-6):
            raise ValueError("target outside [0,1]")


def check_metadata(meta_path: Path) -> None:
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("std_ddof") != 1:
        raise ValueError("std_ddof must be 1")
    if meta.get("quantile_method") not in ("lin-interp-type7", "type7"):
        raise ValueError("quantile_method must be type7")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate bicep_schema_v1")
    parser.add_argument("path")
    parser.add_argument("--format", choices=["csv", "parquet"], help="Override format")
    parser.add_argument("--metadata", required=True, help="Metadata JSON with std/quantile info")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = args.format or ("csv" if path.suffix.lower() == ".csv" else "parquet")
    rows = load_csv(path) if fmt == "csv" else load_parquet(path)
    if not rows:
        raise ValueError("File contains no rows")
    check_order(rows)
    check_values(rows)
    check_metadata(Path(args.metadata))
    print(f"Schema validation passed for {path}")


if __name__ == "__main__":
    main()
