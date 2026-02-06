#!/usr/bin/env python3
import json
import sys
import os
from typing import List


def ok(msg: str):
    print(f"[OK] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def fail(msg: str):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def validate_parquet(path: str):
    columns_required: List[str] = [
        "run_id", "seed", "model", "calc", "dt", "seq_len",
        "sequence_id", "step", "t", "state", "input", "target",
    ]
    try:
        import polars as pl
        df = pl.read_parquet(path)
        cols = set(df.columns)
        missing = [c for c in columns_required if c not in cols]
        if missing:
            fail(f"Parquet missing columns: {missing}")
        if df.height == 0:
            fail("Parquet contains zero rows")
        ok(f"Parquet OK: {path} (rows={df.height}, cols={len(cols)})")
    except Exception as e:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            cols = set(df.columns)
            missing = [c for c in columns_required if c not in cols]
            if missing:
                fail(f"Parquet missing columns: {missing}")
            if len(df) == 0:
                fail("Parquet contains zero rows")
            ok(f"Parquet OK: {path} (rows={len(df)}, cols={len(cols)})")
        except Exception as e2:
            # Fallback: ensure file exists and is non-empty
            if not os.path.exists(path):
                fail(f"Parquet file missing: {path}")
            if os.path.getsize(path) == 0:
                fail(f"Parquet file is empty: {path}")
            warn(f"Parquet schema not validated (need polars or pandas+pyarrow). Basic existence check passed: {path}")


def validate_csv(path: str):
    required = ["sequence_id", "step", "input", "target"]
    import csv
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            fail("CSV has no header/rows")
        missing = [c for c in required if c not in header]
        if missing:
            fail(f"CSV missing columns: {missing}")
        # Count a few rows
        n = 0
        for _ in reader:
            n += 1
            if n >= 5:
                break
        if n == 0:
            warn("CSV appears to have only a header; downstream may fail")
        ok(f"CSV OK: {path}")


def validate_graph(path: str):
    with open(path) as f:
        data = json.load(f)
    for k in ("nodes", "edges"):
        if k not in data:
            fail(f"Graph JSON missing key: {k}")
    nodes = data["nodes"]
    edges = data["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        fail("Graph JSON 'nodes'/'edges' must be lists")
    if len(nodes) == 0:
        warn("Graph has zero nodes")
    if len(edges) == 0:
        warn("Graph has zero edges")
    # Spot-check a few entries
    if nodes:
        if not (isinstance(nodes[0], list) and len(nodes[0]) >= 1):
            warn("Graph node format unexpected; expected list of coords")
    if edges:
        e0 = edges[0]
        if not (isinstance(e0, (list, tuple)) and len(e0) == 3):
            warn("Graph edge format unexpected; expected [i,j,weight]")
    ok(f"Graph OK: {path} (nodes={len(nodes)}, edges={len(edges)})")


def main():
    if len(sys.argv) < 3:
        print("Usage: validate_artifacts.py {parquet|csv|graph} <path>")
        sys.exit(2)
    kind, path = sys.argv[1], sys.argv[2]
    if kind == "parquet":
        validate_parquet(path)
    elif kind == "csv":
        validate_csv(path)
    elif kind == "graph":
        validate_graph(path)
    else:
        print(f"Unknown kind: {kind}")
        sys.exit(2)


if __name__ == "__main__":
    main()
