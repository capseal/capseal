# Acceptance Tests v1.0 (Draft)

Scope: End-to-end checks to enforce determinism and schema contracts across BICEP, ENN, FusionAlpha.

## 1) Bitwise Replay Test

Given frozen artifacts (artifact_set_v1), run full inference/propagation twice.

Assert binary identity for:

- ENN telemetry parquet files
- Fusion graph (nodes/edges) serialization
- Propagated q-values array and selected neighbor/action sequence

## 2) Identity Invariance Test

Permute input ordering (rows/batches) for BICEP/ENN. With identity-bound seeding and sorted adjacency, outputs MUST be identical.

## 3) Graph Determinism Test

Given GraphSpec and buffer snapshot, rebuild the graph.

Assert:

- Adjacency lists sorted and equal byte-for-byte
- kNN neighbor lists stable (distance tie → node_id ascending)

## 4) Stats Consistency Test (BICEP)

For a fixed fixture, compute std (ddof=1) and quantiles using reference implementation.

Assert:

- `state_std` matches within 1e-12 (float64)
- `state_q10`/`state_q90` equal under type-7 quantile

## 5) Calibration Quality Test (ENN)

Validate calibrator on holdout:

- ECE ≤ threshold (e.g., 1.5%)
- NLL ≤ threshold (configured per task)

Persist metrics and thresholds with the calibrator JSON.

