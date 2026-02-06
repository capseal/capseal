# ENN Inference Telemetry Schema v1.0 (Draft)

Scope: Per-observation outputs and diagnostics persisted during ENN inference for calibration and Fusion.

## Determinism

- Inference must be built with deterministic flags (no -ffast-math), single-threaded runtime.
- Toolchain/flags are recorded in the artifact set.

## Table Schema (Parquet/Arrow)

Required columns:

- `model_id` string (hash of weights+config)
- `weights_hash` string (hex)
- `config_hash` string (hex)
- `sequence_id` uint64
- `step` uint32
- `q_pred` float32 (sigmoid(readout))
- `margin` float32 (pre-sigmoid readout)
- `alpha_entropy` float32 (−∑ α log α)
- `alpha_max` float32
- `attention_argmax` uint16 (index)
- `collapse_temp` float32 (exp(log_temp))
- `obs_reliability` float32 ([0,1], post-calibration)
- `calibrator_id` string

Optional:

- `node_id` uint32 (if mapping to graph nodes)
- `env_id` string (scenario tag)

File metadata (key/value):

- `telemetry_schema_version`: "enn_telemetry_v1"
- `feature_schema_hash`
- `inference_flags_hash` (compile/runtime flags)

## NaN/Clamp Policy

- Logits and softmax computed with numerical guards; no NaNs persisted.
- Clamp probabilities to [0,1].

## Calibration Contract

- `obs_reliability = Calibrate(margin or q_pred)` using the calibrator (see calibrator_v1.md).
- Calibrator MUST be version-locked to the exact `model_id` and feature schema.

