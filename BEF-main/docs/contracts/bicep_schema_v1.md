# BICEP Trajectory Schema v1.0 (Draft)

Scope: Parquet/Arrow outputs produced by BICEP generators and samplers for downstream ENN/FusionAlpha.

## Determinism and Identity

- RNG is identity-bound via SeedSpec (see seed_spec_v1.md) — seed derives from instrument/time/ensemble/path.
- Outputs MUST be invariant to input/order permutations when identity is unchanged.

## Table Schema (Parquet/Arrow)

Required columns and types:

- `run_id` string
- `instrument_id` string
- `date_bucket` string (YYYY-MM-DD) OR `epoch_day` int32
- `ensemble_id` uint32
- `path_id` uint64
- `sequence_id` uint64 (if applicable; else equal to `path_id`)
- `step` uint32
- `t` float64
- `state` list<float64> (length = `state_dim`); alternatively `state_0..state_{D-1}` typed float64 columns
- `seed` uint64 (the derived per-path seed)

Optional per-step aggregates (if ensemble/state aggregation is computed):

- `state_mean` float64
- `state_std` float64 (ddof=1)
- `state_q10` float64
- `state_q90` float64
- `aleatoric_unc` float64 (within-config variance, averaged)
- `epistemic_unc` float64 (variance of ensemble means)

File metadata (key/value):

- `state_dim` (int32) if using list-typed `state`
- `bicep_version`, `integrator`, `model_name`, `model_params_hash`
- `trajectory_schema_version`: "bicep_trajectory_v1"

## Statistical Definitions

- Standard deviation: sample std with ddof=1.
- Quantiles: Hyndman & Fan type-7 (“linear” interpolation on sorted values).
- Values MUST be finite. NaNs/Inf are rejected; upstream MUST impute or drop.

## Provenance

- Include `seed_spec_id` and `seed_spec_hash` in a sidecar JSON (see artifact_set_v1.md) or Parquet metadata.
- Include `feature_schema_hash` if states are derived features.

