# Artifact Set v1.0 (Draft)

Scope: Versioned provenance bundle for BICEP → ENN → FusionAlpha.

## JSON Format

```
{
  "schema": "fusion_pipeline_artifacts_v1",
  "bicep": {
    "bicep_version": "...",
    "integrator": "EulerMaruyama|Heun|Milstein",
    "model_params_hash": "...",
    "seed_spec_id": "...",
    "seed_spec_hash": "...",
    "trajectory_schema_hash": "..."
  },
  "enn": {
    "model_id": "...",
    "weights_hash": "...",
    "config_hash": "...",
    "compile_flags_hash": "...",
    "telemetry_schema_hash": "...",
    "calibrator_id": "...",
    "calibrator_hash": "..."
  },
  "fusion": {
    "fusion_version": "...",
    "graph_spec_id": "...",
    "graph_spec_hash": "...",
    "prop_config_hash": "..."
  },
  "features": {
    "feature_schema_hash": "..."
  },
  "toolchain": {
    "compiler": "...",
    "cxxflags": "...",
    "ldflags": "...",
    "os_fingerprint": "...",
    "threads": {"omp": 1, "eigen": 1, "mkl": 1}
  },
  "artifact_set_hash": "blake3(...)"
}
```

## Anchor Hash

- `artifact_set_hash = blake3(ordered_concatenation_of_all_*_hash_fields)`
- Exclude timestamps/UUIDs from the hash inputs; store them in separate metadata if needed.

