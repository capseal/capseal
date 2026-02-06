# FusionAlpha PropagationConfig v1.0 (Draft)

Scope: Risk-aware committor propagation configuration for FusionAlpha.

## JSON Format

```
{
  "schema": "fusion_prop_config_v1",
  "t_max": 50,
  "eps": 1e-4,
  "use_parallel": true,
  "alpha_max": 6.0,
  "risk_aversion": 0.0,
  "step_policy": "fixed" | "risk_scaled"
}
```

## Semantics

- `risk_aversion` ∈ [0,1] sets α = `risk_aversion` × `alpha_max` for risk-sensitive updates.
- `step_policy`:
  - `fixed`: use `t_max` (or caller-provided steps) directly.
  - `risk_scaled`: `t_steps = 1 + floor(risk_aversion × (t_max-1))`.

## Back-compat (severity)

- If only `severity` is provided by caller:
  - `obs_reliability = 1.0 - severity` (linear mapping)
  - `risk_aversion = severity`
  - Marked deprecated; prefer explicit knobs.

## Determinism

- For bitwise identity with `use_parallel=true`, adjacency order MUST be deterministic (see graphspec).
- For strictest replay, `use_parallel=false` is allowed.

