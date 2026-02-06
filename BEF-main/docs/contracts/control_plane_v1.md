# Control Plane v1.0 (Draft)

Scope: Define orthogonal controls for uncertainty vs. risk and their mapping in FusionAlpha.

## Knobs

- `obs_reliability` [0,1]: calibrated probability of correctness from ENN telemetry (clamped). Values below a reliability floor trigger deterministic fallbacks.
- `risk_aversion` [0,1]: policy knob (trader preference) for tail sensitivity.

## Mappings (defaults)

- Priors confidence (eta): `eta = eta_min + (eta_max - eta_min) * obs_reliability` with defaults `eta_min=0.1`, `eta_max=10.0`.
- Risk-sensitive blending: α = `risk_aversion` × `alpha_max`.
- Propagation steps: default `step_policy=fixed`. If enabled, `risk_scaled` uses `risk_aversion` to scale steps.

## Backwards Compatibility

- Legacy `severity` maps to both knobs: `obs_reliability = 1.0 - severity`, `risk_aversion = severity`. Emit deprecation warnings.

## Contracts

- Do not derive `risk_aversion` from telemetry. It is a configuration/policy input.
- `obs_reliability` MUST be produced by a versioned calibrator tied to the exact ENN weights/config and logged with ECE/Brier.
- Guardrails: eta is clamped to [`eta_min`,`eta_max`]; low reliability (< 0.05 by default) forces conservative propagation (max diffusion/min anchoring); propagation steps obey `StepPolicy` bounds.
