# ENN Calibrator v1.0 (Draft)

Scope: Map ENN margins or raw probabilities to calibrated `obs_reliability` in [0,1].

## JSON Format

```
{
  "schema": "enn_calibrator_v1",
  "method": "isotonic" | "platt",
  "horizon": "H20" | 20,
  "fit_on": {
    "model_id": "...",
    "weights_hash": "...",
    "config_hash": "...",
    "feature_schema_hash": "...",
    "train_split_hash": "..."
  },
  "params": { /* method-specific */ },
  "metrics": {
    "ece": 0.0123,
    "nll": 0.345,
    "auc": 0.91,
    "thresholds": {"ece_max": 0.02, "nll_max": 0.5}
  }
}
```

## Methods

- Isotonic regression (monotone, non-parametric): store breakpoints and values.
- Platt scaling (logistic): store A,B s.t. p=1/(1+exp(A*margin+B)).

## Contract

- Calibrator is immutable once shipped; any ENN retrain requires re-calibration and new `calibrator_id`.
- Monotonicity preserved for isotonic; probabilities clamped to [0,1].

## Acceptance

- Validation metrics must satisfy thresholds; record in `metrics.thresholds`.

