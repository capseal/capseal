# AgentEvalBench v1

Agent Evaluation as Active Learning - treating agent evaluation as a structured stochastic simulation space.

## Overview

AgentEvalBench is a framework for systematically evaluating agents under varying environment conditions using Beta posteriors and acquisition-based active sampling. Instead of uniform evaluation, it focuses evaluation budget on the most informative grid points.

### Key Concepts

- **Grid**: 1024 discrete environment configurations (5 parameters × 4 values each)
- **Episodes**: Stochastic agent-environment interactions with Bernoulli success/fail outcomes
- **Posteriors**: Per-grid-point Beta distributions over failure probability `p_fail(x)`
- **Acquisition**: Score-based selection prioritizing uncertain and boundary-near points
- **Tube**: Set of "safe" points where estimated `p_fail <= tau` (default tau=0.2)

## Quick Start

```bash
cd /home/ryan/otherstuff

# 1) Generate the grid (1024 points)
python -m agent_bench.grid --out artifacts/agent_grid_v1.npz

# 2) Create run directory and copy grid
mkdir -p artifacts/agent_test_synth
cp artifacts/agent_grid_v1.npz artifacts/agent_test_synth/grid.npz

# 3) Run synthetic mode (fast, closed-form p_fail)
python -m loop_runner \
  --mode agent_eval \
  --agent-bench toy_v1 \
  --run-dir artifacts/agent_test_synth \
  --rounds 3 \
  --seed 11111 \
  --seed-mode increment \
  --synthetic

# 4) Run real mode (actual env+agent simulation)
mkdir -p artifacts/agent_test_real
cp artifacts/agent_grid_v1.npz artifacts/agent_test_real/grid.npz

python -m loop_runner \
  --mode agent_eval \
  --agent-bench toy_v1 \
  --run-dir artifacts/agent_test_real \
  --rounds 2 \
  --seed 22222 \
  --seed-mode increment
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `bicep` | Loop mode: `bicep` or `agent_eval` |
| `--agent-bench` | `toy_v1` | Agent benchmark (only `toy_v1` for v1) |
| `--episodes-per-budget-unit` | `1` | Episodes per budget unit |
| `--targets-per-round` | `64` | Number of grid points per round (K) |
| `--synthetic` | `False` | Use closed-form p_fail instead of simulation |
| `--run-dir` | required | Directory for artifacts |
| `--rounds` | `5` | Number of rounds to run |
| `--seed` | `12345` | Base seed for reproducibility |

## Grid Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `tool_noise` | 0, 1, 2, 3 | Noise magnitude for lookup() tool |
| `verify_flip` | 0.0, 0.05, 0.10, 0.20 | Probability verify() returns wrong answer |
| `hint_ambiguity` | 0, 1, 2, 3 | Offset range for hint target |
| `distractor_count` | 0, 2, 4, 6 | Number of distractor tokens in hint |
| `memory_tokens` | 16, 32, 64, 128 | Token limit for hint (truncation) |

## Artifacts

### Run-Level Files

| File | Description |
|------|-------------|
| `grid.npz` | Parameter grid (must exist before run) |
| `run_metadata.json` | Run UUID and config |
| `beta_posteriors.npz` | Current posteriors (alpha/beta arrays) |
| `summary.csv` | Per-round summary metrics |

### Per-Round Files

| File | Description |
|------|-------------|
| `agent_results.csv` | Episode-level results with seeds |
| `metrics.json` | Round metrics (COMMIT MARKER) |
| `active_sampling_plan.json` | Acquisition plan |
| `round_pre.json` | Pre-round metadata |
| `round_post.json` | Post-round metadata |

## Metrics

### Tube Metrics

- **tube_var_sum**: Sum of posterior variance in tube
- **tube_coverage**: Fraction of grid points in tube
- **tube_var_delta_prev**: Change from previous round (positive = improvement)
- **tube_var_delta_baseline**: Change from first round

### Status Values

- `FIRST_ROUND`: No comparison possible
- `IMPROVED`: tube_var decreased
- `REGRESSED`: tube_var increased
- `NO_CHANGE`: tube_var unchanged

## Determinism

All randomness is derived deterministically from:
- `run_uuid`: Unique per-run identifier
- `round_num`: Round number (1-indexed)
- `grid_idx`: Grid point index
- `episode_idx`: Episode index within round

Seeds are derived via blake2b hash:
```python
episode_seed = blake2b(f"{run_uuid}:{round_num}:{grid_idx}:{episode_idx}")
agent_seed = blake2b(f"agent:{episode_seed}", person=b'AGENTSEED...')
```

## Resume

Runs automatically resume from the last complete round. A round is complete iff `metrics.json` exists.

```bash
# First run - completes rounds 1-3
python -m loop_runner --mode agent_eval --run-dir artifacts/test --rounds 3

# Resume - continues from round 4
python -m loop_runner --mode agent_eval --run-dir artifacts/test --rounds 3
```

## Testing

```bash
# Run all tests
python tests/test_agent_eval_bench.py

# Run with pytest (if installed)
pytest tests/test_agent_eval_bench.py -v
```

## Verification

Check consistency between metrics.json and summary.csv:

```python
import json
import csv

# Load round metrics
with open('artifacts/agent_test/rounds/R0002_.../metrics.json') as f:
    m = json.load(f)

# Load summary
with open('artifacts/agent_test/summary.csv') as f:
    rows = list(csv.DictReader(f))

# Verify tube_var matches
assert float(rows[1]['tube_var']) == m['tube']['tube_var_sum']
```

## Research Baselines (Future)

For publishable benchmarks, compare against:

| Baseline | Description |
|----------|-------------|
| Passive Uniform | Uniform random sampling |
| Random Top-K | Random selection each round |
| Pure Variance | Only epistemic uncertainty (w2=0) |
| Pure Boundary | Only boundary bonus (w1=0) |

Headline metrics:
- tube_var_sum vs episode budget (learning curve)
- Recall of unsafe points at tau

## Architecture

```
loop_runner.py
  └── --mode agent_eval
        └── agent_bench/runner.py
              ├── agent_bench/grid.py       (grid generation)
              ├── agent_bench/env_toy_v1.py (environment)
              ├── agent_bench/agent_toy_v1.py (agent)
              ├── agent_bench/metrics.py    (tube metrics)
              └── loop_io.py                (atomic I/O)
```

## Commit Order Invariant

For crash safety, artifacts are written in this order:
1. `round_pre.json`
2. `agent_results.csv` (atomic)
3. `beta_posteriors.npz` (atomic)
4. `metrics.json` (atomic) ← COMMIT MARKER
5. `summary.csv` (append)
6. `round_post.json` (atomic)

A round is complete iff `metrics.json` exists. On crash, incomplete rounds are skipped on resume.
