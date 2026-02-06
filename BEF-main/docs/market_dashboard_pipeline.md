# Market Dashboard: CapSeal End-to-End Run

This script runs the full CapSeal loop against `~/projects/market-dashboard` with a
Semgrep-based CI gate and an LLM explainer. It produces:

1. **Base run**: trace → review → DAG → rollup verification.
2. **Head run**: incremental trace reuse from the base run + review + DAG/rollup +
   diff receipt (warning+ gate).
3. **Explain step**: the LLM narrates the verified findings from the head run.

## Prerequisites

- `python -m bef_zk.capsule.cli` must work (activate `.venv` first).
- `capsule` has access to the repo at `~/projects/market-dashboard`.
- Semgrep ruleset `policy=review_v1` is available.
- For the explanation step, set the relevant API key, e.g.
  `export OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`.

## Usage

```bash
bash scripts/run_market_dashboard_review.sh
```

Environment variables (all optional) let you change defaults:

| Var | Purpose | Default |
| --- | --- | --- |
| `PROJECT_DIR` | Repository to review | `~/projects/market-dashboard` |
| `RUNS_DIR` | Where run folders are created | `~/capseal_runs/market_dashboard` |
| `POLICY_ID` | Trace/review policy | `review_v1` |
| `BACKEND` | Review backend for gating | `semgrep` |
| `AGENTS` | Parallel workers | `4` |
| `FAIL_SEVERITY` | `review-diff` gate threshold | `warning` |
| `LLM_PROVIDER` | Provider for explain step | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LLM_MAX_FINDINGS` | Max findings to narrate | `20` |
| `LLM_MIN_SEVERITY` | Min severity for explain | `warning` |

The script prints both run directories along with the diff receipt and explain
receipt so you can inspect the generated artifacts, e.g.:

```bash
ls -R "$HEAD_RUN/reviews" | less
python -m bef_zk.capsule.cli verify-rollup "$HEAD_RUN/workflow/rollup.json" \
  --project-dir "$PROJECT_DIR"
```
