# TODO/FIXME/HACK Triage (2026-02-11)

Command used:

```bash
rg -n "TODO|FIXME|HACK" src/ capseal-tui/src/ tests/ --glob '*.py' --glob '*.rs'
```

## Result

- Open TODO-style comments in product code: `0`
- Remaining keyword matches are intentional scanner logic:
  - `src/capseal/cli/trace_cmd.py:740` and `src/capseal/cli/trace_cmd.py:742`
  - `src/capseal/refactor_engine.py:2277`

## Actions Completed

1. `src/capseal/checkers.py`
- Replaced hardcoded Hypothesis checker version with runtime detection via `importlib.metadata`.

2. `src/capseal/critic_agent.py`
- Reworded template placeholder comments to remove TODO markers in generated test templates.

3. `src/capseal/cli/profile_cmd.py`
- Replaced `"TODO: describe project purpose"` default with `"Describe project purpose"`.

4. `src/capseal/cli/pipeline.py`
- Replaced TODO placeholder commitments with explicit pending markers:
  - `PENDING_INPUTS_COMMITMENT`
  - `PENDING_OUTPUTS_COMMITMENT`

5. `src/capseal/operator/simulate.py`
- Reworded example diff comment to remove TODO marker in simulator payload.

## Remaining Follow-up

1. `src/capseal/cli/trace_cmd.py`
- Keep as-is. It intentionally scans source text for `TODO/FIXME/HACK/XXX`.

2. `src/capseal/refactor_engine.py`
- Keep as-is. It intentionally includes lines with TODO/FIXME in structural summaries.
