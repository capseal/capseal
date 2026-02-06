#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "[WARN] No virtualenv found at $VENV_ACTIVATE; falling back to system python." >&2
fi

PROJECT_DIR=${PROJECT_DIR:-"$HOME/projects/market-dashboard"}
# Default runs dir derives from project basename to avoid cross-project confusion
_proj_name=$(basename "$PROJECT_DIR" | tr '[:space:]' '_' )
RUNS_DIR=${RUNS_DIR:-"$HOME/capseal_runs/${_proj_name}"}
POLICY_ID=${POLICY_ID:-review_v1}
BACKEND=${BACKEND:-semgrep}
AGENTS=${AGENTS:-4}
FAIL_SEVERITY=${FAIL_SEVERITY:-warning}
LLM_PROVIDER=${LLM_PROVIDER:-openai}
LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
LLM_MAX_FINDINGS=${LLM_MAX_FINDINGS:-20}
LLM_MIN_SEVERITY=${LLM_MIN_SEVERITY:-warning}

mkdir -p "$RUNS_DIR"
DATE_SUFFIX=$(date +%s)
BASE_RUN="$RUNS_DIR/base_${DATE_SUFFIX}"
HEAD_RUN="$RUNS_DIR/head_${DATE_SUFFIX}"

py() {
  python -m bef_zk.capsule.cli "$@"
}

echo "=== CapSeal full pipeline ==="
echo "Project : $PROJECT_DIR"
echo "Runs dir: $RUNS_DIR"
echo "Policy  : $POLICY_ID"
echo "Backend : $BACKEND"
echo

# Base pipeline (trace->review->dag->verify)
echo "[BASE] pipeline start -> $BASE_RUN"
py pipeline \
  --project-dir "$PROJECT_DIR" \
  --run "$BASE_RUN" \
  --policy "$POLICY_ID" \
  --backend "$BACKEND" \
  --agents "$AGENTS" \
  --fail-on "$FAIL_SEVERITY"

echo
# Head pipeline with incremental trace + diff gate
echo "[HEAD] pipeline start -> $HEAD_RUN"
py pipeline \
  --project-dir "$PROJECT_DIR" \
  --run "$HEAD_RUN" \
  --policy "$POLICY_ID" \
  --backend "$BACKEND" \
  --agents "$AGENTS" \
  --fail-on "$FAIL_SEVERITY" \
  --diff-base "$BASE_RUN" \
  --incremental-from "$BASE_RUN"

echo
# LLM explanation on verified findings
echo "[EXPLAIN] $LLM_PROVIDER::$LLM_MODEL"
py explain-llm \
  --run "$HEAD_RUN" \
  --llm-provider "$LLM_PROVIDER" \
  --llm-model "$LLM_MODEL" \
  --max-findings "$LLM_MAX_FINDINGS" \
  --min-severity "$LLM_MIN_SEVERITY" \
  --format markdown

summary_dir=$(ls -1d "$HEAD_RUN"/reviews/explain_llm/* 2>/dev/null | tail -n1 || true)
if [[ -n "$summary_dir" ]]; then
  echo "Explain summary: $summary_dir/summary.json"
  if [[ -f "$summary_dir/receipt.json" ]]; then
    echo "Explain receipt: $summary_dir/receipt.json"
  fi
  if [[ -f "$summary_dir/report.md" ]]; then
    echo "Explain report:  $summary_dir/report.md"
  fi
fi

echo
echo "BASE run: $BASE_RUN"
echo "HEAD run: $HEAD_RUN"
if [[ -f "$HEAD_RUN/diff/receipt.json" ]]; then
  echo "Diff receipt: $HEAD_RUN/diff/receipt.json"
fi
if [[ -f "$HEAD_RUN/workflow/rollup.json" ]]; then
  echo "Rollup: $HEAD_RUN/workflow/rollup.json"
fi

echo "Done."
