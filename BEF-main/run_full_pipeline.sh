#!/usr/bin/env bash
set -euo pipefail

# BICEP → ENN-C++ → FusionAlpha pipeline runner
# Produces:
#   - BICEP Parquet: BICEPsrc/BICEPrust/bicep/runs/parity_trajectories.parquet
#   - ENN CSV      : enn-cpp/parity_data.csv
#   - ENN outputs  : enn-cpp/enn_predictions.csv
#   - Fusion graph : fusion_graph.json (at repo root)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SEQUENCES="${SEQUENCES:-200}"
SEQ_LEN="${SEQ_LEN:-15}"
DT="${DT:-0.01}"

echo "=== Pipeline: BICEP → ENN-C++ → FusionAlpha ==="

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1"; exit 1; }
}

require_cmd cargo
require_cmd python3
require_cmd make

# 1) Build + run BICEP parity generator
echo "\n[1/5] Building BICEP examples…"
BICEP_DIR="$ROOT_DIR/BICEPsrc/BICEPrust/bicep"
cd "$BICEP_DIR"
cargo build --release >/dev/null

echo "Running parity_trajectories (sequences=$SEQUENCES, seq_len=$SEQ_LEN, dt=$DT)…"
"$BICEP_DIR/target/release/parity_trajectories" \
  --sequences "$SEQUENCES" \
  --seq-len "$SEQ_LEN" \
  --dt "$DT"

PARQUET_PATH="$BICEP_DIR/runs/parity_trajectories.parquet"
if [[ ! -f "$PARQUET_PATH" ]]; then
  echo "Expected Parquet not found: $PARQUET_PATH"; exit 1;
fi

# 2) Convert Parquet → CSV for ENN
echo "\n[2/5] Converting Parquet → CSV for ENN…"
ENN_DIR="$ROOT_DIR/enn-cpp"
cd "$ENN_DIR"

CSV_OUT="$ENN_DIR/parity_data.csv"
rm -f "$CSV_OUT"

$PYTHON_BIN - "$PARQUET_PATH" "$CSV_OUT" << 'PY'
import sys, os
src, dst = sys.argv[1], sys.argv[2]

def write_with_polars(src, dst):
    import polars as pl
    df = pl.read_parquet(src)
    # For convenience, flatten first state component if present
    if 'state' in df.columns:
        try:
            df = df.with_columns([pl.col('state').list.get(0).alias('state_0')]).drop('state')
        except Exception:
            df = df.rename({'state': 'state_0'})
    df.write_csv(dst)

def write_with_pandas(src, dst):
    import pandas as pd
    try:
        df = pd.read_parquet(src)
    except Exception:
        # Try fallback engine
        df = pd.read_parquet(src, engine='pyarrow')
    if 'state' in df.columns:
        try:
            # Assume list-like state; take first component
            df['state_0'] = df['state'].apply(lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) else v)
            df = df.drop(columns=['state'])
        except Exception:
            df = df.rename(columns={'state': 'state_0'})
    df.to_csv(dst, index=False)

err = None
for writer in (write_with_polars, write_with_pandas):
    try:
        writer(src, dst)
        print(f"Wrote CSV: {dst}")
        err = None
        break
    except Exception as e:
        err = e
        continue

if err is not None:
    print("ERROR: Could not convert Parquet → CSV. Install polars or pandas+pyarrow.", file=sys.stderr)
    raise err
PY

# 3) Build + run ENN trainer app
echo "\n[3/5] Building ENN-C++ …"
make -s apps/bicep_to_enn

echo "Running ENN trainer on parity_data.csv …"
./apps/bicep_to_enn "$CSV_OUT"

ENN_PRED_CSV="$ENN_DIR/enn_predictions.csv"
if [[ ! -f "$ENN_PRED_CSV" ]]; then
  echo "Expected ENN output not found: $ENN_PRED_CSV"; exit 1;
fi

# 4) Build FusionAlpha bindings + generate graph from BICEP trajectories
echo "\n[4/5] Building FusionAlpha (bindings)…"
FUSION_DIR="$ROOT_DIR/FusionAlpha"
cd "$FUSION_DIR"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release -p fusion-bindings >/dev/null || {
  echo "Warning: fusion-bindings build failed. Python demos may not run.";
}
if [[ -f "$FUSION_DIR/target/release/libfusion_alpha.so" ]]; then
  ln -sf libfusion_alpha.so "$FUSION_DIR/target/release/fusion_alpha.so"
fi

echo "Generating fusion_graph.json from BICEP trajectories…"
cd "$ROOT_DIR"
$PYTHON_BIN FusionAlpha/python/bicep_to_edges.py \
  --bicep-csv "$CSV_OUT" \
  --output "$ROOT_DIR/fusion_graph.json" \
  --n-bins 20 --temperature 1.0 \
  --spatial-k 4 \
  --spatial-sigma 0.25 \
  --spatial-weight 0.2 \
  --min-count 1 \
  --spatial-full-grid

if [[ ! -f "$ROOT_DIR/fusion_graph.json" ]]; then
  echo "Expected Fusion graph not found: $ROOT_DIR/fusion_graph.json"; exit 1;
fi

# 5) Validate artifacts (lightweight checks)
echo "\n[5/6] Validating artifacts…"
$PYTHON_BIN scripts/validate_artifacts.py parquet "$PARQUET_PATH"
$PYTHON_BIN scripts/validate_artifacts.py csv "$CSV_OUT"
$PYTHON_BIN scripts/validate_artifacts.py graph "$ROOT_DIR/fusion_graph.json"

# 6) Run FusionAlpha propagation using ENN results
echo "\n[6/6] Propagating committors with FusionAlpha…"
$PYTHON_BIN scripts/run_fusion_propagation.py \
  --graph "$ROOT_DIR/fusion_graph.json" \
  --enn-preds "$ENN_PRED_CSV" \
  --sequence-csv "$CSV_OUT" \
  --out "$ROOT_DIR/fusion_alpha_results.csv"

# Optional EEG pipeline (requires data/bef_runs/* assets)
EEG_NPY_ROOT="$ROOT_DIR/data/bef_runs/npy"
EEG_META_ROOT="$ROOT_DIR/data/bef_runs/meta"
EEG_BEF_ROOT="$ROOT_DIR/data/bef_runs/bef"
EEG_SEQ_OUT="$ROOT_DIR/data/eeg_sequences.csv"
EEG_PRED_OUT="$ROOT_DIR/data/eeg_predictions.csv"
EEG_GRAPH_OUT="$ROOT_DIR/fusion_graph_eeg.json"
EEG_RESULTS_OUT="$ROOT_DIR/fusion_alpha_results_eeg.csv"
EEG_BINS="${EEG_BINS:-20}"

EEG_READY=0
if [[ -d "$EEG_NPY_ROOT" && -d "$EEG_META_ROOT" && -d "$EEG_BEF_ROOT" ]]; then
  echo "\n[EEG] Converting EEG runs → BICEP CSV…"
  $PYTHON_BIN scripts/eeg_to_bicep_csv.py \
    --npy-root "$EEG_NPY_ROOT" \
    --meta-root "$EEG_META_ROOT" \
    --bef-root "$EEG_BEF_ROOT" \
    --seq-out "$EEG_SEQ_OUT" \
    --pred-out "$EEG_PRED_OUT" \
    --window "${EEG_WINDOW:-100}" \
    --hop "${EEG_HOP:-100}" \
    --sequence-steps "${EEG_SEQUENCE_STEPS:-60}" || {
      echo "[EEG] Conversion failed; skipping EEG branch";
      EEG_READY=0
    }

  if [[ -f "$EEG_SEQ_OUT" && -f "$EEG_PRED_OUT" ]]; then
    EEG_READY=1
    echo "\n[EEG] Building FusionAlpha graph from EEG sequences…"
    $PYTHON_BIN FusionAlpha/python/bicep_to_edges.py \
      --bicep-csv "$EEG_SEQ_OUT" \
      --output "$EEG_GRAPH_OUT" \
      --n-bins "$EEG_BINS" \
      --temperature 1.0 \
      --spatial-k 4 \
      --spatial-sigma 0.25 \
      --spatial-weight 0.2 \
      --min-count 1 \
      --spatial-full-grid || EEG_READY=0

    if [[ $EEG_READY -eq 1 && -f "$EEG_GRAPH_OUT" ]]; then
      echo "\n[EEG] Propagating committors over EEG graph…"
      $PYTHON_BIN scripts/run_fusion_propagation.py \
        --graph "$EEG_GRAPH_OUT" \
        --enn-preds "$EEG_PRED_OUT" \
        --sequence-csv "$EEG_SEQ_OUT" \
        --out "$EEG_RESULTS_OUT" || EEG_READY=0
    fi
  fi
else
  echo "\n[EEG] Skipping EEG branch (expected data/bef_runs/{npy,meta,bef})"
fi

echo "\n✅ Pipeline complete"
echo "Artifacts:"
echo "  - Parquet: $PARQUET_PATH"
echo "  - ENN CSV: $CSV_OUT"
echo "  - ENN preds: $ENN_PRED_CSV"
echo "  - Graph: $ROOT_DIR/fusion_graph.json"
echo "  - Propagation: $ROOT_DIR/fusion_alpha_results.csv"
if [[ ${EEG_READY:-0} -eq 1 ]]; then
  echo "  - EEG sequences: $EEG_SEQ_OUT"
  echo "  - EEG preds: $EEG_PRED_OUT"
  echo "  - EEG graph: $EEG_GRAPH_OUT"
  echo "  - EEG propagation: $EEG_RESULTS_OUT"
fi

echo "\nNext:"
echo "  - (Optional) Run FusionAlpha demos: python FusionAlpha/python/fusion_alpha_demo.py"
echo "  - Inspect fusion_alpha_results.csv for propagated committor signals"
if [[ ${EEG_READY:-0} -eq 1 ]]; then
  echo "  - Inspect EEG outputs under data/ and fusion_alpha_results_eeg.csv"
fi
