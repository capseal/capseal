#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/hssa_stc"

echo "[dist] Preparing minimal HSSA/STC bundle at $DIST_DIR"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# 1) GPU accumulator (Python + CUDA)
echo "[dist] Copy gpu_accumulator/"
cp -a "$ROOT_DIR/gpu_accumulator" "$DIST_DIR/"

# 2) Rust verifier workspace (source only, no build artifacts)
echo "[dist] Copy bicep workspace (source only)"
mkdir -p "$DIST_DIR/BICEPsrc/BICEPrust"
cp -a "$ROOT_DIR/BICEPsrc/BICEPrust/bicep" "$DIST_DIR/BICEPsrc/BICEPrust/"
find "$DIST_DIR/BICEPsrc/BICEPrust/bicep" -type d -name target -prune -exec rm -rf {} + || true

# 3) Benches: HSSA + KZG (sources only)
echo "[dist] Copy bench harnesses"
mkdir -p "$DIST_DIR/bench"
cp -a "$ROOT_DIR/bench/bench_hssa.py" "$DIST_DIR/bench/"
cp -a "$ROOT_DIR/bench/compare_hssa_kzg.py" "$DIST_DIR/bench/"
cp -a "$ROOT_DIR/bench/bench_kzg.rs" "$DIST_DIR/bench/" 2>/dev/null || true
if [ -d "$ROOT_DIR/bench/kzg-bench" ]; then
  cp -a "$ROOT_DIR/bench/kzg-bench" "$DIST_DIR/bench/"
  rm -rf "$DIST_DIR/bench/kzg-bench/target" || true
fi

# 4) Docs relevant to STC/HSSA
echo "[dist] Copy docs"
mkdir -p "$DIST_DIR/docs"
for f in \
  BEF-Stream-Accum-summary.md \
  streaming_trace_commitment_formal.md \
  ivc_state_R.md \
  bef_trace_commitment.md \
  hssa_da_protocol.md \
  hssa_vs_kzg_bench.md \
  BEF_CURRENT_STATUS.md \
  BACKEND_READINESS_GAPS.md \
  da_profile_router.md \
  stc_da_profiles.md \
  stc_parameter_guidance.md \
  stc_vm_mapping.md \
  stc_backend_architecture.md \
  stc_pc_backend.md \
; do
  cp -a "$ROOT_DIR/docs/$f" "$DIST_DIR/docs/" 2>/dev/null || true
done

mkdir -p "$DIST_DIR/scripts"
for script in stc_aok.py stc_param_table.py build_vm_trace.py stc_da_sample.py stc_da_swap.py stc_pc_cli.py; do
  cp -a "$ROOT_DIR/scripts/$script" "$DIST_DIR/scripts/" 2>/dev/null || true
done

# 5) Samples (trace + sketch) and benchmark CSVs (small)
echo "[dist] Copy sample trace/sketch and CSVs"
mkdir -p "$DIST_DIR/code/traces" "$DIST_DIR/code/sketches"
cp -a "$ROOT_DIR/code/traces/trace_demo.json" "$DIST_DIR/code/traces/" 2>/dev/null || true
cp -a "$ROOT_DIR/code/sketches/trace_demo_sketch.json" "$DIST_DIR/code/sketches/" 2>/dev/null || true
cp -a "$ROOT_DIR/code/traces/vm_demo.json" "$DIST_DIR/code/traces/" 2>/dev/null || true
cp -a "$ROOT_DIR/code/sketches/vm_demo_sketch.json" "$DIST_DIR/code/sketches/" 2>/dev/null || true

mkdir -p "$DIST_DIR/da_profiles"
cp -a "$ROOT_DIR/da_profiles"/*.json "$DIST_DIR/da_profiles/" 2>/dev/null || true

for csv in \
  "$ROOT_DIR/bench/hssa_results.csv" \
  "$ROOT_DIR/bench/kzg_results.csv" \
  "$ROOT_DIR/bench/combined_results.csv" \
  "$ROOT_DIR/combined_results_colab.csv" \
  "$ROOT_DIR/hssa_results_colab.csv" \
  "$ROOT_DIR/kzg_results.csv" \
; do
  if [ -f "$csv" ]; then
    cp -a "$csv" "$DIST_DIR/bench/" 2>/dev/null || true
  fi
done

# 6) Minimal README for the bundle
cat > "$DIST_DIR/README.md" << 'EOF'
# HSSA/STC Minimal Bundle

This distribution contains the Streaming Trace Commitment (HSSA/STC) components:

- gpu_accumulator/ — CUDA kernels + Python API (`StreamingAccumulatorCUDA`)
- BICEPsrc/BICEPrust/bicep/crates/bicep-crypto — Rust fast verifier + CLI
- bench/ — HSSA and KZG bench harnesses (sources only)
- docs/ — Formal spec, IVC step relation, DA protocol, and backend gaps
- code/traces + code/sketches — sample trace and sketch JSON

Build the verifier CLI:

```
cd BICEPsrc/BICEPrust/bicep
cargo build --release -p bicep-crypto --bin bef_verify_fast_cli
```

Run HSSA bench (requires CUDA):

```
python bench/bench_hssa.py --Ns 1048576 --challenges 4 --chunk-lens 8192 --repeats 1
```

Convert a trace to a sketch:

```
python gpu_accumulator/sketch_trace.py code/traces/trace_demo.json code/sketches/out.json
```
EOF

echo "[dist] Done. Size:"
du -sh "$DIST_DIR" || true
