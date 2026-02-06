#!/bin/bash
# CapSeal E2E Smoke Test
# Exit on first failure
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAPSEAL="$PROJECT_ROOT/capseal"
OUT_DIR="$PROJECT_ROOT/out/e2e_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "CapSeal E2E Smoke Test"
echo "=========================================="
echo "Capseal: $CAPSEAL"
echo "Output:  $OUT_DIR"
echo ""

# 1. Basic demo (installation check)
echo "[1/8] Running demo..."
"$CAPSEAL" demo --verbose > "$OUT_DIR/demo.log" 2>&1
echo "  ✓ Demo passed"

# 2. CRITICAL: Demo → Verify ACCEPT (proves E2E works)
echo "[2/8] Verifying demo receipt (ACCEPT case)..."
"$CAPSEAL" demo -o "$OUT_DIR/demo_receipt.json" > "$OUT_DIR/demo_gen.log" 2>&1
"$CAPSEAL" verify "$OUT_DIR/demo_receipt.json" > "$OUT_DIR/demo_verify.log" 2>&1
if grep -q "VERIFIED" "$OUT_DIR/demo_verify.log"; then
    echo "  ✓ Demo receipt verified (ACCEPT)"
else
    echo "  ✗ FAIL: Demo receipt verification should ACCEPT!"
    cat "$OUT_DIR/demo_verify.log"
    exit 1
fi

# 3. Trace a test project and verify-trace
echo "[3/8] Tracing test project..."
TEST_PROJECT="$OUT_DIR/test_project"
mkdir -p "$TEST_PROJECT"
cat > "$TEST_PROJECT/main.py" << 'EOF'
def calculate(x, y):
    return x + y

if __name__ == "__main__":
    print(calculate(10, 20))
EOF

TRACE_OUT="$OUT_DIR/trace_run"
"$CAPSEAL" trace "$TEST_PROJECT" --out "$TRACE_OUT" > "$OUT_DIR/trace.log" 2>&1
echo "  ✓ Trace completed: $TRACE_OUT"

# Verify the trace against the project
"$CAPSEAL" verify-trace "$TRACE_OUT" --project-dir "$TEST_PROJECT" > "$OUT_DIR/verify_trace.log" 2>&1
if grep -q "PASS" "$OUT_DIR/verify_trace.log"; then
    echo "  ✓ Verify-trace passed"
else
    echo "  ✗ FAIL: verify-trace should pass!"
    cat "$OUT_DIR/verify_trace.log"
    exit 1
fi

# 4. Verify + Inspect golden fixture
echo "[4/8] Verifying golden fixture (production ACCEPT case)..."
GOLDEN="$PROJECT_ROOT/fixtures/golden_run_latest/capsule/strategy_capsule.json"
if [ -f "$GOLDEN" ]; then
    "$CAPSEAL" verify "$GOLDEN" > "$OUT_DIR/golden_verify.log" 2>&1
    if grep -q "VERIFIED" "$OUT_DIR/golden_verify.log"; then
        echo "  ✓ Golden fixture verified (ACCEPT)"
    else
        echo "  ✗ FAIL: Golden fixture verification should ACCEPT!"
        cat "$OUT_DIR/golden_verify.log"
        exit 1
    fi
    "$CAPSEAL" inspect "$GOLDEN" > "$OUT_DIR/inspect.log" 2>&1
    echo "  ✓ Inspect passed"
else
    echo "  ⚠ Golden fixture not found (skipped)"
fi

# 5. Audit golden fixture
echo "[5/8] Auditing golden fixture..."
if [ -f "$GOLDEN" ]; then
    "$CAPSEAL" audit "$GOLDEN" > "$OUT_DIR/audit.log" 2>&1
    echo "  ✓ Audit passed (hash chain valid)"
else
    echo "  ⚠ Golden fixture not found (skipped)"
fi

# 6. Row opening test
echo "[6/8] Testing row opening..."
if [ -f "$GOLDEN" ]; then
    "$CAPSEAL" row "$GOLDEN" --row 0 > "$OUT_DIR/row.log" 2>&1
    echo "  ✓ Row 0 opened with membership proof"
else
    echo "  ⚠ Golden fixture not found (skipped)"
fi

# 7. Tamper detection test (REJECT case)
echo "[7/8] Testing tamper detection (REJECT case)..."
if [ -f "$GOLDEN" ]; then
    TAMPERED="$OUT_DIR/tampered_capsule.json"
    python3 << EOF
import json
with open("$GOLDEN", "r") as f:
    data = json.load(f)
# Tamper with statement_hash
original = data["header"]["statement_hash"]
data["header"]["statement_hash"] = "0000" + original[4:]
with open("$TAMPERED", "w") as f:
    json.dump(data, f, indent=2)
print(f"Tampered: {original[:20]}... -> {data['header']['statement_hash'][:20]}...")
EOF

    # Verify should REJECT tampered capsule
    set +e
    "$CAPSEAL" verify "$TAMPERED" > "$OUT_DIR/tamper_verify.log" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ✓ Tamper detected (exit code $EXIT_CODE)"
        grep -o "E[0-9]\{3\}_[A-Z_]*" "$OUT_DIR/tamper_verify.log" | head -1 || true
    else
        echo "  ✗ FAIL: Tamper not detected!"
        exit 1
    fi
else
    echo "  ⚠ Golden fixture not found (skipped)"
fi

echo ""
echo "=========================================="
echo "All E2E Smoke Tests Passed"
echo "=========================================="
echo "Logs: $OUT_DIR/"
ls -la "$OUT_DIR"/*.log 2>/dev/null || true
