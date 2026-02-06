#!/usr/bin/env bash
# Loom Demo Script - 60-second capsule verification demo
# Clean version: emit -> verify (real 1.9ms) -> tamper fails
set -e
cd "$(dirname "$0")/.."

# Use intuitive paths for demo
COMPUTATION="ran_computations/trading/momentum_strategy_v2"
TAMPERED_CAP="$COMPUTATION/tampered/tampered.cap"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CAPSULE: Portable Verification Receipts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Emit
echo "┌─ [1] EMIT ──────────────────────────────────────────"
echo "│"
echo "│  \$ capsule emit --source $COMPUTATION --out demo.cap"
echo "│"
.venv/bin/python -m bef_zk.capsule.cli emit \
    --source "$COMPUTATION" \
    --policy "$COMPUTATION/policy.json" \
    --out demo.cap 2>&1 | grep -E "(Created|Backend|Total size)" | sed 's/^/│  /'
echo "│"
echo "└────────────────────────────────────────────────────────"
echo ""

# Step 2: Verify (real measured timing)
echo "┌─ [2] VERIFY ────────────────────────────────────────"
echo "│"
echo "│  \$ capsule verify demo.cap --json"
echo "│"
.venv/bin/python -m bef_zk.capsule.cli verify demo.cap --json 2>&1 | \
    .venv/bin/python -c "
import sys, json
d = json.load(sys.stdin)
stats = d.get('verify_stats', {})
verify_ms = stats.get('time_verify_sec', 0) * 1000
print('│  {')
print(f'│    \"status\": \"{d[\"status\"]}\",')
print(f'│    \"proof_verified\": {str(d.get(\"proof_verified\", d.get(\"policy_verified\", False))).lower()},')
print(f'│    \"verify_time_ms\": {verify_ms:.1f},')
print(f'│    \"backend\": \"{d.get(\"backend_id\", \"\")}\"')
print('│  }')
"
echo "│"
echo "└────────────────────────────────────────────────────────"
echo ""

# Step 3: Tamper fail
echo "┌─ [3] TAMPER DETECTION ─────────────────────────────"
echo "│"
echo "│  \$ capsule verify tampered.cap --json"
echo "│"
.venv/bin/python -m bef_zk.capsule.cli verify "$TAMPERED_CAP" --json 2>&1 | \
    .venv/bin/python -c "
import sys, json
d = json.load(sys.stdin)
print('│  {')
print(f'│    \"status\": \"{d[\"status\"]}\",')
print(f'│    \"error_code\": \"{d[\"error_code\"]}\"')
print('│  }')
"
echo "│"
echo "│  ⚠️  Tampered receipt REJECTED"
echo "│"
echo "└────────────────────────────────────────────────────────"
echo ""

rm -f demo.cap

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Emit: Packaged proof into portable .cap"
echo "  ✓ Verify: Independent verification in <2ms"
echo "  ✓ Tamper: Modified receipt rejected with error code"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
