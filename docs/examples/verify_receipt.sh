#!/usr/bin/env bash
# Verify a CapSeal receipt and pretty-print the result.
#
# Usage:
#   ./verify_receipt.sh path/to/session.cap
#   ./verify_receipt.sh .capseal/runs/latest.cap

set -euo pipefail

CAP_FILE="${1:-.capseal/runs/latest.cap}"

if [ ! -f "$CAP_FILE" ]; then
    echo "Error: $CAP_FILE not found"
    echo "Usage: $0 <path-to-cap-file>"
    exit 1
fi

echo "Verifying: $CAP_FILE"
echo "---"

# Run capseal verify with JSON output
OUTPUT=$(capseal verify "$CAP_FILE" --json 2>/dev/null || true)

# Extract the JSON line from output
JSON_LINE=$(echo "$OUTPUT" | grep -m1 '^{' || echo '{}')

if command -v jq &>/dev/null; then
    echo "$JSON_LINE" | jq .
else
    echo "$JSON_LINE"
fi

# Check status
STATUS=$(echo "$JSON_LINE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")

echo "---"
if [ "$STATUS" = "VERIFIED" ]; then
    echo "Result: VERIFIED"
    exit 0
else
    echo "Result: $STATUS"
    exit 1
fi
