#!/bin/bash
# Development daemon for CapSeal local execution
# Runs Flask server with SSE support on localhost:5001

set -e

cd "$(dirname "$0")/.."

export FLASK_APP=server.flask_app:create_app
export FLASK_ENV=development
export CAPSEAL_API_KEYS=""  # No auth for local dev
export CORS_ALLOW_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"

echo "Starting CapSeal daemon on http://localhost:5001"
echo "CORS enabled for: $CORS_ALLOW_ORIGINS"
echo ""
echo "Endpoints:"
echo "  GET  /api/runs                    - List runs"
echo "  GET  /api/runs/:id/events/stream  - SSE event stream"
echo "  POST /api/runs/:id/verify         - Verify run (VerifyReport)"
echo "  GET  /api/runs/:id/audit          - Audit run (AuditReport)"
echo "  GET  /api/runs/:id/evidence       - Evidence index"
echo "  POST /api/run                     - Start new run"
echo ""

.venv/bin/python -m flask run --host=0.0.0.0 --port=5001
