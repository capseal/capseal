# Flask Backend API Contract (Verified by Testing)

## Base URL
`http://localhost:5001/api`

---

## Endpoints

### GET /api/sandbox/status
Check if backend is running and sandbox is available.

**Response:**
```json
{
  "status": "SUCCESS",
  "exitCode": 0,
  "result": {
    "system": "Linux",
    "detected_backend": "bwrap",
    "sandbox_available": true,
    "available": {
      "bubblewrap": true,
      "firejail": false,
      "nsjail": false,
      "sandbox_exec": false
    }
  }
}
```

---

### POST /api/run
Execute a run. **SYNCHRONOUS** - returns after run completes.

**Request:**
```json
{
  "traceId": "my-trace",
  "policyId": "default",
  "policy": {
    "policy": {"version": "1.0", "rules": []}
  },
  "backend": "geom",
  "steps": 64,
  "queries": 8,
  "challenges": 2,
  "datasets": [],
  "profile": "default",
  "sandbox": false,
  "sandboxAllowNetwork": false
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "exitCode": 0,
  "requestId": "...",
  "result": {
    "status": "OK",
    "trace_id": "my-trace",        // <-- THIS is the run ID
    "capsule_hash": "...",
    "capsule_path": "/path/to/strategy_capsule.json",
    "proof_path": "/path/to/adapter_proof.json",
    "output_dir": "/path/to/out/my-trace"
  },
  "artifacts": [
    {
      "type": "capsule_path",
      "path": "...",
      "hash": "...",
      "size": 57260
    }
  ]
}
```

**Key:** Use `result.trace_id` as the run identifier for subsequent calls.

---

### GET /api/runs/{traceId}/events
Get events for a completed run.

**Response:**
```json
{
  "runId": "my-trace",
  "events": [
    "{\"type\":\"run_started\",\"seq\":1,...}",
    "{\"type\":\"trace_simulated\",\"seq\":2,...}",
    ...
  ]
}
```

**CRITICAL:** Events are **JSON strings**, must be parsed:
```javascript
const parsedEvents = data.events.map(e => JSON.parse(e))
```

**Event Structure (after parsing):**
```json
{
  "type": "run_started",
  "seq": 1,
  "event_hash": "...",
  "prev_event_hash": "...",
  "run_id": "my-trace",
  "trace_id": "my-trace",
  "ts_ms": 1769383073713,
  "data": { ... },
  "schema": "bef_capsule_stream_v1",
  "v": 1
}
```

**Event Types:**
1. `run_started` - Run began
2. `trace_simulated` - Trace simulation complete
3. `spec_locked` - Spec locked with policy
4. `row_root_finalized` - Row merkle root computed
5. `statement_locked` - Statement ready for proving
6. `proof_artifact` - Proof file generated
7. `capsule_sealed` - Capsule complete
8. `run_completed` - Run finished

---

### POST /api/verify
Verify a capsule.

**Request:**
```json
{
  "capsulePath": "/path/to/strategy_capsule.json",
  "mode": "proof-only"
}
```

---

### POST /api/audit
Audit a capsule's event chain.

**Request:**
```json
{
  "capsulePath": "/path/to/strategy_capsule.json",
  "format": "summary",
  "verifyChain": true
}
```

---

### POST /api/emit
Emit/export a capsule.

**Request:**
```json
{
  "source": "trace-id",
  "outPath": "/tmp/output.cap",
  "profile": "proof-only"
}
```

---

## Important Notes

1. **Runs are SYNCHRONOUS** - the `/run` endpoint blocks until complete
2. **Events are JSON strings** - always parse them
3. **`trace_id` = run ID** - use this consistently
4. **No SSE streaming** - poll `/runs/{id}/events` for updates
5. **Capsule paths are absolute** - server returns full filesystem paths
