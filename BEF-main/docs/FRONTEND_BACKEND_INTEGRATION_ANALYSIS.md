# Frontend-Backend Integration Analysis

> Generated: 2026-01-27
> Purpose: Document current frontend layout and identify backend integration gaps

---

## 1. Current Frontend Layout Structure

### Route Hierarchy

```
/                        → Launchpad (entry point, no rail)
/workspace               → WorkspaceSelection (project chooser, no rail)
/trace                   → QuickTrace (single-purpose trace view, no rail)
/settings                → SettingsPlaceholder (stub)
/p/:projectId/runs       → ProjectShell (rail visible, runs list + detail)
/p/:projectId/runs/:runId → ProjectShell with run detail
/runs                    → Redirect → /p/default/runs
/runs/:runId             → Redirect → /p/default/runs/:runId
```

### Component Hierarchy

```
App.jsx
├── CommandPalette (global Cmd+K)
├── Launchpad.jsx           → Jobs-like entry, quick actions
├── WorkspaceSelection.jsx  → Project picker with mode=recent|new|open
├── QuickTrace.jsx          → Single trace inspection
├── ProjectShell.jsx        → Main workspace view
│   ├── AppShell            → Layout wrapper
│   ├── Rail                → Navigation sidebar (Runs/Settings/Verify/Evidence)
│   ├── ProjectRunsList     → Filtered runs list
│   └── RunDetailPanel.jsx  → Detail view with tabs
│       ├── InspectSection  → capseal inspect mapping
│       ├── VerifySection   → capseal verify mapping
│       ├── AuditSection    → capseal audit mapping
│       ├── EvidenceSection → capseal row mapping
│       └── ExportSection   → capseal emit mapping
```

### Component Files (ui/src/)

| File | Purpose | Backend Endpoints Used |
|------|---------|------------------------|
| `App.jsx` | Router, route definitions | None |
| `Launchpad.jsx` | Entry point, quick actions | `/api/health`, `/api/runs` |
| `WorkspaceSelection.jsx` | Project picker | `/api/runs` (grouped by track_id) |
| `QuickTrace.jsx` | Trace viewer | `/api/runs/:id/events` |
| `RunsPage.jsx` | Legacy runs list | `/api/runs` |
| `RunDetailPage.jsx` | Legacy detail view | `/api/runs/:id` |
| `RunDetailPanel.jsx` | Main detail component | Multiple (see below) |
| `RunsList.jsx` | Runs list component | `/api/runs` |
| `components/ProjectShell.jsx` | Project workspace | `/api/runs`, `/api/runs/:id` |
| `components/Rail.jsx` | Navigation rail | None (UI only) |
| `components/NewRunDrawer.jsx` | New run form | `/api/run` (POST) |
| `components/EvidenceStrip.jsx` | Verification badges | Uses verification prop |

### UI State Management

- **No Redux/Zustand**: Uses React hooks (`useState`, `useCallback`, `useEffect`)
- **State file**: `ui/src/state/store.js` - minimal shared state
- **Engine pattern**: `ui/src/engine/` abstracts backend calls

---

## 2. Backend API Endpoints

### routes.py (Primary API Blueprint)

| Endpoint | Method | Request Model | Status |
|----------|--------|---------------|--------|
| `/api/fetch` | POST | `FetchRequest` | Active |
| `/api/run` | POST | `RunRequest` | Active |
| `/api/emit` | POST | `EmitRequest` | Active |
| `/api/verify` | POST | `VerifyRequest` | Active |
| `/api/replay` | POST | `ReplayRequest` | Active |
| `/api/audit` | POST | `AuditRequest` | Active |
| `/api/runs/<run_id>/events` | GET | - | Active |
| `/api/sandbox/status` | GET | - | Active |
| `/api/sandbox/test` | POST | `SandboxTestRequest` | Active |
| `/api/jobs/<job_id>` | GET | - | Active |
| `/api/jobs` | GET | - | Active |
| `/api/jobs/<job_id>/cancel` | POST | - | Active |

### sse_routes.py (Contract-Compliant Endpoints)

| Endpoint | Method | Contract Type | Status |
|----------|--------|---------------|--------|
| `/api/runs/<run_id>/events/stream` | GET | SSE stream | Active |
| `/api/runs/<run_id>/verify` | POST | `VerifyReport` | Active |
| `/api/runs/<run_id>/audit` | GET | `AuditReport` | Active |
| `/api/runs/<run_id>/evidence` | GET | `EvidenceIndex` | Active |
| `/api/runs` | GET | `RunSummary[]` | Active |
| `/api/runs/<run_id>/budget` | GET | `BudgetSummary` | Active |

---

## 3. Frontend-Backend Integration Gaps

### CRITICAL: Endpoint Mismatch Issues

#### 3.1 `/api/runs/:id` - GET Single Run (MISSING)

**Frontend expects (flask.js:260-269):**
```javascript
async getRun(runId) {
  const resp = await fetch(`${this._baseUrl}/api/runs/${runId}`)
  // ...expects run data with events, artifacts, verification
}
```

**Backend provides:** NONE - No GET endpoint for single run by ID

**Impact:** `ProjectShell.jsx`, `RunDetailPanel.jsx` - Cannot load run detail
**Fix needed:** Add `GET /api/runs/<run_id>` to sse_routes.py

---

#### 3.2 `/api/runs/:id/artifacts/:name` - Artifact Download (MISSING)

**Frontend expects (flask.js:117-119):**
```javascript
getArtifactUrl(runId, artifactName) {
  return `${this._baseUrl}/api/runs/${runId}/artifacts/${artifactName}`
}
```

**Backend provides:** NONE

**Impact:** `ExportSection` in RunDetailPanel - Download buttons broken
**Fix needed:** Add artifact download endpoint

---

#### 3.3 `/api/runs/:id/capsule` - Capsule Download (MISSING)

**Frontend expects (flask.js:125-127):**
```javascript
getCapsuleUrl(runId) {
  return `${this._baseUrl}/api/runs/${runId}/capsule`
}
```

**Backend provides:** NONE

**Impact:** "Download .cap" button in ExportSection broken
**Fix needed:** Add capsule download endpoint

---

#### 3.4 `/api/runs/:id/rows/:number` - Row Opening (MISSING)

**Frontend expects (flask.js:133-136):**
```javascript
getRowUrl(runId, rowNumber) {
  return `${this._baseUrl}/api/runs/${runId}/rows/${rowNumber}`
}
```

**Backend provides:** NONE

**Impact:** EvidenceSection "Open Row with Proof" feature broken
**Fix needed:** Add row opening endpoint (maps to `capseal row` CLI)

---

#### 3.5 `/api/runs/:id/export` - Export (MISSING)

**Frontend expects (flask.js:349-361):**
```javascript
async exportCapsule(req) {
  const resp = await fetch(`${this._baseUrl}/api/runs/${req.run_id}/export`, {
    method: 'POST',
    // ...
  })
}
```

**Backend provides:** `/api/emit` with different request format

**Impact:** Export functionality mismatch
**Fix needed:** Add wrapper endpoint or update frontend to use `/api/emit`

---

#### 3.6 `/api/health` - Health Check (MISSING)

**Frontend expects (flask.js:141-174):**
```javascript
async getHealth() {
  const resp = await fetch(`${this._baseUrl}/api/health`)
  // expects: { ok, version, capabilities: { run, verify, audit, evidence, export, sse } }
}
```

**Backend provides:** NONE

**Impact:** Engine status check fails, capabilities detection broken
**Fix needed:** Add health endpoint

---

### Contract Format Issues

#### 3.7 Verify Response Format Mismatch

**Contract expects (contracts.ts:70-85):**
```typescript
interface VerifyReport {
  run_id: string;
  status: 'verified' | 'rejected';
  exit_code: number;
  layers: VerificationLayers;  // Object with l0_hash, l1_commitment, etc.
  errors: VerifyError[];
  timings: VerifyTimings;
  // ...
}
```

**Backend provides (sse_routes.py:209-243):** Matches contract

**Frontend normalizer (flask.js:505-528):** Handles both formats

**Status:** OK - Working as designed

---

#### 3.8 Events Response - JSON String Issue

**Backend provides (routes.py:125-129):**
```python
events = get_run_events(run_id, str(event_root))
return jsonify({"runId": run_id, "events": events}), 200
# events are JSON strings, not objects
```

**Frontend must parse (API_CONTRACT.md:100-103):**
```javascript
const parsedEvents = data.events.map(e => JSON.parse(e))
```

**Impact:** Frontend may not be double-parsing
**Status:** Document notes this, verify frontend handles it

---

### State/Data Flow Issues

#### 3.9 RunDetailPanel Data Loading

**Current flow:**
1. `fetchRunDetail()` calls `engine.getRun(runId)` → **FAILS** (endpoint missing)
2. Falls back to `streamEvents()` for SSE
3. `events`, `artifacts`, `verification` not populated from missing endpoint

**Expected:**
1. GET `/api/runs/:id` returns full run data
2. Includes: `{ run_id, events, artifacts, verification, ... }`

---

#### 3.10 Verification Status Not Persisted

**Issue:** `api_list_runs()` always returns `"verification_status": "unverified"`
```python
runs.append({
    # ...
    "verification_status": "unverified",  # Would need to check
})
```

**Impact:** RunsList always shows unverified status
**Fix needed:** Check for verification results in run directory

---

### Missing Backend Features

#### 3.11 Project Filtering Not Implemented

**Frontend expects (flask.js:276-280):**
```javascript
if (query?.project_id) params.set('project_id', query.project_id)
```

**Backend (sse_routes.py:409-452):** Ignores `project_id` query param

**Impact:** ProjectShell filtering by track_id doesn't work
**Fix needed:** Filter runs by `project_id`/`track_id` in backend

---

## 4. Summary: Required Backend Changes

### High Priority (Blocking Features)

| # | Endpoint | Method | Action |
|---|----------|--------|--------|
| 1 | `/api/runs/<run_id>` | GET | Add - returns full run data |
| 2 | `/api/health` | GET | Add - capabilities check |
| 3 | `/api/runs/<run_id>/artifacts/<name>` | GET | Add - artifact download |
| 4 | `/api/runs/<run_id>/capsule` | GET | Add - .cap download |
| 5 | `/api/runs/<run_id>/rows/<number>` | GET | Add - row with proof |

### Medium Priority (Degraded Experience)

| # | Issue | Action |
|---|-------|--------|
| 6 | `verification_status` always unverified | Check for verify results |
| 7 | `project_id` filtering ignored | Implement filter |
| 8 | Export uses different endpoint | Add `/runs/:id/export` wrapper |

### Low Priority (Polish)

| # | Issue | Action |
|---|-------|--------|
| 9 | Events JSON string parsing | Verify frontend handles |
| 10 | SSE Last-Event-ID reconnection | Test reconnection logic |

---

## 5. Frontend Component Dependency Map

```
RunDetailPanel
├── Requires: getRun(id) ← MISSING ENDPOINT
├── Requires: streamEvents(id) ← Works (SSE)
├── InspectSection
│   ├── Uses: meta, summary, verification (from getRun)
│   └── Uses: EvidenceStrip (verification prop)
├── VerifySection
│   ├── Uses: verification (from getRun)
│   └── Calls: verify({ run_id }) ← Works
├── AuditSection
│   └── Uses: events (from getRun or SSE)
├── EvidenceSection
│   └── Calls: getRowUrl(id, n) ← MISSING ENDPOINT
└── ExportSection
    ├── Calls: getArtifactUrl(id, name) ← MISSING ENDPOINT
    └── Calls: getCapsuleUrl(id) ← MISSING ENDPOINT

ProjectShell
├── Requires: listRuns({ project_id }) ← Works (filtering ignored)
└── Requires: getRun(id) ← MISSING ENDPOINT

Launchpad
└── Calls: getHealth() ← MISSING ENDPOINT
```

---

## 6. Contracts Reference

Source of truth: `ui/src/contracts/contracts.ts`

### Key Types Frontend Expects

- `RunSummary` - Run list item
- `VerifyReport` - Verification result with L0-L4 layers
- `AuditReport` - Event chain audit
- `EvidenceIndex` - Openable rows/artifacts
- `BudgetSummary` - Token governance

All backends must conform to these shapes.
