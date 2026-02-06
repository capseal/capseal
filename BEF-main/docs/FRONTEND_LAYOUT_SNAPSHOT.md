# Frontend Layout Snapshot (for Diffing)

> Snapshot Date: 2026-01-27
> UI Version: Pre-integration sweep

---

## File Tree

```
ui/src/
├── main.jsx                    # Entry point, React root
├── App.jsx                     # Router, 7 routes
├── style.css                   # Global styles
│
├── Launchpad.jsx               # / - Entry point
├── WorkspaceSelection.jsx      # /workspace - Project picker
├── QuickTrace.jsx              # /trace - Trace viewer
├── RunsPage.jsx                # Legacy (redirects)
├── RunDetailPage.jsx           # Legacy (redirects)
├── RunsList.jsx                # Runs list component
├── RunDetailPanel.jsx          # Detail view (5 tabs)
│
├── components/
│   ├── index.js                # Barrel exports
│   ├── AppShell.jsx            # Layout: rail + list + detail
│   ├── ProjectShell.jsx        # /p/:id/runs - Main workspace
│   ├── Rail.jsx                # Navigation sidebar
│   │
│   ├── RunRow.jsx              # Single run list item
│   ├── RunHeader.jsx           # Run detail header
│   ├── StatusPill.jsx          # Status badge
│   ├── EvidenceStrip.jsx       # L0-L4 verification badges
│   │
│   ├── NewRunDrawer.jsx        # Create run form
│   ├── Drawer.jsx              # Slide-out panel
│   ├── CommandPalette.jsx      # Cmd+K dialog
│   │
│   ├── OverviewCard.jsx        # Info card
│   ├── Timeline.jsx            # Event timeline
│   ├── MonoBlock.jsx           # Code/JSON display
│   ├── CopyField.jsx           # Copyable text
│   │
│   ├── SearchInput.jsx         # Search box
│   ├── FilterChips.jsx         # Filter pills
│   ├── SegmentedControl.jsx    # Tab selector
│   │
│   ├── Toast.jsx               # Notification
│   ├── Skeleton.jsx            # Loading state
│   ├── EmptyState.jsx          # No data state
│   ├── Welcome.jsx             # First-time message
│   │
│   ├── EngineStatus.jsx        # Backend status
│   ├── ProjectsPage.jsx        # Projects list (placeholder)
│   ├── CircuitsPage.jsx        # Circuits list (placeholder)
│   └── WorkspaceHome.jsx       # Workspace landing
│
├── engine/
│   ├── index.js                # getEngine() export
│   ├── base.js                 # ExecutionEngine interface
│   ├── engine.js               # Engine factory
│   ├── types.js                # JSDoc types
│   ├── impl/
│   │   └── flask.js            # FlaskEngine implementation
│   ├── API_CONTRACT.md         # API documentation
│   ├── ARCHITECTURE.txt        # Engine architecture
│   └── README.md               # Usage guide
│
├── state/
│   └── store.js                # Minimal shared state
│
└── contracts/
    ├── contracts.ts            # TypeScript type definitions
    ├── index.ts                # Type exports
    └── examples/
        ├── run_summary.json
        ├── verify_report.json
        ├── audit_report.json
        └── evidence_index.json
```

---

## Route Configuration

```jsx
// App.jsx routes
<Routes>
  <Route path="/" element={<Launchpad />} />
  <Route path="/workspace" element={<WorkspaceSelection />} />
  <Route path="/trace" element={<QuickTrace />} />
  <Route path="/settings" element={<SettingsPlaceholder />} />
  <Route path="/p/:projectId/runs" element={<ProjectShell />} />
  <Route path="/p/:projectId/runs/:runId" element={<ProjectShell />} />
  <Route path="/runs" element={<Navigate to="/p/default/runs" />} />
  <Route path="/runs/:runId" element={<Navigate to="/p/default/runs/:runId" />} />
</Routes>
```

---

## Screen Layouts

### Launchpad (`/`)
```
┌─────────────────────────────────────────────────┐
│  CapSeal                           [Status]     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Quick Actions:                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ New Run │ │ Verify  │ │ Browse  │           │
│  └─────────┘ └─────────┘ └─────────┘           │
│                                                 │
│  Recent Runs:                                   │
│  ┌─────────────────────────────────────────┐   │
│  │ run-abc123  verified   2 min ago        │   │
│  │ run-def456  running    5 min ago        │   │
│  │ run-ghi789  failed     1 hour ago       │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### ProjectShell (`/p/:projectId/runs`)
```
┌──────────────────────────────────────────────────────────────────┐
│ [Rail]  │         Runs List          │      Run Detail          │
│         │                            │                          │
│ ┌─────┐ │ Project: myproject (12)    │ ┌────────────────────┐   │
│ │Runs │ │ ─────────────────────────  │ │ run-abc123         │   │
│ └─────┘ │ ▶ run-abc123  ✓ verified   │ │ Status: VERIFIED   │   │
│ ┌─────┐ │   run-def456  ◐ running    │ └────────────────────┘   │
│ │Sets │ │   run-ghi789  ✗ failed     │                          │
│ └─────┘ │                            │ [Inspect][Verify][Audit] │
│ ┌─────┐ │                            │ [Evidence][Export]       │
│ │Vrfy │ │                            │                          │
│ └─────┘ │                            │ ┌────────────────────┐   │
│ ┌─────┐ │                            │ │ Overview Cards     │   │
│ │Evid │ │                            │ │ - Integrity        │   │
│ └─────┘ │                            │ │ - Verification     │   │
│         │                            │ │ - Execution        │   │
│         │                            │ └────────────────────┘   │
│         │                            │                          │
│         │                            │ [Evidence Strip L0-L4]   │
└─────────┴────────────────────────────┴──────────────────────────┘
   48px          280px (min)               flex-1 (remaining)
```

### RunDetailPanel Tabs

```
[Inspect] [Verify] [Audit] [Evidence] [Export]
    │         │        │        │         │
    │         │        │        │         └── ExportSection
    │         │        │        │             - Download .cap
    │         │        │        │             - Artifact list
    │         │        │        │             - Artifact legend
    │         │        │        │
    │         │        │        └── EvidenceSection
    │         │        │            - Row number input
    │         │        │            - Open row with proof
    │         │        │            - Row values display
    │         │        │            - Merkle proof levels
    │         │        │
    │         │        └── AuditSection
    │         │            - Hash chain status
    │         │            - Event counts grid
    │         │            - Hash chain visualization
    │         │            - Timeline
    │         │
    │         └── VerifySection
    │             - Status banner (VERIFIED/REJECTED/UNVERIFIED)
    │             - Stats (Proof size, Verify time, Backend)
    │             - L0-L4 checklist
    │             - Run Verify button
    │
    └── InspectSection (default)
        - Info grid (Format, ID, Track, Backend, Profile, Schema, Size)
        - Overview cards (Integrity, Verification, Execution)
        - Evidence strip
```

---

## Component Props Summary

### RunDetailPanel
```typescript
interface Props {
  runId: string
}
// Internal state: meta, events, artifacts, status, verification, activeTab, toast
```

### ProjectShell
```typescript
// Uses useParams(): { projectId, runId }
// Internal state: selectedRunId, runs, loading
```

### FlaskEngine Methods
```typescript
class FlaskEngine {
  getHealth(): Promise<HealthStatus>
  startRun(req: RunStartRequest): Promise<RunStartResponse>
  streamEvents(runId, onEvent, onError): () => void
  getRun(runId): Promise<RunRecord>               // ← ENDPOINT MISSING
  listRuns(query?): Promise<RunSummary[]>
  verify(req): Promise<VerifyReport>
  audit(req): Promise<AuditReport>
  evidence(req): Promise<EvidenceIndex>
  exportCapsule(req): Promise<ExportResult>
  getArtifactUrl(runId, name): string             // ← ENDPOINT MISSING
  getCapsuleUrl(runId): string                    // ← ENDPOINT MISSING
  getRowUrl(runId, rowNumber): string             // ← ENDPOINT MISSING
}
```

---

## CSS Class Structure

### Layout Classes
```
.app-shell          - Main 3-column layout
.rail               - Left navigation (48px)
.list-panel         - Runs list (280px min)
.detail-panel       - Run detail (flex-1)
```

### Component Classes
```
.run-row            - List item
.run-row.selected   - Selected state
.detail-header      - Detail page header
.overview-grid      - 3-column card grid
.inspect-info-grid  - Info key-value grid
.verify-section     - Verify tab content
.audit-section      - Audit tab content
.evidence-section   - Evidence tab content
.export-section     - Export tab content
```

### Status Classes
```
.status-verified    - Green
.status-failed      - Red
.status-running     - Yellow/amber
.status-unverified  - Gray
.status-pending     - Gray
.status-unknown     - Gray
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                          Frontend                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. User navigates to /p/myproject/runs                          │
│     └── ProjectShell mounts                                       │
│         └── fetchRuns() → engine.listRuns({ project_id })         │
│             └── GET /api/runs?project_id=myproject                │
│                                                                   │
│  2. User clicks run-abc123                                        │
│     └── handleSelectRun('run-abc123')                             │
│         └── navigate('/p/myproject/runs/run-abc123')              │
│             └── RunDetailPanel mounts                             │
│                 └── fetchRunDetail()                              │
│                     └── engine.getRun('run-abc123')  ← FAILS      │
│                         └── GET /api/runs/run-abc123 ← MISSING    │
│                                                                   │
│  3. SSE fallback                                                  │
│     └── engine.streamEvents('run-abc123', onEvent)                │
│         └── EventSource /api/runs/run-abc123/events/stream        │
│             └── Events populate, but meta/artifacts missing       │
│                                                                   │
│  4. User clicks "Run Verify" button                               │
│     └── handleVerify()                                            │
│         └── engine.verify({ run_id: 'run-abc123' })               │
│             └── POST /api/runs/run-abc123/verify                  │
│                 └── Returns VerifyReport ← WORKS                  │
│                                                                   │
│  5. User clicks "Download .cap"                                   │
│     └── engine.getCapsuleUrl('run-abc123')                        │
│         └── GET /api/runs/run-abc123/capsule ← FAILS (404)        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Version Tracking

| Component | Version | Last Modified |
|-----------|---------|---------------|
| App.jsx | - | 2026-01-27 |
| RunDetailPanel.jsx | - | 2026-01-27 |
| flask.js (engine) | - | 2026-01-27 |
| contracts.ts | v1 | 2026-01-27 |

---

## Notes for Diffing

When comparing future versions:
1. Check route additions/removals in `App.jsx`
2. Check new components in `components/`
3. Check engine method additions in `flask.js`
4. Check contract changes in `contracts.ts`
5. Check endpoint additions in `sse_routes.py` and `routes.py`
