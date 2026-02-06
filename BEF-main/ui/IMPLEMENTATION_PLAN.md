# Implementation Plan: Fix Frontend-Backend Integration + UX Tightness

## The Real Problems

### Problem 1: Backend Integration is Fundamentally Broken
The Engine Adapter was built on assumptions, not facts. We don't actually know:
- What endpoints the Flask backend exposes and what they return
- Whether the backend is even running
- What the actual request/response shapes are

### Problem 2: UX is Loose (per Design_notes.txt & Solution_notes.txt)
- Empty states are passive ("Select a run") instead of action-driven ("Create first run")
- "Continue Project" dead-ends into empty runs list
- No clear state machine (what should I do next?)
- System ambiguity (is engine connected? local? remote?)

---

## Phase 1: Ground Truth Discovery (MUST DO FIRST)

**Goal**: Understand what actually works before writing more code.

### 1.1 Test Flask Backend Directly
```bash
# Is it running?
curl http://localhost:5001/api/sandbox/status

# What does /run actually return?
curl -X POST http://localhost:5001/api/run \
  -H "Content-Type: application/json" \
  -d '{"traceId":"test", "policyId":"default", "policy":{"policy":{}}, "datasets":[]}'

# What does /runs/{id}/events return?
curl http://localhost:5001/api/runs/test/events
```

### 1.2 Document Actual API Contract
Create `ui/src/engine/API_CONTRACT.md` with:
- Every endpoint that EXISTS and WORKS
- Actual request schemas (from Flask models.py)
- Actual response shapes (from testing)

### 1.3 Fix Engine Adapter Based on Reality
Rewrite flask.js to match the ACTUAL backend, not assumptions.

---

## Phase 2: Make QuickTrace Actually Work

**Goal**: One complete flow working end-to-end.

### 2.1 Simplify QuickTrace
- Remove complex features temporarily
- Just: input path → call backend → show response
- Add proper error display

### 2.2 Add Visible Engine Status
Per Design_notes.txt section 4:
- "Engine: Local (Connected)" / "Engine: Local (Not running)"
- If not running: disable "Run & Seal" button
- Show actual connection errors

### 2.3 Live Timeline During Runs
Per Design_notes.txt section 6:
- When "Run & Seal" clicked, transition to "run in progress" view
- Stream events live
- Show L0-L4 ladder filling in

---

## Phase 3: Fix UX Tightness (per Solution_notes.txt)

### 3.1 Action-Driven Empty States
Change Runs empty state from passive to active:
- "Select a run" → "Create first run" (primary button)
- Add "Quick Trace" secondary button
- Add "Import capsule (.cap)" tertiary

### 3.2 Continue Project Flow
When clicking "Continue Project":
- If project has runs → go to last run
- If no runs → go to empty state with "Create first run"
- Never dead-end

### 3.3 Three Explicit Modes
Lock app into 3 modes with clear visual distinction:
- **Mode A: Launchpad** (`/`) - No context, just actions
- **Mode B: Workspace** (`/workspace`) - Project selection, no rail
- **Mode C: Project Shell** (`/p/:projectId/*`) - Full UI with rail

---

## Phase 4: DockPanel (Terminal + Events)

### 4.1 Create DockPanel Component
- Slides up from bottom
- Ctrl/Cmd+J toggle
- Tabs: Terminal | Events (AI later)
- 35-45% viewport height when open

### 4.2 Terminal Tab
- Shows CLI commands being run
- Every UI action shows corresponding CLI command

### 4.3 Events Tab
- Live event stream from current run
- Hash chain visualization

---

## Implementation Order

1. **Phase 1.1-1.3**: Ground truth + fix Engine Adapter (2-3 hours)
2. **Phase 2.1-2.3**: QuickTrace working E2E (1-2 hours)
3. **Phase 3.1-3.3**: UX tightness fixes (2-3 hours)
4. **Phase 4**: DockPanel (3-4 hours)

---

## Success Criteria

After this plan is complete:
1. QuickTrace works: enter path → run → see events → see result
2. Engine status visible: user knows if backend is connected
3. Empty states are action-driven: user always has a clear next step
4. No dead-ends: every route leads somewhere useful
5. CLI correspondence: user can see what commands are being run

---

## What NOT To Do

- Don't add more features until existing ones work
- Don't build more abstractions without testing them
- Don't assume what the backend does - test it
- Don't patch symptoms - fix root causes
