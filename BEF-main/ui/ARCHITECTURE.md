# CapSeal UI Architecture Checkpoint

> Written for context recovery after conversation compaction.
> Last updated: 2026-01-25

## Project Vision

**Goal**: Build CapSeal into a downloadable desktop application ("governed execution console") that:
- Minimizes token/context and oracle calls
- Maximizes assurance that runs stayed within policy
- Keeps verification cheap
- Preserves parallelism
- Eventually packages as Tauri desktop app with pure Rust `capseal` core

## Typed DAG Workflow Model

The system models governed computations as a typed DAG `G = (V, E)`:

### Node Types
| Type | Description | Verifiable Property |
|------|-------------|---------------------|
| SOURCE | External input (file, API) | Hash commitment |
| COMPUTE | Pure function | Deterministic output |
| ORACLE | LLM/external API call | Context root + budget |
| MERGE | Combine multiple inputs | Aggregation rule |
| GATE | Policy checkpoint | Budget/constraint check |
| SINK | Final output | Receipt generation |

### Edge Semantics
- Edges carry typed data with hash commitments
- Each edge has a `schema_hash` for type verification
- Ordering preserved via topological sort

## L0-L4 Verification Ladder

| Layer | What | Check | Exit Code |
|-------|------|-------|-----------|
| L0 | Hash | Artifact hashes match commitments | 20 |
| L1 | Commitment | Merkle roots valid (STC) | 12 |
| L2 | Constraint | Policy/AIR constraints enforced | 11 |
| L3 | Proximity | FRI low-degree test passed | 10 |
| L4 | Receipt | Capsule format valid | 0 |

UI surfaces layer status via `VerificationLayers` type in `contracts.ts`.

## Token Governance Model

### Context Pack (`bef_zk/capsule/context_pack.py`)
- **ContextChunk**: Single piece of context (file, diff, receipt, tool output)
- **ContextPack**: Hash-addressed bundle with deterministic `context_root`
  - Ensures same inputs → same hash
  - Selection algorithm tracked (manual, auto_relevance, diff_focused)
- **OracleCall**: Record of ORACLE invocation with tokens/cost/latency
- **BudgetLedger**: Cumulative spend tracking with `check_budget()` for GATE enforcement
- **OracleTracker**: Manager combining context packs + budget

### Event Log (`bef_zk/capsule/event_log.py`)
- Hash-chained append-only log
- Event types: `oracle_call`, `context_pack_created`, `budget_checkpoint`, `budget_exceeded`
- `verify_chain()` for tamper detection
- `get_total_token_spend()` for aggregate accounting

### Budget Endpoint
`GET /runs/:id/budget` returns `BudgetSummary`:
```typescript
{
  run_id: string,
  budget: { tokens, oracle_calls, usd },
  spent: { tokens_in, tokens_out, tokens_total, oracle_calls, usd },
  remaining: { tokens, oracle_calls, usd },
  utilization: { tokens_pct, calls_pct, usd_pct },
  oracle_calls: OracleCallRecord[],
  governance_enabled: boolean
}
```

## Key Files

### Contracts (Source of Truth)
- `ui/src/contracts/contracts.ts` - All API response shapes
- `ui/src/contracts/examples/` - Example payloads for testing

### Backend
- `server/flask_app/sse_routes.py` - SSE streaming + contract-compliant endpoints
  - `/runs` - List runs
  - `/runs/:id/events/stream` - SSE event stream
  - `/runs/:id/verify` - Returns `VerifyReport`
  - `/runs/:id/audit` - Returns `AuditReport`
  - `/runs/:id/evidence` - Returns `EvidenceIndex`
  - `/runs/:id/budget` - Returns `BudgetSummary`

### UI Components
- `ui/src/Launchpad.jsx` - Jobs-like entry point
- `ui/src/WorkspaceSelection.jsx` - Project picker with mode support
- `ui/src/components/ProjectShell.jsx` - Rail + RunsList + RunDetailPanel
- `ui/src/components/Rail.jsx` - Navigation rail (Runs/Settings/Verify/Evidence)
- `ui/src/RunsList.jsx` - List of runs with status badges
- `ui/src/RunDetailPanel.jsx` - Detail view with tabs

## Ticket Status

| # | Ticket | Status |
|---|--------|--------|
| 0 | Freeze contracts | DONE |
| 1 | Launchpad + Routes + Rail visibility | DONE |
| 2 | WorkspaceHome project picker | DONE |
| 3 | ProjectShell wiring | DONE |
| 4 | Local daemon + SSE streaming | DONE |
| 5 | Token governance + ORACLE tracking | DONE |
| 6 | Integration sweep | IN PROGRESS |

## Remaining Work

### Ticket 6: Integration Sweep
- [ ] Verify all routes return contract-compliant JSON
- [ ] Test deep links (Launchpad → Project → Run → Tab)
- [ ] Verify SSE reconnection with Last-Event-ID
- [ ] Test budget endpoint with real event logs
- [ ] Fix any remaining evidence endpoint edge cases

### Tauri Packaging (Future)
1. Create Tauri project scaffold
2. Bundle Vite frontend as Tauri webview
3. Implement Tauri commands wrapping `capseal` CLI
4. Replace Flask with direct Rust calls via Tauri IPC
5. Eventually embed pure-Rust `capseal` core (bef_rust)

## Development Commands

```bash
# Start UI dev server
cd ui && npm run dev

# Start Flask daemon
cd server && ../.venv/bin/python -m flask --app flask_app:create_app run --host 0.0.0.0 --port 5001

# Capseal CLI
./capseal run <input> --policy <policy.json> --output out/
./capseal verify out/strategy_capsule.json
./capseal audit out/strategy_capsule.json
./capseal inspect out/strategy_capsule.json
```

## Contract Stability Rules

1. **Never break contracts** - Backend changes must preserve response shapes
2. **Add fields cautiously** - New fields must be optional
3. **Deprecate, don't remove** - Mark fields deprecated before removal
4. **Test with examples** - `ui/src/contracts/examples/` has fixture data
