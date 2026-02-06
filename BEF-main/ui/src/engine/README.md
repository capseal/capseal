# CapsuleTech Engine Adapter Layer

**CRITICAL INFRASTRUCTURE** - This is the abstraction layer between UI components and backend execution engines.

## Architecture

Components NEVER talk directly to Flask/Rust/Remote services. They talk to the `ExecutionEngine` interface, which abstracts all backend details.

```
┌─────────────────────────────────────────────┐
│          UI Components                      │
│  (Launchpad, RunsList, RunDetailPanel)      │
└────────────────┬────────────────────────────┘
                 │
                 │ import { getEngine }
                 │
┌────────────────▼────────────────────────────┐
│      ExecutionEngine Interface              │
│   (engine.js - frozen contract)             │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴────────┬─────────────┐
         │                │             │
         ▼                ▼             ▼
  ┌────────────┐   ┌───────────┐   ┌────────────┐
  │  Flask     │   │   Rust    │   │   Remote   │
  │  Impl      │   │   Impl    │   │   Impl     │
  │ (5001)     │   │ (future)  │   │ (future)   │
  └────────────┘   └───────────┘   └────────────┘
```

## Files

### `/types.js` - Frozen JSON Contracts

All data types that flow between UI and backend. These are **frozen** - they can be extended but never broken.

Key types:
- `EngineConfig`, `HealthStatus` - Engine configuration and capabilities
- `RunStartRequest`, `RunStartResponse` - Run execution
- `RunEvent` - SSE event payloads with hash chain
- `RunSummary`, `RunRecord` - Run data
- `VerifyReport` (L0-L4 trust ladder)
- `AuditReport` (hash chain + policy)
- `EvidenceIndex` - Row/artifact openings
- `BudgetSpec`, `BudgetSpent` - Oracle governance

### `/engine.js` - ExecutionEngine Interface

The interface ALL backends must implement:

```js
class ExecutionEngine {
  async getHealth() // → HealthStatus
  async startRun(req) // → RunStartResponse
  streamEvents(runId, onEvent, onError) // → cleanup function
  async getRun(runId) // → RunRecord
  async listRuns(query) // → RunSummary[]
  async verify(req) // → VerifyReport
  async audit(req) // → AuditReport
  async evidence(req) // → EvidenceIndex
  async exportCapsule(req) // → ExportResult
}
```

Factory functions:
- `getDefaultConfig()` - Returns default config (localhost:5001)
- `createEngine(config)` - Creates engine instance
- `getEngine(config)` - Singleton accessor (use this in components)

### `/impl/flask.js` - Flask Backend Adapter

Bridges between frontend expectations and Flask's RPC-style API.

**Key mappings:**

| Frontend Method | Flask Endpoint | Transformation |
|----------------|---------------|----------------|
| `startRun()` | `POST /run` | Request shape + job polling |
| `streamEvents()` | `GET /runs/:id/events` | Polling (no SSE) |
| `getRun()` | `GET /runs/:id/events` | Construct from events |
| `listRuns()` | Local cache | No backend endpoint |
| `verify()` | `POST /verify` | Request shape transform |
| `audit()` | `POST /audit` | Request shape transform |
| `getHealth()` | `GET /sandbox/status` | Capability mapping |

**Local caching:**
- Runs persist in `localStorage` under `capseal_runs_cache`
- Survives page refresh
- Last 100 runs kept
- Backend doesn't provide list endpoint, so we cache locally

## Usage in Components

### Basic usage (recommended)

```jsx
import { getEngine } from './engine'

function MyComponent() {
  const engine = getEngine()

  useEffect(() => {
    // Load runs
    engine.listRuns().then(setRuns)
  }, [])

  const handleVerify = async (runId) => {
    const report = await engine.verify({ run_id: runId })
    console.log('L0-L4 layers:', report.layers)
  }

  return <div>...</div>
}
```

### Event streaming

```jsx
import { getEngine } from './engine'

function RunDetail({ runId }) {
  useEffect(() => {
    const engine = getEngine()

    const cleanup = engine.streamEvents(
      runId,
      (event) => {
        console.log('Event:', event.type, event.seq)
        setEvents(prev => [...prev, event])
      },
      (error) => {
        console.error('Stream error:', error)
      }
    )

    // Cleanup on unmount
    return cleanup
  }, [runId])

  return <div>...</div>
}
```

### Custom configuration

```jsx
import { createEngine } from './engine'

// Override default (e.g., for remote engine)
const engine = createEngine({
  mode: 'remote',
  baseUrl: 'https://api.capseal.io',
})
```

## Configuration Resolution

Priority order:
1. Explicit config passed to `createEngine(config)`
2. Environment: `window.API_BASE` or `VITE_API_BASE`
3. Default: `http://localhost:5001`

**No silent fallbacks.** If engine is unreachable, components should show explicit "Engine Offline" state.

## Extending for Rust/Remote

To add a new backend implementation:

1. Create `/impl/rust.js` or `/impl/remote.js`
2. Implement `ExecutionEngine` interface
3. Update `createEngine()` in `engine.js`:

```js
export function createEngine(config = {}) {
  const fullConfig = { ...getDefaultConfig(), ...config }

  if (fullConfig.mode === 'rust') {
    const { RustEngine } = require('./impl/rust.js')
    return new RustEngine(fullConfig)
  }

  if (fullConfig.mode === 'remote') {
    const { RemoteEngine } = require('./impl/remote.js')
    return new RemoteEngine(fullConfig)
  }

  // Default: Flask
  const { FlaskEngine } = require('./impl/flask.js')
  return new FlaskEngine(fullConfig)
}
```

**That's it.** No component changes needed.

## Design Principles

1. **Components never construct URLs** - All networking happens in the engine layer
2. **Frozen contracts** - Types can be extended but never broken
3. **Backend-agnostic** - Switching Flask → Rust requires zero component changes
4. **Local-first** - Persistence survives refresh without backend round-trips
5. **Explicit errors** - No silent fallbacks or auto-retries without user knowledge
6. **Trust ladder** - Verification is L0-L4 checkpoints, not binary pass/fail

## Testing

```js
import { createEngine } from './engine'

// Create test engine with mock backend
const testEngine = createEngine({
  mode: 'local',
  baseUrl: 'http://localhost:5001',
})

// Mock responses
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ status: 'ok' }),
  })
)

// Test
const health = await testEngine.getHealth()
expect(health.ok).toBe(true)
```

## Migration Path (Flask → Rust)

1. **Phase 1 (current):** Flask backend, FlaskEngine adapter
2. **Phase 2:** Rust backend available, RustEngine adapter, mode toggle in settings
3. **Phase 3:** Deprecate Flask, default to Rust
4. **Phase 4:** Remove Flask code

Components remain **100% unchanged** throughout.

## Related Files

- `/src/contracts/contracts.ts` - Legacy contracts (being deprecated)
- `/src/components/*` - UI components using this adapter
- `/server/flask_app/routes.py` - Flask backend endpoints

## Questions?

This layer is CRITICAL. If anything is unclear, ask before modifying.
