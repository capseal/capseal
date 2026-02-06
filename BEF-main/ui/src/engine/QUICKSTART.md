# Engine Adapter - Quick Start

Get started with the Engine Adapter in 5 minutes.

## What is this?

The Engine Adapter is a layer that abstracts the backend. Your components talk to it instead of directly calling Flask APIs. This means:

- Components don't care if backend is Flask, Rust, or remote
- Switching backends = zero component changes
- All networking code is centralized
- Types are frozen and stable

## Installation

Already installed! Files are in `/home/ryan/BEF-main/ui/src/engine/`

## Basic Usage

### 1. Import the engine

```jsx
import { getEngine } from './engine'
```

### 2. Use it in your component

```jsx
function MyComponent() {
  const [runs, setRuns] = useState([])

  useEffect(() => {
    const engine = getEngine()
    engine.listRuns().then(setRuns).catch(console.error)
  }, [])

  return (
    <ul>
      {runs.map(run => (
        <li key={run.run_id}>{run.run_id}</li>
      ))}
    </ul>
  )
}
```

That's it! No fetch(), no URL construction, no API configuration.

## Common Operations

### List all runs

```js
const engine = getEngine()
const runs = await engine.listRuns()
```

### Get a specific run

```js
const engine = getEngine()
const run = await engine.getRun('run-id-123')
```

### Start a new run

```js
const engine = getEngine()
const response = await engine.startRun({
  project_id: 'my-project',
  mode: 'project_run',
  input_path: '/path/to/input',
})
console.log('Started:', response.run_id)
```

### Stream events from a running execution

```js
const engine = getEngine()

const cleanup = engine.streamEvents(
  runId,
  (event) => {
    console.log('Event:', event.type)
  },
  (error) => {
    console.error('Error:', error)
  }
)

// Later: cleanup()
```

### Verify a run

```js
const engine = getEngine()
const report = await engine.verify({ run_id: 'run-id-123' })

console.log('Verified:', report.ok)
console.log('Layers:', report.layers) // L0-L4
```

### Audit a run

```js
const engine = getEngine()
const report = await engine.audit({ run_id: 'run-id-123' })

console.log('Chain valid:', report.chain.head_hash)
console.log('Findings:', report.findings)
```

## Configuration

### Default (localhost Flask)

```js
// Uses http://localhost:5001 automatically
const engine = getEngine()
```

### Custom URL

```js
import { createEngine } from './engine'

const engine = createEngine({
  mode: 'local',
  baseUrl: 'http://localhost:5001',
})
```

### Environment variable

```bash
# .env
VITE_API_BASE=http://localhost:5001
```

## Error Handling

Always handle errors:

```js
try {
  const runs = await engine.listRuns()
  setRuns(runs)
} catch (error) {
  console.error('Failed to load runs:', error)
  setError(error.message)
}
```

## Type Safety (JSDoc)

Get type checking without TypeScript:

```jsx
/**
 * @typedef {import('./engine').RunSummary} RunSummary
 */

/**
 * @param {Object} props
 * @param {RunSummary} props.run
 */
function RunCard({ run }) {
  return <div>{run.run_id}</div>
}
```

## Real Example: Complete Component

```jsx
import { getEngine } from './engine'
import { useEffect, useState } from 'react'

/**
 * @typedef {import('./engine').RunSummary} RunSummary
 */

function RunsList() {
  /** @type {[RunSummary[], Function]} */
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const engine = getEngine()

    engine.listRuns()
      .then(setRuns)
      .catch(setError)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  if (runs.length === 0) return <div>No runs yet</div>

  return (
    <div>
      <h2>Runs ({runs.length})</h2>
      <ul>
        {runs.map(run => (
          <li key={run.run_id}>
            <strong>{run.run_id}</strong>
            <span> - {run.status}</span>
            <span> - {new Date(run.started_at).toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default RunsList
```

## Troubleshooting

### "Failed to fetch"

Flask backend not running:
```bash
cd server
python -m flask_app.app
```

### "No runs returned"

Start a run first to populate cache, or the cache is empty.

### "CORS error"

Flask needs CORS enabled for your UI origin.

## Next Steps

1. Read `/home/ryan/BEF-main/ui/src/engine/README.md` for architecture
2. See `/home/ryan/BEF-main/ui/src/engine/EXAMPLES.md` for more examples
3. Follow `/home/ryan/BEF-main/ui/src/engine/INTEGRATION.md` to migrate existing components

## API Reference

### getEngine()

Returns singleton engine instance.

```js
const engine = getEngine()
```

### createEngine(config)

Creates new engine with custom config.

```js
const engine = createEngine({
  mode: 'local',
  baseUrl: 'http://localhost:5001',
  timeoutMs: 120000,
})
```

### ExecutionEngine Methods

- `getHealth()` → HealthStatus
- `startRun(req)` → RunStartResponse
- `streamEvents(runId, onEvent, onError)` → cleanup()
- `getRun(runId)` → RunRecord
- `listRuns(query?)` → RunSummary[]
- `verify(req)` → VerifyReport
- `audit(req)` → AuditReport
- `evidence(req)` → EvidenceIndex
- `exportCapsule(req)` → ExportResult

## Help

Questions? Read the full docs:
- ARCHITECTURE.txt - High-level overview
- README.md - Detailed architecture
- INTEGRATION.md - Migration guide
- EXAMPLES.md - 7 complete examples
