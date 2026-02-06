# Engine Adapter Integration Guide

How to integrate the Engine Adapter into existing components.

## Quick Migration Checklist

If you have existing components that fetch directly from Flask:

- [ ] Remove hardcoded API URLs
- [ ] Import `getEngine` instead of using `fetch()`
- [ ] Replace direct API calls with engine methods
- [ ] Add proper error handling
- [ ] Add loading states
- [ ] Update types to use engine types

## Step-by-Step Migration

### Before (Direct Flask API)

```jsx
// BAD: Direct API access
const API = 'http://localhost:5001'

function RunsList() {
  const [runs, setRuns] = useState([])

  useEffect(() => {
    fetch(`${API}/api/runs`)
      .then(r => r.json())
      .then(data => setRuns(data.runs || []))
  }, [])

  return <div>{/* ... */}</div>
}
```

### After (Engine Adapter)

```jsx
// GOOD: Engine abstraction
import { getEngine } from '../engine'

/**
 * @typedef {import('../engine').RunSummary} RunSummary
 */

function RunsList() {
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

  return <div>{/* ... */}</div>
}
```

## Integration with Existing Components

### 1. RunsList Component

**File:** `/home/ryan/BEF-main/ui/src/RunsList.jsx`

**Current implementation:**
```jsx
const API = window.API_BASE || import.meta.env.VITE_API_BASE || DEFAULT_API
fetch(`${API}/api/runs`)
```

**Migrate to:**
```jsx
import { getEngine } from './engine'

const fetchRuns = useCallback(async () => {
  setStatus('loading')
  try {
    const engine = getEngine()
    const runsList = await engine.listRuns()
    setRuns(runsList)
    setStatus('ready')

    if (onFirstRunLoaded && runsList.length > 0) {
      onFirstRunLoaded(runsList[0].run_id)
    }
  } catch (err) {
    console.error('Failed to fetch runs:', err)
    setStatus('error')
  }
}, [onFirstRunLoaded])
```

### 2. RunDetailPanel Component

**File:** `/home/ryan/BEF-main/ui/src/RunDetailPanel.jsx`

**Current implementation:**
```jsx
const API = window.API_BASE || import.meta.env.VITE_API_BASE || DEFAULT_API

// Multiple fetch calls
fetch(`${API}/api/runs/${runId}`)
fetch(`${API}/api/runs/${runId}/events`)
fetch(`${API}/api/runs/${runId}/verify`, { method: 'POST' })
```

**Migrate to:**
```jsx
import { getEngine } from './engine'

// Get run detail
const fetchRunDetail = useCallback(async () => {
  setStatus('loading')
  try {
    const engine = getEngine()
    const run = await engine.getRun(runId)
    setMeta(run)
    setArtifacts(run.artifacts || [])
    setStatus('ready')
  } catch (err) {
    setStatus(`error: ${err.message}`)
  }
}, [runId])

// Stream events
useEffect(() => {
  const engine = getEngine()

  const cleanup = engine.streamEvents(
    runId,
    (event) => {
      setEvents((prev) => [...prev, event])
    },
    (error) => {
      console.error('Event stream error:', error)
    }
  )

  return cleanup
}, [runId])

// Verify
const handleVerify = async () => {
  setVerifying(true)
  try {
    const engine = getEngine()
    const report = await engine.verify({ run_id: runId })
    setVerification(report)
    setToast({ message: 'Verification complete', type: 'success' })
  } catch (err) {
    setToast({ message: err.message, type: 'error' })
  } finally {
    setVerifying(false)
  }
}
```

### 3. NewRunDrawer Component

**File:** `/home/ryan/BEF-main/ui/src/components/NewRunDrawer.jsx`

**Add engine integration:**
```jsx
import { getEngine } from '../engine'

function NewRunDrawer({ open, onClose, projects, circuits }) {
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (formData) => {
    setStarting(true)
    setError(null)

    try {
      const engine = getEngine()
      const response = await engine.startRun({
        project_id: formData.projectId,
        circuit_id: formData.circuitId,
        mode: 'project_run',
        input_path: formData.inputPath,
        policy_path: formData.policyPath,
      })

      console.log('Run started:', response.run_id)
      onClose()
      // Navigate to run detail page
      window.location.href = `/runs/${response.run_id}`
    } catch (err) {
      console.error('Failed to start run:', err)
      setError(err.message)
    } finally {
      setStarting(false)
    }
  }

  return (
    <Drawer open={open} onClose={onClose}>
      {/* Form UI */}
      {error && <div className="error">{error}</div>}
      <button onClick={handleSubmit} disabled={starting}>
        {starting ? 'Starting...' : 'Start Run'}
      </button>
    </Drawer>
  )
}
```

## Configuration Setup

### Development (Vite)

**File:** `.env.development`
```bash
VITE_API_BASE=http://localhost:5001
```

### Production (Tauri)

**File:** `.env.production`
```bash
VITE_API_BASE=http://localhost:5001
```

### Runtime Override

```js
// In main.jsx or App.jsx, before first render
window.API_BASE = 'http://localhost:5001'
```

## Type Safety with JSDoc

Add types to your components:

```jsx
/**
 * @typedef {import('./engine').RunSummary} RunSummary
 * @typedef {import('./engine').RunRecord} RunRecord
 * @typedef {import('./engine').VerifyReport} VerifyReport
 */

/**
 * Run list component
 * @param {Object} props
 * @param {string} [props.projectId] - Filter by project
 * @param {(runId: string) => void} [props.onSelect] - Selection handler
 */
function RunsList({ projectId, onSelect }) {
  /** @type {[RunSummary[], (runs: RunSummary[]) => void]} */
  const [runs, setRuns] = useState([])

  // ...
}
```

## Testing

### Unit Tests

```js
import { createEngine } from './engine'

describe('Engine Adapter', () => {
  let engine

  beforeEach(() => {
    // Create test engine
    engine = createEngine({
      mode: 'local',
      baseUrl: 'http://localhost:5001',
    })

    // Mock fetch
    global.fetch = jest.fn()
  })

  test('listRuns returns array', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ runs: [] }),
    })

    const runs = await engine.listRuns()
    expect(Array.isArray(runs)).toBe(true)
  })

  test('startRun returns run_id', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ run_id: 'test-123' }),
    })

    const response = await engine.startRun({
      project_id: 'test',
      mode: 'quick_trace',
      input_path: '/test',
    })

    expect(response.run_id).toBe('test-123')
  })
})
```

### Integration Tests

```js
import { getEngine } from './engine'

describe('Engine Integration', () => {
  test('can connect to local Flask', async () => {
    const engine = getEngine()
    const health = await engine.getHealth()

    expect(health.ok).toBe(true)
    expect(health.engine).toBe('flask')
  })

  test('can list runs', async () => {
    const engine = getEngine()
    const runs = await engine.listRuns()

    expect(Array.isArray(runs)).toBe(true)
  })
})
```

## Troubleshooting

### Problem: "Failed to fetch"

**Cause:** Flask backend not running or wrong port

**Solution:**
```bash
# Check if Flask is running
curl http://localhost:5001/sandbox/status

# Start Flask backend
cd server
python -m flask_app.app
```

### Problem: "CORS error"

**Cause:** Flask CORS not configured

**Solution:**
```python
# In server/flask_app/app.py
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])
```

### Problem: "No runs returned"

**Cause:** Local cache empty, no backend endpoint

**Solution:**
- Start a run first to populate cache
- Or manually add test data to localStorage:

```js
localStorage.setItem('capseal_runs_cache', JSON.stringify([
  {
    run_id: 'test-123',
    project_id: 'default',
    started_at: new Date().toISOString(),
    status: 'verified',
  }
]))
```

### Problem: "Events not streaming"

**Cause:** Flask doesn't have SSE endpoint, using polling fallback

**Solution:**
- This is expected behavior
- Events poll every 2 seconds
- Check browser console for errors
- Verify `/runs/:id/events` endpoint works:

```bash
curl http://localhost:5001/runs/test-123/events
```

## Next Steps

1. **Migrate RunsList component** - Replace direct fetch with engine.listRuns()
2. **Migrate RunDetailPanel** - Replace all fetch calls with engine methods
3. **Update NewRunDrawer** - Add engine.startRun() integration
4. **Add error boundaries** - Wrap components with error handlers
5. **Add loading states** - Show spinners during async operations
6. **Test offline mode** - Verify UI gracefully handles engine offline
7. **Add settings page** - Let users configure engine URL

## Best Practices

1. **Always use singleton `getEngine()`** - Don't create multiple instances
2. **Clean up event streams** - Return cleanup function from useEffect
3. **Handle errors explicitly** - Show user-friendly error messages
4. **Use loading states** - Indicate when operations are in progress
5. **Cache locally** - Don't refetch data unnecessarily
6. **Type your components** - Use JSDoc for better IDE support
7. **Test both online and offline** - Verify graceful degradation

## Resources

- `/ui/src/engine/README.md` - Architecture overview
- `/ui/src/engine/EXAMPLES.md` - Usage examples
- `/ui/src/engine/types.js` - Type definitions
- `/server/flask_app/routes.py` - Backend API reference
