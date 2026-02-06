# Engine Adapter Usage Examples

Complete examples showing how to use the Engine Adapter in UI components.

## Example 1: Health Check

```jsx
import { getEngine } from '../engine'
import { useEffect, useState } from 'react'

function HealthBadge() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const engine = getEngine()

    engine.getHealth()
      .then(status => {
        setHealth(status)
        setLoading(false)
      })
      .catch(err => {
        console.error('Health check failed:', err)
        setLoading(false)
      })
  }, [])

  if (loading) return <span className="badge badge-gray">Checking...</span>
  if (!health?.ok) return <span className="badge badge-error">Offline</span>

  return (
    <span className="badge badge-success">
      {health.engine} v{health.version}
    </span>
  )
}
```

## Example 2: List Runs with Filtering

```jsx
import { getEngine } from '../engine'
import { useEffect, useState } from 'react'

function RunsList({ projectId }) {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    const engine = getEngine()

    const query = {}
    if (projectId) query.project_id = projectId
    if (filter !== 'all') query.status = filter

    engine.listRuns(query)
      .then(setRuns)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [projectId, filter])

  if (loading) return <div>Loading runs...</div>

  return (
    <div>
      <select value={filter} onChange={e => setFilter(e.target.value)}>
        <option value="all">All</option>
        <option value="verified">Verified</option>
        <option value="failed">Failed</option>
        <option value="running">Running</option>
      </select>

      <ul>
        {runs.map(run => (
          <li key={run.run_id}>
            {run.run_id} - {run.status}
          </li>
        ))}
      </ul>
    </div>
  )
}
```

## Example 3: Start Run and Stream Events

```jsx
import { getEngine } from '../engine'
import { useState } from 'react'

function NewRunForm({ projectId, onRunStarted }) {
  const [starting, setStarting] = useState(false)
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setStarting(true)
    setError(null)

    const engine = getEngine()

    try {
      // Start the run
      const response = await engine.startRun({
        project_id: projectId,
        mode: 'project_run',
        input_path: '/path/to/input',
        policy_path: '/path/to/policy.json',
      })

      console.log('Run started:', response.run_id)
      onRunStarted?.(response.run_id)

      // Stream events
      const cleanup = engine.streamEvents(
        response.run_id,
        (event) => {
          console.log('Event:', event.type, event.seq)
          setEvents(prev => [...prev, event])

          // Check for completion
          if (event.type === 'run_completed') {
            console.log('Run completed!')
            cleanup()
          } else if (event.type === 'run_failed') {
            console.error('Run failed:', event.message)
            setError(event.message)
            cleanup()
          }
        },
        (err) => {
          console.error('Stream error:', err)
          setError(err.message)
        }
      )

      // Store cleanup function for later
      window._currentRunCleanup = cleanup

    } catch (err) {
      console.error('Failed to start run:', err)
      setError(err.message)
    } finally {
      setStarting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <button type="submit" disabled={starting}>
        {starting ? 'Starting...' : 'Start Run'}
      </button>

      {error && <div className="error">{error}</div>}

      {events.length > 0 && (
        <div className="events">
          <h4>Events ({events.length})</h4>
          <ul>
            {events.map((evt, i) => (
              <li key={i}>
                [{evt.seq}] {evt.type} - {evt.message}
              </li>
            ))}
          </ul>
        </div>
      )}
    </form>
  )
}
```

## Example 4: Run Detail with Verify

```jsx
import { getEngine } from '../engine'
import { useEffect, useState } from 'react'

function RunDetail({ runId }) {
  const [run, setRun] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const [verifyReport, setVerifyReport] = useState(null)

  // Load run on mount
  useEffect(() => {
    const engine = getEngine()

    engine.getRun(runId)
      .then(setRun)
      .catch(console.error)
  }, [runId])

  // Verify handler
  const handleVerify = async () => {
    setVerifying(true)

    try {
      const engine = getEngine()
      const report = await engine.verify({ run_id: runId })

      setVerifyReport(report)
      console.log('Verification:', report.ok ? 'PASS' : 'FAIL')

      // Update run status
      if (run) {
        setRun({
          ...run,
          status: report.ok ? 'verified' : 'failed',
        })
      }
    } catch (err) {
      console.error('Verification failed:', err)
    } finally {
      setVerifying(false)
    }
  }

  if (!run) return <div>Loading...</div>

  return (
    <div className="run-detail">
      <h2>Run {run.run_id}</h2>

      <div className="run-info">
        <p>Status: {run.status}</p>
        <p>Project: {run.project_id}</p>
        <p>Started: {new Date(run.started_at).toLocaleString()}</p>
        {run.capsule_hash && <p>Hash: {run.capsule_hash.slice(0, 16)}...</p>}
      </div>

      <button onClick={handleVerify} disabled={verifying}>
        {verifying ? 'Verifying...' : 'Verify'}
      </button>

      {verifyReport && (
        <div className="verify-report">
          <h3>Verification Report</h3>
          <p className={verifyReport.ok ? 'pass' : 'fail'}>
            {verifyReport.ok ? 'VERIFIED' : 'FAILED'}
          </p>

          <h4>Trust Ladder (L0-L4)</h4>
          <ul>
            {verifyReport.layers.map(layer => (
              <li key={layer.id} className={layer.ok ? 'pass' : 'fail'}>
                {layer.id} {layer.label}: {layer.ok ? '✓' : '✗'}
              </li>
            ))}
          </ul>

          {verifyReport.errors?.length > 0 && (
            <div className="errors">
              <h4>Errors</h4>
              <ul>
                {verifyReport.errors.map((err, i) => (
                  <li key={i}>{err.code}: {err.message}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

## Example 5: Audit Report

```jsx
import { getEngine } from '../engine'
import { useState } from 'react'

function AuditButton({ runId }) {
  const [auditing, setAuditing] = useState(false)
  const [report, setReport] = useState(null)

  const handleAudit = async () => {
    setAuditing(true)

    try {
      const engine = getEngine()
      const auditReport = await engine.audit({
        run_id: runId,
        verify_chain: true,
      })

      setReport(auditReport)
      console.log('Audit:', auditReport.ok ? 'PASS' : 'FAIL')
    } catch (err) {
      console.error('Audit failed:', err)
    } finally {
      setAuditing(false)
    }
  }

  return (
    <div>
      <button onClick={handleAudit} disabled={auditing}>
        {auditing ? 'Auditing...' : 'Run Audit'}
      </button>

      {report && (
        <div className="audit-report">
          <h4>Audit Report</h4>
          <p>Status: {report.ok ? 'PASS' : 'FAIL'}</p>
          <p>Hash Chain: {report.chain.length} events</p>
          <p>Head: {report.chain.head_hash?.slice(0, 16)}...</p>

          {report.findings.length > 0 && (
            <div className="findings">
              <h5>Findings</h5>
              <ul>
                {report.findings.map((f, i) => (
                  <li key={i} className={`severity-${f.severity}`}>
                    [{f.severity.toUpperCase()}] {f.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

## Example 6: Evidence Opening

```jsx
import { getEngine } from '../engine'
import { useEffect, useState } from 'react'

function EvidenceExplorer({ runId }) {
  const [index, setIndex] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const engine = getEngine()

    engine.evidence({
      run_id: runId,
      kind: 'artifacts',
    })
      .then(setIndex)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [runId])

  if (loading) return <div>Loading evidence...</div>
  if (!index) return <div>No evidence available</div>

  return (
    <div className="evidence-explorer">
      <h3>Evidence Index</h3>

      {index.rows && (
        <div className="rows-info">
          <h4>Rows</h4>
          <p>Count: {index.rows.count}</p>
          <p>Chunks: {index.rows.chunks}</p>
        </div>
      )}

      {index.artifacts && index.artifacts.length > 0 && (
        <div className="artifacts-list">
          <h4>Artifacts ({index.artifacts.length})</h4>
          <ul>
            {index.artifacts.map((artifact, i) => (
              <li key={i}>
                <strong>{artifact.name}</strong> ({artifact.artifact_type})
                <br />
                {artifact.size_bytes} bytes
                <br />
                Hash: {artifact.hash?.slice(0, 16)}...
              </li>
            ))}
          </ul>
        </div>
      )}

      {index.events && (
        <div className="events-info">
          <h4>Events</h4>
          <p>Count: {index.events.count}</p>
          {index.events.chain_head && (
            <p>Chain Head: {index.events.chain_head.slice(0, 16)}...</p>
          )}
        </div>
      )}
    </div>
  )
}
```

## Example 7: Custom Engine Configuration

```jsx
import { createEngine, resetEngine } from '../engine'
import { useEffect } from 'react'

function EngineSettings() {
  const [config, setConfig] = useState({
    mode: 'local',
    baseUrl: 'http://localhost:5001',
  })

  const handleSave = () => {
    // Reset existing engine
    resetEngine()

    // Create new engine with custom config
    const engine = createEngine(config)

    // Test connection
    engine.getHealth().then(health => {
      if (health.ok) {
        alert('Engine connected successfully!')
      } else {
        alert('Failed to connect to engine')
      }
    })
  }

  return (
    <div className="engine-settings">
      <h3>Engine Configuration</h3>

      <label>
        Mode:
        <select value={config.mode} onChange={e => setConfig({...config, mode: e.target.value})}>
          <option value="local">Local</option>
          <option value="remote">Remote</option>
        </select>
      </label>

      <label>
        Base URL:
        <input
          type="text"
          value={config.baseUrl}
          onChange={e => setConfig({...config, baseUrl: e.target.value})}
        />
      </label>

      <button onClick={handleSave}>Save & Reconnect</button>
    </div>
  )
}
```

## Common Patterns

### Error Handling

```js
try {
  const engine = getEngine()
  const result = await engine.verify({ run_id: runId })
  // Handle success
} catch (err) {
  if (err.message.includes('HTTP 404')) {
    console.error('Run not found')
  } else if (err.message.includes('timeout')) {
    console.error('Request timed out')
  } else {
    console.error('Verification failed:', err)
  }
}
```

### Loading States

```jsx
function DataComponent() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const engine = getEngine()

    engine.listRuns()
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />
  if (error) return <ErrorBanner error={error} />
  if (!data) return <EmptyState />

  return <DataView data={data} />
}
```

### Event Stream Cleanup

```jsx
useEffect(() => {
  const engine = getEngine()

  const cleanup = engine.streamEvents(
    runId,
    handleEvent,
    handleError
  )

  // IMPORTANT: Return cleanup function
  return cleanup
}, [runId])
```

## Tips

1. **Always use `getEngine()`** - Don't create your own instances
2. **Clean up event streams** - Return cleanup function from useEffect
3. **Handle errors explicitly** - Don't let network errors crash the UI
4. **Cache results locally** - The engine already does this, but you can too
5. **Use loading states** - Network requests take time
6. **Type your props** - Use JSDoc for better IDE support

```jsx
/**
 * @typedef {import('../engine').RunSummary} RunSummary
 *
 * @param {Object} props
 * @param {RunSummary} props.run
 */
function RunCard({ run }) {
  return <div>{run.run_id}</div>
}
```
