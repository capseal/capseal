import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { getEngine } from './engine'
import { useEngine } from './state/EngineContext.jsx'
import EngineHandshakeCard from './components/EngineHandshakeCard'

/**
 * QuickTrace - Single-purpose trace & seal shell.
 *
 * NO rail. Focused UI for:
 * 1. Input: folder path, command, policy selection
 * 2. Execution: start run, stream events live
 * 3. Output: verify button, receipt preview
 *
 * Per Layout_hierarchy.txt section 2.3.D
 */
export default function QuickTrace() {
  const navigate = useNavigate()
  const engineCtx = useEngine()
  const engineReady = useMemo(() => ['online', 'degraded'].includes(engineCtx.status), [engineCtx.status])
  const [step, setStep] = useState('input') // 'input' | 'running' | 'complete'

  // Form state
  const [inputPath, setInputPath] = useState('')
  const [command, setCommand] = useState('')
  const [policyPath, setPolicyPath] = useState('')
  const [execMode, setExecMode] = useState('local')

  // Run state
  const [runId, setRunId] = useState(null)
  const [capsulePath, setCapsulePath] = useState(null)
  const [events, setEvents] = useState([])
  const [runStatus, setRunStatus] = useState(null) // 'running' | 'completed' | 'failed'
  const [verifyResult, setVerifyResult] = useState(null)

  const eventsEndRef = useRef(null)
  const eventSourceRef = useRef(null)

  // Auto-scroll events
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current?.close) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  const handleStartRun = async () => {
    if (!engineReady) {
      engineCtx.openConfig()
      return
    }
    if (!inputPath.trim()) {
      alert('Please enter an input path')
      return
    }

    setStep('running')
    setEvents([])
    setRunStatus('running')
    setVerifyResult(null)

    try {
      const engine = getEngine({ baseUrl: engineCtx.baseUrl })
      const response = await engine.startRun({
        traceId: `quick_trace_${Date.now()}`,
        policy: {
          policyId: 'quick-trace-policy',
          policyPath: policyPath || 'policies/policy_governed_openings.json',
        },
        policyId: 'quick-trace-policy',
        backend: execMode === 'remote' ? 'risc0' : 'geom',
        outputDir: inputPath.trim(),
        steps: 64,
        queries: 8,
        challenges: 2,
        datasets: [],
        profile: 'default',
        sandbox: execMode === 'remote',
        sandboxAllowNetwork: execMode === 'remote',
      })

      const newRunId = response.run_id
      setRunId(newRunId)

      // Store capsule_path for verification (backend returns this synchronously)
      if (response.capsule_path) {
        setCapsulePath(response.capsule_path)
      }

      // Start event streaming
      const cleanup = engine.streamEvents(
        newRunId,
        (event) => {
          setEvents(prev => [...prev, event])
          if (['run_completed', 'run_failed', 'capsule_sealed'].includes(event.type)) {
            setRunStatus(event.type === 'run_failed' ? 'failed' : 'completed')
            setStep('complete')
          }
        },
        (err) => {
          console.error('Event stream error:', err)
        }
      )
      eventSourceRef.current = { close: cleanup }
    } catch (err) {
      console.error('Failed to start run:', err)
      setEvents(prev => [...prev, {
        seq: 0,
        type: 'error',
        data: { message: err.message },
        ts: new Date().toISOString(),
      }])
      setRunStatus('failed')
      setStep('complete')
    }
  }


  const handleVerify = async () => {
    if (!runId) return
    if (!engineReady) {
      engineCtx.openConfig()
      return
    }

    // Need capsule_path for verification - construct from run_id if not available
    const pathToVerify = capsulePath || `/home/ryan/BEF-main/out/${runId}/strategy_capsule.json`

    try {
      const engine = getEngine({ baseUrl: engineCtx.baseUrl })
      const result = await engine.verify({ run_id: runId, capsule_path: pathToVerify })
      setVerifyResult(result)
    } catch (err) {
      console.error('Verify failed:', err)
      setVerifyResult({ status: 'error', ok: false, errors: [{ code: 'ERROR', message: err.message }] })
    }
  }

  const handleReset = () => {
    setStep('input')
    setRunId(null)
    setCapsulePath(null)
    setEvents([])
    setRunStatus(null)
    setVerifyResult(null)
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }
  }

  return (
    <div className="quicktrace-container">
      <EngineHandshakeCard />
      {/* Header */}
      <div className="quicktrace-header">
        <button className="quicktrace-back" onClick={() => navigate('/')}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back
        </button>
        <h1 className="quicktrace-title">Quick Trace</h1>
        <div className="quicktrace-step-indicator">
          {step === 'input' && 'Configure'}
          {step === 'running' && 'Running...'}
          {step === 'complete' && 'Complete'}
        </div>
      </div>

      <div className="quicktrace-content">
        {/* Left: Form / Events */}
        <div className="quicktrace-left">
          {step === 'input' && (
            <div className="quicktrace-form">
              <h2>Run Configuration</h2>
              {!engineReady && (
                <p className="engine-handshake-error">Engine not connected. Configure the backend before starting a run.</p>
              )}

              <div className="form-field">
                <label className="form-label">Input Path *</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="/path/to/project or ./relative/path"
                  value={inputPath}
                  onChange={(e) => setInputPath(e.target.value)}
                />
                <p className="form-hint">Path to the folder or file to process</p>
              </div>

              <div className="form-field">
                <label className="form-label">Command (optional)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="python main.py --arg value"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                />
                <p className="form-hint">Command to execute in the input directory</p>
              </div>

              <div className="form-field">
                <label className="form-label">Policy Path (optional)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="/path/to/policy.json"
                  value={policyPath}
                  onChange={(e) => setPolicyPath(e.target.value)}
                />
                <p className="form-hint">Policy file to enforce during execution</p>
              </div>

              <div className="form-field">
                <label className="form-label">Execution Mode</label>
                <div className="exec-mode-toggle">
                  <button
                    className={`exec-mode-button ${execMode === 'local' ? 'active' : ''}`}
                    onClick={() => setExecMode('local')}
                  >
                    Local
                  </button>
                  <button
                    className={`exec-mode-button ${execMode === 'remote' ? 'active' : ''}`}
                    onClick={() => setExecMode('remote')}
                    disabled
                  >
                    Remote (Coming Soon)
                  </button>
                </div>
                <p className="form-hint">
                  {execMode === 'local'
                    ? 'Runs on your machine, uses filesystem'
                    : 'Uploads inputs, runs hosted sandbox'}
                </p>
              </div>

              <button
                className="btn btn-primary btn-lg"
                onClick={handleStartRun}
                disabled={!inputPath.trim() || !engineReady}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Run & Seal
              </button>
            </div>
          )}

          {(step === 'running' || step === 'complete') && (
            <div className="quicktrace-events">
              <div className="events-header">
                <h2>Event Stream</h2>
                <span className={`events-status ${runStatus}`}>
                  {runStatus === 'running' && 'Live'}
                  {runStatus === 'completed' && 'Completed'}
                  {runStatus === 'failed' && 'Failed'}
                </span>
              </div>

              <div className="events-list">
                {events.map((event, idx) => (
                  <EventRow key={idx} event={event} />
                ))}
                <div ref={eventsEndRef} />
              </div>

              {step === 'complete' && (
                <div className="events-actions">
                  <button className="btn btn-secondary" onClick={handleReset}>
                    New Run
                  </button>
                  {runStatus === 'completed' && (
                    <button className="btn btn-primary" onClick={handleVerify}>
                      Verify Receipt
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Receipt Preview / Verify Result */}
        <div className="quicktrace-right">
          {step === 'input' && (
            <div className="quicktrace-help">
              <h3>How it works</h3>
              <ol>
                <li><strong>Configure</strong> - Set input path and optional command/policy</li>
                <li><strong>Run & Seal</strong> - Execute and generate cryptographic receipt</li>
                <li><strong>Verify</strong> - Check L0-L4 verification layers</li>
                <li><strong>Export</strong> - Share the .cap file for independent verification</li>
              </ol>

              <div className="help-note">
                <strong>What gets committed:</strong>
                <ul>
                  <li>Input file hashes (L0)</li>
                  <li>Execution trace Merkle root (L1)</li>
                  <li>Policy constraint checks (L2)</li>
                  <li>STARK proof (L3)</li>
                  <li>Signed receipt (L4)</li>
                </ul>
              </div>
            </div>
          )}

          {(step === 'running' || step === 'complete') && (
            <div className="quicktrace-receipt">
              <h3>Receipt Preview</h3>

              {runId && (
                <div className="receipt-field">
                  <span className="receipt-label">Run ID</span>
                  <code className="receipt-value">{runId}</code>
                </div>
              )}

              {events.length > 0 && (
                <div className="receipt-field">
                  <span className="receipt-label">Events</span>
                  <span className="receipt-value">{events.length}</span>
                </div>
              )}

              {verifyResult && (
                <div className="verify-result">
                  <h4>Verification Result</h4>
                  <div className={`verify-status ${verifyResult.status}`}>
                    {verifyResult.status === 'verified' && '✓ Verified'}
                    {verifyResult.status === 'rejected' && '✗ Rejected'}
                    {verifyResult.status === 'error' && '⚠ Error'}
                  </div>

                  {verifyResult.layers && (
                    <div className="verify-layers">
                      <LayerRow label="L0 Hash" status={verifyResult.layers.l0_hash?.status} />
                      <LayerRow label="L1 Commitment" status={verifyResult.layers.l1_commitment?.status} />
                      <LayerRow label="L2 Constraint" status={verifyResult.layers.l2_constraint?.status} />
                      <LayerRow label="L3 Proximity" status={verifyResult.layers.l3_proximity?.status} />
                      <LayerRow label="L4 Receipt" status={verifyResult.layers.l4_receipt?.status} />
                    </div>
                  )}

                  {verifyResult.errors?.length > 0 && (
                    <div className="verify-errors">
                      {verifyResult.errors.map((err, i) => (
                        <div key={i} className="verify-error">
                          <code>{err.code}</code>: {err.message}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * EventRow - Single event in the stream
 */
function EventRow({ event }) {
  const typeColors = {
    run_started: 'var(--accent-info)',
    run_completed: 'var(--accent-success)',
    run_failed: 'var(--accent-error)',
    capsule_sealed: 'var(--accent-success)',
    oracle_call: 'var(--accent-warning)',
    connected: 'var(--text-secondary)',
    error: 'var(--accent-error)',
    trace_simulated: 'var(--accent-info)',
    statement_locked: 'var(--accent-info)',
    proof_generated: 'var(--accent-success)',
  }

  // Backend returns 'type', adapter may also add 'event_type' for compatibility
  const eventType = event.type || event.event_type || 'unknown'
  const color = typeColors[eventType] || 'var(--text-primary)'
  const eventHash = event.hash || event.event_hash

  return (
    <div className="event-row">
      <span className="event-seq">{event.seq || '-'}</span>
      <span className="event-type" style={{ color }}>{eventType}</span>
      {eventHash && (
        <span className="event-hash">{eventHash.slice(0, 8)}</span>
      )}
      {event.data?.message && (
        <span className="event-message">{event.data.message}</span>
      )}
    </div>
  )
}

/**
 * LayerRow - L0-L4 verification layer status
 */
function LayerRow({ label, status }) {
  const statusIcon = {
    pass: '✓',
    fail: '✗',
    skipped: '○',
    unknown: '?',
  }

  const statusColor = {
    pass: 'var(--accent-success)',
    fail: 'var(--accent-error)',
    skipped: 'var(--text-secondary)',
    unknown: 'var(--text-secondary)',
  }

  return (
    <div className="layer-row">
      <span className="layer-label">{label}</span>
      <span className="layer-status" style={{ color: statusColor[status] || statusColor.unknown }}>
        {statusIcon[status] || statusIcon.unknown} {status || 'unknown'}
      </span>
    </div>
  )
}
