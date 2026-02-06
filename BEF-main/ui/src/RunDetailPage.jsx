import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  normalizeVerifyReport,
  normalizeJobStatus,
  JobStatus,
  getStatusClass,
  isLayerOk,
} from './engine/impl/flask.js'

const DEFAULT_API = 'https://capsuletech.rochmanofenna.workers.dev'
const API = window.API_BASE || import.meta.env.VITE_API_BASE || DEFAULT_API

function computeMaxSeq(list, current = 0) {
  let m = current
  for (const evt of list || []) {
    const seq = Number(evt.seq || evt.data?.seq || 0)
    if (seq > m) m = seq
  }
  return m
}

function summarize(events) {
  const map = {}
  for (const evt of events || []) {
    map[evt.type] = evt
  }
  const spec = map.spec_locked?.data || {}
  const trace = map.trace_simulated?.data || {}
  const statement = map.statement_locked?.data || {}
  const rowRoot = map.row_root_finalized?.data || {}
  const seal = map.capsule_sealed?.data || {}
  return {
    policyHash: spec.policy_hash,
    traceSpecHash: spec.trace_spec_hash || trace.trace_spec_hash,
    traceRoot: statement.trace_root || rowRoot.trace_root,
    steps: trace.steps,
    challenges: trace.num_challenges,
    capsuleHash: seal.capsule_hash,
    timestamp: map.run_completed?.ts_ms || map.run_started?.ts_ms,
  }
}

function formatBytes(bytes) {
  if (typeof bytes !== 'number') return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function StatusBadge({ status }) {
  const normalized = normalizeJobStatus(status)
  const cssClass = getStatusClass(normalized)
  return <span className={`status-badge ${cssClass}`}>{normalized}</span>
}

function EvidencePanel({ verification }) {
  // Normalize verification report to { ok, layers[], status, errorCode }
  const normalized = useMemo(
    () => normalizeVerifyReport(verification),
    [verification]
  )

  if (!verification) {
    return (
      <div className="evidence-panel">
        <h3>Evidence</h3>
        <p>No verification run yet.</p>
      </div>
    )
  }

  const { ok, layers, status, errorCode } = normalized
  const renderCheck = (value) => (value === true ? '✅' : value === false ? '❌' : '—')

  // Group layers by UI category for display
  // Contract format uses: binding, commitment, policy, proof, receipt
  // Legacy format uses: binding, availability, policy, proof, etc.
  const groupLayers = (layers) => {
    const groups = {
      binding: [], // Hash + commitment checks
      verification: [], // Policy + proof checks
      other: [],
    }

    for (const layer of layers) {
      const uiGroup = layer.meta?.uiGroup
      if (uiGroup === 'binding' || layer.id === 'binding' || layer.id === 'commitment') {
        groups.binding.push(layer)
      } else if (uiGroup === 'policy' || uiGroup === 'proof' || layer.id === 'policy' || layer.id === 'proof') {
        groups.verification.push(layer)
      } else {
        groups.other.push(layer)
      }
    }
    return groups
  }

  const grouped = groupLayers(layers)
  const hasBinding = grouped.binding.length > 0
  const hasVerification = grouped.verification.length > 0

  // Compute aggregate status for UI groups
  const bindingOk = grouped.binding.length > 0 && grouped.binding.every((l) => l.ok)
  const verificationOk = grouped.verification.length > 0 && grouped.verification.every((l) => l.ok)

  return (
    <div className="evidence-panel">
      <h3>Evidence</h3>
      <div className="evidence-summary">
        <span className={`overall-status ${ok ? 'ok' : 'fail'}`}>
          {ok ? 'Verified' : status === JobStatus.VERIFYING ? 'Verifying...' : 'Not Verified'}
        </span>
      </div>
      <div className="grid evidence-grid">
        {/* Show grouped binding status */}
        <div className="evidence-item">
          <strong>Binding</strong> {renderCheck(hasBinding ? bindingOk : null)}
        </div>
        {/* Show grouped verification status */}
        <div className="evidence-item">
          <strong>Verification</strong> {renderCheck(hasVerification ? verificationOk : null)}
        </div>
        {/* Show availability if present (legacy format) */}
        {layers.find((l) => l.id === 'availability') && (
          <div className="evidence-item">
            <strong>Availability</strong> {renderCheck(layers.find((l) => l.id === 'availability')?.ok ?? null)}
          </div>
        )}
      </div>
      {/* Show all individual layers for detail */}
      {layers.length > 0 && (
        <div className="grid evidence-grid evidence-secondary">
          {layers.map((layer) => (
            <div key={layer.id} className="evidence-item">
              <strong>{layer.label}</strong> {renderCheck(layer.ok)}
              {layer.reason && <small className="layer-reason"> ({layer.reason})</small>}
            </div>
          ))}
        </div>
      )}
      {errorCode && (
        <div className="reason-code">
          <strong>Reason:</strong> <code>{errorCode}</code>
        </div>
      )}
    </div>
  )
}

function ArtifactList({ runId, artifacts }) {
  if (!artifacts || !artifacts.length) return null
  return (
    <div className="artifact-list">
      <h3>Artifacts</h3>
      <div className="artifact-grid">
        {artifacts.map((artifact) => (
          <div key={artifact.name} className="artifact-card">
            <div>
              <div className="artifact-name">{artifact.name}</div>
              <small>{formatBytes(artifact.size_bytes)}</small>
            </div>
            <a
              href={`${API}/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.name)}`}
              target="_blank"
              rel="noreferrer"
            >
              Download
            </a>
          </div>
        ))}
      </div>
    </div>
  )
}

function EventLog({ events, bottomRef }) {
  return (
    <div className="event-log" id="event-log">
      {events.map((evt) => (
        <div key={`${evt.seq}-${evt.type}`} className="event-card">
          <div className="event-card-head">
            <span className="type">{evt.type}</span>
            <span className="seq">#{evt.seq}</span>
          </div>
          <div className="event-card-meta">
            {evt.ts_ms ? new Date(evt.ts_ms).toLocaleString() : ''}
          </div>
          <pre>{JSON.stringify(evt.data, null, 2)}</pre>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

export default function RunDetailPage() {
  const { runId } = useParams()
  const [meta, setMeta] = useState(null)
  const [events, setEvents] = useState([])
  const [artifacts, setArtifacts] = useState([])
  const [status, setStatus] = useState('loading')
  const [verification, setVerification] = useState(null)
  const [verifyMessage, setVerifyMessage] = useState('')
  const bottomRef = useRef(null)
  const lastSeqRef = useRef(0)
  const timerRef = useRef(null)

  useEffect(() => () => timerRef.current && clearInterval(timerRef.current), [])

  const fetchRunDetail = useCallback(async () => {
    setStatus('loading')
    try {
      const resp = await fetch(`${API}/api/runs/${encodeURIComponent(runId)}`)
      if (!resp.ok) throw new Error(`Run not found (${resp.status})`)
      const data = await resp.json()
      const runMeta = data.run || data
      setMeta(runMeta)
      setArtifacts(data.artifacts || [])
      const initialEvents = data.events || []
      setEvents(initialEvents)
      setVerification(data.verification || null)
      lastSeqRef.current = data.last_seq || computeMaxSeq(initialEvents)
      setStatus('live')
    } catch (err) {
      setStatus(`error: ${err.message}`)
    }
  }, [runId])

  useEffect(() => {
    async function poll() {
      try {
        const resp = await fetch(
          `${API}/api/runs/${encodeURIComponent(runId)}/events?after_seq=${lastSeqRef.current}`
        )
        if (!resp.ok) return
        const data = await resp.json()
        const newEvents = Array.isArray(data.events) ? data.events : Array.isArray(data) ? data : []
        if (!newEvents.length) return
        lastSeqRef.current = computeMaxSeq(newEvents, lastSeqRef.current)
        setEvents((prev) => [...prev, ...newEvents])
      } catch (err) {
        setStatus(`poll error: ${err.message}`)
      }
    }

    fetchRunDetail()
      .then(() => {
        if (timerRef.current) clearInterval(timerRef.current)
        timerRef.current = setInterval(poll, 1000)
      })
    return () => timerRef.current && clearInterval(timerRef.current)
  }, [runId, fetchRunDetail])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  const summary = useMemo(() => summarize(events), [events])

  async function handleVerify() {
    setVerifyMessage('starting verification...')
    try {
      const resp = await fetch(`${API}/api/runs/${encodeURIComponent(runId)}/verify`, { method: 'POST' })
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok) throw new Error(data.detail || data.error || 'verification failed')
      setVerifyMessage('verification queued; refresh shortly')
      setTimeout(() => {
        fetchRunDetail().catch(() => {})
      }, 2000)
    } catch (err) {
      setVerifyMessage(err.message)
    }
  }

  return (
    <section className="details">
      <div className="section-header">
        <div>
          <h2>{runId}</h2>
          <Link to="/">Back to runs</Link>
        </div>
        <StatusBadge status={meta?.verification_status} />
      </div>
      <div className="receipt-panel">
        <div>
          <p className="eyebrow">Policy-Bound Compute Receipt</p>
          <div className="grid meta-grid">
            <div><strong>Backend:</strong> {meta?.backend || '—'}</div>
            <div><strong>Policy:</strong> {meta?.policy_id || '—'}</div>
            <div><strong>Track:</strong> {meta?.track_id || '—'}</div>
            <div><strong>Created:</strong> {meta?.created_at ? new Date(meta.created_at).toLocaleString() : '—'}</div>
            <div><strong>Steps:</strong> {summary.steps || '—'}</div>
            <div><strong>Challenges:</strong> {summary.challenges || '—'}</div>
            <div><strong>Policy Hash:</strong> <code>{summary.policyHash || '—'}</code></div>
            <div><strong>Trace Spec Hash:</strong> <code>{summary.traceSpecHash || '—'}</code></div>
            <div><strong>Capsule Hash:</strong> <code>{summary.capsuleHash || '—'}</code></div>
          </div>
        </div>
        <div className="actions">
          <button onClick={handleVerify}>Verify</button>
          {verifyMessage && <small>{verifyMessage}</small>}
        </div>
      </div>

      <EvidencePanel verification={verification} />
      <ArtifactList runId={runId} artifacts={artifacts} />
      <div className="events-header">
        <h3>Live Event Stream</h3>
        <span>{status}</span>
      </div>
      <EventLog events={events} bottomRef={bottomRef} />
    </section>
  )
}
