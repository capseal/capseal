import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SkeletonCard, EmptyState, SearchInput } from './index'
import MonoBlock from './MonoBlock'
import { getEngine } from '../engine'

function formatRelativeTime(dateStr) {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return new Date(dateStr).toLocaleDateString()
}

function truncateHash(hash, len = 12) {
  if (!hash) return 'N/A'
  return hash.length > len ? `${hash.slice(0, len)}...` : hash
}

function CircuitListItem({ circuit, isSelected, onClick }) {
  return (
    <button
      className={`circuit-list-item ${isSelected ? 'circuit-list-selected' : ''}`}
      onClick={onClick}
    >
      <div className="circuit-list-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="4" y="4" width="6" height="6" rx="1" />
          <rect x="14" y="4" width="6" height="6" rx="1" />
          <rect x="4" y="14" width="6" height="6" rx="1" />
          <rect x="14" y="14" width="6" height="6" rx="1" />
          <path d="M10 7h4M10 17h4M7 10v4M17 10v4" />
        </svg>
      </div>
      <div className="circuit-list-content">
        <div className="circuit-list-title">{circuit.name}</div>
        <div className="circuit-list-meta">{circuit.runCount} runs</div>
      </div>
    </button>
  )
}

function CircuitEvent({ event, isExpanded, onToggle }) {
  const [showDetails, setShowDetails] = useState(false)

  const getEventColor = (run) => {
    if (run.verification_status === 'verified') return 'verified'
    if (run.verification_status === 'failed') return 'failed'
    return 'default'
  }

  return (
    <div className="circuit-event-node">
      <div className="circuit-event-marker" onClick={onToggle} style={{ cursor: 'pointer' }}>
        <div className={`circuit-event-dot ${getEventColor(event)}`} />
        <div className="circuit-event-line" />
      </div>
      <div className="circuit-event-content">
        <button
          className="circuit-event-header"
          onClick={onToggle}
        >
          <div className="circuit-event-title">
            <span className="circuit-event-name">{event.name || 'Run'}</span>
            <span className={`circuit-event-badge ${getEventColor(event)}`}>
              {event.verification_status || 'pending'}
            </span>
          </div>
          <span className="circuit-event-time">
            {event.created_at ? new Date(event.created_at).toLocaleString() : 'N/A'}
          </span>
          <svg
            className={`circuit-event-chevron ${isExpanded ? 'expanded' : ''}`}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>

        {isExpanded && (
          <div className="circuit-event-details">
            {event.policy_hash && (
              <div className="circuit-event-detail-row">
                <span className="circuit-event-label">Policy Hash:</span>
                <span className="circuit-event-value" title={event.policy_hash}>
                  {truncateHash(event.policy_hash)}
                </span>
              </div>
            )}
            {event.policy_id && (
              <div className="circuit-event-detail-row">
                <span className="circuit-event-label">Policy:</span>
                <span className="circuit-event-value">{event.policy_id}</span>
              </div>
            )}
            <button
              className="circuit-event-raw-toggle"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? 'Hide raw' : 'Show raw'}
            </button>
            {showDetails && <MonoBlock data={event} maxHeight={200} />}
          </div>
        )}
      </div>
    </div>
  )
}

function CircuitGraph({ runs }) {
  // Placeholder for graph view - would be a visual circuit representation
  return (
    <div className="circuit-graph-placeholder">
      <p>Graph view coming soon</p>
      <p className="text-muted">Timeline view is optimized for understanding circuit execution order</p>
    </div>
  )
}

export default function CircuitsPage() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState([])
  const [status, setStatus] = useState('loading')
  const [viewMode, setViewMode] = useState('timeline')
  const [selectedCircuit, setSelectedCircuit] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedRuns, setExpandedRuns] = useState(new Set())

  const loadRuns = useCallback(async () => {
    setStatus('loading')
    try {
      const engine = getEngine()
      const array = await engine.listRuns()
      setRuns(array)
      setStatus('ready')
    } catch (err) {
      setStatus(`error: ${err.message}`)
    }
  }, [])

  useEffect(() => {
    loadRuns()
  }, [loadRuns])

  // Derive unique circuits from runs based on policy_id
  const circuits = useMemo(() => {
    const circuitMap = new Map()

    runs.forEach((run) => {
      const policyId = run.policy_id || 'no_policy'
      const circuitName = policyId === 'no_policy' ? 'No policy' : policyId

      if (!circuitMap.has(policyId)) {
        circuitMap.set(policyId, {
          id: policyId,
          name: circuitName,
          hash: run.policy_hash || null,
          runCount: 0,
          lastUsed: null,
          runs: [],
        })
      }

      const circuit = circuitMap.get(policyId)
      circuit.runCount += 1
      circuit.runs.push(run)

      if (run.policy_hash) {
        circuit.hash = run.policy_hash
      }

      const runTime = run.created_at ? new Date(run.created_at).getTime() : 0
      const lastUsedTime = circuit.lastUsed ? new Date(circuit.lastUsed).getTime() : 0
      if (runTime > lastUsedTime) {
        circuit.lastUsed = run.created_at
      }
    })

    return Array.from(circuitMap.values()).sort((a, b) => {
      const timeA = a.lastUsed ? new Date(a.lastUsed).getTime() : 0
      const timeB = b.lastUsed ? new Date(b.lastUsed).getTime() : 0
      return timeB - timeA
    })
  }, [runs])

  // Filter circuits by search query
  const filteredCircuits = useMemo(() => {
    if (!searchQuery) return circuits
    return circuits.filter((c) =>
      c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.hash?.toLowerCase().includes(searchQuery.toLowerCase())
    )
  }, [circuits, searchQuery])

  // Get selected circuit or first one
  const currentCircuit = useMemo(() => {
    if (selectedCircuit) {
      return circuits.find((c) => c.id === selectedCircuit)
    }
    return circuits[0] || null
  }, [selectedCircuit, circuits])

  // Timeline events for current circuit
  const timelineEvents = useMemo(() => {
    if (!currentCircuit) return []
    // Sort runs by created_at descending (most recent first)
    return [...currentCircuit.runs].sort((a, b) => {
      const timeA = a.created_at ? new Date(a.created_at).getTime() : 0
      const timeB = b.created_at ? new Date(b.created_at).getTime() : 0
      return timeB - timeA
    })
  }, [currentCircuit])

  const toggleRunExpanded = (runId) => {
    const newExpanded = new Set(expandedRuns)
    if (newExpanded.has(runId)) {
      newExpanded.delete(runId)
    } else {
      newExpanded.add(runId)
    }
    setExpandedRuns(newExpanded)
  }

  const handleCircuitClick = (circuit) => {
    if (circuit.id === 'no_policy') {
      navigate('/runs?policy_id=')
    } else {
      navigate(`/runs?policy_id=${encodeURIComponent(circuit.id)}`)
    }
  }

  return (
    <div className="circuit-container">
      {/* Header */}
      <div className="circuit-header">
        <div className="circuit-header-left">
          <h1>Circuits</h1>
          <p className="circuit-header-subtitle">Timeline-first view of policy execution</p>
        </div>
        <div className="circuit-header-actions">
          <button
            className={`circuit-graph-toggle ${viewMode === 'timeline' ? 'active' : ''}`}
            onClick={() => setViewMode('timeline')}
            title="Timeline view"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
            Timeline
          </button>
          <button
            className={`circuit-graph-toggle ${viewMode === 'graph' ? 'active' : ''}`}
            onClick={() => setViewMode('graph')}
            title="Graph view (experimental)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="6" cy="6" r="2" />
              <circle cx="18" cy="6" r="2" />
              <circle cx="12" cy="18" r="2" />
              <line x1="8" y1="7" x2="10" y2="17" />
              <line x1="16" y1="7" x2="14" y2="17" />
            </svg>
            Graph
          </button>
        </div>
      </div>

      {status === 'loading' && (
        <div className="circuit-main">
          <SkeletonCard />
        </div>
      )}

      {status.startsWith('error') && (
        <div className="circuit-main">
          <EmptyState
            title="Failed to load circuits"
            description={status}
            action="Retry"
            onAction={loadRuns}
          />
        </div>
      )}

      {status === 'ready' && circuits.length === 0 && (
        <div className="circuit-main">
          <EmptyState
            title="No circuits yet"
            description="Run your first capsule with a policy to create a circuit. Circuits are derived from run policies."
          />
        </div>
      )}

      {status === 'ready' && circuits.length > 0 && (
        <div className="circuit-main">
          {/* Left Panel - Circuit List */}
          <div className="circuit-panel circuit-left">
            <div className="circuit-panel-header">
              <h2>Circuits</h2>
            </div>
            <SearchInput
              placeholder="Search circuits..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="circuit-list">
              {filteredCircuits.map((circuit) => (
                <CircuitListItem
                  key={circuit.id}
                  circuit={circuit}
                  isSelected={currentCircuit?.id === circuit.id}
                  onClick={() => setSelectedCircuit(circuit.id)}
                />
              ))}
            </div>
          </div>

          {/* Center Panel - Timeline or Graph */}
          <div className="circuit-panel circuit-center">
            {viewMode === 'timeline' && (
              <>
                <div className="circuit-panel-header">
                  <h2>{currentCircuit?.name}</h2>
                  {currentCircuit?.hash && (
                    <span
                      className="circuit-panel-hash"
                      title={currentCircuit.hash}
                    >
                      {truncateHash(currentCircuit.hash)}
                    </span>
                  )}
                </div>

                {timelineEvents.length === 0 ? (
                  <div className="circuit-empty">
                    <p>No events in this circuit</p>
                  </div>
                ) : (
                  <div className="circuit-timeline">
                    {timelineEvents.map((event, idx) => (
                      <CircuitEvent
                        key={event.id || idx}
                        event={event}
                        isExpanded={expandedRuns.has(event.id)}
                        onToggle={() => toggleRunExpanded(event.id)}
                      />
                    ))}
                  </div>
                )}
              </>
            )}

            {viewMode === 'graph' && (
              <>
                <div className="circuit-panel-header">
                  <h2>{currentCircuit?.name}</h2>
                </div>
                <CircuitGraph runs={timelineEvents} />
              </>
            )}
          </div>

          {/* Right Panel - Receipt Inspector Preview */}
          <div className="circuit-panel circuit-right">
            <div className="circuit-panel-header">
              <h2>Inspector</h2>
            </div>
            {currentCircuit ? (
              <div className="circuit-inspector">
                <div className="circuit-inspector-row">
                  <span className="circuit-inspector-label">Name</span>
                  <span className="circuit-inspector-value">{currentCircuit.name}</span>
                </div>
                <div className="circuit-inspector-row">
                  <span className="circuit-inspector-label">Runs</span>
                  <span className="circuit-inspector-value">{currentCircuit.runCount}</span>
                </div>
                <div className="circuit-inspector-row">
                  <span className="circuit-inspector-label">Last used</span>
                  <span className="circuit-inspector-value">
                    {formatRelativeTime(currentCircuit.lastUsed)}
                  </span>
                </div>
                {currentCircuit.hash && (
                  <div className="circuit-inspector-row">
                    <span className="circuit-inspector-label">Hash</span>
                    <span
                      className="circuit-inspector-value circuit-inspector-mono"
                      title={currentCircuit.hash}
                    >
                      {truncateHash(currentCircuit.hash, 16)}
                    </span>
                  </div>
                )}
                <button
                  className="circuit-inspector-action"
                  onClick={() => handleCircuitClick(currentCircuit)}
                >
                  View Runs
                </button>
              </div>
            ) : (
              <p className="text-muted">Select a circuit to inspect</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
