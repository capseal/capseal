import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  StatusPill,
  CopyField,
  OverviewCard,
  EvidenceStrip,
  Timeline,
  SegmentedControl,
  Drawer,
  Toast,
  SkeletonOverviewCard,
  MonoBlock,
  RunHeader,
} from './components'
import { getEngine } from './engine'
import { useEngine } from './state/EngineContext.jsx'

// CLI-aligned tabs: Inspect | Verify | Audit | Evidence | Export
const TABS = [
  { value: 'inspect', label: 'Inspect' },
  { value: 'verify', label: 'Verify' },
  { value: 'audit', label: 'Audit' },
  { value: 'evidence', label: 'Evidence' },
  { value: 'export', label: 'Export' },
]

// Artifact mappings for Export tab
const ARTIFACT_MAP = {
  'strategy_capsule.json': { label: 'Receipt', description: 'Capsule receipt JSON' },
  'capsule.json': { label: 'Receipt', description: 'Capsule receipt JSON' },
  'adapter_proof.bin': { label: 'Proof', description: 'Binary proof data' },
  'proof.json': { label: 'Proof', description: 'Proof JSON' },
  'events.jsonl': { label: 'Audit Log', description: 'Event audit trail' },
  'stc_trace.json': { label: 'Trace', description: 'STC trace data' },
  'trace.bin': { label: 'Trace', description: 'Binary trace data' },
  'manifest.json': { label: 'Manifest', description: 'Capsule manifest' },
  'row_archive/': { label: 'Row Data', description: 'Row archive chunks' },
}

// Verification level definitions for L0-L4 checklist
// Maps to both contract format (l0_hash, etc.) and legacy format (hash_ok, etc.)
const VERIFICATION_LEVELS = [
  { level: 'L0', name: 'Hash', description: 'Artifact hashes match', key: 'hash_ok', contractKey: 'l0_hash' },
  { level: 'L1', name: 'Commitment', description: 'Merkle roots valid', key: 'commitment_ok', contractKey: 'l1_commitment' },
  { level: 'L2', name: 'Constraint', description: 'Policy enforced', key: 'policy_ok', contractKey: 'l2_constraint' },
  { level: 'L3', name: 'Proximity', description: 'FRI low-degree test passed', key: 'fri_ok', contractKey: 'l3_proximity' },
  { level: 'L4', name: 'Receipt', description: 'Capsule format valid', key: 'receipt_ok', contractKey: 'l4_receipt' },
]

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
    format: seal.format || 'capsule-v1',
    schema: spec.schema_version || 'v1',
    profile: spec.profile || 'default',
  }
}

function formatBytes(bytes) {
  if (typeof bytes !== 'number') return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTime(dateStr) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

function formatMs(ms) {
  if (typeof ms !== 'number') return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function truncateHash(hash, len = 8) {
  if (!hash) return '-'
  if (hash.length <= len * 2) return hash
  return `${hash.slice(0, len)}...${hash.slice(-len)}`
}

// =============================================================================
// Icons
// =============================================================================
function CheckIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function XIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

function QuestionIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  )
}

function ReproducePanel({ engineStatus, engineBaseUrl, projectId, runId, backend, policyId, command, onCopy }) {
  const fallbackCommand = `capseal verify ${runId}`
  return (
    <section className="reproduce-panel">
      <div className="reproduce-head">
        <div>
          <p className="eyebrow">Reproduce</p>
          <h3>Engine handshake</h3>
        </div>
        <button className="btn-secondary" onClick={onCopy}>
          Copy command
        </button>
      </div>
      <div className="reproduce-grid">
        <div>
          <span className="label">Engine</span>
          <div className="run-header-engine">
            <span className={`engine-dot status-${engineStatus}`} />
            <span>{engineBaseUrl || 'Not configured'}</span>
          </div>
        </div>
        <div>
          <span className="label">Project</span>
          <div className="value mono">{projectId || '—'}</div>
        </div>
        <div>
          <span className="label">Policy</span>
          <div className="value mono">{policyId || '—'}</div>
        </div>
        <div>
          <span className="label">Backend</span>
          <div className="value mono">{backend || '—'}</div>
        </div>
      </div>
      <div className="reproduce-command">
        <span className="label">CLI command</span>
        <CopyField value={command || fallbackCommand} truncate={false} maxLength={64} />
      </div>
    </section>
  )
}

function MinusIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  )
}

function LinkIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  )
}

function DownloadIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  )
}

function FileIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  )
}

// =============================================================================
// Inspect Tab (maps to `capseal inspect`)
// =============================================================================
function InspectSection({ meta, summary, verification, onEvidenceClick, artifacts }) {
  const totalSize = artifacts?.reduce((sum, a) => sum + (a.size_bytes || 0), 0) || 0

  return (
    <>
      {/* Capsule Info Grid */}
      <div className="inspect-info-grid">
        <div className="inspect-info-item">
          <span className="inspect-label">Format</span>
          <span className="inspect-value">{summary.format || 'capsule-v1'}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">Capsule ID</span>
          <span className="inspect-value mono">{truncateHash(meta?.run_id, 12)}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">Trace ID</span>
          <span className="inspect-value mono">{truncateHash(meta?.track_id, 12)}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">Backend</span>
          <span className="inspect-value">{meta?.backend || '-'}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">Profile</span>
          <span className="inspect-value">{summary.profile || 'default'}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">Schema</span>
          <span className="inspect-value">{summary.schema || 'v1'}</span>
        </div>
        <div className="inspect-info-item">
          <span className="inspect-label">File Size</span>
          <span className="inspect-value">{formatBytes(totalSize)}</span>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="overview-grid">
        <OverviewCard
          title="Integrity"
          items={[
            { label: 'Capsule Hash', value: summary.capsuleHash, mono: true },
            { label: 'Policy Hash', value: summary.policyHash, mono: true },
            { label: 'Trace Spec', value: summary.traceSpecHash, mono: true },
          ]}
        />
        <OverviewCard
          title="Verification"
          items={[
            { label: 'Status', value: meta?.verification_status || 'Unverified' },
            {
              label: 'Last Verified',
              value: verification?.verified_at
                ? formatTime(verification.verified_at)
                : 'Never',
            },
            { label: 'Verifier', value: verification?.verifier_version || '-' },
          ]}
        />
        <OverviewCard
          title="Execution"
          items={[
            { label: 'Steps', value: summary.steps || '-' },
            { label: 'Challenges', value: summary.challenges || '-' },
            { label: 'Track ID', value: meta?.track_id, mono: true },
          ]}
        />
      </div>

      {/* Evidence Strip */}
      <EvidenceStrip
        verification={verification}
        onItemClick={onEvidenceClick}
      />
    </>
  )
}

// =============================================================================
// Verify Tab (maps to `capseal verify`)
// =============================================================================
function VerifySection({ meta, verification, onVerify, verifying }) {
  const status = meta?.verification_status?.toUpperCase() || 'UNVERIFIED'
  const isVerified = status === 'VERIFIED'
  const isRejected = status === 'REJECTED' || status === 'FAILED'

  // Extract verification levels from report
  // Supports both contract format (layers object) and legacy format (flat report fields)
  const getLevelStatus = (key, contractKey) => {
    if (!verification) return 'unknown'

    // Check contract format first: verification.layers.l0_hash, etc.
    const layers = verification.layers
    if (layers && typeof layers === 'object' && !Array.isArray(layers) && contractKey) {
      const layerData = layers[contractKey]
      if (layerData) {
        const layerStatus = layerData.status
        if (layerStatus === 'pass') return 'pass'
        if (layerStatus === 'fail') return 'fail'
        return 'unknown'
      }
    }

    // Fall back to legacy format: verification.report with flat boolean fields
    const report = verification?.report || {}
    const keyMappings = {
      hash_ok: report.hash_ok ?? report.artifact_hashes_ok ?? report.hash_verified,
      commitment_ok: report.commitment_ok ?? report.merkle_ok ?? report.row_index_commitment_ok,
      policy_ok: report.policy_ok ?? report.policy_verified ?? report.enforcement_ok,
      fri_ok: report.fri_ok ?? report.proximity_ok ?? report.low_degree_ok,
      receipt_ok: report.receipt_ok ?? report.format_ok ?? report.capsule_valid,
    }
    const value = keyMappings[key]
    if (value === true) return 'pass'
    if (value === false) return 'fail'
    return 'unknown'
  }

  return (
    <div className="verify-section">
      {/* Status Banner */}
      <div className={`verify-status-banner ${isVerified ? 'verified' : isRejected ? 'rejected' : 'unverified'}`}>
        <div className="verify-status-icon">
          {isVerified ? <CheckIcon /> : isRejected ? <XIcon /> : <QuestionIcon />}
        </div>
        <div className="verify-status-text">
          <span className="verify-status-label">{status}</span>
          {(verification?.errors?.[0]?.code || verification?.error_code) && (
            <span className="verify-error-code">{verification?.errors?.[0]?.code || verification.error_code}</span>
          )}
        </div>
      </div>

      {/* Verification Stats */}
      <div className="verify-stats">
        <div className="verify-stat">
          <span className="verify-stat-label">Proof Size</span>
          <span className="verify-stat-value">{formatBytes(verification?.proof_size_bytes ?? verification?.proof_size)}</span>
        </div>
        <div className="verify-stat">
          <span className="verify-stat-label">Verify Time</span>
          <span className="verify-stat-value">{formatMs(verification?.timings?.total_ms ?? verification?.verify_time_ms)}</span>
        </div>
        <div className="verify-stat">
          <span className="verify-stat-label">Backend</span>
          <span className="verify-stat-value">{verification?.backend_id ?? meta?.backend ?? '-'}</span>
        </div>
      </div>

      {/* L0-L4 Checklist */}
      <div className="verify-checklist">
        <h4 className="verify-checklist-title">Verification Levels</h4>
        <div className="verify-levels">
          {VERIFICATION_LEVELS.map(({ level, name, description, key, contractKey }) => {
            const levelStatus = getLevelStatus(key, contractKey)
            return (
              <div key={level} className={`verify-level verify-level-${levelStatus}`}>
                <div className="verify-level-indicator">
                  {levelStatus === 'pass' ? <CheckIcon size={14} /> :
                   levelStatus === 'fail' ? <XIcon size={14} /> :
                   <MinusIcon size={14} />}
                </div>
                <div className="verify-level-info">
                  <span className="verify-level-name">{level} {name}</span>
                  <span className="verify-level-desc">{description}</span>
                </div>
                <span className={`verify-level-status ${levelStatus}`}>
                  {levelStatus === 'pass' ? 'PASS' : levelStatus === 'fail' ? 'FAIL' : 'N/A'}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Run Verify Button */}
      <div className="verify-action">
        <button
          className="btn btn-primary btn-lg"
          onClick={onVerify}
          disabled={verifying}
        >
          {verifying ? (
            <>
              <span className="btn-spinner" />
              Running Verification...
            </>
          ) : (
            'Run Verify'
          )}
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// Audit Tab (maps to `capseal audit`)
// =============================================================================
function AuditSection({ events }) {
  const sortedEvents = [...(events || [])].sort((a, b) => (a.seq || 0) - (b.seq || 0))

  const eventCounts = {}
  for (const evt of events || []) {
    eventCounts[evt.type] = (eventCounts[evt.type] || 0) + 1
  }

  // Validate hash chain
  const chainValid = useMemo(() => {
    if (!sortedEvents.length) return null
    for (let i = 1; i < sortedEvents.length; i++) {
      const prev = sortedEvents[i - 1]
      const curr = sortedEvents[i]
      if (curr.prev_hash && prev.hash && curr.prev_hash !== prev.hash) {
        return false
      }
    }
    return true
  }, [sortedEvents])

  return (
    <div className="audit-section">
      {/* Hash Chain Status */}
      <div className={`audit-chain-status ${chainValid === true ? 'valid' : chainValid === false ? 'invalid' : 'unknown'}`}>
        <div className="audit-chain-icon">
          {chainValid === true ? <CheckIcon /> : chainValid === false ? <XIcon /> : <LinkIcon />}
        </div>
        <div className="audit-chain-info">
          <span className="audit-chain-label">Hash Chain</span>
          <span className="audit-chain-value">
            {chainValid === true ? 'VALID' : chainValid === false ? 'INVALID' : 'N/A'}
          </span>
        </div>
      </div>

      {/* Event Type Counts */}
      <div className="audit-counts">
        <h4>Event Counts</h4>
        <div className="audit-counts-grid">
          {Object.entries(eventCounts).map(([type, count]) => (
            <div key={type} className="audit-count-item">
              <span className="audit-count-type">{type.replace(/_/g, ' ')}</span>
              <span className="audit-count-value">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Hash Chain Visualization */}
      <div className="audit-chain">
        <h4>Hash Chain</h4>
        <div className="audit-chain-list">
          {sortedEvents.map((evt, idx) => (
            <div key={`${evt.seq}-${evt.type}`} className="audit-chain-item">
              <div className="audit-chain-seq">{evt.seq ?? idx}</div>
              <div className="audit-chain-type">{evt.type}</div>
              <div className="audit-chain-hash mono">{truncateHash(evt.hash, 6)}</div>
              {evt.prev_hash && (
                <div className="audit-chain-link">
                  <LinkIcon size={12} />
                  <span className="mono">{truncateHash(evt.prev_hash, 6)}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Timeline */}
      <div className="audit-timeline">
        <h4>Timeline</h4>
        <Timeline events={events} showLiveIndicator={true} />
      </div>
    </div>
  )
}

// =============================================================================
// Evidence Tab (maps to `capseal row`)
// =============================================================================
function EvidenceSection({ runId, setToast }) {
  const [rowNumber, setRowNumber] = useState('')
  const [rowData, setRowData] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleOpenRow = async () => {
    if (!rowNumber) {
      setToast({ message: 'Please enter a row number', type: 'error' })
      return
    }

    setLoading(true)
    try {
      const engine = getEngine()
      const rowUrl = engine.getRowUrl(runId, rowNumber)
      const resp = await fetch(rowUrl)
      if (!resp.ok) {
        throw new Error(`Row not found (${resp.status})`)
      }
      const data = await resp.json()
      setRowData(data)
    } catch (err) {
      setToast({ message: err.message, type: 'error' })
      setRowData(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="evidence-section">
      {/* Row Input */}
      <div className="evidence-input-group">
        <label className="evidence-label">Open Row</label>
        <div className="evidence-input-row">
          <input
            type="number"
            className="evidence-input"
            placeholder="Enter row number (e.g., 0, 1, 2...)"
            value={rowNumber}
            onChange={(e) => setRowNumber(e.target.value)}
            min="0"
          />
          <button
            className="btn btn-primary"
            onClick={handleOpenRow}
            disabled={loading || !rowNumber}
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                Loading...
              </>
            ) : (
              'Open Row with Proof'
            )}
          </button>
        </div>
      </div>

      {/* Row Data Display */}
      {rowData && (
        <div className="evidence-row-data">
          <h4>Row {rowNumber} Data</h4>

          {/* Row Values */}
          {rowData.values && (
            <div className="evidence-values">
              <h5>Row Values</h5>
              <MonoBlock data={rowData.values} />
            </div>
          )}

          {/* Merkle Proof */}
          {rowData.proof && (
            <div className="evidence-proof">
              <h5>Merkle Proof Levels</h5>
              <div className="evidence-proof-levels">
                {(rowData.proof.path || rowData.proof.siblings || []).map((hash, idx) => (
                  <div key={idx} className="evidence-proof-level">
                    <span className="evidence-proof-idx">Level {idx}</span>
                    <span className="evidence-proof-hash mono">{truncateHash(hash, 12)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Proof Validity */}
          <div className={`evidence-validity ${rowData.valid ? 'valid' : rowData.valid === false ? 'invalid' : 'unknown'}`}>
            <span className="evidence-validity-label">Proof Validity</span>
            <span className="evidence-validity-value">
              {rowData.valid === true ? 'VALID' : rowData.valid === false ? 'INVALID' : 'UNKNOWN'}
            </span>
          </div>

          {/* Archive Chunk Info */}
          {rowData.chunk && (
            <div className="evidence-chunk">
              <h5>Archive Chunk Info</h5>
              <div className="evidence-chunk-info">
                <div className="evidence-chunk-item">
                  <span>Chunk Index</span>
                  <span>{rowData.chunk.index ?? '-'}</span>
                </div>
                <div className="evidence-chunk-item">
                  <span>Offset</span>
                  <span>{rowData.chunk.offset ?? '-'}</span>
                </div>
                <div className="evidence-chunk-item">
                  <span>Size</span>
                  <span>{formatBytes(rowData.chunk.size)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!rowData && !loading && (
        <div className="evidence-empty">
          <p>Enter a row number to view row data with Merkle proof</p>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Export Tab (maps to `capseal emit`)
// =============================================================================
function ExportSection({ runId, artifacts, meta }) {
  const engine = getEngine()
  const downloadUrl = (name) => engine.getArtifactUrl(runId, name)
  const capsuleDownloadUrl = engine.getCapsuleUrl(runId)

  // Categorize artifacts
  const categorized = artifacts.map(artifact => {
    const match = Object.entries(ARTIFACT_MAP).find(([key]) => artifact.name?.includes(key))
    return {
      ...artifact,
      category: match ? match[1] : { label: 'Other', description: 'Additional artifact' },
    }
  })

  return (
    <div className="export-section">
      {/* Primary Download Action */}
      <div className="export-primary">
        <div className="export-primary-info">
          <h4>Download Capsule Archive</h4>
          <p>Download the complete .cap archive containing all artifacts and proofs</p>
        </div>
        <a
          href={capsuleDownloadUrl}
          className="btn btn-primary btn-lg"
          download={`${runId}.cap`}
        >
          <DownloadIcon size={16} />
          Download .cap
        </a>
      </div>

      {/* Artifact List */}
      <div className="export-artifacts">
        <h4>Artifacts</h4>
        <div className="export-artifact-list">
          {categorized.map((artifact) => (
            <div key={artifact.name} className="export-artifact-item">
              <div className="export-artifact-icon">
                <FileIcon />
              </div>
              <div className="export-artifact-info">
                <span className="export-artifact-name">{artifact.name}</span>
                <span className="export-artifact-meta">
                  {artifact.category.label} - {formatBytes(artifact.size_bytes)}
                </span>
              </div>
              <a
                href={downloadUrl(artifact.name)}
                className="btn btn-secondary btn-sm"
                download
              >
                Download
              </a>
            </div>
          ))}
        </div>
      </div>

      {/* Artifact Legend */}
      <div className="export-legend">
        <h4>Artifact Reference</h4>
        <div className="export-legend-grid">
          <div className="export-legend-item">
            <span className="export-legend-name">strategy_capsule.json</span>
            <span className="export-legend-desc">Receipt - Capsule metadata and hashes</span>
          </div>
          <div className="export-legend-item">
            <span className="export-legend-name">adapter_proof.bin</span>
            <span className="export-legend-desc">Proof - Binary STARK proof data</span>
          </div>
          <div className="export-legend-item">
            <span className="export-legend-name">events.jsonl</span>
            <span className="export-legend-desc">Audit Log - Complete event trail</span>
          </div>
          <div className="export-legend-item">
            <span className="export-legend-name">stc_trace.json</span>
            <span className="export-legend-desc">Trace - Execution trace data</span>
          </div>
          <div className="export-legend-item">
            <span className="export-legend-name">row_archive/</span>
            <span className="export-legend-desc">Row Data - Indexed row chunks</span>
          </div>
        </div>
      </div>

      {artifacts.length === 0 && (
        <p className="export-empty">No artifacts available for export.</p>
      )}
    </div>
  )
}

// =============================================================================
// Main Component
// =============================================================================
export default function RunDetailPanel({ runId }) {
  const [meta, setMeta] = useState(null)
  const [events, setEvents] = useState([])
  const [artifacts, setArtifacts] = useState([])
  const [status, setStatus] = useState('loading')
  const [verification, setVerification] = useState(null)
  const [activeTab, setActiveTab] = useState('inspect')
  const [toast, setToast] = useState(null)
  const [evidenceDrawer, setEvidenceDrawer] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const lastSeqRef = useRef(0)
  const timerRef = useRef(null)
  const engineCtx = useEngine()
  const capabilities = engineCtx.health?.capabilities || {}
  const canVerify = capabilities.verify !== false
  const engineBaseUrl = engineCtx.baseUrl

  useEffect(() => () => timerRef.current && clearInterval(timerRef.current), [])

  const fetchRunDetail = useCallback(async () => {
    setStatus('loading')
    try {
      const engine = getEngine()
      const data = await engine.getRun(runId)
      setMeta(data)
      setArtifacts(data.artifacts || [])
      setEvents(data.events || [])
      setVerification(data.verification || null)
      lastSeqRef.current = data.events_root ? 0 : computeMaxSeq(data.events || [])
      setStatus('ready')
    } catch (err) {
      setStatus(`error: ${err.message}`)
    }
  }, [runId])

  useEffect(() => {
    let cleanup = null

    fetchRunDetail().then(() => {
      const engine = getEngine()
      cleanup = engine.streamEvents(
        runId,
        (event) => {
          lastSeqRef.current = Math.max(lastSeqRef.current, event.seq || 0)
          setEvents((prev) => [...prev, event])
        },
        (err) => {
          // Silent poll failure - just like before
        }
      )
    })

    return () => {
      if (cleanup) cleanup()
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [runId, fetchRunDetail])

  const summary = useMemo(() => summarize(events), [events])
  const projectId = meta?.track_id || meta?.project_id || 'default'
  const policyId = meta?.policy_id || meta?.policy
  const reproduceCommand = meta?.command || meta?.cli_command || meta?.capsule?.command

  const handleCopyReproduce = async () => {
    const text = reproduceCommand || `capseal verify ${runId}`
    try {
      await navigator.clipboard.writeText(text)
      setToast({ message: 'Command copied to clipboard', type: 'success' })
    } catch (err) {
      setToast({ message: 'Failed to copy command', type: 'error' })
    }
  }

  const handleVerify = async () => {
    setVerifying(true)
    setToast({ message: 'Starting verification...', type: 'info' })
    try {
      const engine = getEngine()
      const report = await engine.verify({ run_id: runId })
      setVerification({ report, ...report })
      setToast({
        message: report.ok ? 'Verification passed' : 'Verification completed',
        type: report.ok ? 'success' : 'info'
      })
      // Refresh run detail
      setTimeout(() => fetchRunDetail().catch(() => {}), 1000)
    } catch (err) {
      setToast({ message: err.message, type: 'error' })
    } finally {
      setVerifying(false)
    }
  }

  const handleReplay = async () => {
    setToast({ message: 'Replay not yet implemented', type: 'info' })
  }

  const handleDownload = () => {
    const engine = getEngine()
    window.open(engine.getCapsuleUrl(runId), '_blank')
  }

  const handleEvidenceClick = (key) => {
    setEvidenceDrawer(key)
  }

  const handleCopyHash = async () => {
    if (!summary.capsuleHash) {
      setToast({ message: 'Hash not available', type: 'error' })
      return
    }
    try {
      await navigator.clipboard.writeText(summary.capsuleHash)
      setToast({ message: 'Hash copied to clipboard', type: 'success' })
    } catch (err) {
      setToast({ message: 'Failed to copy hash', type: 'error' })
    }
  }

  const handleCopyVerifyCommand = async () => {
    if (!runId) {
      setToast({ message: 'Run ID not available', type: 'error' })
      return
    }
    const command = `capseal verify ${runId}`
    try {
      await navigator.clipboard.writeText(command)
      setToast({ message: 'Verify command copied to clipboard', type: 'success' })
    } catch (err) {
      setToast({ message: 'Failed to copy command', type: 'error' })
    }
  }

  const handleExportCap = async () => {
    if (!runId) {
      setToast({ message: 'Run ID not available', type: 'error' })
      return
    }
    setToast({ message: 'Preparing export...', type: 'info' })
    try {
      const engine = getEngine()
      const result = await engine.exportCapsule({ run_id: runId, format: 'capsule' })
      if (result?.download_url) {
        const url = `${engine.getBaseUrl()}${result.download_url}`
        window.open(url, '_blank', 'noopener,noreferrer')
      }
      setToast({ message: 'Export ready', type: 'success' })
      fetchRunDetail().catch(() => {})
    } catch (err) {
      setToast({ message: err.message || 'Export failed', type: 'error' })
    }
  }

  if (status === 'loading') {
    return (
      <div>
        <div className="detail-header">
          <div className="detail-title">
            <div className="skeleton-line" style={{ width: 120, height: 24 }} />
          </div>
        </div>
        <div className="overview-grid">
          <SkeletonOverviewCard />
          <SkeletonOverviewCard />
          <SkeletonOverviewCard />
        </div>
      </div>
    )
  }

  if (status.startsWith('error')) {
    return (
      <div style={{ color: 'var(--accent-error)' }}>
        <h2>Error</h2>
        <p>{status}</p>
      </div>
    )
  }

  return (
    <div>
      {/* Header - Using RunHeader component */}
      <RunHeader
        runId={runId}
        status={meta?.verification_status}
        backend={meta?.backend}
        policyId={meta?.policy_id}
        createdAt={meta?.created_at}
        onVerify={handleVerify}
        onReplay={handleReplay}
        onDownload={handleDownload}
        verifying={verifying}
        capabilities={capabilities}
        engineStatus={engineCtx.status}
        engineBaseUrl={engineBaseUrl}
        onOpenEngine={engineCtx.openConfig}
      />

      <ReproducePanel
        engineStatus={engineCtx.status}
        engineBaseUrl={engineBaseUrl}
        projectId={projectId}
        runId={runId}
        backend={meta?.backend}
        policyId={policyId}
        command={reproduceCommand}
        onCopy={handleCopyReproduce}
      />

      {/* Share Section */}
      <div className="share-section">
        <button className="btn btn-secondary btn-sm" onClick={handleCopyHash}>
          Copy Hash
        </button>
        <button className="btn btn-secondary btn-sm" onClick={handleCopyVerifyCommand}>
          Copy Verify Command
        </button>
        <button className="btn btn-secondary btn-sm" onClick={handleExportCap}>
          Export .cap
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="detail-content">
        <div style={{ marginBottom: 'var(--space-lg)' }}>
          <SegmentedControl
            options={TABS}
            value={activeTab}
            onChange={setActiveTab}
          />
        </div>

        {/* Inspect Tab (default) */}
        {activeTab === 'inspect' && (
          <InspectSection
            meta={meta}
            summary={summary}
            verification={verification}
            onEvidenceClick={handleEvidenceClick}
            artifacts={artifacts}
          />
        )}

        {/* Verify Tab */}
        {activeTab === 'verify' && (
          <VerifySection
            meta={meta}
            verification={verification}
            onVerify={handleVerify}
            verifying={verifying}
          />
        )}

        {/* Audit Tab */}
        {activeTab === 'audit' && (
          <AuditSection events={events} />
        )}

        {/* Evidence Tab */}
        {activeTab === 'evidence' && (
          <EvidenceSection
            runId={runId}
            setToast={setToast}
          />
        )}

        {/* Export Tab */}
        {activeTab === 'export' && (
          <ExportSection
            runId={runId}
            artifacts={artifacts}
            meta={meta}
          />
        )}
      </div>

      {/* Evidence Drawer */}
      <Drawer
        open={!!evidenceDrawer}
        onClose={() => setEvidenceDrawer(null)}
        title={`Evidence: ${evidenceDrawer}`}
      >
        {evidenceDrawer && (verification?.report || verification?.layers) && (
          <MonoBlock data={verification.report || verification.layers} />
        )}
        {evidenceDrawer && !verification?.report && !verification?.layers && (
          <p style={{ color: 'var(--text-secondary)' }}>
            No verification report available. Run verification to generate evidence.
          </p>
        )}
      </Drawer>

      {/* Toast */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}
