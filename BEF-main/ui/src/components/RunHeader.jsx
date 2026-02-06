import { StatusPill } from './index'

/**
 * RunHeader - Sticky header for run detail panel.
 *
 * Shows: Run ID, status, metadata line, action buttons.
 * Sticks to top of detail panel on scroll.
 */
export default function RunHeader({
  runId,
  status,
  backend,
  policyId,
  createdAt,
  onVerify,
  onReplay,
  onDownload,
  verifying = false,
  capabilities = {},
  engineStatus,
  engineBaseUrl,
  onOpenEngine,
}) {
  const canVerify = capabilities?.verify !== false
  return (
    <header className="run-header">
      <div className="run-header-left">
        {/* Title row */}
        <div className="run-header-title">
          <h1>{runId}</h1>
          <StatusPill status={status} />
        </div>

        {/* Metadata */}
        <div className="run-header-meta">
          {backend}
          <span className="meta-dot" />
          {policyId || 'No policy'}
          <span className="meta-dot" />
          {formatTime(createdAt)}
        </div>

        {engineBaseUrl && (
          <div className="run-header-engine">
            <span className={`engine-dot status-${engineStatus}`} />
            <span>{engineBaseUrl}</span>
            {onOpenEngine && (
              <button className="link-button" onClick={onOpenEngine}>Edit</button>
            )}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="run-header-actions">
        <button
          className="btn btn-primary"
          onClick={onVerify}
          disabled={verifying || !canVerify}
          >
          {verifying ? (
            <>
              <span className="btn-spinner" />
              Verifying
            </>
          ) : canVerify ? (
            'Verify'
          ) : (
            'Verify Unavailable'
          )}
        </button>

        <button className="btn btn-secondary" onClick={onReplay}>
          Replay
        </button>

        <button
          className="btn btn-icon"
          onClick={onDownload}
          title="Download receipt"
          aria-label="Download receipt"
        >
          <DownloadIcon />
        </button>
      </div>
    </header>
  )
}

function formatTime(isoString) {
  if (!isoString) return '\u2014'
  return new Date(isoString).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function DownloadIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  )
}
