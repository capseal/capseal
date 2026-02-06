import { StatusPill } from './index'

/**
 * RunRow - Premium list row for runs.
 *
 * 3-line layout:
 * - Line 1: Run ID + Status pill
 * - Line 2: Backend, policy, time
 * - Line 3: Capsule hash or last event (mono)
 *
 * Strong hover/selected state for "table-list hybrid" feel.
 */
export default function RunRow({ run, selected, onClick }) {
  const timeAgo = formatTimeAgo(run.created_at)

  return (
    <button
      onClick={onClick}
      className={`run-row ${selected ? 'selected' : ''}`}
      aria-selected={selected}
    >
      {/* Line 1: ID + Status */}
      <div className="run-row-header">
        <span className="run-row-id">{run.run_id || run.id}</span>
        <StatusPill status={run.verification_status || run.verification} />
      </div>

      {/* Line 2: Metadata + track_id badge */}
      <div className="run-row-meta">
        {run.track_id && (
          <>
            <span className="track-badge">{run.track_id}</span>
            <span className="meta-dot" />
          </>
        )}
        {run.backend || 'unknown'}
        <span className="meta-dot" />
        {run.policy_id || 'no policy'}
        <span className="meta-dot" />
        {timeAgo}
      </div>

      {/* Line 3: Hash or event (mono) */}
      <div className="run-row-hash">
        {run.capsule_hash
          ? truncateHash(run.capsule_hash)
          : run.last_event || '\u2014'}
      </div>
    </button>
  )
}

function truncateHash(hash) {
  if (!hash) return '\u2014'
  if (hash.length <= 16) return hash
  return hash.slice(0, 12) + '\u2026'
}

function formatTimeAgo(isoString) {
  if (!isoString) return 'unknown'

  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now - date
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHour = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHour / 24)

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHour < 24) return `${diffHour}h ago`
  if (diffDay < 7) return `${diffDay}d ago`

  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
