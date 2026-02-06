import { useState } from 'react'
import MonoBlock from './MonoBlock'

function humanizeEventType(type) {
  const map = {
    run_started: 'Run started',
    spec_locked: 'Specification locked',
    trace_simulated: 'Trace simulated',
    statement_locked: 'Statement locked',
    row_root_finalized: 'Row root finalized',
    capsule_sealed: 'Capsule sealed',
    run_completed: 'Run completed',
    verification_started: 'Verification started',
    verification_completed: 'Verification completed',
  }
  return map[type] || type.replace(/_/g, ' ')
}

function formatRelativeTime(tsMs) {
  if (!tsMs) return ''
  const diff = Date.now() - tsMs
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return new Date(tsMs).toLocaleDateString()
}

function TimelineEvent({ event, isLast }) {
  const [showRaw, setShowRaw] = useState(false)

  return (
    <div className={`timeline-event ${isLast ? 'timeline-latest' : ''}`}>
      <div className="timeline-marker">
        <div className="timeline-dot" />
        {!isLast && <div className="timeline-line" />}
      </div>
      <div className="timeline-content">
        <div className="timeline-header">
          <span className="timeline-type">{humanizeEventType(event.type)}</span>
          <span className="timeline-time" title={event.ts_ms ? new Date(event.ts_ms).toLocaleString() : ''}>
            {formatRelativeTime(event.ts_ms)}
          </span>
        </div>
        {event.data && Object.keys(event.data).length > 0 && (
          <>
            <button
              className="timeline-raw-toggle"
              onClick={() => setShowRaw(!showRaw)}
            >
              {showRaw ? 'Hide details' : 'Show details'}
            </button>
            {showRaw && <MonoBlock data={event.data} maxHeight={200} />}
          </>
        )}
      </div>
    </div>
  )
}

export default function Timeline({ events, showLiveIndicator = false }) {
  const sortedEvents = [...(events || [])].sort((a, b) => (b.ts_ms || 0) - (a.ts_ms || 0))

  if (!sortedEvents.length) {
    return (
      <div className="timeline-empty">
        <p>No events yet</p>
      </div>
    )
  }

  return (
    <div className="timeline">
      {showLiveIndicator && (
        <div className="timeline-live-indicator">
          <span className="live-dot" />
          <span>Live</span>
        </div>
      )}
      <div className="timeline-events">
        {sortedEvents.map((event, idx) => (
          <TimelineEvent
            key={`${event.seq}-${event.type}`}
            event={event}
            isLast={idx === 0}
          />
        ))}
      </div>
    </div>
  )
}
