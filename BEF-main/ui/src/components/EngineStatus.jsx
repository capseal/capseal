import { useMemo } from 'react'
import { useEngine } from '../state/EngineContext.jsx'

export default function EngineStatus() {
  const { status, baseUrl, health, error, refresh, openConfig, lastChecked } = useEngine()

  const statusLabel = useMemo(() => {
    switch (status) {
      case 'online':
        return 'Connected'
      case 'degraded':
        return 'Degraded'
      case 'offline':
        return 'Offline'
      case 'needs-config':
        return 'Not configured'
      default:
        return 'Checking…'
    }
  }, [status])

  const lastCheckedLabel = lastChecked ? formatRelativeTime(lastChecked) : ''

  return (
    <button className={`engine-status-chip status-${status}`} onClick={openConfig} title={lastCheckedLabel ? `Last checked ${lastCheckedLabel}` : undefined}>
      <span className="pulse" aria-hidden="true" />
      <span>
        Engine · {statusLabel}
        {baseUrl && <span className="engine-status-url">{baseUrl}</span>}
      </span>
      <span className="engine-status-actions">
        {status === 'offline' && (
          <span className="engine-status-error" title={error || 'Engine offline'}>Retry</span>
        )}
        {status === 'degraded' && (
          <span className="engine-status-error" title={error || 'Engine degraded'}>Investigate</span>
        )}
        <span className="engine-status-refresh" onClick={(e) => { e.stopPropagation(); refresh() }}>↺</span>
      </span>
      {health?.version && <span className="engine-status-version">v{health.version}</span>}
    </button>
  )
}

function formatRelativeTime(date) {
  try {
    const diff = Date.now() - new Date(date).getTime()
    const seconds = Math.round(diff / 1000)
    if (seconds < 60) return `${seconds}s ago`
    const minutes = Math.round(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.round(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.round(hours / 24)
    return `${days}d ago`
  } catch (err) {
    return ''
  }
}
