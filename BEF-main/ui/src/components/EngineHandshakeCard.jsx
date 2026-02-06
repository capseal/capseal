import EngineStatus from './EngineStatus'
import { useEngine } from '../state/EngineContext.jsx'

export default function EngineHandshakeCard() {
  const { status, health, error, baseUrl, openConfig, refresh, lastChecked } = useEngine()
  const capabilities = health?.capabilities || {}
  const capabilityList = Object.entries(capabilities)

  return (
    <div className={`engine-handshake ${status}`}>
      <div className="engine-handshake-head">
        <div>
          <p className="eyebrow">Engine</p>
          <h2>{baseUrl || 'No base URL configured'}</h2>
          {error && <p className="engine-handshake-error">{error}</p>}
          {lastChecked && (
            <small className="engine-last-checked">Last checked {formatRelativeTime(lastChecked)}</small>
          )}
        </div>
        <EngineStatus />
      </div>
      <div className="engine-handshake-body">
        {capabilityList.length > 0 ? (
          <ul className="capability-list">
            {capabilityList.map(([key, value]) => (
              <li key={key} className={value ? 'ok' : 'missing'}>
                <span>{key}</span>
                <span>{value ? '✔︎' : '—'}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="engine-handshake-empty">Run the backend or set a base URL to continue.</p>
        )}
      </div>
      <div className="engine-handshake-actions">
        <button className="btn-primary" onClick={openConfig}>
          Configure Engine
        </button>
        <button className="btn-secondary" onClick={refresh}>
          Re-check Health
        </button>
      </div>
    </div>
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
