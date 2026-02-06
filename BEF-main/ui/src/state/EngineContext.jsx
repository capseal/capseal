import { createContext, useContext, useCallback, useEffect, useMemo, useState } from 'react'
import { getEngine, resetEngine } from '../engine'
import {
  getPersistedEngineConfig,
  setPersistedEngineConfig,
  normalizeBaseUrl,
} from '../engine/engine.js'

const EngineContext = createContext({
  status: 'checking',
  baseUrl: '',
  health: null,
  error: null,
  lastChecked: null,
  refresh: () => {},
  setBaseUrl: () => {},
  openConfig: () => {},
})

function guessDefaultBaseUrl() {
  if (typeof window === 'undefined') return ''
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:5001'
  }
  return ''
}

export function EngineProvider({ children }) {
  const persisted = getPersistedEngineConfig()
  const [baseUrl, setBaseUrl] = useState(() => persisted?.baseUrl || guessDefaultBaseUrl())
  const [status, setStatus] = useState(baseUrl ? 'checking' : 'needs-config')
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)
  const [lastChecked, setLastChecked] = useState(null)
  const [configOpen, setConfigOpen] = useState(false)

  const applyBaseUrl = useCallback((rawUrl) => {
    let normalized = ''
    if (rawUrl) {
      normalized = normalizeBaseUrl(rawUrl)
    }
    setBaseUrl(normalized)
    setPersistedEngineConfig({ baseUrl: normalized })
    resetEngine()
    setStatus(normalized ? 'checking' : 'needs-config')
    setError(null)
    return normalized
  }, [])

  const refresh = useCallback(async () => {
    if (!baseUrl) {
      setStatus('needs-config')
      setHealth(null)
      setError('Engine base URL not set')
      return
    }

    setStatus('checking')
    try {
      const engine = getEngine({ baseUrl })
      const data = await engine.getHealth()
      setHealth(data)
      setStatus(data.ok ? 'online' : 'degraded')
      setError(data.ok ? null : 'Engine reported degraded state')
    } catch (err) {
      setStatus('offline')
      setHealth(null)
      setError(err?.message || 'Unable to reach engine')
    } finally {
      setLastChecked(new Date())
    }
  }, [baseUrl])

  useEffect(() => {
    if (!baseUrl) return
    refresh()
    const interval = setInterval(refresh, 30000)
    return () => clearInterval(interval)
  }, [baseUrl, refresh])

  const value = useMemo(() => ({
    status,
    baseUrl,
    health,
    error,
    lastChecked,
    refresh,
    setBaseUrl: applyBaseUrl,
    openConfig: () => setConfigOpen(true),
    closeConfig: () => setConfigOpen(false),
  }), [status, baseUrl, health, error, lastChecked, refresh, applyBaseUrl])

  return (
    <EngineContext.Provider value={value}>
      {children}
      {configOpen && (
        <EngineConfigDialog
          baseUrl={baseUrl}
          status={status}
          error={error}
          lastChecked={lastChecked}
          onClose={() => setConfigOpen(false)}
          onSave={(url) => {
            applyBaseUrl(url)
            setConfigOpen(false)
            setTimeout(refresh, 100)
          }}
        />
      )}
    </EngineContext.Provider>
  )
}

function EngineConfigDialog({ baseUrl, status, error, onSave, onClose, lastChecked }) {
  const [draft, setDraft] = useState(baseUrl || '')
  const [saving, setSaving] = useState(false)
  const [validationError, setValidationError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      setSaving(true)
      await onSave(draft.trim())
      setValidationError(null)
    } catch (err) {
      setValidationError(err?.message || 'Invalid URL')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="engine-config-overlay" onClick={onClose}>
      <div className="engine-config-card" onClick={(e) => e.stopPropagation()}>
        <div className="engine-config-head">
          <div>
            <h2>Engine Connection</h2>
            <p>Set the base URL for your Capseal backend.</p>
            {lastChecked && (
              <small className="engine-last-checked">Last checked {formatRelativeTime(lastChecked)}</small>
            )}
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close engine configuration">
            ×
          </button>
        </div>
        <form onSubmit={handleSubmit} className="engine-config-form">
          <label htmlFor="engineBaseUrl">Base URL</label>
          <input
            id="engineBaseUrl"
            type="text"
            placeholder="http://localhost:5001"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="engine-config-details">
            <span className={`status-pill status-${status}`}>
              {status === 'online' && 'Online'}
              {status === 'checking' && 'Checking…'}
              {status === 'offline' && 'Offline'}
              {status === 'needs-config' && 'Needs configuration'}
              {status === 'degraded' && 'Degraded'}
            </span>
            {(error || validationError) && (
              <span className="engine-config-error">{validationError || error}</span>
            )}
          </div>
          <div className="engine-config-actions">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? 'Saving…' : 'Save & Reconnect'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export function useEngine() {
  return useContext(EngineContext)
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
