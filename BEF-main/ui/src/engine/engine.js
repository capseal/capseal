/**
 * CapsuleTech Engine Adapter - Core Interface
 *
 * This is the ONLY interface components should use to talk to the backend.
 * Backend implementation (Flask, Rust, remote) is transparent to components.
 *
 * @typedef {import('./types.js').EngineConfig} EngineConfig
 * @typedef {import('./types.js').HealthStatus} HealthStatus
 * @typedef {import('./types.js').RunStartRequest} RunStartRequest
 * @typedef {import('./types.js').RunStartResponse} RunStartResponse
 * @typedef {import('./types.js').RunEvent} RunEvent
 * @typedef {import('./types.js').RunRecord} RunRecord
 * @typedef {import('./types.js').RunSummary} RunSummary
 * @typedef {import('./types.js').RunListQuery} RunListQuery
 * @typedef {import('./types.js').VerifyRequest} VerifyRequest
 * @typedef {import('./types.js').VerifyReport} VerifyReport
 * @typedef {import('./types.js').AuditRequest} AuditRequest
 * @typedef {import('./types.js').AuditReport} AuditReport
 * @typedef {import('./types.js').EvidenceRequest} EvidenceRequest
 * @typedef {import('./types.js').EvidenceIndex} EvidenceIndex
 * @typedef {import('./types.js').ExportRequest} ExportRequest
 * @typedef {import('./types.js').ExportResult} ExportResult
 */

// Re-export ExecutionEngine from base.js for backwards compatibility
export { ExecutionEngine } from './base.js'

// Import FlaskEngine (safe - flask.js imports from base.js, not engine.js)
import { FlaskEngine } from './impl/flask.js'

// Re-export config error for consumers
export { EngineConfigError, resolveConfig } from './impl/flask.js'

const ENGINE_CONFIG_STORAGE_KEY = 'capseal.engine.config'

function readStoredConfig() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null
  }
  try {
    const raw = window.localStorage.getItem(ENGINE_CONFIG_STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch (err) {
    console.warn('Failed to read engine config from storage', err)
    return null
  }
}

function writeStoredConfig(config) {
  if (typeof window === 'undefined' || !window.localStorage) {
    return
  }
  try {
    window.localStorage.setItem(ENGINE_CONFIG_STORAGE_KEY, JSON.stringify(config))
  } catch (err) {
    console.warn('Failed to write engine config to storage', err)
  }
}

export function getPersistedEngineConfig() {
  return readStoredConfig()
}

export function setPersistedEngineConfig(partial) {
  const current = readStoredConfig() || {}
  const next = { ...current, ...partial }
  writeStoredConfig(next)
  return next
}

export function clearPersistedEngineConfig() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return
  }
  window.localStorage.removeItem(ENGINE_CONFIG_STORAGE_KEY)
}

export function normalizeBaseUrl(input) {
  if (!input) return ''
  let value = input.trim()
  if (!value) return ''
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value)) {
    value = `http://${value}`
  }
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol)) {
      throw new Error('Only http:// or https:// URLs are supported')
    }
    url.hash = ''
    if (!url.pathname || url.pathname === '/') {
      url.pathname = ''
    }
    let normalized = url.toString()
    if (normalized.endsWith('/')) {
      normalized = normalized.slice(0, -1)
    }
    return normalized
  } catch (err) {
    throw new Error('Invalid base URL')
  }
}

/**
 * Get default engine configuration
 *
 * Configuration sources (in priority order):
 * 1. Runtime: window.API_BASE
 * 2. Environment: VITE_API_BASE
 * 3. Backend: /api/config response (future)
 *
 * NO HARDCODED FALLBACKS - missing baseUrl will throw EngineConfigError
 * when createEngine() or getEngine() is called.
 *
 * @returns {Partial<EngineConfig>}
 */
export function getDefaultConfig() {
  // Collect config from available sources - NO hardcoded fallback
  let baseUrl = undefined

  const stored = getPersistedEngineConfig()
  if (stored?.baseUrl) {
    baseUrl = stored.baseUrl
  }

  if (typeof window !== 'undefined' && window.API_BASE) {
    baseUrl = window.API_BASE
  } else if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) {
    baseUrl = import.meta.env.VITE_API_BASE
  }

  if (!baseUrl && typeof window !== 'undefined') {
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    if (isLocalhost) {
      baseUrl = 'http://localhost:5001'
    }
  }
  // NOTE: No hardcoded fallback - baseUrl may be undefined
  // FlaskEngine constructor will throw EngineConfigError if missing

  return {
    mode: 'local',
    baseUrl, // May be undefined - that's intentional
    timeoutMs: 120000, // 2 minutes
    ssePath: '/events/stream',
  }
}

/**
 * Create an execution engine instance
 *
 * Factory function that returns the appropriate engine implementation
 * based on configuration.
 *
 * @param {Partial<EngineConfig>} [config] - Optional configuration overrides
 * @returns {import('./base.js').ExecutionEngine}
 */
export function createEngine(config = {}) {
  const fullConfig = { ...getDefaultConfig(), ...config }

  // For now, we only have Flask implementation
  // Later: check fullConfig.mode and return appropriate implementation
  // if (fullConfig.mode === 'rust') return new RustEngine(fullConfig)
  // if (fullConfig.mode === 'remote') return new RemoteEngine(fullConfig)

  return new FlaskEngine(fullConfig)
}

/**
 * Singleton engine instance
 * Components should use this instead of creating their own instances
 */
let _engineInstance = null

/**
 * Get the global engine instance
 *
 * @param {Partial<EngineConfig>} [config] - Optional configuration overrides
 * @returns {import('./base.js').ExecutionEngine}
 */
export function getEngine(config) {
  if (!_engineInstance) {
    _engineInstance = createEngine(config)
  }
  return _engineInstance
}

/**
 * Reset the global engine instance
 * Useful for testing or when configuration changes
 */
export function resetEngine() {
  _engineInstance = null
}
