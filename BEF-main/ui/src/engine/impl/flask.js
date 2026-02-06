/**
 * Flask/Relay backend adapter for verification reports.
 *
 * Normalizes verification data from the backend into a consistent shape:
 *   { ok: boolean, layers: VerifyLayer[] }
 *
 * @typedef {'PENDING' | 'VERIFYING' | 'VERIFIED' | 'FAILED' | 'UNKNOWN'} JobStatus
 *
 * @typedef {Object} VerifyLayer
 * @property {string} id - Unique layer identifier
 * @property {string} label - Human-readable layer name
 * @property {boolean} ok - Whether this layer passed verification
 * @property {string} [reason] - Error reason code if failed
 * @property {Object} [meta] - Additional metadata for this layer
 */

import { ExecutionEngine } from '../base.js'

/**
 * Configuration error for missing or invalid engine settings.
 * Thrown when required configuration is not provided.
 */
export class EngineConfigError extends Error {
  /**
   * @param {string} message - Error message
   * @param {string} [configKey] - The missing/invalid config key
   */
  constructor(message, configKey = null) {
    super(message)
    this.name = 'EngineConfigError'
    this.configKey = configKey
  }
}

/**
 * Validate and resolve engine configuration.
 * Throws EngineConfigError if required config is missing.
 *
 * Configuration sources (in priority order):
 * 1. Explicit config.baseUrl parameter
 * 2. Runtime window.API_BASE
 * 3. Environment VITE_API_BASE
 * 4. Backend /api/config response (if available)
 *
 * NO HARDCODED FALLBACKS - missing config is an error.
 *
 * @param {import('../types.js').EngineConfig} config
 * @returns {import('../types.js').EngineConfig}
 * @throws {EngineConfigError}
 */
export function resolveConfig(config) {
  // Try to resolve baseUrl from various sources
  let baseUrl = config?.baseUrl

  if (!baseUrl && typeof window !== 'undefined') {
    baseUrl = window.API_BASE
  }

  if (!baseUrl && typeof import.meta !== 'undefined') {
    baseUrl = import.meta.env?.VITE_API_BASE
  }

  // NO HARDCODED FALLBACK - throw clear error if missing
  if (!baseUrl) {
    throw new EngineConfigError(
      'Engine baseUrl is required but not configured. ' +
      'Set one of: config.baseUrl, window.API_BASE, or VITE_API_BASE environment variable.',
      'baseUrl'
    )
  }

  // Validate URL format
  try {
    new URL(baseUrl)
  } catch {
    throw new EngineConfigError(
      `Invalid baseUrl format: "${baseUrl}". Must be a valid URL.`,
      'baseUrl'
    )
  }

  return {
    mode: config?.mode || 'local',
    baseUrl,
    timeoutMs: config?.timeoutMs || 120000,
    ssePath: config?.ssePath || '/events/stream',
  }
}

/**
 * Flask backend engine implementation.
 * Implements ExecutionEngine interface for Flask/Relay backend.
 */
export class FlaskEngine extends ExecutionEngine {
  /**
   * @param {import('../types.js').EngineConfig} config
   * @throws {EngineConfigError} If config.baseUrl is not provided
   */
  constructor(config) {
    super()
    this.config = resolveConfig(config)
    this._baseUrl = this.config.baseUrl.replace(/\/$/, '') // Remove trailing slash
  }

  /**
   * @returns {string}
   */
  getBaseUrl() {
    return this._baseUrl
  }

  /**
   * @param {string} runId
   * @param {string} artifactName
   * @returns {string}
   */
  getArtifactUrl(runId, artifactName) {
    return `${this._baseUrl}/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactName)}`
  }

  /**
   * @param {string} runId
   * @returns {string}
   */
  getCapsuleUrl(runId) {
    return `${this._baseUrl}/api/runs/${encodeURIComponent(runId)}/capsule`
  }

  /**
   * @param {string} runId
   * @param {number} rowNumber
   * @returns {string}
   */
  getRowUrl(runId, rowNumber) {
    return `${this._baseUrl}/api/runs/${encodeURIComponent(runId)}/rows/${rowNumber}`
  }

  /**
   * @returns {Promise<import('../types.js').HealthStatus>}
   */
  async getHealth() {
    const resp = await fetch(`${this._baseUrl}/api/health`, {
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      return {
        ok: false,
        engine: 'flask',
        version: 'unknown',
        capabilities: {
          run: false,
          verify: false,
          audit: false,
          evidence: false,
          export: false,
          sse: false,
        },
      }
    }
    const data = await resp.json()
    return {
      ok: data.ok ?? true,
      engine: 'flask',
      version: data.version || 'unknown',
      capabilities: {
        run: true,
        verify: true,
        audit: true,
        evidence: true,
        export: true,
        sse: data.sse ?? false,
      },
    }
  }

  /**
   * @param {import('../types.js').RunStartRequest} req
   * @returns {Promise<import('../types.js').RunStartResponse>}
   */
  async startRun(req) {
    const resp = await fetch(`${this._baseUrl}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || err.error || `Run start failed: ${resp.status}`)
    }
    return resp.json()
  }

  /**
   * @param {string} runId
   * @param {(event: import('../types.js').RunEvent) => void} onEvent
   * @param {(error: any) => void} [onError]
   * @returns {() => void}
   */
  streamEvents(runId, onEvent, onError) {
    const url = `${this._baseUrl}/api/runs/${encodeURIComponent(runId)}${this.config.ssePath}`
    let active = true

    // Try SSE first, fall back to polling
    const eventSource = new EventSource(url)

    eventSource.onmessage = (evt) => {
      if (!active) return
      try {
        const event = JSON.parse(evt.data)
        onEvent(event)
      } catch (err) {
        onError?.(err)
      }
    }

    eventSource.onerror = () => {
      if (!active) return
      // Fall back to polling
      eventSource.close()
      this._pollEvents(runId, onEvent, onError, () => active)
    }

    return () => {
      active = false
      eventSource.close()
    }
  }

  /**
   * @private
   */
  async _pollEvents(runId, onEvent, onError, isActive) {
    let lastSeq = 0
    while (isActive()) {
      try {
        const resp = await fetch(
          `${this._baseUrl}/api/runs/${encodeURIComponent(runId)}/events?after_seq=${lastSeq}`,
          { signal: AbortSignal.timeout(30000) }
        )
        if (!resp.ok) continue
        const data = await resp.json()
        const events = Array.isArray(data.events) ? data.events : Array.isArray(data) ? data : []
        for (const event of events) {
          onEvent(event)
          const seq = Number(event.seq || 0)
          if (seq > lastSeq) lastSeq = seq
        }
      } catch (err) {
        onError?.(err)
      }
      await new Promise((r) => setTimeout(r, 1000))
    }
  }

  /**
   * @param {string} runId
   * @returns {Promise<import('../types.js').RunRecord>}
   */
  async getRun(runId) {
    const resp = await fetch(`${this._baseUrl}/api/runs/${encodeURIComponent(runId)}`, {
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      throw new Error(`Run not found: ${runId}`)
    }
    const data = await resp.json()
    return data.run || data
  }

  /**
   * @param {import('../types.js').RunListQuery} [query]
   * @returns {Promise<import('../types.js').RunSummary[]>}
   */
  async listRuns(query) {
    const params = new URLSearchParams()
    if (query?.project_id) params.set('project_id', query.project_id)
    if (query?.status) params.set('status', query.status)
    if (query?.limit) params.set('limit', String(query.limit))
    if (query?.offset) params.set('offset', String(query.offset))

    const url = `${this._baseUrl}/api/runs${params.toString() ? '?' + params : ''}`
    const resp = await fetch(url, {
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      throw new Error(`Failed to list runs: ${resp.status}`)
    }
    const data = await resp.json()
    return Array.isArray(data) ? data : data.runs || []
  }

  /**
   * @param {import('../types.js').VerifyRequest} req
   * @returns {Promise<import('../types.js').VerifyReport>}
   */
  async verify(req) {
    const resp = await fetch(`${this._baseUrl}/api/runs/${encodeURIComponent(req.run_id)}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) {
      throw new Error(data.detail || data.error || `Verification failed: ${resp.status}`)
    }
    // Normalize the response to VerifyReport format
    return normalizeVerifyReport(data)
  }

  /**
   * @param {import('../types.js').AuditRequest} req
   * @returns {Promise<import('../types.js').AuditReport>}
   */
  async audit(req) {
    const resp = await fetch(`${this._baseUrl}/api/runs/${encodeURIComponent(req.run_id)}/audit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || err.error || `Audit failed: ${resp.status}`)
    }
    return resp.json()
  }

  /**
   * @param {import('../types.js').EvidenceRequest} req
   * @returns {Promise<import('../types.js').EvidenceIndex>}
   */
  async evidence(req) {
    const resp = await fetch(
      `${this._baseUrl}/api/runs/${encodeURIComponent(req.run_id)}/evidence/${req.kind}`,
      { signal: AbortSignal.timeout(this.config.timeoutMs) }
    )
    if (!resp.ok) {
      throw new Error(`Evidence request failed: ${resp.status}`)
    }
    return resp.json()
  }

  /**
   * @param {import('../types.js').ExportRequest} req
   * @returns {Promise<import('../types.js').ExportResult>}
   */
  async exportCapsule(req) {
    const resp = await fetch(`${this._baseUrl}/api/runs/${encodeURIComponent(req.run_id)}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: req.format }),
      signal: AbortSignal.timeout(this.config.timeoutMs),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || err.error || `Export failed: ${resp.status}`)
    }
    return resp.json()
  }
}

/**
 * Unified job status enum.
 * Use these values consistently across the UI.
 */
export const JobStatus = Object.freeze({
  PENDING: 'PENDING',
  VERIFYING: 'VERIFYING',
  VERIFIED: 'VERIFIED',
  FAILED: 'FAILED',
  UNKNOWN: 'UNKNOWN',
});

/**
 * Normalize raw job status strings to JobStatus enum.
 * Handles case-insensitivity and legacy status values.
 *
 * @param {string | null | undefined} rawStatus
 * @returns {JobStatus}
 */
export function normalizeJobStatus(rawStatus) {
  if (!rawStatus) return JobStatus.UNKNOWN;
  const upper = String(rawStatus).toUpperCase();
  switch (upper) {
    case 'PENDING':
    case 'QUEUED':
      return JobStatus.PENDING;
    case 'VERIFYING':
    case 'RUNNING':
    case 'IN_PROGRESS':
      return JobStatus.VERIFYING;
    case 'VERIFIED':
    case 'PASSED':
    case 'SUCCESS':
    case 'OK':
      return JobStatus.VERIFIED;
    case 'FAILED':
    case 'ERROR':
    case 'REJECTED':
      return JobStatus.FAILED;
    default:
      return JobStatus.UNKNOWN;
  }
}

/**
 * Layer definitions for verification report normalization.
 * Maps backend field names to UI layer definitions.
 *
 * Supports TWO formats:
 * 1. Contract format (from sse_routes.py): layers object with l0_hash, l1_commitment, etc.
 * 2. Legacy format: flat boolean fields like row_index_commitment_ok, proof_verified, etc.
 */

// Contract-compliant layer mapping (L0-L4 verification ladder)
// Maps backend layer keys to UI display
const CONTRACT_LAYER_MAP = {
  l0_hash: { id: 'binding', label: 'Hash Binding', uiGroup: 'binding' },
  l1_commitment: { id: 'commitment', label: 'Commitment', uiGroup: 'binding' },
  l2_constraint: { id: 'policy', label: 'Policy', uiGroup: 'policy' },
  l3_proximity: { id: 'proof', label: 'Proof', uiGroup: 'proof' },
  l4_receipt: { id: 'receipt', label: 'Receipt', uiGroup: 'receipt' },
};

// Legacy field definitions (for backward compatibility)
const LAYER_DEFS = [
  {
    id: 'binding',
    label: 'Binding',
    fields: ['row_index_commitment_ok', 'binding_ok', 'capsule_hash_ok'],
  },
  {
    id: 'proof',
    label: 'Proof',
    fields: ['proof_verified', 'proof_ok'],
  },
  {
    id: 'policy',
    label: 'Policy',
    fields: ['policy_verified', 'policy_ok', 'policy_rules_satisfied'],
  },
  {
    id: 'authorship',
    label: 'Authorship',
    fields: ['authorship_verified', 'authorship_ok'],
  },
  {
    id: 'acl',
    label: 'ACL',
    fields: ['acl_authorized', 'acl_ok'],
  },
  {
    id: 'availability',
    label: 'Availability',
    fields: ['da_audit_verified', 'availability_ok', 'da_ok'],
  },
  {
    id: 'events',
    label: 'Events',
    fields: ['events_verified', 'events_ok'],
  },
  {
    id: 'header',
    label: 'Header',
    fields: ['header_verified', 'header_ok'],
  },
];

/**
 * Extract a boolean value from a report object using multiple field aliases.
 *
 * @param {Object} report - The raw verification report
 * @param {string[]} fields - Field names to check, in priority order
 * @returns {{ value: boolean | null, field: string | null }}
 */
function extractBool(report, fields) {
  for (const field of fields) {
    if (field in report) {
      return { value: Boolean(report[field]), field };
    }
  }
  return { value: null, field: null };
}

/**
 * Normalize a verification report from the backend.
 *
 * Transforms various backend report shapes into a consistent format:
 *   { ok: boolean, layers: VerifyLayer[] }
 *
 * Supports TWO formats:
 * 1. Contract format (from sse_routes.py):
 *    { status, layers: { l0_hash: {status, message}, ... }, errors, timings }
 * 2. Legacy format:
 *    { report: { row_index_commitment_ok: true, ... } }
 *
 * The `layers` array is ALWAYS an array, never an object.
 * Components should iterate over `layers` without assuming key/value shape.
 *
 * @param {Object | null} verification - Raw verification object from backend
 * @returns {{ ok: boolean, layers: VerifyLayer[], status: JobStatus, errorCode: string | null }}
 */
export function normalizeVerifyReport(verification) {
  // Handle null/undefined verification
  if (!verification) {
    return {
      ok: false,
      layers: [],
      status: JobStatus.UNKNOWN,
      errorCode: null,
    };
  }

  // Detect format: contract format has .layers as object with l0_hash, etc.
  const hasContractLayers = verification.layers &&
    typeof verification.layers === 'object' &&
    !Array.isArray(verification.layers) &&
    ('l0_hash' in verification.layers || 'l1_commitment' in verification.layers);

  if (hasContractLayers) {
    return normalizeContractFormat(verification);
  }

  // Fall back to legacy format
  return normalizeLegacyFormat(verification);
}

/**
 * Normalize contract-compliant verification format (from sse_routes.py).
 * @private
 */
function normalizeContractFormat(verification) {
  const rawStatus = verification.status;
  const status = normalizeJobStatus(rawStatus);

  // Extract error code from errors array
  const errors = verification.errors || [];
  const errorCode = errors.length > 0 ? errors[0].code : null;

  // Build layers array from contract layers object
  const layers = [];
  const backendLayers = verification.layers || {};

  for (const [key, layerData] of Object.entries(backendLayers)) {
    const mapping = CONTRACT_LAYER_MAP[key];
    if (!mapping) continue;

    const layerStatus = layerData?.status;
    const ok = layerStatus === 'pass';
    const failed = layerStatus === 'fail';

    layers.push({
      id: mapping.id,
      label: mapping.label,
      ok: ok,
      reason: failed ? (layerData?.message || null) : null,
      meta: {
        sourceKey: key,
        uiGroup: mapping.uiGroup,
        rawStatus: layerStatus,
      },
    });
  }

  // Compute overall ok from status and layers
  const hasFailedLayer = layers.some((layer) => layer.ok === false);
  const ok = (status === JobStatus.VERIFIED || rawStatus === 'verified') && !hasFailedLayer;

  return {
    ok,
    layers,
    status,
    errorCode: typeof errorCode === 'string' ? errorCode : null,
  };
}

/**
 * Normalize legacy verification format (flat boolean fields).
 * @private
 */
function normalizeLegacyFormat(verification) {
  const report = verification.report || {};
  const rawStatus = verification.status;
  const status = normalizeJobStatus(rawStatus);

  // Extract error code from various possible locations
  const errorCode = report.error_code || report.error || report.status || null;

  // Build layers array from report fields
  const layers = [];
  for (const def of LAYER_DEFS) {
    const { value, field } = extractBool(report, def.fields);

    // Only include layers that have data in the report
    if (value !== null) {
      layers.push({
        id: def.id,
        label: def.label,
        ok: value,
        reason: value ? null : (report[`${def.id}_error`] || report.error_code || null),
        meta: { sourceField: field },
      });
    }
  }

  // Compute overall ok: true if status is VERIFIED and no layers failed
  const hasFailedLayer = layers.some((layer) => layer.ok === false);
  const ok = status === JobStatus.VERIFIED && !hasFailedLayer;

  return {
    ok,
    layers,
    status,
    errorCode: typeof errorCode === 'string' ? errorCode : null,
  };
}

/**
 * Get a layer by ID from a normalized report.
 *
 * @param {VerifyLayer[]} layers - Array of layers
 * @param {string} id - Layer ID to find
 * @returns {VerifyLayer | undefined}
 */
export function getLayerById(layers, id) {
  return layers.find((layer) => layer.id === id);
}

/**
 * Check if a specific layer passed verification.
 *
 * @param {VerifyLayer[]} layers - Array of layers
 * @param {string} id - Layer ID to check
 * @returns {boolean | null} - true if passed, false if failed, null if layer not present
 */
export function isLayerOk(layers, id) {
  const layer = getLayerById(layers, id);
  return layer ? layer.ok : null;
}

/**
 * CSS class mapping for job statuses.
 * Use with status badge components.
 */
export const STATUS_CLASSES = Object.freeze({
  [JobStatus.PENDING]: 'status-pending',
  [JobStatus.VERIFYING]: 'status-verifying',
  [JobStatus.VERIFIED]: 'status-verified',
  [JobStatus.FAILED]: 'status-failed',
  [JobStatus.UNKNOWN]: 'status-unknown',
});

/**
 * Get the CSS class for a job status.
 *
 * @param {JobStatus} status
 * @returns {string}
 */
export function getStatusClass(status) {
  return STATUS_CLASSES[status] || 'status-unknown';
}
