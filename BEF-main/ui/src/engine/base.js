/**
 * CapsuleTech Engine Adapter - Base Interface
 *
 * This file contains ONLY the base class to avoid circular imports.
 * Implementations import from here, not from engine.js.
 */

/**
 * ExecutionEngine interface
 *
 * All backend implementations must conform to this interface.
 *
 * @interface
 */
export class ExecutionEngine {
  /**
   * Get engine health status and capabilities
   * @returns {Promise<import('./types.js').HealthStatus>}
   */
  async getHealth() {
    throw new Error('Not implemented')
  }

  /**
   * Start a new run
   * @param {import('./types.js').RunStartRequest} req - Run start request
   * @returns {Promise<import('./types.js').RunStartResponse>}
   */
  async startRun(req) {
    throw new Error('Not implemented')
  }

  /**
   * Stream events from a running execution
   * @param {string} runId - Run identifier
   * @param {(event: import('./types.js').RunEvent) => void} onEvent - Event handler
   * @param {(error: any) => void} [onError] - Error handler
   * @returns {() => void} Cleanup function to stop streaming
   */
  streamEvents(runId, onEvent, onError) {
    throw new Error('Not implemented')
  }

  /**
   * Get a single run record
   * @param {string} runId - Run identifier
   * @returns {Promise<import('./types.js').RunRecord>}
   */
  async getRun(runId) {
    throw new Error('Not implemented')
  }

  /**
   * List runs with optional filtering
   * @param {import('./types.js').RunListQuery} [query] - Query parameters
   * @returns {Promise<import('./types.js').RunSummary[]>}
   */
  async listRuns(query) {
    throw new Error('Not implemented')
  }

  /**
   * Verify a run (L0-L4 trust ladder)
   * @param {import('./types.js').VerifyRequest} req - Verification request
   * @returns {Promise<import('./types.js').VerifyReport>}
   */
  async verify(req) {
    throw new Error('Not implemented')
  }

  /**
   * Audit a run (hash chain + policy compliance)
   * @param {import('./types.js').AuditRequest} req - Audit request
   * @returns {Promise<import('./types.js').AuditReport>}
   */
  async audit(req) {
    throw new Error('Not implemented')
  }

  /**
   * Get evidence index (rows, artifacts, events)
   * @param {import('./types.js').EvidenceRequest} req - Evidence request
   * @returns {Promise<import('./types.js').EvidenceIndex>}
   */
  async evidence(req) {
    throw new Error('Not implemented')
  }

  /**
   * Export capsule artifacts
   * @param {import('./types.js').ExportRequest} req - Export request
   * @returns {Promise<import('./types.js').ExportResult>}
   */
  async exportCapsule(req) {
    throw new Error('Not implemented')
  }

  /**
   * Get the base URL for the engine
   * Used for constructing direct download links
   * @returns {string}
   */
  getBaseUrl() {
    throw new Error('Not implemented')
  }

  /**
   * Get artifact download URL
   * @param {string} runId - Run identifier
   * @param {string} artifactName - Artifact filename
   * @returns {string}
   */
  getArtifactUrl(runId, artifactName) {
    throw new Error('Not implemented')
  }

  /**
   * Get capsule download URL
   * @param {string} runId - Run identifier
   * @returns {string}
   */
  getCapsuleUrl(runId) {
    throw new Error('Not implemented')
  }

  /**
   * Get row data URL
   * @param {string} runId - Run identifier
   * @param {number} rowNumber - Row number
   * @returns {string}
   */
  getRowUrl(runId, rowNumber) {
    throw new Error('Not implemented')
  }
}
