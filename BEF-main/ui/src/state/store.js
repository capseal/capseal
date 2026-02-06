/**
 * Persistence & Cache Layer for CapsuleTech UI
 *
 * Purpose:
 * - Cache run data, verification reports, and audit reports
 * - Reduce unnecessary refetches from the backend
 * - Persist data across page refreshes and sessions
 * - Maintain "sealed receipt" model (append-only, immutable reads)
 *
 * Storage Strategy:
 * - IndexedDB for large data (runs, verify_reports, audit_reports)
 * - localStorage for small metadata (projects)
 * - Graceful fallback to memory cache if IndexedDB unavailable
 *
 * Data Model:
 * What persists:
 *   ✓ Run summaries (id, timestamp, status, hashes)
 *   ✓ Verification results
 *   ✓ Audit reports
 *   ✓ Project metadata
 *
 * What does NOT persist:
 *   ✗ Live execution streams
 *   ✗ In-progress runs
 *   ✗ Ephemeral logs
 */

// ═══════════════════════════════════════════════════════════════════════════
// TYPES (JSDoc for type hints)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @typedef {Object} RunSummary
 * @property {string} run_id
 * @property {string} project_id
 * @property {string} [circuit_id]
 * @property {string} started_at - ISO timestamp
 * @property {string} [ended_at] - ISO timestamp
 * @property {string} status - "running" | "verified" | "failed" | "unverified"
 * @property {string} [capsule_hash]
 * @property {string} [proof_hash]
 * @property {Object} [budgets]
 */

/**
 * @typedef {RunSummary & Object} RunRecord
 * @property {Array} artifacts
 * @property {string} [events_root]
 * @property {string} [receipt_path]
 * @property {string} [logs_path]
 */

/**
 * @typedef {Object} VerifyReport
 * @property {string} [run_id]
 * @property {string} capsule_hash
 * @property {string} verified_at - ISO timestamp
 * @property {boolean} ok
 * @property {Array} layers - VerifyLayer[]
 * @property {Object} [timings_ms]
 * @property {Array} [errors]
 */

/**
 * @typedef {Object} AuditReport
 * @property {string} run_id
 * @property {boolean} ok
 * @property {string} [policy_id]
 * @property {string} checked_at - ISO timestamp
 * @property {Object} chain - { head_hash, length }
 * @property {Array} findings
 */

/**
 * @typedef {Object} Project
 * @property {string} id
 * @property {string} name
 * @property {string} [inputPath]
 * @property {string} [template]
 * @property {string} [policy]
 * @property {string} [createdAt] - ISO timestamp
 * @property {string} [updatedAt] - ISO timestamp
 */

// ═══════════════════════════════════════════════════════════════════════════
// INDEXEDDB MANAGER
// ═══════════════════════════════════════════════════════════════════════════

const DB_NAME = 'capseal_store'
const DB_VERSION = 1

// Store names and their indices
const STORES = {
  runs: {
    name: 'runs',
    keyPath: 'run_id',
    indices: [
      { name: 'project_id', keyPath: 'project_id' },
      { name: 'status', keyPath: 'status' },
      { name: 'created_at', keyPath: 'started_at' },
    ],
  },
  verify_reports: {
    name: 'verify_reports',
    keyPath: 'run_id',
    indices: [
      { name: 'verified_at', keyPath: 'verified_at' },
    ],
  },
  audit_reports: {
    name: 'audit_reports',
    keyPath: 'run_id',
    indices: [
      { name: 'checked_at', keyPath: 'checked_at' },
    ],
  },
}

/**
 * Initialize IndexedDB with schema
 * @private
 * @returns {Promise<IDBDatabase>}
 */
async function initIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = event.target.result

      // Create or clear stores
      Object.values(STORES).forEach((store) => {
        // Delete if exists (fresh migration)
        if (db.objectStoreNames.contains(store.name)) {
          db.deleteObjectStore(store.name)
        }

        // Recreate with correct schema
        const objectStore = db.createObjectStore(store.name, { keyPath: store.keyPath })

        // Add indices
        store.indices.forEach((idx) => {
          objectStore.createIndex(idx.name, idx.keyPath, { unique: false })
        })
      })
    }
  })
}

/**
 * Get a transaction on one or more stores
 * @private
 * @param {IDBDatabase} db
 * @param {string|string[]} storeName
 * @param {string} mode - "readonly" | "readwrite"
 * @returns {IDBTransaction}
 */
function getTransaction(db, storeName, mode = 'readonly') {
  return db.transaction(
    Array.isArray(storeName) ? storeName : [storeName],
    mode,
  )
}

/**
 * Promise wrapper for IDBRequest
 * @private
 * @param {IDBRequest} request
 * @returns {Promise}
 */
function promiseRequest(request) {
  return new Promise((resolve, reject) => {
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)
  })
}

// ═══════════════════════════════════════════════════════════════════════════
// IN-MEMORY FALLBACK CACHE
// ═══════════════════════════════════════════════════════════════════════════

class MemoryCache {
  constructor() {
    this.runs = new Map()
    this.verifyReports = new Map()
    this.auditReports = new Map()
  }

  clear() {
    this.runs.clear()
    this.verifyReports.clear()
    this.auditReports.clear()
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// STORE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

class Store {
  constructor() {
    this.db = null
    this.memoryCache = new MemoryCache()
    this.useIndexedDB = false
    this._initPromise = this._init()
  }

  /**
   * Initialize store (IndexedDB or fallback to memory)
   * @private
   */
  async _init() {
    try {
      // Check if IndexedDB is available
      if (typeof indexedDB === 'undefined') {
        console.warn('IndexedDB not available, using memory cache')
        return
      }

      this.db = await initIndexedDB()
      this.useIndexedDB = true
      console.log('IndexedDB initialized successfully')
    } catch (err) {
      console.warn('Failed to initialize IndexedDB, using memory cache:', err.message)
      this.useIndexedDB = false
    }
  }

  /**
   * Ensure store is initialized
   * @private
   */
  async _ensureInit() {
    await this._initPromise
  }

  // ═════════════════════════════════════════════════════════════════════════
  // RUN OPERATIONS
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Get all runs (optionally filtered by project)
   * @param {Object} [query]
   * @param {string} [query.project_id] - Filter by project
   * @param {string} [query.status] - Filter by status
   * @returns {Promise<RunSummary[]>}
   */
  async getRuns(query = {}) {
    await this._ensureInit()

    if (!this.useIndexedDB) {
      return Array.from(this.memoryCache.runs.values()).filter((run) => {
        if (query.project_id && run.project_id !== query.project_id) return false
        if (query.status && run.status !== query.status) return false
        return true
      })
    }

    try {
      const tx = getTransaction(this.db, STORES.runs.name)
      let objectStore = tx.objectStore(STORES.runs.name)

      // Apply filter if project_id specified
      if (query.project_id) {
        objectStore = objectStore.index('project_id')
      }

      const request = objectStore.getAll()
      const runs = await promiseRequest(request)

      // Apply status filter in memory if needed
      if (query.status) {
        return runs.filter((r) => r.status === query.status)
      }

      return runs
    } catch (err) {
      console.error('Failed to get runs from IndexedDB:', err)
      return Array.from(this.memoryCache.runs.values())
    }
  }

  /**
   * Get a single run by ID
   * @param {string} runId
   * @returns {Promise<RunRecord|null>}
   */
  async getRun(runId) {
    await this._ensureInit()

    if (!this.useIndexedDB) {
      return this.memoryCache.runs.get(runId) || null
    }

    try {
      const tx = getTransaction(this.db, STORES.runs.name)
      const request = tx.objectStore(STORES.runs.name).get(runId)
      const run = await promiseRequest(request)
      return run || null
    } catch (err) {
      console.error(`Failed to get run ${runId} from IndexedDB:`, err)
      return this.memoryCache.runs.get(runId) || null
    }
  }

  /**
   * Save or update a run
   * @param {RunRecord} run
   * @returns {Promise<void>}
   */
  async saveRun(run) {
    await this._ensureInit()

    if (!run || !run.run_id) {
      throw new Error('Run must have a run_id')
    }

    // Always save to memory cache
    this.memoryCache.runs.set(run.run_id, run)

    if (!this.useIndexedDB) return

    try {
      const tx = getTransaction(this.db, STORES.runs.name, 'readwrite')
      const request = tx.objectStore(STORES.runs.name).put(run)
      await promiseRequest(request)
    } catch (err) {
      console.error(`Failed to save run ${run.run_id} to IndexedDB:`, err)
      // Continue - memory cache has the data
    }
  }

  /**
   * Update run status only
   * @param {string} runId
   * @param {string} status
   * @returns {Promise<void>}
   */
  async updateRunStatus(runId, status) {
    const run = await this.getRun(runId)
    if (!run) {
      throw new Error(`Run ${runId} not found`)
    }

    run.status = status
    run.updated_at = new Date().toISOString()

    await this.saveRun(run)
  }

  // ═════════════════════════════════════════════════════════════════════════
  // VERIFICATION REPORT OPERATIONS
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Get verification report for a run
   * @param {string} runId
   * @returns {Promise<VerifyReport|null>}
   */
  async getVerifyReport(runId) {
    await this._ensureInit()

    if (!this.useIndexedDB) {
      return this.memoryCache.verifyReports.get(runId) || null
    }

    try {
      const tx = getTransaction(this.db, STORES.verify_reports.name)
      const request = tx.objectStore(STORES.verify_reports.name).get(runId)
      const report = await promiseRequest(request)
      return report || null
    } catch (err) {
      console.error(`Failed to get verify report for run ${runId}:`, err)
      return this.memoryCache.verifyReports.get(runId) || null
    }
  }

  /**
   * Save a verification report
   * @param {string} runId
   * @param {VerifyReport} report
   * @returns {Promise<void>}
   */
  async saveVerifyReport(runId, report) {
    await this._ensureInit()

    if (!report) {
      throw new Error('Report cannot be empty')
    }

    // Ensure report has run_id
    const reportWithId = {
      ...report,
      run_id: runId,
    }

    // Always save to memory cache
    this.memoryCache.verifyReports.set(runId, reportWithId)

    if (!this.useIndexedDB) return

    try {
      const tx = getTransaction(this.db, STORES.verify_reports.name, 'readwrite')
      const request = tx.objectStore(STORES.verify_reports.name).put(reportWithId)
      await promiseRequest(request)
    } catch (err) {
      console.error(`Failed to save verify report for run ${runId}:`, err)
      // Continue - memory cache has the data
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // AUDIT REPORT OPERATIONS
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Get audit report for a run
   * @param {string} runId
   * @returns {Promise<AuditReport|null>}
   */
  async getAuditReport(runId) {
    await this._ensureInit()

    if (!this.useIndexedDB) {
      return this.memoryCache.auditReports.get(runId) || null
    }

    try {
      const tx = getTransaction(this.db, STORES.audit_reports.name)
      const request = tx.objectStore(STORES.audit_reports.name).get(runId)
      const report = await promiseRequest(request)
      return report || null
    } catch (err) {
      console.error(`Failed to get audit report for run ${runId}:`, err)
      return this.memoryCache.auditReports.get(runId) || null
    }
  }

  /**
   * Save an audit report
   * @param {string} runId
   * @param {AuditReport} report
   * @returns {Promise<void>}
   */
  async saveAuditReport(runId, report) {
    await this._ensureInit()

    if (!report) {
      throw new Error('Report cannot be empty')
    }

    // Ensure report has run_id
    const reportWithId = {
      ...report,
      run_id: runId,
    }

    // Always save to memory cache
    this.memoryCache.auditReports.set(runId, reportWithId)

    if (!this.useIndexedDB) return

    try {
      const tx = getTransaction(this.db, STORES.audit_reports.name, 'readwrite')
      const request = tx.objectStore(STORES.audit_reports.name).put(reportWithId)
      await promiseRequest(request)
    } catch (err) {
      console.error(`Failed to save audit report for run ${runId}:`, err)
      // Continue - memory cache has the data
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PROJECT OPERATIONS (localStorage)
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Get all projects from localStorage
   * @returns {Project[]}
   */
  getProjects() {
    try {
      const data = localStorage.getItem('capseal_projects')
      return data ? JSON.parse(data) : []
    } catch (err) {
      console.error('Failed to load projects from localStorage:', err)
      return []
    }
  }

  /**
   * Save a project to localStorage
   * @param {Project} project
   * @returns {void}
   */
  saveProject(project) {
    if (!project || !project.id) {
      throw new Error('Project must have an id')
    }

    try {
      const projects = this.getProjects()
      const existingIndex = projects.findIndex((p) => p.id === project.id)

      if (existingIndex >= 0) {
        // Update existing
        projects[existingIndex] = {
          ...projects[existingIndex],
          ...project,
          updatedAt: new Date().toISOString(),
        }
      } else {
        // Add new
        projects.unshift({
          ...project,
          createdAt: new Date().toISOString(),
        })
      }

      localStorage.setItem('capseal_projects', JSON.stringify(projects))
    } catch (err) {
      console.error('Failed to save project to localStorage:', err)
    }
  }

  /**
   * Delete a project from localStorage
   * @param {string} projectId
   * @returns {void}
   */
  deleteProject(projectId) {
    try {
      const projects = this.getProjects().filter((p) => p.id !== projectId)
      localStorage.setItem('capseal_projects', JSON.stringify(projects))
    } catch (err) {
      console.error('Failed to delete project from localStorage:', err)
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // CACHE CONTROL
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Invalidate (delete) a run from cache
   * @param {string} runId
   * @returns {Promise<void>}
   */
  async invalidateRun(runId) {
    await this._ensureInit()

    // Always remove from memory
    this.memoryCache.runs.delete(runId)
    this.memoryCache.verifyReports.delete(runId)
    this.memoryCache.auditReports.delete(runId)

    if (!this.useIndexedDB) return

    try {
      const tx = getTransaction(
        this.db,
        [STORES.runs.name, STORES.verify_reports.name, STORES.audit_reports.name],
        'readwrite',
      )

      await promiseRequest(tx.objectStore(STORES.runs.name).delete(runId))
      await promiseRequest(tx.objectStore(STORES.verify_reports.name).delete(runId))
      await promiseRequest(tx.objectStore(STORES.audit_reports.name).delete(runId))
    } catch (err) {
      console.error(`Failed to invalidate run ${runId} from IndexedDB:`, err)
      // Continue - memory cache already cleared
    }
  }

  /**
   * Clear all cached data (IndexedDB and memory)
   * CAUTION: This is destructive. Consider invalidateRun for targeted invalidation.
   * @returns {Promise<void>}
   */
  async clearAll() {
    await this._ensureInit()

    // Clear memory
    this.memoryCache.clear()

    if (!this.useIndexedDB) return

    try {
      // Clear all stores
      const tx = getTransaction(
        this.db,
        [STORES.runs.name, STORES.verify_reports.name, STORES.audit_reports.name],
        'readwrite',
      )

      await promiseRequest(tx.objectStore(STORES.runs.name).clear())
      await promiseRequest(tx.objectStore(STORES.verify_reports.name).clear())
      await promiseRequest(tx.objectStore(STORES.audit_reports.name).clear())
    } catch (err) {
      console.error('Failed to clear IndexedDB:', err)
      // Continue - memory cache already cleared
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // UTILITY / DEBUGGING
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Get cache statistics (for debugging)
   * @returns {Promise<Object>}
   */
  async getStats() {
    await this._ensureInit()

    const memoryStats = {
      runs: this.memoryCache.runs.size,
      verifyReports: this.memoryCache.verifyReports.size,
      auditReports: this.memoryCache.auditReports.size,
    }

    if (!this.useIndexedDB) {
      return {
        backend: 'memory',
        memory: memoryStats,
        indexeddb: null,
      }
    }

    try {
      const tx = getTransaction(
        this.db,
        [STORES.runs.name, STORES.verify_reports.name, STORES.audit_reports.name],
      )

      const runCount = await promiseRequest(tx.objectStore(STORES.runs.name).count())
      const verifyCount = await promiseRequest(
        tx.objectStore(STORES.verify_reports.name).count(),
      )
      const auditCount = await promiseRequest(
        tx.objectStore(STORES.audit_reports.name).count(),
      )

      return {
        backend: 'indexeddb',
        memory: memoryStats,
        indexeddb: {
          runs: runCount,
          verifyReports: verifyCount,
          auditReports: auditCount,
        },
      }
    } catch (err) {
      console.error('Failed to get IndexedDB stats:', err)
      return {
        backend: 'memory (fallback)',
        memory: memoryStats,
        indexeddb: null,
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SINGLETON EXPORT
// ═══════════════════════════════════════════════════════════════════════════

export const store = new Store()

// ═══════════════════════════════════════════════════════════════════════════
// USAGE EXAMPLES (in comments)
// ═══════════════════════════════════════════════════════════════════════════

/*
 * EXAMPLE 1: Save a run from backend
 * ──────────────────────────────────
 *
 * import { store } from '@/state/store'
 *
 * async function fetchAndCacheRun(runId) {
 *   const run = await backend.getRun(runId)
 *   await store.saveRun(run)
 *   return run
 * }
 *
 *
 * EXAMPLE 2: Use cached data with fallback to network
 * ───────────────────────────────────────────────────
 *
 * async function getRun(runId) {
 *   // Try cache first (instant)
 *   const cached = await store.getRun(runId)
 *   if (cached && !needsRefresh) {
 *     return cached
 *   }
 *
 *   // If not in cache or needs refresh, fetch from backend
 *   const fresh = await backend.getRun(runId)
 *   await store.saveRun(fresh)
 *   return fresh
 * }
 *
 *
 * EXAMPLE 3: Populate runs list from cache
 * ────────────────────────────────────────
 *
 * async function getRuns(projectId) {
 *   // Check cache first
 *   const cached = await store.getRuns({ project_id: projectId })
 *   if (cached.length > 0) {
 *     setRuns(cached)
 *   }
 *
 *   // Optionally refresh from backend
 *   const fresh = await backend.listRuns(projectId)
 *   for (const run of fresh) {
 *     await store.saveRun(run)
 *   }
 *   setRuns(fresh)
 * }
 *
 *
 * EXAMPLE 4: Save verification report
 * ───────────────────────────────────
 *
 * async function verifyAndCache(runId) {
 *   const report = await backend.verify(runId)
 *   await store.saveVerifyReport(runId, report)
 *   return report
 * }
 *
 *
 * EXAMPLE 5: Projects (localStorage, synchronous)
 * ───────────────────────────────────────────────
 *
 * // Get all projects (sync)
 * const projects = store.getProjects()
 *
 * // Save a project (sync)
 * store.saveProject({
 *   id: 'my-project',
 *   name: 'My Project',
 *   template: 'backtest',
 * })
 *
 * // Delete a project (sync)
 * store.deleteProject('my-project')
 */
