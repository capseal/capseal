/**
 * CapsuleTech Engine Adapter - Type Definitions
 *
 * FROZEN JSON CONTRACTS
 * These types are the canonical interface between UI and execution engine.
 * They must remain stable across backend implementations (Flask → Rust).
 *
 * Uses JSDoc for type annotations (no TypeScript in this project).
 */

// =============================================================================
// Engine Configuration
// =============================================================================

/**
 * @typedef {'local' | 'remote'} EngineMode
 */

/**
 * @typedef {Object} EngineConfig
 * @property {EngineMode} mode - Execution mode
 * @property {string} baseUrl - API base URL (default: http://localhost:5001)
 * @property {number} [timeoutMs] - Request timeout in milliseconds
 * @property {string} [ssePath] - SSE path for event streaming
 */

// =============================================================================
// Health Status
// =============================================================================

/**
 * @typedef {Object} HealthCapabilities
 * @property {boolean} run - Can execute runs
 * @property {boolean} verify - Can verify proofs
 * @property {boolean} audit - Can audit event chains
 * @property {boolean} evidence - Can provide evidence/openings
 * @property {boolean} export - Can export capsules
 * @property {boolean} sse - Supports Server-Sent Events
 */

/**
 * @typedef {Object} HealthStatus
 * @property {boolean} ok - Overall health status
 * @property {'flask' | 'rust' | 'remote'} engine - Engine implementation
 * @property {string} version - Version string
 * @property {HealthCapabilities} capabilities - Feature capabilities
 */

// =============================================================================
// Run Start Request/Response
// =============================================================================

/**
 * @typedef {Object} BudgetSpec
 * @property {number} tokens_per_node - Max tokens per oracle node
 * @property {number} tokens_total - Total token budget
 * @property {number} oracle_calls - Max oracle calls
 * @property {number} proof_size_kb - Max proof size in KB
 * @property {number} verify_time_ms - Max verification time
 * @property {number} context_window - Context window size
 * @property {number} parallel_branches - Max parallel branches
 */

/**
 * @typedef {Object} RunStartRequest
 * @property {string} project_id - Stable logical project (track_id)
 * @property {string} [circuit_id] - Optional policy/circuit reference
 * @property {'quick_trace' | 'project_run'} mode - Run mode
 * @property {string} input_path - Local path or repo path
 * @property {string} [command] - Optional command to execute
 * @property {string} [policy_path] - Optional explicit policy file path
 * @property {string} [policy_id] - Optional policy reference
 * @property {Partial<BudgetSpec>} [budgets] - Override budget caps
 * @property {Record<string, string>} [tags] - User labels
 */

/**
 * @typedef {Object} EventsStream
 * @property {'sse'} type - Stream type
 * @property {string} url - Stream URL
 */

/**
 * @typedef {Object} EngineInfo
 * @property {EngineMode} mode - Engine mode
 * @property {string} baseUrl - Base URL
 */

/**
 * @typedef {Object} RunStartResponse
 * @property {string} run_id - UUID or hash
 * @property {string} started_at - ISO timestamp
 * @property {'queued' | 'running'} status - Initial status
 * @property {EngineInfo} engine - Engine info
 * @property {EventsStream} events_stream - Event stream info
 */

// =============================================================================
// Run Events (SSE payloads)
// =============================================================================

/**
 * @typedef {Object} BudgetSpent
 * @property {number} tokens - Tokens consumed
 * @property {number} oracle_calls - Oracle calls made
 * @property {number} [wall_time_ms] - Wall clock time
 */

/**
 * @typedef {Object} RunEvent
 * @property {string} run_id - Run identifier
 * @property {number} seq - Sequence number (strictly increasing)
 * @property {string} ts - ISO timestamp
 * @property {string} type - Event type
 * @property {string} [hash] - Event hash (for audit chain)
 * @property {string} [prev_hash] - Previous event hash
 * @property {string} [message] - UI-friendly message
 * @property {Record<string, any>} [data] - Event payload
 * @property {BudgetSpent} [budget_spent] - Budget consumed so far
 */

// =============================================================================
// Run Summary/Record
// =============================================================================

/**
 * @typedef {'running' | 'verified' | 'failed' | 'unverified'} RunStatus
 */

/**
 * @typedef {Object} BudgetInfo
 * @property {BudgetSpent} spent - Budget spent
 * @property {BudgetSpec} [limits] - Budget limits
 */

/**
 * @typedef {Object} RunSummary
 * @property {string} run_id - Run identifier
 * @property {string} project_id - Project identifier
 * @property {string} [circuit_id] - Circuit identifier
 * @property {string} started_at - ISO timestamp
 * @property {string} [ended_at] - ISO timestamp
 * @property {RunStatus} status - Run status
 * @property {string} [capsule_hash] - Capsule hash
 * @property {string} [proof_hash] - Proof hash
 * @property {BudgetInfo} [budgets] - Budget information
 */

/**
 * @typedef {Object} ArtifactRef
 * @property {string} name - Artifact name
 * @property {string} path - File path
 * @property {string} hash - Content hash
 * @property {number} size_bytes - Size in bytes
 * @property {'receipt' | 'proof' | 'trace' | 'events' | 'manifest' | 'other'} artifact_type - Type
 */

/**
 * @typedef {RunSummary & {
 *   artifacts: ArtifactRef[],
 *   events_root?: string,
 *   receipt_path?: string,
 *   logs_path?: string
 * }} RunRecord
 */

// =============================================================================
// Verification (L0-L4 Trust Ladder)
// =============================================================================

/**
 * @typedef {'L0' | 'L1' | 'L2' | 'L3' | 'L4'} VerifyLayerId
 */

/**
 * @typedef {Object} EvidencePointer
 * @property {'merkle_opening' | 'file' | 'event'} kind - Evidence kind
 * @property {string} ref - Reference (path, event_seq, or opening id)
 */

/**
 * @typedef {Object} VerifyLayer
 * @property {VerifyLayerId} id - Layer ID
 * @property {boolean} ok - Layer passed
 * @property {string} label - Layer label (e.g., "SHA-256 commitments")
 * @property {string} [details] - Short explanation
 * @property {Record<string, number>} [metrics] - Metrics (ms, kb, etc.)
 * @property {EvidencePointer[]} [evidence] - Evidence pointers
 */

/**
 * @typedef {Object} VerifyError
 * @property {string} code - Error code
 * @property {string} message - Error message
 */

/**
 * @typedef {Object} VerifyReport
 * @property {string} [run_id] - Run identifier
 * @property {string} capsule_hash - Capsule hash
 * @property {string} verified_at - ISO timestamp
 * @property {boolean} ok - Overall verification result
 * @property {VerifyLayer[]} layers - Verification layers (L0-L4)
 * @property {Record<string, number>} [timings_ms] - Timing breakdown
 * @property {VerifyError[]} [errors] - Errors encountered
 */

/**
 * @typedef {Object} VerifyRequest
 * @property {string} run_id - Run identifier
 * @property {string} [mode] - Verification mode
 */

// =============================================================================
// Audit Report
// =============================================================================

/**
 * @typedef {Object} AuditFinding
 * @property {'info' | 'warn' | 'fail'} severity - Finding severity
 * @property {string} code - Finding code
 * @property {string} message - Finding message
 * @property {number} [event_seq] - Related event sequence
 * @property {EvidencePointer[]} [evidence] - Evidence pointers
 */

/**
 * @typedef {Object} AuditChain
 * @property {string} head_hash - Chain head hash
 * @property {number} length - Chain length
 */

/**
 * @typedef {Object} AuditReport
 * @property {string} run_id - Run identifier
 * @property {boolean} ok - Audit passed
 * @property {string} [policy_id] - Policy identifier
 * @property {string} checked_at - ISO timestamp
 * @property {AuditChain} chain - Hash chain info
 * @property {AuditFinding[]} findings - Audit findings
 */

/**
 * @typedef {Object} AuditRequest
 * @property {string} run_id - Run identifier
 * @property {boolean} [verify_chain] - Verify hash chain
 */

// =============================================================================
// Evidence Index
// =============================================================================

/**
 * @typedef {Object} EvidenceRows
 * @property {number} count - Row count
 * @property {number} chunks - Chunk count
 * @property {string} [chunk_roots_path] - Path to chunk roots
 */

/**
 * @typedef {Object} EvidenceEvents
 * @property {number} count - Event count
 * @property {string} [chain_head] - Chain head hash
 */

/**
 * @typedef {Object} EvidenceIndex
 * @property {string} run_id - Run identifier
 * @property {EvidenceRows} [rows] - Row evidence
 * @property {ArtifactRef[]} [artifacts] - Artifact evidence
 * @property {EvidenceEvents} [events] - Event evidence
 */

/**
 * @typedef {Object} EvidenceRequest
 * @property {string} run_id - Run identifier
 * @property {'rows' | 'artifacts' | 'events'} kind - Evidence kind
 */

// =============================================================================
// Export
// =============================================================================

/**
 * @typedef {Object} ExportRequest
 * @property {string} run_id - Run identifier
 * @property {'capsule' | 'receipt' | 'proof' | 'full'} format - Export format
 */

/**
 * @typedef {Object} ExportResult
 * @property {string} path - Export path
 * @property {number} size_bytes - Export size
 * @property {string} hash - Export hash
 */

// =============================================================================
// Run List Query
// =============================================================================

/**
 * @typedef {Object} RunListQuery
 * @property {string} [project_id] - Filter by project
 * @property {RunStatus} [status] - Filter by status
 * @property {number} [limit] - Result limit
 * @property {number} [offset] - Result offset
 */

export {}
