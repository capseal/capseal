/**
 * CapSeal UI↔Backend Contracts
 *
 * CANONICAL SOURCE OF TRUTH
 *
 * All API responses must conform to these shapes exactly.
 * Backend changes (Python → Rust) must preserve these contracts.
 *
 * Naming convention: snake_case (matches Python backend)
 */

// =============================================================================
// Layer Status (L0-L4 Verification Ladder)
// =============================================================================

export type LayerStatus = 'pass' | 'fail' | 'skipped' | 'unknown';

export interface LayerCheck {
  status: LayerStatus;
  message?: string;
  evidence_ref?: string;  // Reference to artifact/row for detailed inspection
}

export interface VerificationLayers {
  l0_hash: LayerCheck;        // Artifact hashes match commitments
  l1_commitment: LayerCheck;  // Merkle roots valid
  l2_constraint: LayerCheck;  // Policy/AIR constraints enforced
  l3_proximity: LayerCheck;   // FRI low-degree test passed
  l4_receipt: LayerCheck;     // Capsule format valid, all fields present
}

// =============================================================================
// RunSummary - Used in RunsList
// =============================================================================

export type VerificationStatus = 'verified' | 'failed' | 'running' | 'unverified';

export interface RunSummary {
  run_id: string;
  project_id?: string;        // Derived from track_id or explicit
  track_id?: string;
  policy_id?: string;
  policy_hash?: string;
  backend: string;            // e.g., "geom_stc_fri", "risc0", "mock"
  capsule_hash?: string;
  verification_status: VerificationStatus;
  created_at: string;         // ISO 8601
  last_event?: string;        // Most recent event type
  event_count?: number;
}

// =============================================================================
// VerifyReport - Returned by POST /runs/:id/verify
// =============================================================================

export interface VerifyError {
  code: string;               // e.g., "E052_PROOF_HASH_MISMATCH"
  message: string;
  hint?: string;
  evidence_ref?: string;      // Reference to inspect for details
}

export interface VerifyTimings {
  total_ms: number;
  parse_ms?: number;
  proof_verify_ms?: number;
  merkle_verify_ms?: number;
}

export interface VerifyReport {
  run_id: string;
  status: 'verified' | 'rejected';
  exit_code: number;          // 0=verified, 10=proof fail, 11=policy, etc.

  layers: VerificationLayers;
  errors: VerifyError[];

  timings: VerifyTimings;
  proof_size_bytes?: number;

  backend_id: string;
  proof_system_id: string;    // e.g., "stark_fri", "groth16"

  verified_at: string;        // ISO 8601
}

// =============================================================================
// AuditReport - Returned by GET /runs/:id/audit
// =============================================================================

export interface EventSummary {
  seq: number;
  event_type: string;
  event_hash: string;
  prev_hash: string;
  timestamp?: string;
}

export interface EventTypeCounts {
  [event_type: string]: number;
}

export interface AuditReport {
  run_id: string;

  chain_valid: boolean;
  chain_length: number;
  genesis_hash: string;       // Should be zeros
  head_hash: string;          // Latest event hash

  event_counts: EventTypeCounts;

  timeline: EventSummary[];   // Ordered by seq

  first_event?: EventSummary;
  last_event?: EventSummary;

  audited_at: string;         // ISO 8601
}

// =============================================================================
// EvidenceIndex - Returned by GET /runs/:id/evidence
// =============================================================================

export interface OpenableRow {
  row_index: number;
  chunk_file: string;
  has_proof: boolean;
}

export interface OpenableArtifact {
  name: string;
  path: string;
  hash: string;
  size_bytes: number;
  artifact_type: 'receipt' | 'proof' | 'trace' | 'events' | 'manifest' | 'other';
}

export interface RowOpening {
  row_index: number;
  values: (string | number)[];  // Field values
  proof_levels: number;
  proof_valid: boolean;
  chunk_file: string;
  merkle_root: string;
}

export interface EvidenceIndex {
  run_id: string;

  // Commitment roots
  capsule_hash: string;
  row_root: string;
  events_root?: string;

  // What can be opened/inspected
  openable_rows: OpenableRow[];
  row_count: number;
  tree_arity: number;

  artifacts: OpenableArtifact[];

  // Evidence strip status (pre-computed for UI)
  evidence_status: {
    binding: LayerStatus;
    availability: LayerStatus;
    enforcement: LayerStatus;
    determinism: LayerStatus;
    replayable: LayerStatus;
  };
}

// =============================================================================
// BudgetSummary - Returned by GET /runs/:id/budget
// =============================================================================

export interface OracleCallRecord {
  seq: number;
  oracle_id: string;
  call_id: string;
  context_root: string;           // First 16 chars
  tokens_in: number;
  tokens_out: number;
  model: string;
  cost_usd: number;
  latency_ms?: number;
  success: boolean;
  timestamp?: number;
}

export interface BudgetLimits {
  tokens: number;                 // Max tokens allowed
  oracle_calls: number;           // Max oracle calls
  usd: number;                    // Max cost in USD
}

export interface BudgetSpent {
  tokens_in: number;
  tokens_out: number;
  tokens_total: number;
  oracle_calls: number;
  usd: number;
}

export interface BudgetRemaining {
  tokens: number;
  oracle_calls: number;
  usd: number;
}

export interface BudgetUtilization {
  tokens_pct: number;             // 0-100
  calls_pct: number;
  usd_pct: number;
}

export interface BudgetSummary {
  run_id: string;

  budget: BudgetLimits;
  spent: BudgetSpent;
  remaining: BudgetRemaining;
  utilization: BudgetUtilization;

  oracle_calls: OracleCallRecord[];
  governance_enabled: boolean;    // Whether full governance module is active
}

export function isBudgetSummary(obj: unknown): obj is BudgetSummary {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.run_id === 'string' &&
    typeof o.budget === 'object' &&
    typeof o.spent === 'object' &&
    Array.isArray(o.oracle_calls)
  );
}

// =============================================================================
// Run Request/Response - For POST /run
// =============================================================================

export type ExecutionMode = 'local' | 'remote';

export interface RunRequest {
  project_id?: string;
  policy_path: string;
  input_path: string;
  mode: ExecutionMode;
  backend?: string;
  track_id?: string;
}

export interface RunStartResponse {
  run_id: string;
  output_dir: string;
  started_at: string;
}

// =============================================================================
// SSE Event - For GET /runs/:id/events (streaming)
// =============================================================================

export interface SSEEvent {
  seq: number;
  event_type: string;
  event_hash: string;
  prev_hash: string;
  timestamp: string;
  data?: Record<string, unknown>;
}

// =============================================================================
// Error Codes (stable across implementations)
// =============================================================================

export const ERROR_CODES = {
  // Parse errors (20)
  E001_PARSE_FAILED: { code: 'E001_PARSE_FAILED', exit: 20 },
  E002_SCHEMA_UNSUPPORTED: { code: 'E002_SCHEMA_UNSUPPORTED', exit: 20 },

  // Capsule integrity (20)
  E011_CAPSULE_HASH_MISMATCH: { code: 'E011_CAPSULE_HASH_MISMATCH', exit: 20 },

  // Authorship (20)
  E021_SIGNATURE_INVALID: { code: 'E021_SIGNATURE_INVALID', exit: 20 },

  // Policy (11)
  E031_POLICY_NOT_FOUND: { code: 'E031_POLICY_NOT_FOUND', exit: 11 },
  E032_POLICY_VERSION_MISMATCH: { code: 'E032_POLICY_VERSION_MISMATCH', exit: 11 },
  E033_POLICY_HASH_MISMATCH: { code: 'E033_POLICY_HASH_MISMATCH', exit: 11 },

  // Proof binding (10)
  E051_PROOF_MISSING: { code: 'E051_PROOF_MISSING', exit: 10 },
  E052_PROOF_HASH_MISMATCH: { code: 'E052_PROOF_HASH_MISMATCH', exit: 10 },
  E053_PROOF_VERIFICATION_FAILED: { code: 'E053_PROOF_VERIFICATION_FAILED', exit: 10 },

  // Commitment/Index (12)
  E061_ROW_ROOT_MISMATCH: { code: 'E061_ROW_ROOT_MISMATCH', exit: 12 },
  E062_CHUNK_MISSING: { code: 'E062_CHUNK_MISSING', exit: 12 },
  E063_CHUNK_HASH_MISMATCH: { code: 'E063_CHUNK_HASH_MISMATCH', exit: 12 },
  E064_MERKLE_PROOF_INVALID: { code: 'E064_MERKLE_PROOF_INVALID', exit: 12 },

  // DA Audit (13)
  E071_EVENTS_MISSING: { code: 'E071_EVENTS_MISSING', exit: 13 },
  E072_CHAIN_BROKEN: { code: 'E072_CHAIN_BROKEN', exit: 13 },
  E073_GENESIS_INVALID: { code: 'E073_GENESIS_INVALID', exit: 13 },
  E074_AVAILABILITY_FAILED: { code: 'E074_AVAILABILITY_FAILED', exit: 13 },

  // Replay (14)
  E081_REPLAY_DIVERGED: { code: 'E081_REPLAY_DIVERGED', exit: 14 },
} as const;

export type ErrorCode = keyof typeof ERROR_CODES;

// =============================================================================
// Exit Codes (stable across implementations)
// =============================================================================

export const EXIT_CODES = {
  VERIFIED: 0,
  PROOF_FAILED: 10,
  POLICY_MISMATCH: 11,
  COMMITMENT_FAILED: 12,
  DA_AUDIT_FAILED: 13,
  REPLAY_DIVERGED: 14,
  PARSE_ERROR: 20,
} as const;

// =============================================================================
// Type Guards (runtime validation helpers)
// =============================================================================

export function isVerifyReport(obj: unknown): obj is VerifyReport {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.run_id === 'string' &&
    (o.status === 'verified' || o.status === 'rejected') &&
    typeof o.exit_code === 'number' &&
    typeof o.layers === 'object' &&
    Array.isArray(o.errors)
  );
}

export function isAuditReport(obj: unknown): obj is AuditReport {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.run_id === 'string' &&
    typeof o.chain_valid === 'boolean' &&
    typeof o.chain_length === 'number' &&
    Array.isArray(o.timeline)
  );
}

export function isEvidenceIndex(obj: unknown): obj is EvidenceIndex {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.run_id === 'string' &&
    typeof o.capsule_hash === 'string' &&
    Array.isArray(o.openable_rows) &&
    Array.isArray(o.artifacts)
  );
}

export function isRunSummary(obj: unknown): obj is RunSummary {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.run_id === 'string' &&
    typeof o.backend === 'string' &&
    typeof o.verification_status === 'string'
  );
}
