const EVIDENCE_ITEMS = [
  { key: 'binding', label: 'Binding', icon: '1' },
  { key: 'availability', label: 'Availability', icon: '2' },
  { key: 'enforcement', label: 'Enforcement', icon: '3' },
  { key: 'determinism', label: 'Determinism', icon: '4' },
  { key: 'replayable', label: 'Replayable', icon: '5' },
]

/**
 * Get status for an evidence key from verification data.
 * Supports both contract format (layers object) and legacy format (flat report fields).
 */
function getStatus(verification, key) {
  if (!verification) return 'unknown'

  // Check for contract format: layers object with l0_hash, l1_commitment, etc.
  const layers = verification.layers
  if (layers && typeof layers === 'object' && !Array.isArray(layers)) {
    // Map evidence keys to contract layer keys
    const contractMapping = {
      binding: ['l0_hash', 'l1_commitment'], // hash + commitment = binding
      availability: [], // Not in L0-L4, check evidence_status if present
      enforcement: ['l2_constraint'], // policy constraint
      determinism: [], // Not in L0-L4
      replayable: [], // Not in L0-L4
    }

    const layerKeys = contractMapping[key] || []
    if (layerKeys.length > 0) {
      // Check if all mapped layers pass
      const statuses = layerKeys.map((k) => layers[k]?.status)
      if (statuses.every((s) => s === 'pass')) return 'pass'
      if (statuses.some((s) => s === 'fail')) return 'fail'
      return 'unknown'
    }

    // Check evidence_status fallback (from /evidence endpoint)
    const evidenceStatus = verification.evidence_status?.[key]
    if (evidenceStatus === 'pass') return 'pass'
    if (evidenceStatus === 'fail') return 'fail'
  }

  // Fall back to legacy format
  const report = verification.report || {}
  const checks = {
    binding: report.row_index_commitment_ok ?? report.binding_ok,
    availability: report.da_audit_verified ?? report.availability_ok,
    enforcement: report.policy_verified ?? report.policy_ok,
    determinism: report.determinism_ok ?? true,
    replayable: report.replay_ok ?? report.replayable,
  }

  const value = checks[key]
  if (value === true) return 'pass'
  if (value === false) return 'fail'
  return 'unknown'
}

export default function EvidenceStrip({ verification, onItemClick }) {
  return (
    <div className="evidence-strip">
      <div className="evidence-strip-header">
        <h4>Assurance</h4>
      </div>
      <div className="evidence-strip-items">
        {EVIDENCE_ITEMS.map(({ key, label, icon }) => {
          const status = getStatus(verification, key)
          return (
            <button
              key={key}
              className={`evidence-item evidence-${status}`}
              onClick={() => onItemClick?.(key)}
              title={`${label}: ${status}`}
            >
              <span className="evidence-marker" />
              <span className="evidence-label">{label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
