import { useEffect, useState } from 'react'

const STATUS_STYLES = {
  verified: 'status-verified',
  failed: 'status-failed',
  running: 'status-running',
  unverified: 'status-unverified',
  unknown: 'status-unknown',
}

export default function StatusPill({ status }) {
  const normalized = (status || 'unknown').toLowerCase()
  const className = STATUS_STYLES[normalized] || STATUS_STYLES.unknown

  return (
    <span className={`status-pill ${className}`}>
      {normalized === 'running' && <span className="status-dot" />}
      {status || 'Unknown'}
    </span>
  )
}
