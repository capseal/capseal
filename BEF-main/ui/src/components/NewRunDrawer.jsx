import { useState, useCallback } from 'react'
import Drawer from './Drawer'
import Toast from './Toast'

const BACKEND_OPTIONS = [
  { value: 'geom', label: 'Geom (STARK/FRI)' },
  { value: 'risc0', label: 'RISC0' },
  { value: 'mock', label: 'Mock' },
]

const MODE_OPTIONS = [
  { value: 'lab', label: 'Lab', desc: 'Relaxed constraints, for testing' },
  { value: 'production', label: 'Production', desc: 'Strict policy enforcement' },
]

/**
 * NewRunDrawer - Circuit-builder feel drawer for starting a new run
 *
 * Props:
 * - open: boolean - whether drawer is visible
 * - onClose: function - callback to close the drawer
 * - projects: array of track_ids from existing runs
 * - circuits: array of policy_ids from existing runs
 */
export default function NewRunDrawer({ open, onClose, projects = [], circuits = [] }) {
  const [project, setProject] = useState('')
  const [circuit, setCircuit] = useState('')
  const [backend, setBackend] = useState('geom')
  const [mode, setMode] = useState('lab')
  const [toast, setToast] = useState(null)

  const generateTimestamp = () => {
    const now = new Date()
    return now.toISOString().replace(/[:.]/g, '-').slice(0, 19)
  }

  const generateCommand = useCallback(() => {
    const timestamp = generateTimestamp()
    const parts = ['capseal run']

    if (circuit) {
      parts.push(`-p ${circuit}`)
    }
    if (project) {
      parts.push(`--track-id ${project}`)
    }
    parts.push(`--backend ${backend}`)
    if (mode === 'production') {
      parts.push('--strict')
    }
    parts.push(`-o ./out/${timestamp}`)

    return parts.join(' ')
  }, [project, circuit, backend, mode])

  const handleCopyCommand = async () => {
    const command = generateCommand()
    try {
      await navigator.clipboard.writeText(command)
      setToast({ message: 'CLI command copied to clipboard', type: 'success' })
    } catch (err) {
      // Fallback for older browsers
      const textarea = document.createElement('textarea')
      textarea.value = command
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setToast({ message: 'CLI command copied to clipboard', type: 'success' })
    }
  }

  const handleClose = () => {
    // Reset form state
    setProject('')
    setCircuit('')
    setBackend('geom')
    setMode('lab')
    onClose?.()
  }

  return (
    <>
      <Drawer open={open} onClose={handleClose} title="Run & Seal">
        <div className="new-run-form">
          {/* Project Dropdown */}
          <div className="form-field">
            <label className="form-label">Project</label>
            <select
              className="form-select"
              value={project}
              onChange={(e) => setProject(e.target.value)}
            >
              <option value="">Select a project...</option>
              {projects.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
              <option value="_new">+ New project...</option>
            </select>
          </div>

          {/* Circuit/Policy Dropdown */}
          <div className="form-field">
            <label className="form-label">Circuit</label>
            <select
              className="form-select"
              value={circuit}
              onChange={(e) => setCircuit(e.target.value)}
            >
              <option value="">Select a circuit...</option>
              {circuits.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
              <option value="_new">+ Create new circuit...</option>
            </select>
          </div>

          {/* Backend Dropdown */}
          <div className="form-field">
            <label className="form-label">Backend</label>
            <select
              className="form-select"
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
            >
              {BACKEND_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <span className="form-hint">
              {backend === 'geom' && 'Fast polynomial commitment proofs'}
              {backend === 'risc0' && 'zkVM-based proving'}
              {backend === 'mock' && 'No proof generation (dev only)'}
            </span>
          </div>

          {/* Mode Toggle */}
          <div className="form-field">
            <label className="form-label">Mode</label>
            <div className="mode-toggle">
              {MODE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`mode-option ${mode === opt.value ? 'active' : ''}`}
                  onClick={() => setMode(opt.value)}
                >
                  <span className="mode-label">{opt.label}</span>
                  <span className="mode-desc">{opt.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Generated Command Preview */}
          <div className="form-field">
            <label className="form-label">CLI Command</label>
            <div className="command-preview">
              <code>{generateCommand()}</code>
            </div>
          </div>

          {/* Actions */}
          <div className="drawer-actions">
            <button className="btn btn-secondary" onClick={handleClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleCopyCommand}>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              Copy CLI Command
            </button>
          </div>
        </div>
      </Drawer>

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </>
  )
}
