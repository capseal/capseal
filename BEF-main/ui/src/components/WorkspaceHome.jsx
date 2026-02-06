import { useCallback, useEffect, useState, useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { EmptyState } from './index'
import { getEngine } from '../engine'

/**
 * WorkspaceHome - Standalone project chooser page.
 *
 * Supports URL query params:
 * - ?mode=recent  : Show recent projects list (default)
 * - ?mode=new     : New Project flow
 * - ?mode=open    : Open existing project
 *
 * No rail visible - renders as full-screen standalone component.
 */
export default function WorkspaceHome() {
  const [searchParams] = useSearchParams()
  const mode = searchParams.get('mode') || 'recent'

  return (
    <div className="workspace-home-container">
      {mode === 'new' && <NewProjectMode />}
      {mode === 'open' && <OpenProjectMode />}
      {(mode === 'recent' || !mode) && <RecentProjectsMode />}
    </div>
  )
}

/**
 * RecentProjectsMode - Shows recent projects derived from runs.
 * Groups runs by track_id, displays as cards, allows navigation to project.
 */
function RecentProjectsMode() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState([])
  const [status, setStatus] = useState('loading')

  const fetchRuns = useCallback(async () => {
    setStatus('loading')
    try {
      const engine = getEngine()
      const runsList = await engine.listRuns()
      setRuns(runsList)
      setStatus('ready')
    } catch (err) {
      console.error('Failed to fetch runs:', err)
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  // Derive projects from runs, grouped by track_id
  const projects = useMemo(() => {
    const projectMap = new Map()

    runs.forEach((run) => {
      const trackId = run.track_id || 'default'
      const projectName = run.track_id || 'Default Project'

      if (!projectMap.has(trackId)) {
        projectMap.set(trackId, {
          id: trackId,
          name: projectName,
          runs: [],
          lastRun: null,
          verified: 0,
          failed: 0,
        })
      }

      const project = projectMap.get(trackId)
      project.runs.push(run)

      // Update last run timestamp
      if (!project.lastRun || new Date(run.created_at) > new Date(project.lastRun)) {
        project.lastRun = run.created_at
      }

      // Count statuses
      const status = (run.verification_status || run.verification || '').toLowerCase()
      if (status === 'verified') project.verified++
      if (status === 'failed') project.failed++
    })

    // Convert to array and sort by most recent
    return Array.from(projectMap.values()).sort(
      (a, b) => new Date(b.lastRun || 0) - new Date(a.lastRun || 0)
    )
  }, [runs])

  const handleProjectClick = (projectId) => {
    navigate(`/p/${projectId}/runs`)
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  return (
    <div className="workspace-home-page">
      {/* Header */}
      <div className="workspace-header">
        <div className="workspace-header-content">
          <h1 className="workspace-title">Projects</h1>
          <p className="workspace-subtitle">Choose a project to continue working</p>
        </div>

        {/* Mode Buttons */}
        <div className="workspace-mode-buttons">
          <button
            className="workspace-mode-btn workspace-mode-btn-active"
            onClick={() => handleModeChange('recent')}
          >
            Recent
          </button>
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('new')}
          >
            New
          </button>
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('open')}
          >
            Open
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="workspace-content">
        {status === 'loading' && (
          <div className="workspace-loading">
            <div className="loading-spinner" />
            <p>Loading projects...</p>
          </div>
        )}

        {status === 'error' && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <div className="empty-state-title">Failed to load projects</div>
            <div className="empty-state-text">
              There was an error loading your projects. Please try again.
            </div>
            <button className="btn btn-primary" onClick={fetchRuns}>
              Retry
            </button>
          </div>
        )}

        {status === 'ready' && projects.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <div className="empty-state-title">No projects yet</div>
            <div className="empty-state-text">
              Create your first project by running a computation.
            </div>
            <button className="btn btn-primary" onClick={() => handleModeChange('new')}>
              Create New Project
            </button>
          </div>
        )}

        {status === 'ready' && projects.length > 0 && (
          <div className="projects-grid-recent">
            {projects.slice(0, 10).map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => handleProjectClick(project.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * NewProjectMode - Create a new project (stub implementation).
 */
function NewProjectMode() {
  const navigate = useNavigate()
  const [projectName, setProjectName] = useState('')
  const [isCreating, setIsCreating] = useState(false)

  const handleCreate = async () => {
    if (!projectName.trim()) {
      alert('Please enter a project name')
      return
    }

    setIsCreating(true)
    try {
      // Stub: In a real implementation, this would create a project via API
      // For now, we'll just navigate to the project with a default track_id
      const trackId = projectName.toLowerCase().replace(/\s+/g, '-')
      setTimeout(() => {
        navigate(`/p/${trackId}/runs`)
      }, 500)
    } finally {
      setIsCreating(false)
    }
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isCreating) {
      handleCreate()
    }
  }

  return (
    <div className="workspace-home-page">
      {/* Header */}
      <div className="workspace-header">
        <div className="workspace-header-content">
          <h1 className="workspace-title">Create New Project</h1>
          <p className="workspace-subtitle">Start a new computation project</p>
        </div>

        {/* Mode Buttons */}
        <div className="workspace-mode-buttons">
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('recent')}
          >
            Recent
          </button>
          <button
            className="workspace-mode-btn workspace-mode-btn-active"
            onClick={() => handleModeChange('new')}
          >
            New
          </button>
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('open')}
          >
            Open
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="workspace-content">
        <div className="workspace-form-container">
          <div className="workspace-form">
            <div className="form-field">
              <label className="form-label">Project Name</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g., My First Proof"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isCreating}
              />
              <p className="form-hint">
                This will be used as the project identifier. You can rename it later.
              </p>
            </div>

            <div className="workspace-form-actions">
              <button
                className="btn btn-secondary"
                onClick={() => handleModeChange('recent')}
                disabled={isCreating}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={isCreating || !projectName.trim()}
              >
                {isCreating ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </div>

          {/* Help Text */}
          <div className="workspace-form-help">
            <h3 className="help-title">About Projects</h3>
            <p className="help-text">
              Projects are containers for related runs. Each run can belong to exactly one project (identified by track_id).
              Group related computations together to maintain organization and audit trails.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * OpenProjectMode - Open existing project by path/ID (stub implementation).
 */
function OpenProjectMode() {
  const navigate = useNavigate()
  const [projectPath, setProjectPath] = useState('')
  const [isOpening, setIsOpening] = useState(false)

  const handleOpen = async () => {
    if (!projectPath.trim()) {
      alert('Please enter a project path or ID')
      return
    }

    setIsOpening(true)
    try {
      // Stub: In a real implementation, this would verify the project exists
      // For now, we'll just navigate with the provided ID
      const trackId = projectPath.trim().toLowerCase()
      setTimeout(() => {
        navigate(`/p/${trackId}/runs`)
      }, 500)
    } finally {
      setIsOpening(false)
    }
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isOpening) {
      handleOpen()
    }
  }

  return (
    <div className="workspace-home-page">
      {/* Header */}
      <div className="workspace-header">
        <div className="workspace-header-content">
          <h1 className="workspace-title">Open Project</h1>
          <p className="workspace-subtitle">Enter a project ID or path to open</p>
        </div>

        {/* Mode Buttons */}
        <div className="workspace-mode-buttons">
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('recent')}
          >
            Recent
          </button>
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('new')}
          >
            New
          </button>
          <button
            className="workspace-mode-btn workspace-mode-btn-active"
            onClick={() => handleModeChange('open')}
          >
            Open
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="workspace-content">
        <div className="workspace-form-container">
          <div className="workspace-form">
            <div className="form-field">
              <label className="form-label">Project ID or Path</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g., my-project or /path/to/project"
                value={projectPath}
                onChange={(e) => setProjectPath(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isOpening}
              />
              <p className="form-hint">
                Enter the project identifier or file path. This will open the project dashboard.
              </p>
            </div>

            <div className="workspace-form-actions">
              <button
                className="btn btn-secondary"
                onClick={() => handleModeChange('recent')}
                disabled={isOpening}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleOpen}
                disabled={isOpening || !projectPath.trim()}
              >
                {isOpening ? 'Opening...' : 'Open Project'}
              </button>
            </div>
          </div>

          {/* Help Text */}
          <div className="workspace-form-help">
            <h3 className="help-title">Finding Your Project</h3>
            <p className="help-text">
              You can find your project ID in the dashboard or in your project files. Projects are identified by their track_id,
              which is assigned when the project is first created.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * ProjectCard - Display a project with metadata.
 */
function ProjectCard({ project, onClick }) {
  const runCount = project.runs.length
  const verifiedCount = project.verified || 0

  return (
    <button className="workspace-project-card" onClick={onClick}>
      <div className="project-card-header">
        <div className="project-card-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
        </div>
        <h3 className="project-card-title">{project.name}</h3>
      </div>

      <div className="project-card-meta">
        <div className="meta-item">
          <span className="meta-label">Runs</span>
          <span className="meta-value">{runCount}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Verified</span>
          <span className="meta-value verified">{verifiedCount}</span>
        </div>
        {project.lastRun && (
          <div className="meta-item">
            <span className="meta-label">Last Run</span>
            <span className="meta-value">{formatTimeAgo(project.lastRun)}</span>
          </div>
        )}
      </div>

      <div className="project-card-arrow">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </div>
    </button>
  )
}

function formatTimeAgo(isoString) {
  if (!isoString) return 'unknown'

  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now - date
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHour = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHour / 24)

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHour < 24) return `${diffHour}h ago`
  if (diffDay < 7) return `${diffDay}d ago`

  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
