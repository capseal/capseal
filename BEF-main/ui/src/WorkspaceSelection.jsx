import { useCallback, useEffect, useState, useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { getEngine } from './engine'
import { useEngine } from './state/EngineContext.jsx'
import EngineHandshakeCard from './components/EngineHandshakeCard'

// ═══════════════════════════════════════════════════════════════════════════
// LOCAL PROJECT STORAGE
// Persists projects to localStorage so they survive page reloads.
// ═══════════════════════════════════════════════════════════════════════════

const STORAGE_KEY = 'capseal_projects'

function getStoredProjects() {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch (err) {
    console.error('Failed to load projects from localStorage:', err)
    return []
  }
}

function saveProjects(projects) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
  } catch (err) {
    console.error('Failed to save projects to localStorage:', err)
  }
}

function addProject(project) {
  const projects = getStoredProjects()
  // Check if project with same ID already exists
  const existingIndex = projects.findIndex(p => p.id === project.id)
  if (existingIndex >= 0) {
    projects[existingIndex] = { ...projects[existingIndex], ...project, updatedAt: new Date().toISOString() }
  } else {
    projects.unshift({ ...project, createdAt: new Date().toISOString() })
  }
  saveProjects(projects)
  return projects
}

function deleteProject(projectId) {
  const projects = getStoredProjects().filter(p => p.id !== projectId)
  saveProjects(projects)
  return projects
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * WorkspaceSelection - Standalone project chooser page.
 *
 * Supports URL query params:
 * - ?mode=recent  : Show recent projects list (default)
 * - ?mode=new     : New Project flow
 * - ?mode=open    : Open existing project
 *
 * No rail visible - renders as full-screen standalone component.
 * Navigates to /p/:projectId/runs on project selection.
 */
export default function WorkspaceSelection() {
  const [searchParams] = useSearchParams()
  const mode = searchParams.get('mode') || 'recent'
  const [localProjects, setLocalProjects] = useState(getStoredProjects)
  const engineCtx = useEngine()

  const refreshProjects = useCallback(() => {
    setLocalProjects(getStoredProjects())
  }, [])

  const handleProjectCreated = useCallback((project) => {
    const updated = addProject(project)
    setLocalProjects(updated)
  }, [])

  const handleProjectDeleted = useCallback((projectId) => {
    const updated = deleteProject(projectId)
    setLocalProjects(updated)
  }, [])

  return (
    <div className="workspace-home-container">
      <EngineHandshakeCard />
      {mode === 'new' && <NewProjectMode onProjectCreated={handleProjectCreated} />}
      {mode === 'open' && <OpenProjectMode />}
      {(mode === 'recent' || !mode) && (
        <RecentProjectsMode
          localProjects={localProjects}
          onRefresh={refreshProjects}
          onDelete={handleProjectDeleted}
          engineStatus={engineCtx.status}
        />
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// RECENT PROJECTS MODE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * RecentProjectsMode - Shows both local projects and API-derived projects.
 */
function RecentProjectsMode({ localProjects, onRefresh, onDelete, engineStatus }) {
  const navigate = useNavigate()
  const [apiRuns, setApiRuns] = useState([])
  const [status, setStatus] = useState('loading')
  const [deleteConfirm, setDeleteConfirm] = useState(null)

  const fetchRuns = useCallback(async () => {
    setStatus('loading')
    try {
      if (!['online', 'degraded'].includes(engineStatus)) {
        setStatus('ready')
        return
      }
      const engine = getEngine()
      const runsList = await engine.listRuns()
      setApiRuns(runsList)
      setStatus('ready')
    } catch (err) {
      console.error('Failed to fetch runs:', err)
      setStatus('ready') // Still show local projects even if API fails
    }
  }, [engineStatus])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  // Merge local projects with API-derived projects
  const allProjects = useMemo(() => {
    const projectMap = new Map()

    // First, add local projects
    localProjects.forEach((proj) => {
      projectMap.set(proj.id, {
        id: proj.id,
        name: proj.name,
        inputPath: proj.inputPath,
        template: proj.template,
        policy: proj.policy,
        createdAt: proj.createdAt,
        updatedAt: proj.updatedAt,
        runs: [],
        verified: 0,
        failed: 0,
        isLocal: true,
      })
    })

    // Then, merge/enhance with API runs
    apiRuns.forEach((run) => {
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
          isLocal: false,
        })
      }

      const project = projectMap.get(trackId)
      project.runs.push(run)

      // Update last run timestamp
      if (!project.lastRun || new Date(run.created_at) > new Date(project.lastRun)) {
        project.lastRun = run.created_at
      }

      // Count statuses
      const runStatus = (run.verification_status || run.verification || '').toLowerCase()
      if (runStatus === 'verified') project.verified++
      if (runStatus === 'failed') project.failed++
    })

    // Convert to array and sort by most recent activity
    return Array.from(projectMap.values()).sort((a, b) => {
      const aDate = new Date(a.lastRun || a.updatedAt || a.createdAt || 0)
      const bDate = new Date(b.lastRun || b.updatedAt || b.createdAt || 0)
      return bDate - aDate
    })
  }, [localProjects, apiRuns])

  const handleProjectClick = (projectId) => {
    navigate(`/p/${projectId}/runs`)
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  const handleDelete = (e, projectId) => {
    e.stopPropagation()
    if (deleteConfirm === projectId) {
      onDelete(projectId)
      setDeleteConfirm(null)
    } else {
      setDeleteConfirm(projectId)
      // Auto-cancel after 3 seconds
      setTimeout(() => setDeleteConfirm(null), 3000)
    }
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
            + New
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

        {status === 'ready' && allProjects.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <div className="empty-state-title">No projects yet</div>
            <div className="empty-state-description">
              Create your first project to get started with cryptographic proofs.
            </div>
            <button className="btn btn-primary" onClick={() => handleModeChange('new')}>
              Create New Project
            </button>
          </div>
        )}

        {status === 'ready' && allProjects.length > 0 && (
          <div className="projects-grid-recent">
            {allProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => handleProjectClick(project.id)}
                onDelete={project.isLocal ? (e) => handleDelete(e, project.id) : null}
                deleteConfirm={deleteConfirm === project.id}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW PROJECT MODE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * NewProjectMode - Create a new project via 4-step stepper wizard.
 *
 * Steps:
 * 1. Template - Choose template (Backtest / Code audit / Research run / Custom)
 * 2. Inputs - Project name + input path
 * 3. Policy - Choose policy preset (default, strict, permissive) with "Advanced" expand
 * 4. Confirm - Review selections + Create button
 */
function NewProjectMode({ onProjectCreated }) {
  const navigate = useNavigate()
  const [step, setStep] = useState(1)
  const [template, setTemplate] = useState('')
  const [projectName, setProjectName] = useState('')
  const [inputPath, setInputPath] = useState('')
  const [policy, setPolicy] = useState('default')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [oracleBudget, setOracleBudget] = useState('1000')
  const [tokenLimit, setTokenLimit] = useState('50000')
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState(null)

  const templates = [
    { id: 'backtest', name: 'Backtest', description: 'Algorithmic trading backtest', icon: '📈' },
    { id: 'audit', name: 'Code Audit', description: 'Security code review', icon: '🔍' },
    { id: 'research', name: 'Research Run', description: 'Data science research', icon: '🔬' },
    { id: 'custom', name: 'Custom', description: 'Start from scratch', icon: '⚙️' },
  ]

  const policies = [
    { id: 'default', name: 'Default', description: 'Balanced verification and performance' },
    { id: 'strict', name: 'Strict', description: 'Maximum security, all checks enabled' },
    { id: 'permissive', name: 'Permissive', description: 'Fast verification, minimal checks' },
  ]

  const canGoNext = () => {
    switch (step) {
      case 1:
        return !!template
      case 2:
        return !!projectName.trim() && !!inputPath.trim()
      case 3:
        return true
      case 4:
        return false
      default:
        return false
    }
  }

  const handleNext = () => {
    setError(null)
    if (step < 4 && canGoNext()) {
      setStep(step + 1)
    }
  }

  const handleBack = () => {
    setError(null)
    if (step > 1) {
      setStep(step - 1)
    }
  }

  const handleCreate = async () => {
    if (!projectName.trim()) {
      setError('Please enter a project name')
      return
    }

    if (!inputPath.trim()) {
      setError('Please enter an input path')
      return
    }

    setIsCreating(true)
    setError(null)

    try {
      // Generate a URL-safe project ID
      const projectId = projectName
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
        || `project-${Date.now()}`

      // Create project object
      const project = {
        id: projectId,
        name: projectName.trim(),
        inputPath: inputPath.trim(),
        template,
        policy,
        settings: {
          oracleBudget: parseInt(oracleBudget, 10) || 1000,
          tokenLimit: parseInt(tokenLimit, 10) || 50000,
        },
      }

      // Save to localStorage
      onProjectCreated(project)

      // Small delay for UX feedback
      await new Promise(resolve => setTimeout(resolve, 300))

      // Navigate to the new project
      navigate(`/p/${projectId}/runs`)
    } catch (err) {
      console.error('Failed to create project:', err)
      setError(err.message || 'Failed to create project')
      setIsCreating(false)
    }
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isCreating && step < 4 && canGoNext()) {
      handleNext()
    }
  }

  return (
    <div className="workspace-home-page">
      <div className="workspace-header">
        <div className="workspace-header-content">
          <h1 className="workspace-title">Create New Project</h1>
          <p className="workspace-subtitle">Set up your project in 4 simple steps</p>
        </div>

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
            + New
          </button>
          <button
            className="workspace-mode-btn"
            onClick={() => handleModeChange('open')}
          >
            Open
          </button>
        </div>
      </div>

      <div className="workspace-content">
        <div className="stepper-container">
          {/* Stepper Indicator */}
          <div className="stepper-indicator">
            {[1, 2, 3, 4].map((s) => (
              <div key={s} className="stepper-step-group">
                <button
                  className={`stepper-dot ${s === step ? 'active' : ''} ${s < step ? 'completed' : ''}`}
                  onClick={() => s < step && setStep(s)}
                  disabled={s > step}
                >
                  {s < step ? '✓' : s}
                </button>
                <div className="stepper-label">
                  {s === 1 && 'Template'}
                  {s === 2 && 'Inputs'}
                  {s === 3 && 'Policy'}
                  {s === 4 && 'Confirm'}
                </div>
              </div>
            ))}
          </div>

          {/* Error Display */}
          {error && (
            <div className="stepper-error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}

          {/* Step Content */}
          <div className="stepper-content">
            {/* Step 1: Template */}
            {step === 1 && (
              <div className="stepper-step-content">
                <h2 className="step-title">Choose a Template</h2>
                <p className="step-description">Select a template to get started quickly</p>

                <div className="template-grid">
                  {templates.map((t) => (
                    <button
                      key={t.id}
                      className={`template-card ${template === t.id ? 'selected' : ''}`}
                      onClick={() => setTemplate(t.id)}
                    >
                      <div className="template-icon">{t.icon}</div>
                      <div className="template-name">{t.name}</div>
                      <div className="template-description">{t.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 2: Inputs */}
            {step === 2 && (
              <div className="stepper-step-content">
                <h2 className="step-title">Project Details</h2>
                <p className="step-description">Enter your project information</p>

                <div className="form-field">
                  <label className="form-label">Project Name *</label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="e.g., My First Proof"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isCreating}
                    autoFocus
                  />
                  <p className="form-hint">
                    This will be used as the project identifier. You can rename it later.
                  </p>
                </div>

                <div className="form-field">
                  <label className="form-label">Input Path *</label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="e.g., /data/input or ./backtest_data"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isCreating}
                  />
                  <p className="form-hint">
                    Path to your input data or repository. Can be local or remote.
                  </p>
                </div>

                {/* Path Preview */}
                {inputPath && (
                  <div className="path-preview">
                    <div className="path-preview-label">Path Preview</div>
                    <code className="path-preview-value">{inputPath}</code>
                  </div>
                )}
              </div>
            )}

            {/* Step 3: Policy */}
            {step === 3 && (
              <div className="stepper-step-content">
                <h2 className="step-title">Choose a Policy</h2>
                <p className="step-description">Select a policy preset for verification rules</p>

                <div className="policy-grid">
                  {policies.map((p) => (
                    <button
                      key={p.id}
                      className={`policy-card ${policy === p.id ? 'selected' : ''}`}
                      onClick={() => setPolicy(p.id)}
                    >
                      <div className="policy-name">{p.name}</div>
                      <div className="policy-description">{p.description}</div>
                    </button>
                  ))}
                </div>

                <div className="advanced-section">
                  <button
                    className="advanced-toggle"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                  >
                    <span>{showAdvanced ? '▼' : '▶'}</span>
                    <span>Advanced Settings</span>
                  </button>

                  {showAdvanced && (
                    <div className="advanced-content">
                      <div className="form-field">
                        <label className="form-label">Oracle Budget</label>
                        <input
                          type="number"
                          className="form-input"
                          placeholder="1000"
                          value={oracleBudget}
                          onChange={(e) => setOracleBudget(e.target.value)}
                          disabled={isCreating}
                        />
                        <p className="form-hint">Maximum oracle calls allowed</p>
                      </div>

                      <div className="form-field">
                        <label className="form-label">Token Limit</label>
                        <input
                          type="number"
                          className="form-input"
                          placeholder="50000"
                          value={tokenLimit}
                          onChange={(e) => setTokenLimit(e.target.value)}
                          disabled={isCreating}
                        />
                        <p className="form-hint">Maximum token count for computation</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Step 4: Confirm */}
            {step === 4 && (
              <div className="stepper-step-content">
                <h2 className="step-title">Review & Create</h2>
                <p className="step-description">Verify your project configuration</p>

                <div className="review-card">
                  <div className="review-section">
                    <div className="review-label">Template</div>
                    <div className="review-value">
                      {templates.find((t) => t.id === template)?.name || 'N/A'}
                    </div>
                  </div>

                  <div className="review-section">
                    <div className="review-label">Project Name</div>
                    <div className="review-value">{projectName}</div>
                  </div>

                  <div className="review-section">
                    <div className="review-label">Input Path</div>
                    <div className="review-value">
                      <code>{inputPath}</code>
                    </div>
                  </div>

                  <div className="review-section">
                    <div className="review-label">Policy</div>
                    <div className="review-value">
                      {policies.find((p) => p.id === policy)?.name || 'N/A'}
                    </div>
                  </div>

                  {showAdvanced && (
                    <>
                      <div className="review-section">
                        <div className="review-label">Oracle Budget</div>
                        <div className="review-value">{oracleBudget}</div>
                      </div>
                      <div className="review-section">
                        <div className="review-label">Token Limit</div>
                        <div className="review-value">{tokenLimit}</div>
                      </div>
                    </>
                  )}
                </div>

                <p className="review-note">
                  Your project will be saved locally and available from the Projects page.
                </p>
              </div>
            )}
          </div>

          {/* Navigation Buttons */}
          <div className="stepper-nav">
            <button
              className="btn btn-secondary"
              onClick={() => (step === 1 ? handleModeChange('recent') : handleBack())}
              disabled={isCreating}
            >
              {step === 1 ? 'Cancel' : 'Back'}
            </button>

            {step < 4 && (
              <button
                className="btn btn-primary"
                onClick={handleNext}
                disabled={!canGoNext() || isCreating}
              >
                Next
              </button>
            )}

            {step === 4 && (
              <button
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={isCreating}
              >
                {isCreating ? (
                  <>
                    <span className="btn-spinner" />
                    Creating...
                  </>
                ) : (
                  'Create Project'
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// OPEN PROJECT MODE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * OpenProjectMode - Open existing project by path/ID.
 */
function OpenProjectMode() {
  const navigate = useNavigate()
  const [projectPath, setProjectPath] = useState('')
  const [isOpening, setIsOpening] = useState(false)
  const [error, setError] = useState(null)

  const handleOpen = async () => {
    if (!projectPath.trim()) {
      setError('Please enter a project path or ID')
      return
    }

    setIsOpening(true)
    setError(null)

    try {
      // Clean the input - could be a path or just an ID
      const trackId = projectPath
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9-_]/g, '-')
        .replace(/^-|-$/g, '')

      if (!trackId) {
        throw new Error('Invalid project identifier')
      }

      // Small delay for UX
      await new Promise(resolve => setTimeout(resolve, 200))
      navigate(`/p/${trackId}/runs`)
    } catch (err) {
      setError(err.message || 'Failed to open project')
      setIsOpening(false)
    }
  }

  const handleModeChange = (newMode) => {
    navigate(`/workspace?mode=${newMode}`)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isOpening && projectPath.trim()) {
      handleOpen()
    }
  }

  return (
    <div className="workspace-home-page">
      <div className="workspace-header">
        <div className="workspace-header-content">
          <h1 className="workspace-title">Open Project</h1>
          <p className="workspace-subtitle">Enter a project ID or path to open</p>
        </div>

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
            + New
          </button>
          <button
            className="workspace-mode-btn workspace-mode-btn-active"
            onClick={() => handleModeChange('open')}
          >
            Open
          </button>
        </div>
      </div>

      <div className="workspace-content">
        <div className="workspace-form-container">
          <div className="workspace-form">
            {error && (
              <div className="form-error">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {error}
              </div>
            )}

            <div className="form-field">
              <label className="form-label">Project ID or Path</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g., my-project or /path/to/project"
                value={projectPath}
                onChange={(e) => {
                  setProjectPath(e.target.value)
                  setError(null)
                }}
                onKeyDown={handleKeyDown}
                disabled={isOpening}
                autoFocus
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

          <div className="workspace-form-help">
            <h3 className="help-title">Finding Your Project</h3>
            <p className="help-text">
              You can find your project ID in the dashboard or in your project files. Projects are identified by their track_id,
              which is assigned when the project is first created.
            </p>
            <div className="help-examples">
              <div className="help-example">
                <span className="help-example-label">Project ID:</span>
                <code>my-backtest-v1</code>
              </div>
              <div className="help-example">
                <span className="help-example-label">Path:</span>
                <code>/home/user/projects/trading</code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// PROJECT CARD
// ═══════════════════════════════════════════════════════════════════════════

/**
 * ProjectCard - Display a project with metadata.
 */
function ProjectCard({ project, onClick, onDelete, deleteConfirm }) {
  const runCount = project.runs?.length || 0
  const verifiedCount = project.verified || 0

  return (
    <button className="workspace-project-card" onClick={onClick}>
      <div className="project-card-header">
        <div className="project-card-icon">
          {project.isLocal ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            </svg>
          )}
        </div>
        <div className="project-card-title-section">
          <h3 className="project-card-title">{project.name}</h3>
          {project.isLocal && <span className="project-card-badge">Local</span>}
        </div>
      </div>

      {/* Show input path for local projects */}
      {project.inputPath && (
        <div className="project-card-path">
          <code>{project.inputPath}</code>
        </div>
      )}

      <div className="project-card-meta">
        {runCount > 0 && (
          <div className="meta-item">
            <span className="meta-label">Runs</span>
            <span className="meta-value">{runCount}</span>
          </div>
        )}
        {verifiedCount > 0 && (
          <div className="meta-item">
            <span className="meta-label">Verified</span>
            <span className="meta-value verified">{verifiedCount}</span>
          </div>
        )}
        {project.template && (
          <div className="meta-item">
            <span className="meta-label">Template</span>
            <span className="meta-value">{project.template}</span>
          </div>
        )}
        {(project.lastRun || project.createdAt) && (
          <div className="meta-item">
            <span className="meta-label">{project.lastRun ? 'Last Run' : 'Created'}</span>
            <span className="meta-value">{formatTimeAgo(project.lastRun || project.createdAt)}</span>
          </div>
        )}
      </div>

      {/* Delete button for local projects */}
      {onDelete && (
        <button
          className={`project-card-delete ${deleteConfirm ? 'confirm' : ''}`}
          onClick={onDelete}
          title={deleteConfirm ? 'Click again to confirm delete' : 'Delete project'}
        >
          {deleteConfirm ? 'Confirm?' : '×'}
        </button>
      )}

      <div className="project-card-arrow">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </div>
    </button>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

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
