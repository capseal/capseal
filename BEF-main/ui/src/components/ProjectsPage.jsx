import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { SkeletonCard, EmptyState } from './index'
import { getEngine } from '../engine'

function formatRelativeTime(dateStr) {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return new Date(dateStr).toLocaleDateString()
}

function ProjectCard({ project, onClick, isActive }) {
  return (
    <button
      className={`project-card ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      <div className="project-card-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
      </div>
      <div className="project-card-content">
        <div className="project-card-title">{project.name}</div>
        <div className="project-card-meta">
          <span>{project.runCount} run{project.runCount !== 1 ? 's' : ''}</span>
          <span className="meta-dot" />
          <span>Last run {formatRelativeTime(project.lastRun)}</span>
        </div>
      </div>
      <div className="project-card-arrow">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </div>
    </button>
  )
}

export default function ProjectsPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const activeProject = searchParams.get('project')

  const [runs, setRuns] = useState([])
  const [status, setStatus] = useState('loading')

  const loadRuns = useCallback(async () => {
    setStatus('loading')
    try {
      const engine = getEngine()
      const array = await engine.listRuns()
      setRuns(array)
      setStatus('ready')
    } catch (err) {
      setStatus(`error: ${err.message}`)
    }
  }, [])

  useEffect(() => {
    loadRuns()
  }, [loadRuns])

  // Derive unique projects from runs based on track_id
  const projects = useMemo(() => {
    const projectMap = new Map()

    runs.forEach((run) => {
      const trackId = run.track_id || 'unassigned'
      const projectName = trackId === 'unassigned' ? 'Unassigned' : trackId

      if (!projectMap.has(trackId)) {
        projectMap.set(trackId, {
          id: trackId,
          name: projectName,
          runCount: 0,
          lastRun: null,
          runs: [],
        })
      }

      const project = projectMap.get(trackId)
      project.runCount += 1
      project.runs.push(run)

      // Update last run time
      const runTime = run.created_at ? new Date(run.created_at).getTime() : 0
      const lastRunTime = project.lastRun ? new Date(project.lastRun).getTime() : 0
      if (runTime > lastRunTime) {
        project.lastRun = run.created_at
      }
    })

    // Sort by last run time (most recent first)
    return Array.from(projectMap.values()).sort((a, b) => {
      const timeA = a.lastRun ? new Date(a.lastRun).getTime() : 0
      const timeB = b.lastRun ? new Date(b.lastRun).getTime() : 0
      return timeB - timeA
    })
  }, [runs])

  const handleProjectClick = (project) => {
    // Navigate to runs page filtered by track_id
    if (project.id === 'unassigned') {
      navigate('/runs?track_id=')
    } else {
      navigate(`/runs?track_id=${encodeURIComponent(project.id)}`)
    }
  }

  return (
    <div className="page-content">
      <div className="page-header">
        <h1>Projects</h1>
        <p className="page-subtitle">
          Organize runs by track ID. Click a project to view its runs.
        </p>
      </div>

      <div className="projects-grid">
        {status === 'loading' && (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        )}

        {status === 'ready' && projects.length === 0 && (
          <EmptyState
            title="No projects yet"
            description="Run your first capsule to create a project. Projects are derived from run track IDs."
          />
        )}

        {status === 'ready' && projects.length > 0 && (
          projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              isActive={activeProject === project.id}
              onClick={() => handleProjectClick(project)}
            />
          ))
        )}

        {status.startsWith('error') && (
          <EmptyState
            title="Failed to load projects"
            description={status}
            action="Retry"
            onAction={loadRuns}
          />
        )}
      </div>
    </div>
  )
}
