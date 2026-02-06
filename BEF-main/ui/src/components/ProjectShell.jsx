import { useState, useCallback, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { AppShell, Rail, EmptyState, SkeletonRunRow } from './index'
import RunDetailPanel from '../RunDetailPanel'
import { getEngine } from '../engine'

/**
 * ProjectShell - Scoped runs experience for a single project.
 *
 * Route: /p/:projectId/runs or /p/:projectId/runs/:runId
 *
 * Features:
 * - Shows rail (navigation)
 * - Left panel: project header + scoped runs list
 * - Right panel: run detail
 * - Filters runs by track_id (unless projectId is "default")
 * - Deep linking support: /p/myproject/runs/run123
 * - URL state sync: selecting a run updates the URL
 */
export default function ProjectShell() {
  const { projectId, runId } = useParams()
  const navigate = useNavigate()
  const [selectedRunId, setSelectedRunId] = useState(runId || null)
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)

  // Fetch all runs once
  const fetchRuns = useCallback(async () => {
    setLoading(true)
    try {
      const engine = getEngine()
      const runsList = await engine.listRuns({ project_id: projectId })
      setRuns(Array.isArray(runsList) ? runsList : runsList.runs || [])
    } catch (err) {
      console.error('Failed to fetch runs:', err)
      setRuns([])
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  // Handle run selection - update URL and state
  const handleSelectRun = useCallback((id) => {
    setSelectedRunId(id)
    navigate(`/p/${projectId}/runs/${id}`, { replace: true })
  }, [projectId, navigate])

  // Handle first run loaded from filtered list
  const handleFirstRunLoaded = useCallback((id) => {
    if (!selectedRunId && id) {
      setSelectedRunId(id)
    }
  }, [selectedRunId])

  return (
    <AppShell
      rail={<Rail />}
      list={
        <ProjectRunsList
          projectId={projectId}
          selectedId={selectedRunId || runId}
          onSelect={handleSelectRun}
          onFirstRunLoaded={handleFirstRunLoaded}
          runs={runs}
          loading={loading}
          onRefresh={fetchRuns}
        />
      }
      detail={
        selectedRunId || runId ? (
          <RunDetailPanel runId={selectedRunId || runId} />
        ) : (
          <EmptyDetail />
        )
      }
    />
  )
}

/**
 * ProjectRunsList - Scoped version of RunsList.
 * Filters runs by projectId (track_id), unless projectId is "default".
 */
function ProjectRunsList({
  projectId,
  selectedId,
  onSelect,
  onFirstRunLoaded,
  runs,
  loading,
  onRefresh,
}) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')

  const FILTER_OPTIONS = [
    { value: 'all', label: 'All' },
    { value: 'verified', label: 'Verified' },
    { value: 'failed', label: 'Failed' },
    { value: 'running', label: 'Running' },
    { value: 'unverified', label: 'Unverified' },
  ]

  // Filter runs by projectId
  const scopedRuns = useMemo(() => {
    if (projectId === 'default') {
      return runs
    }
    return runs.filter((run) => run.track_id === projectId)
  }, [runs, projectId])

  // Apply search and status filters
  const filteredRuns = useMemo(() => {
    return scopedRuns.filter((run) => {
      // Search filter
      const runId = (run.run_id || run.id || '').toLowerCase()
      if (search && !runId.includes(search.toLowerCase())) {
        return false
      }

      // Status filter
      if (filter !== 'all') {
        const runStatus = (run.verification_status || run.verification || 'unverified').toLowerCase()
        if (runStatus !== filter) {
          return false
        }
      }

      return true
    })
  }, [scopedRuns, search, filter])

  // Auto-select first run on initial load
  useEffect(() => {
    if (!loading && filteredRuns.length > 0) {
      const firstId = filteredRuns[0].run_id || filteredRuns[0].id
      onFirstRunLoaded(firstId)
    }
  }, [loading, filteredRuns, onFirstRunLoaded])

  return (
    <>
      {/* Header with project info */}
      <div className="list-topbar">
        <div className="list-topbar-header">
          <div className="project-header">
            <h2>
              {projectId === 'default' ? 'All Runs' : `Project: ${projectId}`}
            </h2>
            {projectId !== 'default' && (
              <span className="project-track-id">{scopedRuns.length} runs</span>
            )}
          </div>
        </div>
      </div>

      {/* Runs list */}
      <div className="runs-list">
        {loading && (
          <>
            <SkeletonRunRow />
            <SkeletonRunRow />
            <SkeletonRunRow />
            <SkeletonRunRow />
          </>
        )}

        {!loading && filteredRuns.length === 0 && (
          <EmptyState
            title={
              search || filter !== 'all'
                ? 'No matching runs'
                : scopedRuns.length === 0
                ? `No runs for project "${projectId}"`
                : 'No runs yet'
            }
            description={
              search || filter !== 'all'
                ? 'Try adjusting your search or filters.'
                : scopedRuns.length === 0
                ? `Create a run with track_id="${projectId}" to see it here.`
                : 'Run your first capsule to see it here.'
            }
          />
        )}

        {!loading &&
          filteredRuns.map((run) => {
            const id = run.run_id || run.id
            return (
              <RunRow
                key={id}
                run={run}
                selected={id === selectedId}
                onClick={() => onSelect(id)}
              />
            )
          })}
      </div>
    </>
  )
}

/**
 * RunRow - Single run list item (imported pattern from RunsList).
 * Renders a clickable run entry with status and metadata.
 */
function RunRow({ run, selected, onClick }) {
  const id = run.run_id || run.id
  const status = (run.verification_status || run.verification || 'unverified').toLowerCase()

  const statusColor = {
    verified: 'var(--accent-success)',
    failed: 'var(--accent-error)',
    rejected: 'var(--accent-error)',
    running: 'var(--accent-warning)',
    unverified: 'var(--text-secondary)',
  }[status] || 'var(--text-secondary)'

  return (
    <div
      className={`run-row ${selected ? 'selected' : ''}`}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      <div className="run-row-header">
        <span className="run-row-id mono">{id?.slice(0, 12)}</span>
        <span
          className="run-row-status"
          style={{
            color: statusColor,
            textTransform: 'capitalize',
            fontSize: '12px',
            fontWeight: 500,
          }}
        >
          {status}
        </span>
      </div>
      {run.track_id && (
        <div className="run-row-meta">
          <span className="run-row-label">Track:</span>
          <span className="run-row-value mono">{run.track_id}</span>
        </div>
      )}
      {run.policy_id && (
        <div className="run-row-meta">
          <span className="run-row-label">Policy:</span>
          <span className="run-row-value mono">{run.policy_id?.slice(0, 12)}</span>
        </div>
      )}
      {run.created_at && (
        <div className="run-row-meta">
          <span className="run-row-label">Created:</span>
          <span className="run-row-value">
            {new Date(run.created_at).toLocaleDateString()}
          </span>
        </div>
      )}
    </div>
  )
}

/**
 * EmptyDetail - Shown when no run is selected
 */
function EmptyDetail() {
  return (
    <div className="empty-state" style={{ height: '100%' }}>
      <div className="empty-state-icon">
        <svg
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <rect x="3" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="3" width="7" height="7" rx="1" />
          <rect x="3" y="14" width="7" height="7" rx="1" />
          <rect x="14" y="14" width="7" height="7" rx="1" />
        </svg>
      </div>
      <div className="empty-state-title">Select a run</div>
      <div className="empty-state-text">Choose a run from the list to view details.</div>
    </div>
  )
}
