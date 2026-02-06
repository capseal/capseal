import { useCallback, useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  RunRow,
  SearchInput,
  FilterChips,
  SkeletonRunRow,
  EmptyState,
} from './components'
import NewRunDrawer from './components/NewRunDrawer'
import { getEngine } from './engine'

const FILTER_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'verified', label: 'Verified' },
  { value: 'failed', label: 'Failed' },
  { value: 'running', label: 'Running' },
  { value: 'unverified', label: 'Unverified' },
]

/**
 * RunsList - List panel for the runs shell.
 *
 * Features:
 * - Search by run ID
 * - Filter by verification status
 * - Premium RunRow components
 * - Auto-selects first run on load
 */
export default function RunsList({ selectedId, onSelect, onFirstRunLoaded }) {
  const navigate = useNavigate()
  const [runs, setRuns] = useState([])
  const [status, setStatus] = useState('loading')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [drawerOpen, setDrawerOpen] = useState(false)

  const fetchRuns = useCallback(async () => {
    setStatus('loading')
    try {
      const engine = getEngine()
      const runsList = await engine.listRuns()
      setRuns(runsList)
      setStatus('ready')

      // Auto-select first run if callback provided and we have runs
      if (onFirstRunLoaded && runsList.length > 0) {
        const firstId = runsList[0].run_id || runsList[0].id
        onFirstRunLoaded(firstId)
      }
    } catch (err) {
      console.error('Failed to fetch runs:', err)
      setStatus('error')
    }
  }, [onFirstRunLoaded])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  // Filter and search
  const filteredRuns = runs.filter((run) => {
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

  // Derive unique projects (track_ids) and circuits (policy_ids) from runs
  const { projects, circuits } = useMemo(() => {
    const projectSet = new Set()
    const circuitSet = new Set()

    runs.forEach((run) => {
      if (run.track_id) projectSet.add(run.track_id)
      if (run.policy_id) circuitSet.add(run.policy_id)
    })

    return {
      projects: Array.from(projectSet),
      circuits: Array.from(circuitSet),
    }
  }, [runs])

  return (
    <>
      {/* Top bar with search and filters */}
      <div className="list-topbar">
        <div className="list-topbar-header">
          <h2>Runs</h2>
          <button
            className="new-run-cta"
            onClick={() => setDrawerOpen(true)}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="16" />
              <line x1="8" y1="12" x2="16" y2="12" />
            </svg>
            New Run
          </button>
        </div>
        <SearchInput
          placeholder="Search runs..."
          onSearch={setSearch}
          debounceMs={200}
        />
        <FilterChips
          options={FILTER_OPTIONS}
          value={filter}
          onChange={setFilter}
        />
      </div>

      {/* Runs list */}
      <div className="runs-list">
        {status === 'loading' && (
          <>
            <SkeletonRunRow />
            <SkeletonRunRow />
            <SkeletonRunRow />
            <SkeletonRunRow />
          </>
        )}

        {status === 'error' && (
          <EmptyState
            title="Failed to load runs"
            description="Check your connection and try again."
            action="Retry"
            onAction={fetchRuns}
          />
        )}

        {status === 'ready' && filteredRuns.length === 0 && (
          search || filter !== 'all' ? (
            <EmptyState
              title="No matching runs"
              description="Try adjusting your search or filters."
              action="Clear filters"
              onAction={() => { setSearch(''); setFilter('all') }}
            />
          ) : (
            <EmptyState
              title="No runs yet"
              description="Create your first cryptographic run to see it here."
              actions={[
                {
                  label: 'Create first run',
                  onClick: () => setDrawerOpen(true),
                  primary: true,
                },
                {
                  label: 'Quick Trace',
                  onClick: () => navigate('/quicktrace'),
                  secondary: true,
                },
                {
                  label: 'Import .cap file',
                  onClick: () => {
                    // TODO: Implement import dialog
                    alert('Import feature coming soon')
                  },
                },
              ]}
            />
          )
        )}

        {status === 'ready' &&
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

      {/* New Run Drawer */}
      <NewRunDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        projects={projects}
        circuits={circuits}
      />
    </>
  )
}
