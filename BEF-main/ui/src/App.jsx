import { Routes, Route, Navigate } from 'react-router-dom'
import Launchpad from './Launchpad'
import ProjectShell from './components/ProjectShell'
import WorkspaceSelection from './WorkspaceSelection'
import QuickTrace from './QuickTrace'
import CommandPalette from './components/CommandPalette'
import './style.css'

/**
 * App - Main application with hierarchical routing.
 *
 * Route structure (per Design_notes.txt & Layout_hierarchy.txt):
 * / → Launchpad (Jobs-minimal, no rail)
 * /workspace → WorkspaceSelection (project chooser, no rail)
 * /workspace?mode=recent|new|open → mode-specific views
 * /p/:projectId/runs → ProjectShell (rail visible)
 * /p/:projectId/runs/:runId → ProjectShell with detail
 * /trace → QuickTrace (no rail)
 * /settings → Settings (placeholder)
 * /runs, /runs/:runId → backward compat redirects
 */
export default function App() {
  return (
    <>
      <CommandPalette />
      <Routes>
        {/* Launchpad - Jobs-minimal first impression (no rail) */}
        <Route path="/" element={<Launchpad />} />

        {/* Workspace selection - project chooser (no rail) */}
        <Route path="/workspace" element={<WorkspaceSelection />} />

        {/* Quick Trace - single-purpose trace inspection (no rail) */}
        <Route path="/trace" element={<QuickTrace />} />

        {/* Settings - placeholder for now */}
        <Route path="/settings" element={<SettingsPlaceholder />} />

        {/* Project hierarchy - rail visible here ONLY */}
        <Route path="/p/:projectId/runs" element={<ProjectShell />} />
        <Route path="/p/:projectId/runs/:runId" element={<ProjectShell />} />

        {/* Backward compatibility: /runs → /p/default/runs */}
        <Route path="/runs" element={<Navigate to="/p/default/runs" replace />} />
        <Route path="/runs/:runId" element={<Navigate to="/p/default/runs/:runId" replace />} />
      </Routes>
    </>
  )
}

/**
 * SettingsPlaceholder - Minimal settings page
 */
function SettingsPlaceholder() {
  return (
    <div className="settings-page">
      <div className="settings-container">
        <h1 className="settings-title">Settings</h1>
        <p className="settings-subtitle">Configuration options coming soon.</p>
        <a href="/" className="btn btn-secondary" style={{ marginTop: '24px' }}>
          Back to Launchpad
        </a>
      </div>
    </div>
  )
}
