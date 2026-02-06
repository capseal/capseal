/**
 * EmptyState - Action-driven empty state component
 *
 * Per Solution_notes.txt:
 * - Every empty state needs a primary action
 * - Supports multiple action buttons (primary, secondary, tertiary)
 * - Never passive ("Select a run") - always active ("Create first run")
 */
export default function EmptyState({
  title,
  description,
  icon,
  // Single action (backwards compatible)
  action,
  onAction,
  // Multiple actions
  actions = [],
}) {
  // Build actions array from either props style
  const allActions = action && onAction
    ? [{ label: action, onClick: onAction, primary: true }]
    : actions

  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        {icon || (
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <rect x="8" y="12" width="32" height="24" rx="4" stroke="currentColor" strokeWidth="2" />
            <path d="M16 22h16M16 28h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        )}
      </div>
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {allActions.length > 0 && (
        <div className="empty-state-actions">
          {allActions.map((act, idx) => (
            <button
              key={idx}
              className={`empty-state-action ${act.primary ? 'primary' : act.secondary ? 'secondary' : 'tertiary'}`}
              onClick={act.onClick}
            >
              {act.icon && <span className="action-icon">{act.icon}</span>}
              {act.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
