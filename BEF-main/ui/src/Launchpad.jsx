import { useNavigate } from 'react-router-dom'
import EngineHandshakeCard from './components/EngineHandshakeCard'

/**
 * Launchpad - Jobs-style minimal entry point
 *
 * Centered card with 4 action buttons:
 * - Start New Project → /workspace?mode=new
 * - Continue Project → /workspace?mode=recent
 * - Open Existing Project → /workspace?mode=open
 * - Quick Trace → /trace
 *
 * Settings link (small, secondary) → /settings
 * NO rail. Steve Jobs minimal design.
 */
export default function Launchpad() {
  const navigate = useNavigate()

  const actions = [
    {
      title: 'Start New Project',
      icon: 'star',
      onClick: () => navigate('/workspace?mode=new'),
    },
    {
      title: 'Continue Project',
      icon: 'refresh',
      onClick: () => navigate('/workspace?mode=recent'),
    },
    {
      title: 'Open Existing Project',
      icon: 'folder',
      onClick: () => navigate('/workspace?mode=open'),
    },
    {
      title: 'Quick Trace',
      icon: 'zap',
      onClick: () => navigate('/trace'),
    },
  ]

  return (
    <div className="launchpad">
      <div className="launchpad-card">
        <EngineHandshakeCard />

        {/* Logo */}
        <div className="launchpad-logo">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2L2 7v10l10 5 10-5V7L12 2z"
              stroke="url(#launchpadGrad)"
              strokeWidth="1.5"
              fill="none"
            />
            <path
              d="M12 22V12M2 7l10 5 10-5"
              stroke="url(#launchpadGrad)"
              strokeWidth="1.5"
            />
            <circle cx="12" cy="12" r="3" fill="url(#launchpadGrad)" opacity="0.6" />
            <defs>
              <linearGradient id="launchpadGrad" x1="2" y1="2" x2="22" y2="22">
                <stop stopColor="#6366f1" />
                <stop offset="1" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Heading */}
        <h1 className="launchpad-title">Capseal</h1>
        <p className="launchpad-subtitle">
          Cryptographic proof that your code ran exactly as written.
        </p>

        {/* Action Buttons */}
        <div className="launchpad-actions">
          {actions.map((action, idx) => (
            <button
              key={idx}
              className="launchpad-btn"
              onClick={action.onClick}
            >
              <span className="launchpad-btn-icon">{getIcon(action.icon)}</span>
              <span className="launchpad-btn-text">{action.title}</span>
            </button>
          ))}
        </div>

        {/* Settings Link (small, secondary) */}
        <button
          className="launchpad-settings-link"
          onClick={() => navigate('/settings')}
        >
          Settings
        </button>
      </div>
    </div>
  )
}

/**
 * Icon helper
 */
function getIcon(name) {
  const icons = {
    star: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M13 2l3.293 6.707a1 1 0 0 0 .896.496h7.411l-6 4.591a1 1 0 0 0-.368 1.126l2.296 7.307L12 18.07l-6.228 4.757 2.296-7.307a1 1 0 0 0-.368-1.126l-6-4.591h7.41a1 1 0 0 0 .896-.496L13 2z" />
      </svg>
    ),
    refresh: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M21 2v6h-6M3 22v-6h6" />
        <path d="M3 11.5a9 9 0 0 1 15.84-3.79M21 12.5a9 9 0 0 1-15.84 3.79" />
      </svg>
    ),
    folder: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
      </svg>
    ),
    zap: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
  }
  return icons[name] || null
}
