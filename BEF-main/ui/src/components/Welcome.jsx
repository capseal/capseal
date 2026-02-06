/**
 * @deprecated Use WorkspaceHome instead. This full-page landing has been
 * replaced by WorkspaceHome which fits inside AppShell and shows useful info.
 *
 * Welcome - First-run landing experience (DEPRECATED).
 *
 * Shows 3 core actions:
 * 1. Create Circuit - Define what your code should prove
 * 2. Run & Seal - Run code and generate cryptographic proof
 * 3. View Runs - See verification history
 */
export default function Welcome({ onAction }) {
  return (
    <div className="welcome">
      <div className="welcome-content">
        {/* Hero */}
        <div className="welcome-hero">
          <div className="welcome-logo">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2L2 7v10l10 5 10-5V7L12 2z"
                stroke="url(#logoGrad)"
                strokeWidth="1.5"
                fill="none"
              />
              <path
                d="M12 22V12M2 7l10 5 10-5"
                stroke="url(#logoGrad)"
                strokeWidth="1.5"
              />
              <circle cx="12" cy="12" r="3" fill="url(#logoGrad)" opacity="0.6" />
              <defs>
                <linearGradient id="logoGrad" x1="2" y1="2" x2="22" y2="22">
                  <stop stopColor="#6366f1" />
                  <stop offset="1" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1 className="welcome-title">Capseal</h1>
          <p className="welcome-subtitle">
            Cryptographic proof that your code ran exactly as written.
          </p>
        </div>

        {/* Action Cards */}
        <div className="welcome-actions">
          <button
            className="action-card"
            onClick={() => onAction('policy')}
          >
            <div className="action-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 12h6M9 16h6M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h6l6 6v10a2 2 0 01-2 2z" />
                <path d="M13 3v6h6" />
              </svg>
            </div>
            <div className="action-content">
              <span className="action-title">Generate Policy</span>
              <span className="action-desc">Define verification rules for your pipeline</span>
            </div>
            <div className="action-arrow">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>
          </button>

          <button
            className="action-card"
            onClick={() => onAction('proof')}
          >
            <div className="action-icon accent">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="M9 12l2 2 4-4" />
              </svg>
            </div>
            <div className="action-content">
              <span className="action-title">Create Proof</span>
              <span className="action-desc">Run your code and generate verifiable proof</span>
            </div>
            <div className="action-arrow">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>
          </button>

          <button
            className="action-card"
            onClick={() => onAction('runs')}
          >
            <div className="action-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
              </svg>
            </div>
            <div className="action-content">
              <span className="action-title">View Runs</span>
              <span className="action-desc">Browse verification history and proofs</span>
            </div>
            <div className="action-arrow">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>
          </button>
        </div>

        {/* Footer hint */}
        <div className="welcome-footer">
          <kbd>Ctrl</kbd> + <kbd>K</kbd> for command palette
        </div>
      </div>
    </div>
  )
}
