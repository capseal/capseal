import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useEngine } from '../state/EngineContext.jsx'

/**
 * Fuzzy search helper - simple character matching
 */
function fuzzyMatch(query, text) {
  const lowerQuery = query.toLowerCase()
  const lowerText = text.toLowerCase()

  if (!lowerQuery) return true
  if (lowerText.includes(lowerQuery)) return true

  // Simple fuzzy: all query chars must appear in text in order
  let queryIdx = 0
  for (let i = 0; i < lowerText.length && queryIdx < lowerQuery.length; i++) {
    if (lowerText[i] === lowerQuery[queryIdx]) {
      queryIdx++
    }
  }
  return queryIdx === lowerQuery.length
}

/**
 * CommandPalette - VS Code-style command palette
 *
 * Features:
 * - Opens with Cmd+K (Mac) or Ctrl+K (Windows)
 * - Fuzzy search filtering
 * - Arrow keys to navigate
 * - Enter to execute
 * - Escape to close
 */
export default function CommandPalette() {
  const navigate = useNavigate()
  const { openConfig, baseUrl, refresh } = useEngine()
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef(null)

  // Command definitions
  const commands = [
    {
      id: 'launchpad',
      name: 'Go to Launchpad',
      description: 'Return to home',
      action: () => navigate('/')
    },
    {
      id: 'workspace',
      name: 'Go to Projects',
      description: 'Open project workspace',
      action: () => navigate('/workspace')
    },
    {
      id: 'trace',
      name: 'Quick Trace',
      description: 'Inspect trace data',
      action: () => navigate('/trace')
    },
    {
      id: 'engine-config',
      name: 'Configure Engine',
      description: 'Set backend base URL',
      action: () => openConfig()
    },
    {
      id: 'engine-copy-url',
      name: 'Copy Engine URL',
      description: baseUrl || 'Set base URL first',
      disabled: !baseUrl,
      action: async () => {
        if (!baseUrl) {
          openConfig()
          return
        }
        try {
          await navigator.clipboard.writeText(baseUrl)
        } catch (err) {
          console.error('Failed to copy engine URL', err)
        }
      }
    },
    {
      id: 'engine-open-health',
      name: 'Open Engine Health JSON',
      description: baseUrl ? `${baseUrl}/api/health` : 'Set base URL first',
      disabled: !baseUrl,
      action: () => {
        if (!baseUrl) {
          openConfig()
          return
        }
        window.open(`${baseUrl}/api/health`, '_blank', 'noopener,noreferrer')
      }
    },
    {
      id: 'engine-refresh',
      name: 'Re-check Engine Health',
      description: 'Ping /api/health now',
      action: () => refresh()
    },
    {
      id: 'settings',
      name: 'Settings',
      description: 'Open settings',
      action: () => navigate('/settings')
    },
    {
      id: 'new-project',
      name: 'New Project',
      description: 'Create a new project',
      action: () => navigate('/workspace?mode=new')
    }
  ]

  // Filter commands based on query
  const filteredCommands = commands.filter(cmd =>
    fuzzyMatch(query, cmd.name) || fuzzyMatch(query, cmd.description)
  )

  // Reset selected index when filtered commands change
  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  // Focus input when palette opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  // Global keyboard handler
  useEffect(() => {
    function handleKeyDown(e) {
      // Open with Cmd+K (Mac) or Ctrl+K (Windows)
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen(prev => !prev)
      }

      if (!isOpen) return

      // Close with Escape
      if (e.key === 'Escape') {
        setIsOpen(false)
        return
      }

      // Navigate with arrow keys
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex(prev =>
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        )
        return
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex(prev =>
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        )
        return
      }

      // Execute with Enter
      if (e.key === 'Enter' && filteredCommands.length > 0) {
        e.preventDefault()
        const selected = filteredCommands[selectedIndex]
        selected.action()
        setIsOpen(false)
        setQuery('')
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, selectedIndex, filteredCommands, navigate])

  if (!isOpen) return null

  return (
    <div className="command-palette-overlay" onClick={() => setIsOpen(false)}>
      <div className="command-palette-card" onClick={e => e.stopPropagation()}>
        {/* Search Input */}
        <input
          ref={inputRef}
          type="text"
          placeholder="Command palette (search or type >)"
          value={query}
          onChange={e => setQuery(e.target.value)}
          className="command-palette-input"
        />

        {/* Commands List */}
        <div className="command-palette-list">
          {filteredCommands.length > 0 ? (
            filteredCommands.map((cmd, idx) => (
              <button
                key={cmd.id}
                className={`command-palette-item ${
                  idx === selectedIndex ? 'selected' : ''
                } ${cmd.disabled ? 'disabled' : ''}`}
                onClick={() => {
                  if (cmd.disabled) return
                  cmd.action()
                  setIsOpen(false)
                  setQuery('')
                }}
                onMouseEnter={() => setSelectedIndex(idx)}
              >
                <div className="command-palette-item-name">{cmd.name}</div>
                <div className="command-palette-item-description">
                  {cmd.description}
                </div>
              </button>
            ))
          ) : (
            <div className="command-palette-empty">No matching commands</div>
          )}
        </div>

        {/* Footer */}
        <div className="command-palette-footer">
          <span className="command-palette-hint">
            <kbd>↑</kbd> <kbd>↓</kbd> navigate • <kbd>Enter</kbd> select • <kbd>Esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  )
}
