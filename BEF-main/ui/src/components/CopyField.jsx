import { useState } from 'react'

export default function CopyField({ value, truncate = true, maxLength = 12 }) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)

  if (!value) return <span className="copy-field empty">-</span>

  const displayValue = truncate && !expanded && value.length > maxLength
    ? `${value.slice(0, maxLength)}...`
    : value

  const handleCopy = async (e) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (err) {
      console.error('Copy failed:', err)
    }
  }

  return (
    <span
      className="copy-field"
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Click to copy'}
    >
      <code>{displayValue}</code>
      <span className={`copy-indicator ${copied ? 'copied' : ''}`}>
        {copied ? 'Copied' : 'Copy'}
      </span>
    </span>
  )
}
