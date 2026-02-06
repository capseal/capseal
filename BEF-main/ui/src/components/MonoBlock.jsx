import { useState } from 'react'

export default function MonoBlock({ data, maxHeight = 300 }) {
  const [copied, setCopied] = useState(false)

  const content = typeof data === 'string' ? data : JSON.stringify(data, null, 2)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (err) {
      console.error('Copy failed:', err)
    }
  }

  return (
    <div className="mono-block">
      <div className="mono-block-header">
        <button className="mono-copy-btn" onClick={handleCopy}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre style={{ maxHeight }}>{content}</pre>
    </div>
  )
}
