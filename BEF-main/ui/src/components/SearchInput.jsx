import { useState, useEffect } from 'react'

const SearchIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
)

export default function SearchInput({ placeholder = 'Search...', onSearch, debounceMs = 300 }) {
  const [value, setValue] = useState('')

  useEffect(() => {
    const timer = setTimeout(() => {
      onSearch?.(value)
    }, debounceMs)
    return () => clearTimeout(timer)
  }, [value, debounceMs, onSearch])

  return (
    <div className="search-input">
      <span className="search-icon"><SearchIcon /></span>
      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      {value && (
        <button
          className="search-clear"
          onClick={() => setValue('')}
        >
          &times;
        </button>
      )}
    </div>
  )
}
