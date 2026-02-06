/**
 * FilterChips - Simple filter pill selector.
 *
 * Props:
 * - options: Array of { value, label }
 * - value: Currently selected value
 * - onChange: Callback with new value
 */
export default function FilterChips({ options = [], value, onChange }) {
  return (
    <div className="filter-chips">
      {options.map((option) => (
        <button
          key={option.value}
          className={`filter-chip ${value === option.value ? 'active' : ''}`}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
