import CopyField from './CopyField'

export default function OverviewCard({ title, icon, items }) {
  return (
    <div className="overview-card">
      <div className="overview-card-header">
        {icon && <span className="overview-card-icon">{icon}</span>}
        <h4>{title}</h4>
      </div>
      <div className="overview-card-content">
        {items.map(({ label, value, mono }) => (
          <div key={label} className="overview-item">
            <span className="overview-label">{label}</span>
            {mono ? (
              <CopyField value={value} />
            ) : (
              <span className="overview-value">{value || '-'}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
