export function SkeletonLine({ width = '100%', height = 16 }) {
  return (
    <div
      className="skeleton-line"
      style={{ width, height }}
    />
  )
}

export function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <SkeletonLine width="60%" height={14} />
      <SkeletonLine width="40%" height={12} />
      <SkeletonLine width="80%" height={12} />
    </div>
  )
}

export function SkeletonRunRow() {
  return (
    <div className="skeleton-run-row">
      <div className="skeleton-run-main">
        <SkeletonLine width="120px" height={16} />
        <SkeletonLine width="60px" height={20} />
      </div>
      <SkeletonLine width="200px" height={12} />
    </div>
  )
}

export function SkeletonOverviewCard() {
  return (
    <div className="skeleton-overview-card">
      <SkeletonLine width="80px" height={14} />
      <div className="skeleton-items">
        <SkeletonLine width="100%" height={12} />
        <SkeletonLine width="100%" height={12} />
        <SkeletonLine width="70%" height={12} />
      </div>
    </div>
  )
}
