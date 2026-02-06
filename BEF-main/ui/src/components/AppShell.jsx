/**
 * AppShell - The single shell that wraps the entire app.
 *
 * 3-column layout:
 * - Rail (64px): Navigation icons
 * - List (360px): Runs list with search/filters
 * - Detail (flex): Selected run content
 *
 * This makes the app feel like "one product" instead of "two pages."
 */
export default function AppShell({ rail, list, detail, listCollapsed = false }) {
  return (
    <div className="app-shell">
      {/* Subtle noise texture overlay */}
      <div className="noise-overlay" aria-hidden="true" />

      <div className={`shell-grid ${listCollapsed ? 'list-collapsed' : ''}`}>
        {/* Column A: Rail */}
        <aside className="shell-rail">
          {rail}
        </aside>

        {/* Column B: List panel */}
        <section className="shell-list">
          {list}
        </section>

        {/* Column C: Detail panel */}
        <main className="shell-detail">
          {detail}
        </main>
      </div>
    </div>
  )
}
