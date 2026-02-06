import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { normalizeJobStatus, getStatusClass } from './engine/impl/flask.js'

const DEFAULT_API = 'https://capsuletech.rochmanofenna.workers.dev'
const API = window.API_BASE || import.meta.env.VITE_API_BASE || DEFAULT_API

export default function RunsPage() {
  const [runs, setRuns] = useState([])
  const [status, setStatus] = useState('loading')

  async function load() {
    setStatus('loading')
    try {
      const resp = await fetch(`${API}/api/runs`)
      const data = await resp.json()
      const array = Array.isArray(data) ? data : data.runs || []
      setRuns(array)
      setStatus('ready')
    } catch (err) {
      setStatus(`error: ${err.message}`)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <section className="runs">
      <div className="section-header">
        <h2>Runs</h2>
        <button onClick={load}>Refresh</button>
      </div>
      {status !== 'ready' && <p>{status}</p>}
      <ul>
        {runs.map((run) => (
          <li key={run.run_id} className="run-item">
            <Link to={`/runs/${run.run_id}`}>
              <strong>{run.run_id}</strong>
              <div><small>{run.backend}</small></div>
              <div><small>{run.policy_id}</small></div>
              <div><small>{run.created_at ? new Date(run.created_at).toLocaleString() : '—'}</small></div>
              {run.verification_status && (
                <div>
                  <small className={`run-status ${getStatusClass(normalizeJobStatus(run.verification_status))}`}>
                    Status: {normalizeJobStatus(run.verification_status)}
                  </small>
                </div>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </section>
  )
}
