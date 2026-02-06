# CapSeal Flask API

Production-shaped Flask + Gunicorn wrapper around the CapSeal CLI. Every HTTP
call shells into the canonical CLI (fetch / run / emit / verify / replay /
audit / sandbox) and responds with a normalized JSON envelope:
`{requestId,status,exitCode,errorCode,command,stdout,stderr,result,artifacts}`.

Key features:

- Redis-backed async jobs (RQ) with `/api/jobs`, `/api/jobs/<id>`, and cancel
  support – workers survive API restarts (falls back to an in-memory queue if
  redis/rq are unavailable).
- Content-addressed artifact cache with disk quotas + GC (local) or optional S3
  backend with signed URLs and a local LRU cache.
- Event ingestion automatically copies `out/<trace_id>/events.jsonl` into
  `server_data/events/<trace_id>.jsonl` so dashboards can stream events via
  `/api/runs/<id>/events`.
- Pydantic validation for every request, API key auth (`X-API-Key`), configurable
  CORS, and Redis-backed rate limiting (silently disabled if Redis client is
  missing).
- OpenAPI spec at `/openapi.json` for SDK generation.
- Optional DA profile pinning – attach trusted relay/manifests JSON via
  `daProfileId` or `daProfilePath` in `/api/verify` and the server exports the
  trusted keys as environment variables before invoking the CLI.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[server]'

# Redis powers async jobs + rate limits
redis-server --save "" --appendonly no &

export FLASK_APP=server.flask_app.app:app
export CAPSEAL_API_KEYS=demo-key
export REDIS_URL=redis://127.0.0.1:6379/0

# API (Gunicorn) + RQ worker in separate shells
FLASK_ENV=production gunicorn -w 4 -b 0.0.0.0:5000 server.flask_app.app:app --timeout 180
python -m server.flask_app.worker
```

Every async-capable endpoint accepts `?async=true`. When present, the handler
returns HTTP `202` with `{status: SUBMITTED, jobId, location}`, and the RQ worker
executes the CLI request. Job metadata persists in Redis and can be listed or
polled via `/api/jobs` and `/api/jobs/<id>`.

### Configuration knobs

| Env | Purpose |
| --- | --- |
| `CAPSEAL_PROJECT_ROOT` | Overrides repo root auto-detection |
| `CAPSEAL_EVENT_ROOT` | Where events are copied (default `server_data/events`) |
| `CAPSEAL_ARTIFACT_ROOT` | Local artifact cache root |
| `CAPSEAL_API_KEYS` | `X-API-Key` allowlist (`key1,key2`) |
| `REDIS_URL` | Redis connection for jobs + rate limits |
| `JOB_QUEUE_NAME` / `JOB_RESULT_TTL` | RQ queue name + retention |
| `CORS_ALLOW_ORIGINS` | Whitelist origins (`*` or `https://foo`) |
| `API_RATE_LIMITS` | e.g. `120 per minute, 10 per second` |
| `ARTIFACT_STORE` | `local` (default) or `s3` |
| `CAPSEAL_ARTIFACT_MAX_BYTES|FILES|AGE` | Local cache quotas |
| `S3_BUCKET` / `AWS_*` | Required when `ARTIFACT_STORE=s3` |
| `ARTIFACT_SIGNED_URL_TTL` | Signed URL lifetime in seconds |

### Docker / Compose

```
docker compose up --build
```

The compose file builds the API image, runs Redis, the API, and an RQ worker. It
bind-mounts `./server_data` and `./out` so artifacts persist across restarts.

## API surface

| Endpoint | Description |
| --- | --- |
| `POST /api/fetch` | Governed dataset fetch (async-aware) |
| `POST /api/run` | Run CapSeal proof pipeline (`async=true` to enqueue) |
| `POST /api/emit` | Emit `.cap` archives |
| `POST /api/verify` | Verify proofs, manifests, DA (supports `daProfileId`) |
| `POST /api/replay` | Deterministic replay with tolerance knobs |
| `POST /api/audit` | Export audits (summary/json/jsonl/csv) |
| `GET /api/runs/<runId>/events` | Stream stored event log |
| `GET /api/sandbox/status` | Sandbox posture |
| `POST /api/sandbox/test` | Launch built-in sandbox smoke test |
| `GET /api/jobs?limit=20` | Recent async jobs |
| `GET /api/jobs/<id>` | Inspect job metadata + CLI result |
| `POST /api/jobs/<id>/cancel` | Cancel queued/running job |

Responses for CLI-backed operations always include the structured envelope with
raw stdout/stderr plus parsed JSON (when the CLI emitted JSON). `artifacts` lists
content-addressed files (hash + relative path + signed URL in S3 mode).

## Front-end dashboard

A Vite/React dashboard (`ui/`) drives the HTTP API end-to-end: submit runs,
monitor jobs, trigger verify/replay/audit, exercise the sandbox, and tail events.

```
cd ui
npm install
npm run dev  # or npm run build && serve dist/
```

Configure the API base via `VITE_API_BASE` or by defining `window.API_BASE` in
`index.html`. Provide an API key in the UI to authorize requests.

## Tests / linting

- Smoke test the Flask app via `python - <<'PY'` snippet in CI (hits `/health`,
  `/ready`, and a short `/api/run`).
- UI bundle verified via `npm run build`.
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs pytest, linting, and
  builds/pushes the Docker image; `.github/workflows/e2e.yml` spins up the stack
  with docker compose, waits for `/ready`, and exercises `run → verify → audit`.

## Artifact storage

Local mode keeps artifacts under `CAPSEAL_ARTIFACT_ROOT` using `sha256` content
addresses and a GC thread that enforces age/count/byte quotas. When
`ARTIFACT_STORE=s3`, uploads stream directly to S3/GCS-compatible buckets and
responses include a signed URL for clients to download proof artifacts while the
local cache maintains a warm copy for replays.
