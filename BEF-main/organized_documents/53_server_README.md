# CapsuleBench Relay (Prototype)

This FastAPI app ingests live events from `capsule-bench run` and replays them to any
subscribed clients. It persists events per `run_id` under `server_data/events/`.

## Usage

```
uvicorn server.relay:app --reload --port 8000
```

- `ws://localhost:8000/ws/ingest/{run_id}` accepts canonical JSON events from the runner.
- `ws://localhost:8000/ws/subscribe/{run_id}` streams history + live updates to UIs.
- `GET /runs/{run_id}/events?since_seq=<int>` returns JSON events for replay/debugging.

The prototype uses a simple file-backed store (`server/event_store.py`). Replace it with Redis
or another persistent store for production deployments.
