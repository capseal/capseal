CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
  run_id   TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  seq      BIGINT NOT NULL,
  ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
  payload  JSONB NOT NULL,
  raw_line TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);

CREATE INDEX IF NOT EXISTS events_run_id_seq_idx ON events(run_id, seq);
