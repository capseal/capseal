# STC / Geometry Backend

Hash-only STC + FRI pipeline for streaming traces, including TraceSpecV1/StatementV1 bindings, policy registry pinning, and a policy-aware DA audit.

## Key Features

- STC-backed vector + polynomial commitments
- TraceSpecV1: canonical, content-addressed trace spec stored in every capsule (`trace_spec`, `trace_spec_hash`)
- StatementV1: binds trace spec, policy hash, trace root, public inputs, anchors; hashed and fed to the STARK transcript (`statement`, `statement_hash`)
- Policy registry tooling (`scripts/build_policy_registry.py`) and policy-aware DA audit (retry/timeout + E074 availability codes)
- Authorship (secp256k1) and ACL enforcement

## Repo Layout

```
BEF/
├── bef_zk/          # AIR, STC backends, codec, TraceSpec/Statement helpers
├── scripts/         # prover, verifier, policy tooling, DA sampling
├── docs/            # backend architecture, DA protocol, TraceSpec docs
├── tests/           # regression/unit tests (verifier, DA audit)
└── bench/           # benchmarking harnesses
```

## TraceSpecV1 & StatementV1

See `docs/trace_statement_spec.md` for serialization + hashing details.

- `trace_spec`: describes trace format (schema ref, encoding, field modulus)
- `statement`: binds trace spec hash, policy hash, trace root, public inputs, anchors, optional external commitments
- `trace_spec_hash` / `statement_hash`: canonical CBOR hashes used both in the capsule and in the STARK transcript

## Running the Pipeline

```
PYTHONPATH=. .venv/bin/python scripts/run_pipeline.py --backend geom \
    --steps 4096 --num-challenges 4 --num-queries 32 \
    --output-dir out/geom_demo --trace-id geom_demo \
    --policy policies/policy1.txt --policy-id rollup_policy --policy-version v1
```

> Demo tip: if you're presenting on a GPU-equipped laptop, use the demo policy
> bundle (`policies/demo_policy_v1.json`, `policies/demo_acl.json`, and
> `demo_assets/demo_private_key.hex`). It allows GPUs, relaxes deterministic
> build requirements, and gives you a deterministic signer so ACL enforcement
> succeeds during verification.

Outputs:

- `out/.../geom_proof.{json,bin}` – STARK proof (includes row commitment metadata)
- `out/.../strategy_capsule.{json,bin}` – capsule with trace spec, statement, policy hash, DA policy
- `out/.../artifact_manifest.json` – content-addressed artifact index
- `out/.../row_archive/` – STC chunk archive, Merkle roots/digest

## Verifying a Capsule

```
PYTHONPATH=. .venv/bin/python scripts/verify_capsule.py out/geom_demo/strategy_capsule.json \
    --policy policies/benchmark_policy_v1.json \
    --manifest-root out/capsule_runs/<run_id>/manifests \
    --acl-path policies/demo_acl.json
```

- Checks capsule hash, trace spec hash, statement hash, policy registry, proof payload hashes, row commitment, authorship/ACL, Nova state (if present), and DA audit (policy-aware retries/timeouts).
- Returns JSON:
  - Success: proof stats, DA audit flag, etc.
  - Failure: `{"status": "REJECT", "error_code": "E0xx"}`

## Tests

```
PYTHONPATH=. .venv/bin/python -m pytest tests/test_da_provider.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_capsule_verify.py -k da --maxfail=1
```

## Docs

- `docs/stc_backend_architecture.md` – STC as VC/PC backend
- `docs/trace_statement_spec.md` – TraceSpecV1/StatementV1 spec
- `docs/roadmap.md` – consolidated roadmap for TraceAdapters, DA providers, and policy tooling
- `docs/benchmark_policy_schema.json` – JSON schema for `bef_benchmark_policy_v1`
- `docs/hssa_da_protocol.md` – DA sampling protocol and guarantees
- `docs/stc_da_profiles.md` – DA policy profiles
- `docs/security_model.md` – adversary model, binding points and security claims for Capsules + STC
- `server/README.md` – FastAPI relay for CapsuleBench live event streams

## CapsuleBench CLI

The `capsule-bench` CLI wraps the pipeline, captures hardware/toolchain manifests, and assembles
the canonical `capsulepack.tgz` artifact.

```
. .venv/bin/capsule-bench run \
    --backend geom \
    --policy policies/benchmark_policy_v1.json \
    --policy-id baseline_policy_v1 \
    --track-id baseline_no_accel \
    --docker-image-digest sha256:<digest> \
    --manifest-signer-id my_lab_manifest \
    --manifest-signer-key secrets/manifest_signer.hex

. .venv/bin/capsule-bench pack --run-dir out/capsule_runs/run_YYYYMMDD_HHMMSS
```

`run` captures manifests and executes `scripts/run_pipeline.py`; `pack` enforces the canonical
capsulepack layout and writes `<run_id>.capsulepack.tgz`.

For a fast signed demo run on a GPU-enabled laptop:

```
capsule-bench run \
    --backend geom \
    --policy policies/demo_policy_v1.json \
    --policy-id demo_policy_v1 \
    --policy-version 1.0 \
    --track-id demo_geom_fast \
    --private-key demo_assets/demo_private_key.hex \
    --manifest-signer-id demo_manifest \
    --manifest-signer-key demo_assets/demo_manifest_key.hex
```

Then verify with policy + ACL enforcement:

```
PYTHONPATH=. .venv/bin/python scripts/verify_capsule.py \
    out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
    --policy policies/demo_policy_v1.json \
    --manifest-root out/capsule_runs/<run_id>/manifests \
    --acl-path policies/demo_acl.json
```

> The signing key in `demo_assets/demo_private_key.hex` is public and intended
> only for local demos. Generate and protect your own keys for anything real.

### Manifest signatures and policy enforcement

`capsule-bench run` writes `manifests/manifest_signature.json` whenever you
provide `--manifest-signer-id` (or `CAPSULE_MANIFEST_SIGNER_ID`) together with
`--manifest-signer-key` (or `CAPSULE_MANIFEST_SIGNER_KEY`). The signer id must
match an entry in the verifier’s `config/manifest_signers.json`, and the key can
be specified either as a path to a hex file or an inline hex string. Without a
signature the capsule still verifies at `proof_only`, but policy enforcement
fails closed with `E106_MANIFEST_SIGNATURE_MISSING`.

For demos, generate a throwaway secp256k1 key pair, add the public key to your
verifier config, and point `--manifest-signer-key` at the private key so the
manifests are authenticated.

If you run `scripts/run_pipeline.py` directly, collect manifests with
`capsule_bench.manifests.collect_manifests` (or any equivalent process), then
sign them via:

```
PYTHONPATH=. python scripts/sign_manifest.py ./manifests \
    --signer-id my_lab_manifest \
    --private-key secrets/manifest_signer.hex
```

Trusted relay/manifest registries also need reproducible hashes. Use
`scripts/compute_trust_roots.py` to print the SHA-256 root after editing
`config/trusted_relays.json` or `config/manifest_signers.json`; the verifier’s
`--trusted-*-root` flags must match these values when you roll keys.

### Streaming events to the relay

`capsule-bench run` always writes a chained `events.jsonl`. To push those events to
the FastAPI relay while the prover runs, forward them to the ingest socket:

```
PYTHONPATH=. .venv/bin/python scripts/relay_forward.py \
    out/capsule_runs/<run_id>/events.jsonl \
    ws://localhost:8000/ws/ingest/<run_id>
```

Subscribers (`ws://.../ws/subscribe/<run_id>`) will now see the same live stream
that is recorded on disk.

## Web UI (prototype)

`ui/` hosts a static dashboard that talks to the Cloudflare Worker API. Serve it
locally (e.g., `python -m http.server ui`) or deploy it via Cloudflare Pages with
the project root set to `ui/`. If the UI and Worker live on different hosts,
add `window.API_BASE = "https://your-worker.workers.dev";` before loading
`app.js` in `ui/index.html`.

### Worker proxy configuration

The worker proxies `/api/runs`, `/api/runs/:id`, `/api/runs/:id/events`, and
`/api/runs/:id/artifacts/*` to your FastAPI relay. Set `RELAY_BASE` in
`wrangler.toml` (or in the Cloudflare dashboard) to the base URL of the relay,
e.g. `https://relay.capsuletech.com`. Redeploy with `npx wrangler deploy` so
the UI reflects live data.

The relay itself can point at a Postgres database by setting the
`DATABASE_URL` environment variable; if it is unset, it falls back to local
JSONL files under `server_data/events/`. Artifact downloads are served from
`ARTIFACTS_ROOT` (default: `server_data/artifacts/`). To expose capsule packs,
proofs, or traces for a given run, place them under
`ARTIFACTS_ROOT/<run_id>/` before replaying the events.

## Benchmarks

See `bench/` scripts (e.g., `bench/bench_geom_stc.py`, `bench/bench_streaming_backends.py`) for commitment throughput and STC vs KZG back-ends.
