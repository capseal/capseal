# CapSeal

Portable cryptographic receipts for off-chain computation.

A `.cap` receipt cryptographically binds **code + inputs + config + output claim**. Verification fails if any part is modified.

## Quickstart

```bash
# Run a computation and seal it
capseal run ./strategy.py

# Verify the receipt (milliseconds)
capseal verify run.cap

# Inspect what was sealed
capseal inspect run.cap

# (Optional) Fetch full audit bundle
capseal audit run.cap --fetch

# FusionAlpha demos (requires cargo build -p fusion-bindings --release)
python FusionAlpha/python/fusion_alpha_demo.py
python FusionAlpha/python/ogbench_integration.py

# ENN training + telemetry + calibrator
cd enn-cpp
python ../scripts/validate_bicep_schema.py ~/sample_bicep_enn_demo.csv --metadata ~/sample_bicep_enn_demo.meta.json
make apps/bicep_to_enn
./apps/bicep_to_enn ~/sample_bicep_enn_demo.csv --metadata ~/sample_bicep_enn_demo.meta.json --telemetry /tmp/out.csv
python scripts/fit_calibrator.py /tmp/out.csv /tmp/calibrator.json --model-id demo
python scripts/evaluate_signals.py /tmp/out.csv --calibrator /tmp/calibrator.json

# Verify determinism
python scripts/run_replay_test.py --csv ~/sample_bicep_enn_demo.csv

# Collect artifact set
python scripts/collect_artifact_set.py --telemetry /tmp/out.csv --calibrator /tmp/calibrator.json \
  --graph-spec /tmp/humanoid_spec.json --graph-json /tmp/humanoid_graph.json \
  --output artifact_set_demo.json

# Monitor telemetry drift vs baseline
python scripts/monitor_telemetry.py baseline.csv /tmp/out.csv

# Export deterministic graph artifacts (example)
python FusionAlpha/python/export_graph.py humanoid --current 1.0,1.0,0.0,0.0,0.0 \
  --goal 8.0,8.0,0.0,0.0,0.0 --spec-out /tmp/humanoid_spec.json --graph-out /tmp/humanoid_graph.json
```

## Verification Levels

| Level | What's Checked |
|-------|----------------|
| **Proof-Only** | Cryptographic binding intact |
| **Policy-Enforced** | Hardware/toolchain manifests match policy |
| **Audit-Ready** | Full replay artifacts retrievable on demand |

## What CapSeal Proves

CapSeal proves that **this exact code, inputs, config, and environment produced the claimed output** — not that the code itself is correct.

Tampering or mismatch fails verification.

## Typed Attestations

The `capsule attest` command runs semantic checks over a traced repository using typed tracers (e.g., tabular datasets, tensors) without changing the underlying `project_trace_v1` format. Create a profile file describing which files to include and the constraints to enforce:

```json
{
  "profiles": {
    "data_quality_v1": {
      "tabular_v1": {
        "include": ["data/**/*.csv"],
        "max_missing_frac": 0.01,
        "columns": {
          "age": {"min": 0, "max": 120},
          "price": {"min": 0}
        }
      }
    }
  }
}
```

Run attestations and inspect the summary + per-file receipts:

```bash
capsule trace ./repo --out /tmp/run
capsule attest --run /tmp/run --project-dir ./repo \
  --profile profiles/data_quality.json --profile-id data_quality_v1
capsule attest-diff --base /tmp/base_run --head /tmp/head_run \
  --profile-id data_quality_v1 --fail-on warning
```

All attestation outputs live under `attestations/<profile_id>/` and are keyed to the same `trace_root`, so they can be stitched into workflow DAGs and diffed just like code-review receipts.

## LLM Explanation Reports

Once review packets exist, you can ask an LLM to narrate the top issues (and suggested fixes) without re-running the shards:

```bash
# Summarize warning+ findings with OpenAI (human readable)
capsule explain-llm --run /tmp/run_dir --llm-provider openai \
  --llm-model gpt-4o-mini --max-findings 20 --min-severity warning

# Outputs:
#   reviews/explain_llm/prompt_entries.json  (inputs)
#   reviews/explain_llm/prompt.txt          (rendered prompt)
#   reviews/explain_llm/raw.txt             (raw LLM reply)
#   reviews/explain_llm/summary.json        (parsed explanations)
```

This keeps the primary receipts lean (chunk-hash anchored) while providing a human-friendly action plan as a derived artifact.

## Audit Bundles

Audit bundles are:
- Content-addressed
- Integrity-checked against receipt commitments
- Retrievable only if authorized by policy

---

# Backend: `geom_stc_fri` (HSSA-STC v3)

The sections below document the underlying STC + FRI pipeline for streaming traces, including TraceSpecV1/StatementV1 bindings, policy registry pinning, and policy-aware DA audit.

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

```bash
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

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_capsule.py out/geom_demo/strategy_capsule.json \
    --policy policies/benchmark_policy_v1.json \
    --manifest-root out/capsule_runs/<run_id>/manifests \
    --acl-path policies/demo_acl.json
```

- Checks capsule hash, trace spec hash, statement hash, policy registry, proof payload hashes, row commitment, authorship/ACL, Nova state (if present), and DA audit (policy-aware retries/timeouts).
- Returns JSON:
  - Success: proof stats, DA audit flag, etc.
  - Failure: `{"status": "REJECT", "error_code": "E0xx"}`

### Hermetic verification (.cap)

Use the CapSeal CLI to package a run into a portable receipt and verify anywhere:

```bash
# Package a run into a .cap archive
capseal emit \
  --capsule out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --artifacts out/capsule_runs/<run_id>/pipeline \
  --policy out/capsule_runs/<run_id>/policy.json \
  --out /tmp/receipt.cap

# Verify hermetically (safe extraction + sandboxed materialization)
capseal verify /tmp/receipt.cap --json
```

The `.cap` verifier enforces safe extraction (no traversal, no links, size limits), writes artifacts to the rel paths recorded in the capsule, validates sizes/hashes, then runs the canonical verifier. See `docs/guides/cli.md` and `docs/spec/10_cap_format.md`.

## Tests

```bash
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
- `docs/guides/cli.md` – CapSeal CLI (emit/verify/inspect) and `.cap` usage
- `docs/spec/10_cap_format.md` – portable `.cap` format and hermetic verification
- `server/README.md` – FastAPI relay for CapsuleBench live event streams

### Backends

- Geom backend: `docs/backends/geom.md`
- Risc0 backend: `docs/backends/risc0.md`

## CapsuleBench CLI

The `capseal-bench` CLI wraps the pipeline, captures hardware/toolchain manifests, and assembles
the canonical `capsulepack.tgz` artifact.

```bash
capseal-bench run \
    --backend geom \
    --policy policies/benchmark_policy_v1.json \
    --policy-id baseline_policy_v1 \
    --track-id baseline_no_accel \
    --docker-image-digest sha256:<digest> \
    --manifest-signer-id my_lab_manifest \
    --manifest-signer-key secrets/manifest_signer.hex

capseal-bench pack --run-dir out/capsule_runs/run_YYYYMMDD_HHMMSS
```

`run` captures manifests and executes `scripts/run_pipeline.py`; `pack` enforces the canonical
capsulepack layout and writes `<run_id>.capsulepack.tgz`.

For a fast signed demo run on a GPU-enabled laptop:

```bash
capseal-bench run \
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

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_capsule.py \
    out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
    --policy policies/demo_policy_v1.json \
    --manifest-root out/capsule_runs/<run_id>/manifests \
    --acl-path policies/demo_acl.json
```

## BEF EEG Pipeline (v46 Submission)

The BEF (BICEP → ENN → FusionAlpha) implementation from the NeurIPS v46
submission is vendored under `v46_submission/` and exposed as a Python package.
If you have BioSemi BDF recordings (e.g., the `R7_mini` dataset), convert them to
NumPy tensors first:

```bash
python scripts/convert_bdf_to_numpy.py \
    --bdf R7_mini/.../sub-XXXX_task-YYYY_eeg.bdf \
    --out data/sub-XXXX_task-YYYY.npy \
    --meta data/sub-XXXX_task-YYYY.json
```

Then run stochastic EEG inference end-to-end with:

```bash
python scripts/run_bef_eeg.py \
    --input data/sub-XXXX_task-YYYY.npy \
    --checkpoint v46_submission/c1_bef.pt \
    --output bef_outputs.json
```

`run_bef_eeg.py` instantiates `BEF_EEG`, loads checkpoints, executes adaptive
BICEP sampling, entangled neural encoding, and FusionAlpha graph fusion, then
writes predictions plus aleatoric/epistemic uncertainty diagnostics to JSON.

> The signing key in `demo_assets/demo_private_key.hex` is public and intended
> only for local demos. Generate and protect your own keys for anything real.

### Manifest signatures and policy enforcement

`capseal-bench run` writes `manifests/manifest_signature.json` whenever you
provide `--manifest-signer-id` (or `CAPSEAL_MANIFEST_SIGNER_ID`) together with
`--manifest-signer-key` (or `CAPSEAL_MANIFEST_SIGNER_KEY`). The signer id must
match an entry in the verifier's `config/manifest_signers.json`, and the key can
be specified either as a path to a hex file or an inline hex string. Without a
signature the capsule still verifies at `proof_only`, but policy enforcement
fails closed with `E106_MANIFEST_SIGNATURE_MISSING`.

For demos, generate a throwaway secp256k1 key pair, add the public key to your
verifier config, and point `--manifest-signer-key` at the private key so the
manifests are authenticated.

If you run `scripts/run_pipeline.py` directly, collect manifests with
`capseal_bench.manifests.collect_manifests` (or any equivalent process), then
sign them via:

```bash
PYTHONPATH=. python scripts/sign_manifest.py ./manifests \
    --signer-id my_lab_manifest \
    --private-key secrets/manifest_signer.hex
```

Trusted relay/manifest registries also need reproducible hashes. Use
`scripts/compute_trust_roots.py` to print the SHA-256 root after editing
`config/trusted_relays.json` or `config/manifest_signers.json`; the verifier's
`--trusted-*-root` flags must match these values when you roll keys.

### Data availability (FULL)

For FULL verification, the verifier accepts a signed DA challenge (challenge v1) issued by a trusted challenger key id pinned in `config/trusted_relays.json` (or CLI overrides). The challenge binds `{capsule_commit_hash, seed, k, chunk_len, chunk_tree_arity, issued_at}`; the verifier samples deterministically and checks Merkle openings to `row_root`. See `docs/spec/06_protocol.md` and `docs/spec/05_profiles.md`.

### Streaming events to the relay

`capseal-bench run` always writes a chained `events.jsonl`. To push those events to
the FastAPI relay while the prover runs, forward them to the ingest socket:

```bash
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

## Quickstart: Market Dashboard CI run

Run `scripts/run_market_dashboard_review.sh` to execute the full CapSeal loop
(trace → Semgrep review → rollup → diff gate → LLM explain) against
`~/projects/market-dashboard`. See `docs/market_dashboard_pipeline.md` for the
exact steps, required environment variables, and output locations.
