# BEF Roadmap: TraceAdapter, DA Providers, and Policy Enforcement

This note distills the guidance from Benedikt, Dan, and Joseph into an actionable roadmap. It
sits alongside the code so we can tie ongoing work to concrete deliverables.

## Initiative 1 — Universal TraceAdapter API (Q1 focus)

* Goal: completely decouple the capsule pipeline from the `geom` VM so any zkVM/IVC backend can
  plug in with zero bespoke glue code.
* Security mandate (Boneh): every adapter **must** inject the `statement_hash` we provide into the
  prover transcript/Fiat–Shamir oracle. This prevents relabeling attacks.
* Work items:
  1. Define the `TraceAdapter` ABC (`bef_zk/adapter.py`) with clear responsibilities:
     * expose trace format/schema ids,
     * produce `TraceSpecV1` metadata and BEF-formatted traces,
     * call back into the pipeline for `statement_hash` binding,
     * surface proof/row-commitment artifacts for capsule assembly,
     * run the backend verifier for sanity checks.
  2. Implement `GeomTraceAdapter` as the reference adapter.
  3. Replace `scripts/run_geom_pipeline.py` with `scripts/run_pipeline.py --backend <name>` so the
     runner is pure orchestration.
  4. v0.2 milestone: integrate a second backend (`Risc0TraceAdapter`). The adapter must hook into
     the risc0 transcript so the provided `statement_hash` seeds the receipt.

## Initiative 2 — Networked Data Availability Layer

* Goal: make DA a swappable service so rollups can anchor traces to Ethereum blobs, Celestia, etc.
  (Bonneau’s mandate: no hard-coded DA opinion.)
* API (`bef_zk/da_provider.py`):
  ```python
  class DAProvider(ABC):
      def submit_blob(self, data: bytes) -> str: ...
      def fetch_chunk(self, blob_id: str, chunk_index: int) -> Tuple[bytes, Any]: ...
      def verify_inclusion(self, blob_commitment: Any, chunk: bytes, proof: Any) -> bool: ...
  ```
* First target: EIP-4844 (Type-3 blobs) via `web3.py`.
  * `submit_blob` crafts a blob tx and returns the versioned hash/tx hash.
  * `fetch_chunk` relies on a blob archive/indexer; we must document that trust assumption.
  * `verify_inclusion` uses a KZG verifier against the versioned hash.

## Initiative 3 — Policy Enforcement & `capsule-bench`

* Goal: give policies operational “teeth” and ship a first-class CLI for verifiable benchmarking.
* CLI shape (`capsule-bench` using `click`):
  * `run`: capture machine manifests (CPU/GPU, toolchains, git commits) **before** proving, invoke
    the adapter, hash manifests, embed hashes in the capsule anchors, and emit a standard
    `out/<run_id>/` bundle.
  * `pack`: compress the bundle into `*.capsulepack.tgz` for upload.
* Verifier upgrades:
  * load the policy referenced in the capsule,
  * interpret machine-readable rules (e.g. `forbid_gpu`),
  * re-hash/parse attached manifests before checking proofs,
  * emit precise policy violation codes (e.g. `E101_POLICY_VIOLATION_GPU_FORBIDDEN`).
* Manifests follow explicit schemas such as:
  ```json
  {
    "schema": "bef_hardware_manifest_v1",
    "cpu": {"model": "AMD EPYC 7B13", "cores": 64},
    "memory_gb": 1024,
    "gpu": {"model": "NVIDIA A100-SXM4-80GB", "count": 8, "driver_version": "535.104.05"}
  }
  ```

## Timeline Snapshot

* **Q1 (next 6 weeks)**: finish Initiative 1 + ship `capsule-bench run/pack` w/ manifest capture.
* **Q2 (next 8 weeks)**: deliver `Risc0TraceAdapter`, `EIP4844Provider`, and launch the
  “verifiable benchmark leaderboard” that gates entries on capsule verification.
* **Q3**: onboard at least one external zkVM/prover team using the TraceAdapter docs + leaderboard
  and convert them into recurring capsule producers.

The capsule is our “zip format” for proofs. These initiatives turn it from a single-product demo
into the neutral backend layer the ecosystem can standardize on.
