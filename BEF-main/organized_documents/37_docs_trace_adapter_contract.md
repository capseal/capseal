# Trace Adapter Contract

CapsuleBench uses the `TraceAdapter` interface (see `bef_zk/adapter.py`) to plug any backend prover into the pipeline. An adapter must implement the following methods:

1. `simulate(trace_id: str, callbacks: Optional[ProgressSink]) -> TraceSimResult`
   * Run the workload and produce a `TraceSimResult` containing:
     - `trace_spec`: the TraceSpecV1 description (rows, columns, names).
     - `public_inputs`: a list of public outputs / named fields.
     - `row_archive`: metadata describing how to load row chunks (paths, roots).
   * Emit progress callbacks so events (`run_started`, `trace_simulated`, etc.) stream to the relay.

2. `extract_public_inputs(sim_result: TraceSimResult) -> StatementV1`
   * Build the `StatementV1` object from the simulation result.
   * Include anchors (policy hash, manifests, trace commitment) in the statement.

3. `commit_to_trace(sim_result: TraceSimResult) -> TraceCommitmentResult`
   * Commit to the row trace by producing Merkle roots/chunks.
   * Return a `TraceCommitmentResult` with:
     - `row_root`, `chunk_meta`, `row_index_ref`
     - `chunk_handles` (files for each chunk)
     - `chunk_roots` (hashes per chunk)

4. `generate_proof(sim_result: TraceSimResult, commitment: TraceCommitmentResult, callbacks: Optional[ProgressSink]) -> ProofArtifacts`
   * Run the backend prover (e.g. geom, risc0) and produce proof artifacts.
   * Proofs must bind to the `statement_hash` / row commitments.
   * Return `ProofArtifacts` describing proof files (JSON/bin), payload hashes, stats, etc., and emit events (`proof_artifact`, `capsule_sealed`).

5. `verify_proof(proof_artifacts: ProofArtifacts, statement: StatementV1) -> None`
   * Verify the backend proof against the statement.
   * Raise if the proof doesn’t bind to the statement hash/commitment.

Adapters may optionally use the `ProgressSink` to emit granular progress, but they **must** ensure the final proof transcript binds to `TraceSpecV1` + `StatementV1` via `statement_hash` and row commitment parameters. The pipeline handles event logging, packaging, policy verification, DA audit, and artifact upload — you just implement this contract.
