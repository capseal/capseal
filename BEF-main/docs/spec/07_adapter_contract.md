# Spec 07 — Adapter Contract (normative)

API (pseudo‑signature)
- simulate_trace(args) -> TraceArtifacts
- commit_to_trace(artifacts, row_archive_dir) -> TraceCommitment
- generate_proof(artifacts, commitment, *, statement_hash: bytes, binding_hash: bytes, ...) -> ProofArtifacts
- verify(proof_json_or_bytes, statement_hash: bytes, artifacts, *, binding_hash: bytes) -> (ok: bool, stats, time)

MUSTs
- `binding_hash` MUST be treated as a public input (SNARK) or absorbed into the Fiat–Shamir transcript (STARK). Ignoring it is non‑conformant.
- Row parameters in the proof (`root, chunk_len, chunk_tree_arity`) MUST match `row_index_ref` and `chunk_meta` in the capsule.
- `program_hash, vk_hash, air_params_hash, fri_params_hash` MUST be recomputed by the verifier and compared to capsule header fields.

Mutation tests (adapters MUST fail):
1. Flip any of: vk_hash, statement_hash, trace_spec_hash, row_root.
2. Change any of: chunk_len, num_chunks, tree_arity.
3. Change any of: air_params_hash, fri_params_hash, program_hash, backend_id.

