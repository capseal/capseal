# Spec 01 — Objects and Fields (normative)

All objects are serialized canonically (see 02_canonicalization.md). Hashes and signatures MUST be computed over canonical bytes.

TraceSpecV1 (object)
- spec_version: string
- trace_format_id: string
- record_schema_ref: string (e.g., `sha256:<hex>` or `inline`)
- encoding_id: string (e.g., `dag_cbor_canonical_v1`)
- field_modulus_id: string

StatementV1 (object)
- statement_version: string
- trace_spec_hash: hex (sha256)
- policy_hash: hex (sha256)
- trace_root: hex (row Merkle root)
- public_inputs: list of {name: string, value: int|string}
- anchors: list of AnchorView (see below)

AnchorView (object)
- anchor_rule_id: string (e.g., `capsule_bench_manifest_v1`)
- anchor_ref: string (`capsulebench_manifest_v1:<sha256>`)
- track_id: string
- event_chain_head: hex (optional)

ChunkMeta (object)
- chunk_len: int (rows per chunk)
- num_chunks: int
- chunk_size_bytes: int
- data_length_bytes: int
- chunking_rule_id: string

RowIndexRef (object)
- commitment_type: `merkle_root`
- commitment: hex (root)
- tree_arity: int (k‑ary)
- proof_fetch_rule_id: string
- pointer: { path: `row_archive`, provider_root: abs path or URI }

Capsule Header (object)
- schema: `capsule_header_v*`
- vm_id, backend_id, circuit_id
- trace_id, prev_capsule_hash (optional)
- trace_spec_hash, statement_hash
- row_tree_arity, row_index_ref_hash, chunk_meta_hash, chunk_handles_root
- policy_ref: {policy_id, policy_version, policy_hash, track_id}
- da_policy_hash
- anchor: AnchorView + {events_log_hash?, events_log_len?}
- proof_system: {air_params_hash, fri_params_hash, program_hash, vk_hash}
- manifest_hash
- air_params_hash, fri_params_hash, program_hash
- payload_hash
- verification_profile

Capsule Payload (object)
- trace_spec, statement
- params (AIR params view)
- da_policy, chunk_meta, row_index_ref, hashing
- proofs (formats and default)
- row_archive (artifact map)
- artifacts: {manifest, events_log?, da_challenge?}

Capsule Hashes (values)
- payload_hash = H(DST_CAPSULE_PAYLOAD_V2 || canonical_payload_bytes)
- header_commit_hash = H(DST_CAPSULE_HEADER_COMMIT_V2 || canonical_header_commit_bytes)
- capsule_hash = H(DST_CAPSULE_ID_V2 || header_commit_hash || payload_hash)

