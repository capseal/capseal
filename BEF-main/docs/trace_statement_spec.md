# TraceSpecV1 & StatementV1 Overview

## TraceSpecV1 (Trace Definition)

Every capsule now embeds a `trace_spec` object plus its canonical hash. The fields are:

- `spec_version`: e.g. `"1.0"`
- `trace_format_id`: logical name such as `"GEOM_AIR_V1"`
- `record_schema_ref`: content-addressed reference (`sha256:<hash>`) to the column schema
- `encoding_id`: canonical encoding for records (`dag_cbor_canonical_v1`, `dag_cbor_compact_fields_v1`, ...)
- `field_modulus_id`: identifier for the prime field (e.g. `"goldilocks_61"`)

The prover uses canonical CBOR‐encoding with a domain-separated prefix to compute `trace_spec_hash`.
The capsule carries both the object and the hash so verifiers can recompute the hash before trusting the spec.

## StatementV1 (Binding Claim)

`statement` captures what is being proven:

- `statement_version`
- `trace_spec_hash`
- `policy_hash`
- `trace_root` (row index Merkle root)
- `public_inputs` (array of `{name,value}` pairs describing AIR public inputs)
- `anchors` (array of external commitments, e.g. L1 block hash, dataset hash)

The prover constructs `StatementV1`, hashes it canonically (domain-separated), and the STARK transcript
absorbs that hash before any challenges are sampled. Capsules store both the serialized object and `statement_hash`.
The verifier recomputes the hash from the capsule and feeds it to `zk_verify_geom`, preventing proof reuse under a different statement.

## Policy Integration

`da_policy` now includes an explicit `verification_level` (currently `"probabilistic_da_sampling"`).
Policy hashes are content-addressed; verifiers compare `statement.policy_hash` against the capsule’s policy
and optional policy registry proofs.

## Content-Addressed References

- Record schemas: `record_schema_ref = sha256:<digest>`
- Policies: `policy_hash = sha256(...)`
- Trace spec / statement: canonical CBOR hashes with domain-separated prefixes

Resolvers (manifest entries) can map these references to actual files when needed, but the capsule primarily binds
hashes so the data can live out-of-band.
