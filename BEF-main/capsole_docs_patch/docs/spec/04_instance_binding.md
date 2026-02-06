# Instance binding (normative, aligned to current code)

## Goal

Prevent **statement/trace/spec drift**: a proof for one capsule must not be replayable as a proof for a different capsule’s statement, trace spec, root, or acceptance-affecting parameters.

Backends MUST absorb/check a verifier-supplied binding value (called `binding_hash` in adapter APIs).

## Definition: `instance_hash`

Implementations MUST compute:

```
instance_hash = H(
  b"CAPSULE_INSTANCE_V1::" ||
  vk_hash || statement_hash || trace_spec_hash || row_root ||
  params_hash || chunk_meta_hash || row_tree_arity ||
  air_params_hash || fri_params_hash || program_hash
)
```

Where:
- `vk_hash` identifies the verification key.
- `statement_hash` binds the public statement.
- `trace_spec_hash` binds the trace format/semantics declaration.
- `row_root` binds the committed row-major trace table.
- `params_hash` binds acceptance-affecting config not otherwise captured (backend-specific knobs).
- `chunk_meta_hash` binds chunking metadata (num_chunks, chunk_len, etc.).
- `row_tree_arity` binds the Merkle arity used for `row_root`.
- `air_params_hash`, `fri_params_hash`, `program_hash` bind the proof system identity.

### Backend requirement

A backend verifier MUST:
- take `instance_hash` (aka `binding_hash`) as an explicit input; and
- ensure a proof cannot verify under a different `instance_hash`.

In a Fiat–Shamir STARK backend, this means `instance_hash` MUST be absorbed into the transcript before challenges are derived.

## Conformance tests (required)

For any backend adapter, the following mutations MUST cause verification to fail:

- Change any one of: `vk_hash`, `statement_hash`, `trace_spec_hash`, `row_root`
- Change any acceptance-affecting parameter covered by `params_hash`
- Change `chunk_meta_hash` or `row_tree_arity`
- Change any of: `air_params_hash`, `fri_params_hash`, `program_hash`

Adapters that do not fail closed on these mutations are non-conformant.
