# Capsules + STC — Consolidated Documentation

This file concatenates the Markdown docs under `docs/` (including `docs/spec/` and `docs/guides/`) in a stable order.

## Table of Contents
- [docs/index.md](#docs-indexmd)
- [docs/primer.md](#docs-primermd)
- [docs/guides/cli.md](#docs-guides-climd)
- [docs/guides/integration_guide.md](#docs-guides-integration_guidemd)
- [docs/spec/00_overview.md](#docs-spec-00_overviewmd)
- [docs/spec/01_objects.md](#docs-spec-01_objectsmd)
- [docs/spec/02_canonicalization.md](#docs-spec-02_canonicalizationmd)
- [docs/spec/03_domain_tags.md](#docs-spec-03_domain_tagsmd)
- [docs/spec/04_instance_binding.md](#docs-spec-04_instance_bindingmd)
- [docs/spec/05_profiles.md](#docs-spec-05_profilesmd)
- [docs/spec/06_protocol.md](#docs-spec-06_protocolmd)
- [docs/spec/07_adapter_contract.md](#docs-spec-07_adapter_contractmd)
- [docs/spec/08_registries.md](#docs-spec-08_registriesmd)
- [docs/spec/09_reason_codes.md](#docs-spec-09_reason_codesmd)
- [docs/spec/10_cap_format.md](#docs-spec-10_cap_formatmd)
- [docs/spec/SPEC.md](#docs-spec-SPECmd)
- [docs/security_model.md](#docs-security_modelmd)
- [docs/security/theorems.md](#docs-security-theoremsmd)
- [docs/SECURITY_ANALYSIS.md](#docs-SECURITY_ANALYSISmd)
- [docs/backends/geom.md](#docs-backends-geommd)
- [docs/backends/risc0.md](#docs-backends-risc0md)
- [docs/trace_adapter_contract.md](#docs-trace_adapter_contractmd)
- [docs/trace_statement_spec.md](#docs-trace_statement_specmd)
- [docs/stc_backend_architecture.md](#docs-stc_backend_architecturemd)
- [docs/stc_da_profiles.md](#docs-stc_da_profilesmd)
- [docs/hssa_da_protocol.md](#docs-hssa_da_protocolmd)
- [docs/roadmap.md](#docs-roadmapmd)
- [docs/README.md](#docs-READMEmd)
- [docs/notes/dec18_plan.md](#docs-notes-dec18_planmd)
- [docs/archive/BACKEND_READINESS_GAPS.md](#docs-archive-BACKEND_READINESS_GAPSmd)
- [docs/archive/BEF-Stream-Accum-summary.md](#docs-archive-BEF-Stream-Accum-summarymd)
- [docs/archive/BEF_CURRENT_STATUS.md](#docs-archive-BEF_CURRENT_STATUSmd)
- [docs/archive/bef_compact_v1_spec.md](#docs-archive-bef_compact_v1_specmd)
- [docs/archive/bef_trace_commitment.md](#docs-archive-bef_trace_commitmentmd)
- [docs/archive/da_profile_router.md](#docs-archive-da_profile_routermd)
- [docs/archive/encoding.md](#docs-archive-encodingmd)
- [docs/archive/fri_parameter_guidance.md](#docs-archive-fri_parameter_guidancemd)
- [docs/archive/geom_backend_analysis.md](#docs-archive-geom_backend_analysismd)
- [docs/archive/hssa_vs_kzg_bench.md](#docs-archive-hssa_vs_kzg_benchmd)
- [docs/archive/ivc_state_R.md](#docs-archive-ivc_state_Rmd)
- [docs/archive/operator_flow.md](#docs-archive-operator_flowmd)
- [docs/archive/stc_parameter_guidance.md](#docs-archive-stc_parameter_guidancemd)
- [docs/archive/stc_pc_backend.md](#docs-archive-stc_pc_backendmd)
- [docs/archive/stc_vm_mapping.md](#docs-archive-stc_vm_mappingmd)
- [docs/archive/streaming_trace_commitment_formal.md](#docs-archive-streaming_trace_commitment_formalmd)

---


---

# docs/index.md

<a name="docs-indexmd"></a>

# Capsules + STC Documentation Map

This repository’s docs are split into four layers. Read what you need, and only that:

- Primer (human intro): `docs/primer.md`
- Spec (normative; single source of truth): `docs/spec/`
- Security model (hostile‑review predicates): `docs/security_model.md` and `docs/security/theorems.md`
- Guides (integration and CLI): `docs/guides/`

CLI quickstart:
- CLI usage: `docs/guides/cli.md`
- Portable `.cap` format: `docs/spec/10_cap_format.md`

If you find any definition duplicated outside of `docs/spec/`, treat it as non‑normative. The spec is the law.

Backend guides:
- Geom: `docs/backends/geom.md`
- Risc0: `docs/backends/risc0.md`



---

# docs/primer.md

<a name="docs-primermd"></a>

# Capsules + STC: Primer

This primer explains the core primitive (the capsule), how it binds traces and proofs, and the end‑to‑end flow including Data Availability (DA) auditing. It is written to support hostile review: what is bound, in what order, and how it is verified.

## Capsule Anatomy

A capsule is a content‑addressed object: a header, a payload, and canonical hashes that give it a stable identity.

```
Capsule (bef_capsule_v1)
┌───────────────────────────────────────────────────────────────┐
│ header: {                                                     │
│   schema: HEADER_SCHEMA,                                      │
│   trace_spec_hash, statement_hash,                            │
│   row_index_ref: { commitment(root), tree_arity },            │
│   chunk_meta: { chunk_len, num_chunks },                      │
│   proof_system: {                                             │
│     air_params_hash, fri_params_hash, program_hash, vk_hash   │
│   },                                                          │
│   instance_hash = H("CAPSULE_INSTANCE_V1" ||                 │
│                    vk_hash || statement_hash ||               │
│                    trace_spec_hash || row_root),              │
│   policy_ref: { policy_id, version, policy_hash, track_id },  │
│   da_ref, artifact_manifest_hash, events_log_hash/len         │
│ }                                                             │
│                                                               │
│ payload: {                                                    │
│   trace_spec (canonical),                                     │
│   statement  (canonical),                                     │
│   proofs + proof artifacts,                                   │
│   row archive manifest pointers,                              │
│   policy view, da_challenge (optional)                        │
│ }                                                             │
│                                                               │
│ payload_hash        = H("CAPSULE_PAYLOAD_V2" || Enc(payload)) │
│ header_commit_hash  = H("CAPSULE_HEADER_COMMIT_V2" || Enc(..))│
│ capsule_hash        = H("CAPSULE_ID_V2" ||                    │
│                       header_commit_hash || payload_hash)     │
└───────────────────────────────────────────────────────────────┘
```

The capsule may carry an authorship signature (secp256k1 over `capsule_hash`) and an ACL policy for the signer. For portable verification, capsules can be packaged into a `.cap` archive (see `docs/spec/10_cap_format.md`) and verified hermetically via the CLI (`docs/guides/cli.md`).

## Binding Surfaces

```
TraceSpecV1  --hash-->  trace_spec_hash
StatementV1  --hash-->  statement_hash (binds trace_root, policy_hash, anchors)
STC rows     --merkle--> row_root (k‑ary tree with arity, chunk_len, num_chunks)

Proof System:
  air_params_hash, fri_params_hash, program_hash, vk_hash
  instance_hash = H("CAPSULE_INSTANCE_V1" || vk_hash || statement_hash || trace_spec_hash || row_root)

Backends absorb the binding:
  • STARK (geom): absorb `instance_hash` (or `statement_hash`) into Fiat–Shamir
  • RISC0: journal == binding bytes (equality check in prover/verifier)
```

Manifests (hardware/os/toolchain) are hashed into a *manifest anchor* and then signed:

```
manifests/*.json  --sha256-->  hashes map
anchor_payload = { schema: "capsule_bench_manifest_anchor_v1", hashes }
anchor_ref = "capsulebench_manifest_v1:" + sha256(anchor_payload)

Signature: manifests/manifest_signature.json
  { schema: "capsule_manifest_signature_v1",
    signer_id, signature = secp256k1_recoverable(anchor_ref) }
```

## DA Commit → Challenge → Open (flow)

```
        (1) commit                           (2) challenge                         (3) open samples
Prover ─────────────► Relay/Registry ─────────────────────────► Verifier ─────────────► Provider
         header_commit_hash,               signed challenge{ relay_pubkey_id,         fetch indices k
         chunk_handles_root                capsule_commit_hash, nonce, expiry }       verify Merkle paths

seed = derive_da_seed(capsule_hash, challenge)
indices = deterministic_sample(seed, total=num_chunks, k)
```

The verifier accepts only challenges signed by a relay key included in the pinned relay registry root.

## End‑to‑End Verification

1) Capsule integrity: payload/hash/header hashes match; schemas consistent.
2) Spec/statement: recompute `trace_spec_hash`, `statement_hash`; match header.
3) Instance binding: recompute `air_params_hash`, `fri_params_hash`, `program_hash`, `vk_hash`; build `instance_hash` and feed to backend verifier.
4) Row commitment: chunk_len/num_chunks/tree_arity consistent; sampled chunks open to `row_root` (if DA audited).
5) Manifests/policy: recompute anchor; verify manifest signature against trusted registry; apply policy over signed manifests.
6) Authorship/ACL: signature over `capsule_hash`; evaluate ACL if provided.

The result is a structured verdict: PROOF_ONLY / POLICY_SELF_REPORTED / POLICY_ENFORCED / FULLY_VERIFIED with error codes if any predicate fails.

## Threat Model (abridged)

- Adversary can fabricate traces, manifests, and capsules but cannot break SHA‑256 or STARK/RISC0 soundness.
- Verifier is honest; trust roots (relay + signer registries) are pinned and not attacker‑controlled.
- DA audit requires commit before unpredictable challenge; relay keys are authentic.

## Demo Commands (quick)

```
# Run a signed demo
capsule-bench run \
  --backend geom \
  --policy policies/demo_policy_v1.json \
  --policy-id demo_policy_v1 --policy-version 1.0 \
  --track-id demo_geom_fast \
  --manifest-signer-id test_manifest \
  --manifest-signer-key 3333333333333333333333333333333333333333333333333333333333333333

# Verify
PYTHONPATH=. python scripts/verify_capsule.py \
  out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --policy policies/demo_policy_v1.json \
  --manifest-root out/capsule_runs/<run_id>/manifests
```

For a live demo that shows PASS → TAMPER → REJECT and prints key hashes, use `scripts/demo_run_and_verify.py` (see below).



---

# docs/guides/cli.md

<a name="docs-guides-climd"></a>

# Capsule CLI

Hermetic, portable verification via a single file.

## Commands

### Emit
```
capsule emit \
  --capsule out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --artifacts out/capsule_runs/<run_id>/pipeline \
  --policy out/capsule_runs/<run_id>/policy.json \
  --out /tmp/receipt.cap
```

Produces a `.cap` archive containing:
- `capsule.json` (full capsule)
- `proof.bin.zst` (compressed proof payload)
- `commitments.json` (root/chunk metadata)
- `artifact_manifest.json` (content-addressed artifact index)
- `events/events.jsonl` (event chain, optional)
- `archive/` (row archive, optional)
- `policy.json` (optional)

### Verify
```
# Raw capsule.json (requires policy/manifests on disk)
PYTHONPATH=. python scripts/verify_capsule.py <capsule.json> \
  --policy <policy.json> --manifest-root <manifests/>

# Hermetic .cap (uses embedded artifacts and safe extraction)
capsule verify /tmp/receipt.cap --json
```

Exit codes:
- 0 verified
- 10 proof invalid (E05x/E3xx)
- 11 policy mismatch (E03x/E10x)
- 12 commitment/index failed (E06x/E2xx)
- 13 DA audit failed (E07x)
- 20 malformed/parse error (E001–E004)

### Inspect
```
capsule inspect /tmp/receipt.cap --json
```

## Safety & Semantics

- Extraction is sandboxed: no symlinks/hardlinks, no traversal/absolute paths, size limits per entry.
- Materialization writes proof/archive/events to their recorded `rel_path` and validates sizes/hashes before invoking the canonical verifier.
- `.cap` verification matches raw `scripts/verify_capsule.py` behavior and reason codes.

See `docs/spec/10_cap_format.md` and `docs/spec/06_protocol.md`.




---

# docs/guides/integration_guide.md

<a name="docs-guides-integration_guidemd"></a>

# Integration Guide

Quick links: `docs/primer.md` (intro) • `docs/spec/` (normative) • `docs/security_model.md` (predicates)

## Run
```
capsule-bench run --backend <id> --policy <path> --policy-id <id> --track-id <track> \
  --manifest-signer-id <id> --manifest-signer-key <hex|path>
```

## Verify (raw capsule.json)
```
PYTHONPATH=. python scripts/verify_capsule.py <capsule.json> \
  --policy <policy.json> --manifest-root <manifests/>
```

## Package as a portable .cap
```
# Create a hermetic verification artifact
capsule emit \
  --capsule out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --artifacts out/capsule_runs/<run_id>/pipeline \
  --policy out/capsule_runs/<run_id>/policy.json \
  --out /tmp/receipt.cap
```

## Verify a .cap (hermetic)
```
# Self-contained: extractor enforces path safety; verifier uses embedded artifacts
capsule verify /tmp/receipt.cap --json

# Or provide policy/manifests explicitly
capsule verify /tmp/receipt.cap --policy policy.json --manifests manifests/
```

Notes:
- `.cap` verification reconstructs the expected rel-path layout in a sandboxed temp dir and validates sizes/hashes before calling the canonical verifier.
- See `docs/spec/10_cap_format.md` for the archive format and safety guarantees.

See `docs/spec/07_adapter_contract.md` for backend integration and mutation tests.



---

# docs/spec/00_overview.md

<a name="docs-spec-00_overviewmd"></a>

# Spec 00 — Overview and Normative Scope

This folder is the single source of truth. Every definition here is normative and versioned. Primer and guides MAY summarize but MUST NOT redefine.

Conformance language follows RFC 2119 (MUST/SHOULD/MAY).

Scope:
- Objects/fields that constitute a capsule and its header/payload.
- Canonical encodings and domain tags for hashes/signatures.
- Instance binding (the tuple and how backends MUST absorb it).
- Verifier profiles and acceptance predicates.
- Adapter contract (prove/verify API + mutation tests).
- Registries/trust roots and override rules.
- DA protocol messages and ordering constraints.

Out of scope:
- Performance and parameter tuning; storage SLAs; UI/relay UX.




---

# docs/spec/01_objects.md

<a name="docs-spec-01_objectsmd"></a>

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




---

# docs/spec/02_canonicalization.md

<a name="docs-spec-02_canonicalizationmd"></a>

# Canonicalization (normative)

All hashes and signatures in CapSole are computed over **canonical bytes**. Implementations MUST hash bytes, not parsed objects.

## Normative encoding: `dag_cbor_compact_fields_v1`

The canonical encoding for all hash/signature inputs is **DAG‑CBOR** with the project’s “compact fields” ruleset:

- Deterministic / canonical CBOR encoding (no floats, no indefinite lengths).
- Stable map key ordering.
- Field normalization (“compact fields”) applied consistently (exact rules belong to the encoder implementation).

If two implementations produce different bytes for the same semantic object, they are **non‑conformant**.

## Non‑normative views

JSON may be used for human inspection, debugging, or UI, but:
- JSON encodings are not authoritative
- JSON bytes MUST NOT be hashed for verification



---

# docs/spec/03_domain_tags.md

<a name="docs-spec-03_domain_tagsmd"></a>

# Domain tags (normative)

This file is the single source of truth for hash/signature domain separation strings used by the current codebase.

## Hash prefixes / domain tags

| Name | Value (bytes) | Used for |
| --- | --- | --- |
| Payload hash | `b"BEF_CAPSULE_V1"` | `payload_hash = H(tag || Enc(payload_view))` |
| Audit seed | `b"BEF_AUDIT_SEED_V1"` | `seed = H(tag || capsule_hash || challenge_bytes)` |
| Header hash | `b"CAPSULE_HEADER_V2::"` | hashing the full header (if present) |
| Header commit hash | `b"CAPSULE_HEADER_COMMIT_V1::"` | `header_commit_hash` |
| Capsule ID hash | `b"CAPSULE_ID_V2::"` | `capsule_hash` |
| Instance binding hash | `b"CAPSULE_INSTANCE_V1::"` | `instance_hash` |

## Notes

- Version numbers in tags are not “marketing”; they are part of the security boundary.
- If you change any domain tag, you MUST treat it as a breaking format change.



---

# docs/spec/04_instance_binding.md

<a name="docs-spec-04_instance_bindingmd"></a>

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



---

# docs/spec/05_profiles.md

<a name="docs-spec-05_profilesmd"></a>

# Verification profiles (normative)

Profiles are *verifier predicates*: the verifier chooses the requested level and the verification procedure must fail-closed if prerequisites are missing.

## Profiles

### PROOF_ONLY

Verifies:
- Capsule integrity hashes (payload/hash/header)
- `statement_hash`, `trace_spec_hash` recomputation
- `instance_hash` recomputation and backend proof verification bound to it

Guarantee: *the statement holds for the committed `row_root` under the bound proof system parameters.*

### POLICY_ENFORCED

Includes PROOF_ONLY, plus:
- Manifest anchor recomputation
- Manifest signature verification against a pinned signer registry root
- Policy evaluation over the signed anchor + statement
- Authorship signature and ACL checks if required by the policy

Guarantee: PROOF_ONLY + *policy rules were evaluated over a signed manifest anchor under pinned trust roots.*

### FULL

Includes POLICY_ENFORCED, plus:
- Relay challenge verification against a pinned relay registry root
- Deterministic sampling and k sample-open checks to `row_root`
- Enforced commit→challenge→open ordering (challenge excluded from header_commit_hash)

Challenge requirements:
- The challenge MUST be `challenge_v1` as defined in Spec 06, signed by a trusted challenger key id present in the pinned registry. Unsigned/mismatched/expired challenges MUST result in a failure, not a downgrade.

Guarantee: POLICY_ENFORCED + *availability sampling succeeded under a trusted relay challenge.*



---

# docs/spec/06_protocol.md

<a name="docs-spec-06_protocolmd"></a>

# Spec 06 — Protocol (normative)

## Ordering constraints (DA)
1. Prover commits: publishes `(header_commit_hash, chunk_handles_root)` and (optionally) receives a signed commit receipt with timestamp T_c.
2. Challenger issues challenge: signed JSON with `{capsule_commit_hash, seed, k, chunk_len, chunk_tree_arity, issued_at, key_id}`. Challenge time T_ch > T_c.
3. Verifier derives seed and sampling deterministically; provider serves sampled chunks; verifier checks Merkle paths against `row_root`.

`header_commit_hash` MUST exclude the challenge so commit→challenge remains acyclic.

## Messages (canonical JSON)
- Commit: `{capsule_commit_hash, payload_hash, chunk_handles_root, num_chunks}`
- Challenge v1: `{
    scheme: "da_sample_v1",
    capsule_commit_hash,
    seed, k, chunk_len, chunk_tree_arity,
    issued_at,
    issuer: { key_id },
    sig
  }`

Verifier MUST reject if: missing/expired/unsigned challenge; challenge hash mismatches; registry root mismatch; any sampled chunk missing or Merkle path invalid.



---

# docs/spec/07_adapter_contract.md

<a name="docs-spec-07_adapter_contractmd"></a>

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




---

# docs/spec/08_registries.md

<a name="docs-spec-08_registriesmd"></a>

# Registries & signatures (normative, aligned to current code)

## Registry pinning

Registries (manifest signers, trusted relays) are pinned by a root hash computed over a canonical `id → pubkey` map.

- The verifier must know the expected root hash out-of-band (config pin).
- Overriding a registry requires supplying both:
  1) the new key map and
  2) the expected root for that map

If the computed root mismatches, verification MUST fail.

## Manifest anchor signing (current)

- The manifest bundle is reduced to an anchor payload and then to an `anchor_digest = sha256(Enc(anchor_payload))`.
- The manifest signature is a secp256k1 signature over the **anchor_digest bytes**.

No additional domain tag is prepended in current code. If you want a DST, you must introduce it as a breaking change and update verifiers accordingly.

## Relay challenge signing (summary)

Relay challenges are accepted only if:
- signed by a relay key id present in the pinned relay registry
- unexpired (if expiry is modeled)
- bound to the capsule commit/capsule hash as specified in `docs/spec/06_protocol.md`



---

# docs/spec/09_reason_codes.md

<a name="docs-spec-09_reason_codesmd"></a>

# Spec 09 — Reason Codes (stable)

The verifier emits machine‑readable error codes. The canonical list is defined in `bef_zk/verifier_errors.py`. Implementations MUST NOT repurpose an existing code; new codes MUST be appended.

Examples (non‑exhaustive):
- E010_CAPSULE_HASH_MISSING
- E011_CAPSULE_HASH_MISMATCH
- E012_CAPSULE_HEADER_MISSING
- E013_CAPSULE_HEADER_MISMATCH
- E050_PROOF_MISSING
- E052_PROOF_HASH_MISMATCH
- E053_PROOF_STATEMENT_MISMATCH
- E064_MERKLE_PROOF_INVALID
- E065_CHUNK_ROOT_MISMATCH
- E071_DA_CHALLENGE_MISSING
- E073_DA_CHALLENGE_UNTRUSTED
- E101_POLICY_VIOLATION_FORBID_GPU
- E106_MANIFEST_SIGNATURE_MISSING
- E109_MANIFEST_REGISTRY_MISMATCH




---

# docs/spec/10_cap_format.md

<a name="docs-spec-10_cap_formatmd"></a>

# Spec 10 — `.cap` format (normative)

A `.cap` is a hermetic verification archive. It contains the capsule and the minimum artifacts required for verification, with strict extraction safety rules.

## Contents

- `capsule.json` — the full capsule descriptor
- `proof.bin.zst` — compressed proof payload (optional if proofs are small/JSON)
- `commitments.json` — root/chunk metadata (optional; convenience)
- `artifact_manifest.json` — content-addressed artifact index (optional but RECOMMENDED)
- `events/events.jsonl` — event stream (optional)
- `archive/` — row archive snapshots (optional)
- `policy.json` — policy document (optional)

## Safe extraction (required)

Implementations MUST enforce the following when extracting a `.cap`:

- Reject absolute paths, `..` traversal, and backslashes as separators.
- Reject symlinks and hardlinks.
- Allow only directories and regular files.
- Enforce a maximum member size (configurable; default ≤ 512 MiB).
- Reject duplicate normalized paths.
- Refuse to overwrite an existing path in the sandbox.

## Materialization (required)

After extraction into a sandbox directory, implementations MUST:

- Write the proof blob (if present) to the rel path recorded in the capsule’s `proofs[...].formats[*].rel_path` (or a default); validate size and `sha256_payload_hash` before use.
- Relocate `archive/` to the row archive’s recorded `rel_path` (default `row_archive/`).
- If `artifacts.row_archive.rel_path` differs, mirror or alias accordingly.
- Preserve `events/events.jsonl` and `artifact_manifest.json` so verifier can recompute anchors.

## Verification mapping

- The verifier MUST use the materialized paths within the sandbox and MUST NOT read outside the sandbox.
- Size/hash mismatches for materialized artifacts MUST produce a hard failure.
- The behavior and reason codes MUST match the canonical verifier run over the original `strategy_capsule.json`.

This spec complements Spec 06 (Protocol) and Spec 08 (Registries) by standardizing portable verification.




---

# docs/spec/SPEC.md

<a name="docs-spec-SPECmd"></a>

# BEF Capsule Specification v0.2

**Status:** Frozen as `v0.2-adapters-cli-fixtures`
**Date:** 2025-12-27

This document defines the capsule format, verification semantics, and backend interface
as of v0.2. Changes to these contracts require a version bump.

---

## 1. Capsule Structure

A capsule is a cryptographic receipt that binds:
- A **trace** (execution record)
- A **proof** (cryptographic attestation)
- A **policy** (rules governing valid execution)
- **Metadata** (timestamps, identifiers, anchors)

### 1.1 Core Fields

```
capsule.json
├── schema: "bef_capsule_v1"
├── trace_id: string
├── trace_spec: TraceSpecV1
├── trace_spec_hash: hex
├── policy: PolicyRef
├── params: {row_width, ...}
├── da_policy: DAPolicy
├── chunk_meta: ChunkMeta
├── row_index_ref: RowIndexRef
├── statement: StatementV1
├── statement_hash: hex
├── header: CapsuleHeaderV2
├── header_hash: hex
├── capsule_hash: hex
├── proofs: {primary: ProofEntry}
└── artifacts: {events_log, row_archive, ...}
```

### 1.2 Header Schema (capsule_header_v2)

The header contains commitment hashes for all mutable fields:

```
header
├── schema: "capsule_header_v2"
├── trace_id: string
├── trace_spec_hash: hex
├── policy_id: string
├── policy_hash: hex
├── params_hash: hex
├── chunk_meta_hash: hex
├── row_index_ref_hash: hex
├── da_policy_hash: hex
├── statement_hash: hex
├── proof_system: ProofSystemMeta
├── row_commitment: RowCommitmentMeta
├── verification_profile: string
└── anchor: AnchorMeta
```

---

## 2. Hash Binding Rules

All hashes use SHA256 with domain separation prefixes.

### 2.1 Domain Prefixes

| Hash Type | Prefix |
|-----------|--------|
| Header | `CAPSULE_HEADER_V2::` |
| Header Commit | `CAPSULE_HEADER_COMMIT_V1::` |
| Params | `CAPSULE_PARAMS_V1::` |
| Capsule ID | `CAPSULE_ID_V2::` |
| Chunk Meta | `CAPSULE_CHUNK_META_V1::` |
| Row Index Ref | `CAPSULE_ROW_INDEX_REF_V1::` |
| DA Policy | `CAPSULE_DA_POLICY_V1::` |
| Chunk Manifest | `CAPSULE_CHUNK_MANIFEST_V1::` |
| Proof System | `CAPSULE_PROOF_SYSTEM_V1::` |
| AIR Params | `CAPSULE_AIR_PARAMS_V1::` |
| FRI Config | `CAPSULE_FRI_CONFIG_V1::` |
| Program | `CAPSULE_PROGRAM_V1::` |
| Instance | `CAPSULE_INSTANCE_V1::` |
| Statement | `BEF_STATEMENT_V1::` |
| Trace Spec | `BEF_TRACE_SPEC_V1::` |

### 2.2 Hash Computation

```python
def canonical_hash(prefix: bytes, payload: Any) -> str:
    encoded = canonical_encode(payload, encoding_id="dag_cbor_compact_fields_v1")
    return sha256(prefix + encoded).hexdigest()
```

### 2.3 Capsule Hash

```
capsule_hash = sha256(CAPSULE_ID_V2:: || header_hash || payload_hash)
```

Where:
- `header_hash = canonical_hash(CAPSULE_HEADER_V2::, header)`
- `payload_hash = canonical_hash(payload_fields)`

---

## 3. Verification Profiles

Three verification levels, each building on the previous:

### 3.1 PROOF_ONLY

**Requirements:**
- Capsule structure is valid
- Header hash matches content
- Statement hash matches statement
- Proof verifies against statement

**Exit code:** 0 if verified

### 3.2 POLICY_ENFORCED

**Requirements (in addition to PROOF_ONLY):**
- Policy document is present and hash matches
- Policy rules are evaluated against manifests
- If `require_attestation`: manifest signature is valid

**Exit code:** 0 if verified, 11 if policy mismatch

### 3.3 FULL

**Requirements (in addition to POLICY_ENFORCED):**
- DA challenge is present and signed by trusted challenger
- Challenge binds to `commit_root` and `payload_hash`
- Row archive passes DA sampling audit

**Exit code:** 0 if verified, 13 if DA failed

---

## 4. TraceAdapter Interface

Backends must implement the `TraceAdapter` ABC:

```python
class TraceAdapter(ABC):
    name: str = "unknown"

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Hook for adapter-specific CLI arguments."""

    @abstractmethod
    def simulate_trace(self, args: Any) -> TraceArtifacts:
        """Produce trace with metadata."""

    @abstractmethod
    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict]:
        """Return public inputs from prepared trace."""

    @abstractmethod
    def commit_to_trace(
        self, artifacts: TraceArtifacts, *, row_archive_dir: Path
    ) -> TraceCommitment:
        """Commit to trace and export STC artifacts."""

    @abstractmethod
    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: TraceCommitment,
        *,
        statement_hash: bytes,
        binding_hash: bytes | None = None,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        """Generate backend proof using finalized statement hash."""

    @abstractmethod
    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
        *,
        binding_hash: bytes | None = None,
    ) -> tuple[bool, dict, float]:
        """Run backend verifier."""
```

### 4.1 Backend Binding Requirements

Each backend must bind into the capsule:

| Field | Description |
|-------|-------------|
| `statement_hash` | Hash of the statement (trace_root, policy_hash, trace_spec_hash) |
| `params_hash` | Hash of backend-specific parameters |
| `trace_spec_hash` | Hash of the trace specification |
| `proof_system.scheme_id` | Backend identifier (e.g., "geom_stc_fri", "risc0_receipt_v1") |
| `proof_system.backend_id` | Backend name (e.g., "geom", "risc0") |

### 4.2 Registered Backends

| Backend | scheme_id | binding_hash |
|---------|-----------|--------------|
| Geom (Python) | `geom_stc_fri` | `H(INSTANCE_V1 \|\| ...)` |
| Geom (Rust) | `geom_stc_rust` | `H(INSTANCE_V1 \|\| ...)` |
| RISC0 | `risc0_receipt_v1` | `H(RISC0_BIND_V1 \|\| image_id \|\| journal_digest \|\| statement_hash)` |

---

## 5. Exit Codes

Stable exit codes for CI integration:

| Code | Meaning |
|------|---------|
| 0 | Verified successfully |
| 10 | Proof invalid (E054) |
| 11 | Policy mismatch (E03x, E1xx) |
| 12 | Commitment/binding failed (E05x, E06x) |
| 13 | DA audit failed (E07x) |
| 14 | Replay diverged (E08x) |
| 20 | Malformed input (E002, E003) |

---

## 6. .cap File Format

Portable capsule archive (gzipped tarball):

```
.cap
├── manifest.json       # CapManifest with metadata
├── capsule.json        # Full capsule data
├── commitments.json    # Root commitment, chunk info
├── proof.bin.zst       # Compressed proof (zstd)
├── policy.json         # Policy document (optional)
├── artifact_manifest.json # Encoding metadata
├── events/             # Events log (optional)
│   └── events.jsonl
├── archive/            # Row archive (optional)
│   └── chunk_*.bin
└── signatures/         # Detached signatures (optional)
```

### 6.1 Manifest Schema

```json
{
  "schema": "cap_manifest_v1",
  "capsule_id": "hex16",
  "trace_id": "string",
  "policy_id": "string",
  "policy_hash": "hex",
  "backend": "string",
  "verification_profile": "PROOF_ONLY|POLICY_ENFORCED|FULL",
  "root_hex": "hex",
  "num_chunks": int,
  "proof_size": int,
  "archive_format": "json|binary",
  "created_at": "ISO8601"
}
```

---

## 7. CLI Commands

### 7.1 capsule emit

```bash
capsule emit --capsule <path> --out <path.cap> [--policy <path>]
```

Packages capsule into portable `.cap` archive.

### 7.2 capsule verify

```bash
capsule verify <capsule.cap|capsule.json> [--mode proof-only|da|replay] [--json]
```

Verifies capsule with stable exit codes.

### 7.3 capsule inspect

```bash
capsule inspect <capsule.cap|capsule.json> [--json]
```

Displays capsule metadata without verification.

---

## 8. Future (v0.3+)

Not part of this spec, but planned:

- **Signed DA Challenge v1**: Non-prover-picked challenge with issuer signature
- **Policy attestation**: Signed binding of policy + manifest
- **EIP-4844 anchoring**: Public challenge via Ethereum blobs

---

## Changelog

- **v0.2** (2025-12-27): Initial frozen spec
  - TraceAdapter ABC with Geom and RISC0 backends
  - Capsule CLI (emit/verify/inspect)
  - Golden fixtures with tamper detection
  - Stable exit codes for CI



---

# docs/security_model.md

<a name="docs-security_modelmd"></a>

# Capsules + STC Security Model

This note records the threat model, binding requirements and security claims that the verifier now enforces. It is designed to be hostile-review ready: every predicate is explicit, and every assumption is called out.

## 1. Adversary model

* **Malicious prover** – controls the adapter, local filesystem and OS, can fabricate traces, proofs, manifests and DA responses. Cannot break collision resistance or the STC/FRI soundness assumptions.
* **Malicious policy author / registry** – may attempt to publish conflicting policies or lie about enforcement rules.
* **Malicious verifier** – may downgrade required levels locally, but cannot forge relay signatures or alter the canonical capsule payload without changing the payload hash. When verifying `.cap` archives, extraction/materialization occurs inside a sandbox with path traversal, symlink/hardlink, and size checks.
* **Malicious DA provider** – can withhold chunks, serve garbage data or replay old archives, but cannot break Merkle binding or the relay signature scheme.

The verifier runs on an honest machine: we trust its runtime, hashing, and the coincurve implementation when available.

## 2. Binding points and what now enforces them

* **Capsule identity** – three hashes exist:
  * `payload_hash = H("CAPSULE_PAYLOAD_V2" || Enc(payload\ {authorship, da_challenge}))`
  * `header_commit_hash = H("CAPSULE_HEADER_COMMIT_V2" || Enc(header\ {da_ref.challenge_hash}))`
  * `capsule_hash = H("CAPSULE_ID_V2" || header_commit_hash || payload_hash)`
  The header records `payload_hash`; the outer capsule carries both `payload_hash` and `capsule_hash`. Any mutation to the payload or header that isn’t followed by a recompute is caught as `E011/E013`.
* **Header contents** – `trace_spec_hash`, `statement_hash`, row-commitment metadata, proof-system hashes, `policy_ref`, `da_ref`, `artifact_manifest_hash`, and the `(events_log_hash, events_log_len)` pair. The DA challenge is intentionally excluded from `header_commit_hash` so commit-then-challenge remains acyclic.
* **Proof parameters + instance binding** – `air_params_hash`, `fri_params_hash`, `program_hash`, and `vk_hash` live in the header. The backend MUST absorb the instance tuple defined normatively in `docs/spec/04_instance_binding.md` (V2 for POLICY_ENFORCED/FULL). The verifier recomputes all values from the proof object (`GeomAIRParams`, `FRIConfig`, program descriptor, row commitment) before calling the backend and fails with `E301_PROOF_SYSTEM_MISMATCH` if any hash differs.
* **Statement binding** – the proof object carries a full `StatementV1`. The verifier recomputes its hash and ensures it matches the header; `zk_verify_geom` also absorbs the caller-provided `statement_hash` into the Fiat–Shamir transcript so a receipt tied to one statement cannot be replayed under another without breaking soundness.
* **Row commitments & manifests** – the Merkle root referenced in the statement is re-derived from the chunk roots and compared to the proof’s `row_commitment`. Chunk manifests are canonicalised, hashed, and every handle is confined to the extracted archive root; the verifier recomputes each chunk’s size and SHA-256 before any DA I/O happens.
* **Policy binding / assurance** – `policy_ref` commits to `(policy_id, version, hash, track_id)`. Policy enforcement now requires a `manifest_signature.json` beside the manifests; it must be a secp256k1 recoverable signature over the manifest anchor hash, signed by a trusted manifest authority listed in `config/manifest_signers.json` (or CLI overrides). The verifier re-derives the registry hash, requires the signer id to be allowed, and returns `E106-E109` otherwise. Merkle inclusion proofs against a trusted policy registry root upgrade `policy_assurance` to `ATTESTED`; without them enforcement is capped at `POLICY_SELF_REPORTED`.
* **Event log** – verification treats the log as a tamper-evident transcript. The predicate is purely structural (hash chain, monotone sequence numbers, required event types once each). It enforces non-equivocation/integrity and emits `E201_EVENT_LOG_MISMATCH` warnings but does **not** block FULL verification because the emitter is unauthenticated.
* **DA audit** – FULL verification is a verifier-chosen predicate; it requires a relay-issued challenge signed over `canonical_encode(challenge\{signature})`, hashed via SHA-256 before secp256k1 signing. The verifier only accepts challenges whose `relay_pubkey_id` appears in a pinned registry hash `CAPSULE_TRUSTED_RELAYS_ROOT` (computed as the SHA-256 of the sorted id→pubkey map). A default registry lives at `config/trusted_relays.json`; overriding it requires supplying both the new key map and the matching root. Missing/expired/unsigned challenges or registry hash mismatches raise `E073` instead of silently downgrading.
* **Authorship/ACL** – any capsule claiming more than PROOF_ONLY must include a secp256k1 recoverable signature over `capsule_hash`. ACLs bind signer pubkeys to policy IDs; if signatures are missing or not in the ACL, status drops to PROOF_ONLY.

## 3. Verification predicate

Let `level ∈ {PROOF_ONLY, POLICY_ENFORCED, FULLY_VERIFIED}` be selected by the *verifier/operator* (the capsule merely advertises an intended profile). Define the booleans:

```
proof_ok  := header_ok ∧ VerifyProof(instance_hash, proof) = 1
policy_ok := CheckPolicy(policy_ref, policy_doc, manifests, statement)
acl_ok    := (no ACL) ∨ signer ∈ ACL(policy_id)
da_ok     := DA audit predicate (challenge verified + sampling succeeds)
```

Then:

```
if ¬proof_ok:              status = REJECTED
elif level = PROOF_ONLY:   status = PROOF_ONLY
elif level = POLICY_ENFORCED:
    status = POLICY_ENFORCED      if policy_ok ∧ acl_ok ∧ policy_assurance=ATTESTED
    status = POLICY_SELF_REPORTED if policy_ok ∧ acl_ok but assurance is self-reported
elif level = FULLY_VERIFIED:
    require policy_ok ∧ acl_ok ∧ da_ok (with attested policy)
    status = FULLY_VERIFIED
else:                       status = E302_VERIFICATION_PROFILE_UNSATISFIED
```

Any missing prerequisite short-circuits with the corresponding `E0xx` code (e.g. `E071/E073` for DA). Event log issues are reported as warnings instead of fatal errors because they are not authenticated data sources.

### DA Sampling Bound (FULL)

Let δ be the (adversarial) fraction of rows unavailable at audit time, k the number of sampled chunks, p the field modulus, n_max the maximum row/trace length relevant to the soundness of the sketch/consistency checks, m the number of independent algebraic checks (e.g. random linear constraints), `Adv_H_crh` the adversary’s advantage against SHA‑256 collision resistance, `Adv_backend` the advantage against the backend’s proof soundness with instance absorption, and `Adv_sig` the advantage against the signature scheme (secp256k1 UF‑CMA). Under the commit→unpredictable‑challenge→open ordering, the acceptance probability for a cheating prover is bounded by:

```
Pr[cheat] ≤ (1 − δ)^k + Adv_H_crh + ((p − 1)/n_max − 1)^m + Adv_backend + Adv_sig
```

This isolates the three levers under operator control: (i) availability via k, (ii) algebraic collision via m (and field size p versus problem scale n_max), and (iii) the trust roots (hash, backend, signatures).

## 4. Attack surfaces & mitigations

| Surface | Mitigation |
| --- | --- |
| Payload/header drift | Structural payload view with explicit key list plus `capsule_hash = H(header_commit || payload_hash)`. Any mutation without recomputing both hashes trips `E011/E013`.
| Proof parameter swaps | Header pins `air/fri/program/vk` hashes; verifier recomputes and raises `E301` on mismatch.
| Fake DA randomness | Relay issues commit-then-challenge receipts and signs them. FULL verification requires a trusted relay key; insecure local challenges are rejected.
| Manifest spoofing | `artifact_manifest_hash` recomputed; lack of attestation is surfaced via `policy_assurance = SELF_REPORTED` and downgrades status.
| Event log tampering | Hash chain recomputed and file length checked; failures emit `E201` warnings but do not affect the verification status because the emitter is untrusted.
| Capsulepack traversal / tarbomb | `_extract_capsulepack` validates members (no abs paths, dot-dot, symlinks) and enforces a size cap and hash check via `pack_meta` before verification runs.
| Legacy DA fallback | FULL verification now refuses legacy/insecure challenges even if the capsule declares `verification_profile=FULL`.

## 5. Security claims (informal)

1. **Capsule binding** – Producing two different payload/header pairs that verify under the same `capsule_hash` implies either a SHA-256 collision or a breach of the proof/commitment primitives (identical to the standard commitment “two openings” experiment).
2. **Statement binding** – Any adversary that makes the verifier accept while the header’s `statement_hash` differs from the proof’s statement must break the STC/FRI soundness guarantees or produce a collision in the canonical Statement hash.
3. **DA audit soundness** – Given an honest relay key and unpredictable challenges, the probability that an unavailable fraction ≥δ of the committed chunks evades detection is at most `(1-δ)^k` plus negligible collision probability in the vector-commitment/Merkle layers (the exact bound in HSSA’s DA section).
4. **Event log non-equivocation** – Under collision resistance, no two distinct logs can yield the same `(event_chain_head, events_log_hash, events_log_len)` triple, so tampering always surfaces as `E201_EVENT_LOG_MISMATCH` warnings even though the verifier no longer blocks FULL status on unauthenticated logs.
5. **Policy assurance honesty** – Without an inclusion proof under a trusted registry root, the verifier never claims `POLICY_ENFORCED`/`FULLY_VERIFIED`; all output explicitly labels such runs as `POLICY_SELF_REPORTED` and includes machine-readable warnings.

Future work (attested manifests, alternate backend adapters) fits into this skeleton by adding new hash commitments or elevating `policy_assurance` to `ATTESTED` once trusted attestations are plumbed through.

## 6. Canonical security experiments

Just as in Katz–Lindell and HSSA, each security notion is phrased as an experiment against an adversary with advantage measured relative to standard primitives. The reduction targets for Capsules + STC are:

1. **CapsuleBinding** – Challenger samples a payload/header pair, computes `capsule_hash`, and gives the adversary oracle access to the canonical hash function. The adversary wins if it outputs `(header, payload), (header', payload')` with `(header, payload) ≠ (header', payload')` that both pass `_verify_capsule_core` under the same `capsule_hash`. This reduces to either a collision in SHA-256 (payload hash or header commit hash) or a violation of the proof/statement binding conditions (i.e. STC/Fri soundness). Advantage is bounded by `Adv_bind ≤ Adv_crh + Adv_proof`.
2. **CapsuleAuthenticity** – Challenger exposes a signing oracle for a registered policy id/ACL. The adversary may query capsules to be signed; it wins if it outputs a capsule that `_verify_capsule_core` classifies as `POLICY_ENFORCED`/`FULLY_VERIFIED` without having requested a signature on that capsule hash (UF-CMA). Advantage is bounded by the secp256k1 UF-CMA advantage plus any ACL/policy registry assumptions.
3. **DASoundness** – Challenger commits to a capsule (fixing `payload_hash`, `header_commit_hash`, row roots) and then samples a relay challenge uniformly at random, signs it under the trusted relay key, and reveals it. The adversary controls the DA provider and wins if it convinces the verifier while at least a δ fraction of chunk roots are unavailable. The failure probability is bounded by `(1-δ)^k + Adv_crh + Adv_sketch`, matching the HSSA DA experiment with the additional assumption that the relay key is bound to `CAPSULE_TRUSTED_RELAYS_ROOT`.
4. **EventNonEquivocation** – Challenger exposes the canonical event hash/length predicate; the adversary wins if it produces two different event logs that both verify under the same `(events_log_hash, events_log_len, event_chain_head)` triple. Advantage reduces to finding a collision in SHA-256 over the chained hash or forging `events_log_len` (i.e. a length/tamper check bypass).

Stating the security goal as a combination of these experiments makes it clear that any break in Capsules + STC must either (a) collide SHA-256 / Merkle hashes, (b) forge a signature, (c) break the backend proof system, or (d) subvert the DA hash chain – all standard cryptographic assumptions.



---

# docs/security/theorems.md

<a name="docs-security-theoremsmd"></a>

# Security Theorems (non‑normative statements)

Capsule Binding (informal)
- Under SHA‑256 collision resistance and backend instance‑binding soundness, any adversary that produces two distinct (header, payload) that verify under the same `capsule_hash` breaks CRH or backend soundness.

Instance Soundness (informal)
- If `_verify_capsule_core` accepts at PROOF_ONLY for a capsule whose backend absorbed the instance tuple (V1/V2), then the statement proven by the backend is true for the bound row commitment.

DA Sampling Bound (FULL)
- Let δ be the unavailable fraction; k be the sample count; p the field modulus; n_max the problem scale; m the number of algebraic checks; Adv terms as in the security model. Under commit→unpredictable‑challenge→open ordering,

```
Pr[cheat] ≤ (1 − δ)^k + Adv_H_crh + ((p − 1)/n_max − 1)^m + Adv_backend + Adv_sig
```

Authorship/Registry Authenticity
- Accept at ≥ POLICY_ENFORCED implies a valid secp256k1 signature over `capsule_hash` by a pubkey authorized in the ACL (if provided) and a manifest anchor signature by a signer in the pinned registry root; otherwise reduces to UF‑CMA or registry‑auth failure.




---

# docs/SECURITY_ANALYSIS.md

<a name="docs-SECURITY_ANALYSISmd"></a>

# BEF Security Analysis & Cryptographic Model

## 1. Executive Summary
The Benchmark Execution Framework (BEF) implements a **Succinct Verifiable Computing** system based on the **Streaming Trace Commitment (STC)** protocol and **FRI** (Fast Reed-Solomon Interactive Oracle Proof).

This document formally defines the security model, analyzes the impact of the Rust/GPU optimization architecture, and details the asymptotic complexity of the system.

**Note on Zero-Knowledge:** While the system uses STARK-like mechanics (FRI, AIR), it currently implements a **succinct integrity proof**, not a zero-knowledge proof. The protocol does not yet implement witness blinding or masking polynomials required for zero-knowledge.

## 2. Security Model

### 2.1 The Adversary
*   **Goal:** The adversary attempts to convince the Verifier to accept a proof $\pi$ for a false statement $S$ (e.g., an invalid execution trace).
*   **Capabilities:**
    *   Computationally bounded (cannot find collisions in SHA-256).
    *   Full control over the Prover's hardware (CPU, GPU, FPGA).
    *   Can choose the trace data and the commitments freely.
*   **Success Condition:** The Verifier outputs `ACCEPT`.

### 2.2 Verifier Independence (The Golden Rule)
The core security guarantee of the BEF system is **Verifier Independence**.
*   The Verifier algorithm is deterministic and purely mathematical.
*   It checks probabilistic relations between the **Commitment** $C$ and the **Openings** $O$ provided in the proof.
*   **Hardware Agnosticism:** The verifier does not know or care if the commitment was computed by a Python script, a Rust binary, or a CUDA kernel.
*   **Soundness:** Soundness error is a function of the configured FRI parameters (blowup factor, number of queries $Q$, field size $F$). For our standard profile ($Q=32$, Blowup=4, Goldilocks Field), the soundness error $\epsilon$ is cryptographically negligible against classical adversaries.

## 3. Architecture & TCB Analysis

We have transitioned from a pure Python prover to a hybrid **Rust + CUDA** architecture.

### 3.1 Trusted Computing Base (TCB)
The set of components that *must* be correct for the system to maintain its security properties.

| Component | Status | Impact on Soundness (Forgery) | Impact on Liveness (Failure) |
| :--- | :--- | :--- | :--- |
| **Verifier (Python)** | **Trusted** | **Critical** | Critical |
| **Prover (Python)** | Untrusted | None | High |
| **Prover (Rust)** | Untrusted | None | High |
| **Prover (CUDA)** | Untrusted | None | Medium |
| **NVIDIA Driver** | Untrusted | None | Medium |

### 3.2 Impact of Optimizations
*   **Soundness:** Moving logic to Rust/CUDA **does not** weaken soundness. If the GPU computes $2+2=5$, the STC/FRI checks will fail, and the verifier will reject.
*   **Liveness:** The complexity of the build chain (Rust + NVCC) increases the risk of *proof generation failure* (e.g., driver mismatch, compilation error).
*   **Side Channels:** Offloading witness data to GPU VRAM introduces potential timing/power side-channels in multi-tenant environments.
    *   *Mitigation:* This is acceptable for Data Availability (DA) layers where data is public. For private compute, isolated hardware is recommended.

## 4. Asymptotic Complexity

### 4.1 Prover Complexity
Let $N$ be the trace length (number of steps).

*   **Trace Generation:** $O(N)$
*   **Row Commitment (STC):**
    *   **Sketching:** $O(N \cdot m)$ scalar multiplications (where $m$ is number of challenges).
    *   **Merkle Tree:** $O(N)$ hashes.
    *   **Optimized:** Perfectly linear $O(N)$ on CPU/GPU.
*   **FRI Proving:**
    *   **Folding:** $O(N \log N)$ total field operations across all layers (FFT-like structure).
    *   **Merkle Trees:** $O(N)$ hashes (sum of geometric series).
*   **Total Prover Time:** $O(N \log N)$.

**Benchmarks:** The system scales near-linearly in the tested range ($2^{12}$ to $2^{18}$), dominated by the $O(N)$ commitment phase overhead.

### 4.2 Verifier Complexity
The verifier is **succinct**.
*   **Time:** $O(\log^2 N)$. The verifier performs $Q$ queries, each requiring $O(\log N)$ Merkle checks.
*   **Space:** $O(\log N)$.
*   **Benchmarks:** Verification takes ~3-4ms regardless of trace size.

## 5. Adversary Experiments

We validated the security model against active tampering using reason codes defined in `bef_zk/verifier_errors.py`:

1.  **Bit-Flip Attack:**
    *   *Action:* Modified 1 bit in the capsule payload.
    *   *Result:* `E011_CAPSULE_HASH_MISMATCH` (Detected).
2.  **Signature Forgery:**
    *   *Action:* Provided valid payload but invalid signature.
    *   *Result:* `E107_MANIFEST_SIGNATURE_INVALID` (Detected).
3.  **Statement Mismatch:**
    *   *Action:* Provided valid proof for a different statement.
    *   *Result:* `E053_PROOF_STATEMENT_MISMATCH` (Detected).
4.  **Path Traversal Attack:**
    *   *Action:* Provided chunk handle with `../../../etc/passwd`.
    *   *Result:* `ValueError: Path traversal not allowed` (Blocked by `safe_join`).

## 6. Formal Security Bounds

### 6.1 Cheating Probability

The probability that a malicious prover convinces the verifier to accept while at least δ fraction of chunks are unavailable is bounded by:

$$\Pr[\text{cheat}] \leq (1-\delta)^k + \epsilon_{CRH} + \epsilon_{STC} + \epsilon_{FRI} + \epsilon_{sig}$$

Where:
*   $(1-\delta)^k$ — DA sampling miss probability ($k$ = sampled chunks)
*   $\epsilon_{CRH}$ — SHA-256 collision resistance advantage (negligible, ~$2^{-128}$)
*   $\epsilon_{STC}$ — STC algebraic soundness: $\frac{d}{p}$ per check (Schwartz-Zippel), with $m$ independent checks: $\left(\frac{d}{p}\right)^m$ where $d$ is polynomial degree and $p = 2^{61}-1$
*   $\epsilon_{FRI}$ — FRI soundness error: $\left(\frac{1}{\rho}\right)^Q$ where $\rho$ is blowup factor (4) and $Q$ is query count (32)
*   $\epsilon_{sig}$ — secp256k1 UF-CMA advantage (negligible)

### 6.2 Standard Parameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| Field modulus $p$ | $2^{61}-1$ | $\epsilon_{STC} \approx 2^{-122}$ for $m=2$, $d \leq 2^{18}$ |
| FRI queries $Q$ | 32 | $\epsilon_{FRI} \approx 2^{-64}$ |
| FRI blowup $\rho$ | 4 | |
| DA samples $k$ | configurable | $(1-\delta)^k \leq 2^{-40}$ for $k=80$, $\delta=0.5$ |

### 6.3 Caveat on Optimization Claims

"Optimizing prover computation does not change the *cryptographic* security reduction, assuming verifier logic and transcript derivation remain unchanged and correct."

Prover bugs (GPU/Rust) may cause:
*   Implementation bugs that produce invalid proofs (detected by verifier)
*   Mismatched transcripts or domain separation (detected by verifier)
*   Non-soundness bugs unrelated to cryptography (e.g., path traversal — now mitigated)

## 7. Implementation Security Mitigations

### 7.1 Path Traversal Protection

All chunk archive access is confined to the archive root via `safe_join()`:

```python
# bef_zk/stc/archive.py
def safe_join(root: Path, rel: str) -> Path:
    # Reject absolute paths, .., null bytes
    # Verify resolved path is under root
    candidate.relative_to(root_resolved)  # raises ValueError if escape
```

Attack vectors blocked:
*   `../../../etc/passwd` — Rejected with "Path traversal not allowed"
*   `/etc/passwd` — Rejected with "Absolute path not allowed"
*   `chunk.json\x00/etc/passwd` — Rejected with "Null bytes not allowed"

### 7.2 Capsulepack Extraction

Capsule verification extracts archives with `_confine_path()` checks:
*   Absolute paths rejected
*   Parent directory traversal (`..`) rejected
*   Symlinks rejected
*   Size limits enforced

## 8. Conclusion

The BEF system maintains **provable security** while achieving **2x-3x performance gains** through Rust/GPU acceleration. The integrity of the verification process ensures that the optimized prover cannot degrade the system's trust model.

All security bounds are explicitly stated with reduction targets. Implementation-level protections (path confinement, size limits) complement the cryptographic guarantees.



---

# docs/backends/geom.md

<a name="docs-backends-geommd"></a>

# Geom Backend

Purpose: didactic STARK backend for demonstrating instance binding and capsule composition. Not optimized for performance.

What it proves: a small AIR over a 2×2 matrix + counter; public outputs include `final_cnt` for policy checks.

Bindings enforced: row params (root, chunk_len, arity) and `instance_hash` absorption; verifier re‑derives program/vk/AIR/FRI hashes.

Limitations: Python implementation; JSON I/O; non‑native hash for STC; small parameters; redundant commitments (STC + FRI).




---

# docs/backends/risc0.md

<a name="docs-backends-risc0md"></a>

# Risc0 Backend

Purpose: demonstrate backend generality by binding a Risc0 receipt to the capsule instance tuple and verifying it under the same verifier profiles.

## Binding

- `image_id` and `journal_digest` MUST bind to the capsule’s `statement_hash` via the backend’s claim.
- The capsule’s `proof_system` MUST include `scheme_id="risc0_receipt_v1"`, and `vk`/program identities as appropriate.
- The verifier recomputes `instance_hash` and ensures the receipt claim/journal binding occurs before any Fiat–Shamir challenges.

## Adapter contract

- `simulate_trace`: loads journal/public outputs; supplies a minimal `TraceSpecV1` describing the image.
- `commit_to_trace`: uses receipt claim (or a synthetic stub) to populate `row_root` and artifact pointers.
- `generate_proof`: returns the receipt as JSON/bin; the adapter must enforce binding to `statement_hash`.
- `verify`: calls the Risc0 verifier and rejects if claim/journal do not match the bound statement.

## Limitations

- No row-level archive; the commitment is a receipt claim (merkle root), not a full STC archive.
- DA sampling is N/A for Risc0-only proofs; use POLICY_ENFORCED or legacy DA profiles until a challenger-backed DA is available.




---

# docs/trace_adapter_contract.md

<a name="docs-trace_adapter_contractmd"></a>

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



---

# docs/trace_statement_spec.md

<a name="docs-trace_statement_specmd"></a>

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



---

# docs/stc_backend_architecture.md

<a name="docs-stc_backend_architecturemd"></a>

# STC as Vector Commitment and Polynomial Commitment Backend

We treat HSSA/STC as a two-layer commitment system:

1. **Vector commitment (VC):** streaming commitment to a trace/evaluation vector.
2. **Polynomial commitment (PC):** plug the VC into an IOP+FRI stack, exactly as
   Merkle trees back classic STARK PCs.

## VC Interface

- `Commit_vec(pp, w)` streams the evaluation vector `w ∈ F^n` into STC buckets,
  producing the public commitment `(n, root, r⃗, s⃗)` plus bucket metadata.
- `Open_vec(pp, C, i)` returns `(w_i, proof_i)` where `proof_i` contains Merkle
  paths inside the bucket tree and from bucket root to the global root.
- `Verify_vec(pp, C, i, w_i, proof_i)` recomputes leaves and Merkle paths to
  ensure inclusion. Sketches stay global; they detect bucket tampering or DA
  faults, not per-index openings.

The only twist vs. vanilla Merkle VC is that STC buckets are uneven/streaming
and every bucket records per-challenge sketch contributions. All hashing remains
SHA-256 (or any CRH).

## PC Interface (FRI-backed)

To commit to a low-degree polynomial `p(X)`:

1. Evaluate `p` over the FRI domain `H` to obtain `w`.
2. Run `Commit_vec` to get the base commitment `C = C^(0)`.
3. During the FRI prover loop, each round’s codeword is also committed with STC,
   giving `C^(1) … C^(r)`.

To open at a point `z`:

1. Run the standard FRI prover and record all query indices.
2. For every queried index in every round, invoke `Open_vec` on the corresponding
   STC commitment to supply `(w_i, proof_i)`.
3. Return `y = p(z)` together with the full FRI transcript plus all STC openings.

The verifier runs `Verify_vec` for every opened index, then executes the usual
FRI algebraic checks to link the base codeword to the final small-degree
polynomial and to enforce `p(z) = y`.

## What’s new vs. stock Merkle+FRI

- **Streaming/GPU:** `Commit_vec` is identical to the CUDA accumulator, so the
  commitment phase consumes huge traces on GPU without ever materializing the
  entire vector in CPU memory.
- **Uneven buckets + sketches:** per-bucket sketches and SHA-256 roots give cheap
  integrity/DA checks and mesh with the DA sampling protocol in
  `docs/hssa_da_protocol.md`.
- **Hash-only/PQ:** No pairings or trusted setup; the PC inherits STARK-style
  post-quantum security.

The algebraic security story is unchanged from Merkle-backed FRI PCs. The
systems story is what changes: STC provides an optimized VC backend tailored to
streaming traces, GPU acceleration, and DA requirements.



---

# docs/stc_da_profiles.md

<a name="docs-stc_da_profilesmd"></a>

# STC Data Availability Profiles

A DA profile is a small policy word bound to every STC sketch via the
`da_profile` field. It specifies how the raw trace is stored and what sampling
parameters verifiers must satisfy.

```
{
  "version": 1,
  "mode": "LOCAL_FILE | L1_BLOB | EXTERNAL_DA | COMMITTEE",
  "sampling": {
    "delta": 0.1,      # targeted withholding fraction
    "epsilon": 1e-6,   # failure probability budget
    "k_min": 96        # minimum random chunk samples (derived)
  },
  "provider": {
    "path": "code/traces/vm_demo.json"
  }
}
```

## Example profiles

### Minimal-Local

- mode: `LOCAL_FILE`
- delta = 0.2, epsilon = 1e-3 → `k_min ≈ 33`
- provider: `{ "path": "code/traces/vm_demo.json" }`
- Use for demos/tests; run `scripts/stc_da_sample.py` to spot-check chunks.

### Rollup-L1Blob (hypothetical)

- mode: `L1_BLOB`
- delta = 0.1, epsilon = 1e-9 → `k_min ≈ 207`
- provider: `{ "chain_id": 1, "blob_tx": "0xdead..." }`
- Intended for rollup posting data to Ethereum blobs; samplers re-fetch blob and
  verify sampled buckets.

### External-DA (Celestia/Eigen)

- mode: `EXTERNAL_DA`
- delta = 0.05, epsilon = 1e-12 → `k_min ≈ 552`
- provider: `{ "namespace": "0xabc...", "height": 123456 }`
- Samplers use the DA backend API to fetch random chunks.

Profiles are stored in `da_profiles/*.json` for reference. Clients may always
run more samples than `k_min`; the profile just encodes the minimum guarantee.



---

# docs/hssa_da_protocol.md

<a name="docs-hssa_da_protocolmd"></a>

Data Availability via Streaming Trace Commitments (STC)

1) Generic DA via STC (abstract)
- Goal: Given a large blob B, a sequencer convinces light clients that (1) data is available and (2) everyone agrees on a single committed version, using a Streaming Trace Commitment (STC) layer.
- STC algorithms: (Init, Update, Commit, Open, VerifyOpen). GlobalCheck is optional for fast probabilistic checks.

- Actors:
  - Sequencer (S): forms blobs and posts commitments.
  - Full nodes: store full blobs, serve openings.
  - Light clients: see on-chain state; run cheap sampling.

- Data model:
  - Encode blob B to field vector v ∈ F^n and partition into chunks chunk_t ∈ F^L for t=0..T−1.

- Commit phase (per blob):
  1) Encoding: B → v ∈ F^n; partition into chunks {chunk_t}.
  2) Streaming commitment: st0 ← Init(pp). For t in order: st_{t+1} ← Update(pp, st_t, chunk_t). Final (C, meta) ← Commit(pp, st_T).
  3) On-chain posting: post C (and minimal metadata), blob ID / index, optional short hash.

- Availability sampling (light client):
  1) Sample k random chunk indices {t_1..t_k} using a public seed (e.g., block hash).
  2) Request openings: for each sampled t, fetch the chunk values and openings π (per-index Merkle paths + chunk inclusion to the committed structure).
  3) Verify: for each i in sampled chunks, VerifyOpen(pp, C, i, v_i, π_i) ?= 1. If any fails → unavailable.
  4) Decision: if all pass, accept availability with soundness error ≤ (1−δ)^k for target corruption fraction δ.

2) HSSA-based DA Instantiation (concrete)
- State after all chunks: st_T = (n, root, (r_j)_{j=1..m}, (s_j)_{j=1..m}, (pow_j)_{j=1..m}).
- Commitment C = (n, root, (r_j), (s_j)).
- Metadata (off-chain or partially on-chain): per-chunk (offset, length, chunkRoot_t, optional per-chunk sketch contributions); global (T, field/param IDs).

- Merkle/hash structure:
  - Each chunk has a Merkle root.
  - Global root is a hash chain (or top Merkle tree) over chunk roots: root_{t+1} = H(root_t || n_t || chunkRoot_t).

- Open / VerifyOpen (index i in chunk t):
  - Open returns (v_i), Merkle path leaf→chunkRoot_t, and chain from chunkRoot_t into root; includes offset/length metadata.
  - VerifyOpen recomputes chunkRoot_t and root, then checks equality with C.root and consistency with n and index i.

- GlobalCheck via sketches (optional, fast):
  - s_j = Σ_{i=0}^{n−1} v_i r_j^i.
  - Reconstruct expected global sketches ŝ_j from per-chunk contributions (or recompute) and offsets.
  - Accept iff ŝ_j = s_j for all j.

3) Security sketch (DA + sketches)
- Sampling error: if δ fraction of chunks are bad and k random chunks are sampled, Pr[miss] ≤ (1−δ)^k.
- Sketch error: for any v≠v′ of length ≤ n_max, with m independent challenges over F_p,
  Pr[(s_j(v)=s_j(v′) for all j)] ≤ ((n_max−1)/p)^m (Schwartz–Zippel).
- Combined (ignoring hash collisions):
  Pr[cheat] ≤ (1−δ)^k + ((n_max−1)/p)^m.
- With a b‑bit CRH for the root chain, add Adv_crh(H) ≈ 2^{−b}:
  Pr[cheat] ≤ Adv_crh(H) + (1−δ)^k + ((n_max−1)/p)^m.




---

# docs/roadmap.md

<a name="docs-roadmapmd"></a>

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



---

# docs/README.md

<a name="docs-READMEmd"></a>

# Documentation Index

## Primary references
- [Trace & Statement Spec](trace_statement_spec.md)
- [STC Backend Architecture](stc_backend_architecture.md)
- [HSSA DA Protocol](hssa_da_protocol.md)
- [STC DA Profiles](stc_da_profiles.md)
- [Roadmap](roadmap.md)
- [Benchmark Policy Schema](benchmark_policy_schema.json)

## Archive & deep dives
Older research notes, design spikes, and supporting slide decks were moved to [`docs/archive/`](archive/). They remain available for historical context but are not required for day-to-day development.

## Notes & supplemental material
- Planning notes live under [`docs/notes/`](notes/).
- Long-form PDFs are collected in [`docs/papers/`](papers/).



---

# docs/notes/dec18_plan.md

<a name="docs-notes-dec18_planmd"></a>

### Executive Summary of the Gap

The current system is a **strong proof-of-concept**, accurately described in the PDF as a "capsule interface + geom demo driver." It successfully demonstrates the *structure* of a verifiable computation pipeline. However, the pitch presents it as a mature, general-purpose, and production-ready framework.

The primary gap is between a **demo-shaped system** and a **general-purpose, adoptable platform**. We have the blueprints for a powerful engine, but we've only built a single, hardwired model car around it.

---

### Detailed Gap Analysis: What We Don't Have Yet

Here is a point-by-point breakdown of the missing pieces, referencing the claims in the pitch and the evidence from the `pipeline.pdf`.

#### 1. The "Complete, End-to-End Pipeline" is a Demo, Not a Platform

*   **Claim:** "A complete, end-to-end pipeline for creating, verifying, and managing 'proof capsules.'"
*   **Reality (`pipeline.pdf`):** "It's a capsule format + (demo) pipeline harness for one concrete VM/AIR ('geom')." and "Only one AIR (GEOM_AIR_V1) is concretely wired."
*   **The Gap:** The pipeline is not "complete" in a general sense. It's a specific, hardcoded implementation for a single, simple VM ("geom"). It cannot handle other types of computation or different zkVMs without significant engineering effort.
*   **What Needs to Get Done:**
    *   **Trace Adapter API:** This is the single most critical missing piece, as highlighted in the PDF. We need to design and implement a standard interface (`TraceAdapter`) that allows new VMs and computations to be plugged into the framework. This involves defining how the pipeline should:
        1.  Define a trace schema.
        2.  Extract public inputs.
        3.  Generate a `TraceSpecV1` and `StatementV1`.
        4.  Invoke the correct prover/verifier for that specific backend.
    *   **Refactor the Pipeline Runner:** The `run_geom_pipeline.py` script needs to be refactored to use this `TraceAdapter` interface, removing the hardcoded "geom" logic.

#### 2. The "Auditable Data Availability" is a Local Simulation

*   **Claim:** "We've integrated a data availability (DA) sampling protocol directly into our pipeline."
*   **Reality (`pipeline.pdf`):** "DA is implemented as sampling + local archive provider, not a full general DA layer story."
*   **The Gap:** The current DA mechanism only works by reading from a local filesystem. It proves *retrievability* from a known, local archive, but it does not solve the problem of convincing a light client that the data is available on a *decentralized network*. The pitch implies a solution for the latter.
*   **What Needs to Get Done:**
    *   **DA Provider Interface:** Define a clear interface for DA providers.
    *   **Implement Networked DA Providers:** To make this useful for a real rollup, we would need to implement providers for actual DA layers like Celestia, EigenDA, or Ethereum blobs. This involves handling network requests, data serialization, and error handling.
    *   **Honest Positioning:** As the PDF suggests, we need to be precise in our language. We should call it "**audited retrievability**" for now, not "data availability."

#### 3. "Policy-Bound Execution" Lacks Semantic Depth

*   **Claim:** "Every capsule is bound to a specific policy, which can define everything from the underlying hardware to the specific version of the software."
*   **Reality (`pipeline.pdf`):** "Right now policy is mostly 'bound,' not 'interpreted,' except via ACL and DA parameters."
*   **The Gap:** We are hashing the policy and including it in the `StatementV1`, which is great for binding. However, the *verifier does not actually enforce the rules within the policy*. For example, if the policy says "no JIT," the verifier doesn't check if a JIT was used. The enforcement is currently based on trust or out-of-band mechanisms.
*   **What Needs to Get Done:**
    *   **Define Enforceable Policy Semantics:** We need to decide which policy rules can and should be cryptographically enforced. The PDF gives an example: requiring a standardized container/harness for baseline benchmarks.
    *   **Implement Policy Enforcement in the Verifier:** The `verify_capsule.py` script needs to be extended to read and interpret the policy file and then check the corresponding anchors and manifests to ensure the rules were followed. For example, it would need to check the `docker_image_digest` anchor if the policy requires a specific container.

#### 4. The "Extensible and Adaptable" Claim is an Aspiration, Not a Feature

*   **Claim:** "Our capsule format is designed to be extensible... can be adapted to any zkVM."
*   **Reality (`pipeline.pdf`):** "The 'adoption surface' for other traces is implied, not implemented in this snapshot."
*   **The Gap:** This is directly related to the missing `TraceAdapter` API. Without a clear, documented interface for integration, adapting the system to a new zkVM would require a deep understanding of the entire codebase and significant custom engineering. It's not the clean, plug-and-play experience the pitch suggests.
*   **What Needs to Get Done:**
    *   The **Trace Adapter API** is the answer here as well. This is the "adoption surface" that needs to be built.

#### 5. The "Mature, Well-Documented Contract" is a High-Quality Prototype

*   **Claim:** "We have a mature, well-documented 'proof object / verification contract' that is ready for adoption."
*   **Reality (`pipeline.pdf`):** The PDF rates the "'Proof object / verification contract' maturity" as **high**, but the "'General-purpose platform / others can adopt without you' maturity" as **medium-low**.
*   **The Gap:** The *contract itself*—the structure of the capsule and the verification logic—is indeed strong. However, its maturity is limited by its tight coupling to the "geom" demo. A contract is only truly mature when it has been proven to work across multiple, diverse implementations.
*   **What Needs to Get Done:**
    *   **Implement a Second Backend:** The best way to prove the maturity and generality of the contract is to implement a `TraceAdapter` for a second, real-world zkVM (e.g., RISC Zero, SP1). This would force us to confront the real-world challenges of integration and would battle-harden the capsule format and verification logic.
    *   **Public Documentation and Tooling:** Create clear, public documentation for the `TraceAdapter` API and provide tooling (like a `capsule-bench` CLI mentioned in the PDF) to make integration as easy as possible.

### Summary of Work to Be Done

To bridge the gap between the pitch and the current reality, we need to move from a **demo** to a **platform**. The work can be summarized in three major initiatives:

1.  **Build the "Adoption Surface" (The `TraceAdapter` API):** This is the highest priority. It's the key that unlocks the "general-purpose" and "extensible" claims.
2.  **Implement Real-World Integrations:**
    *   Integrate with at least one other zkVM to prove the `TraceAdapter` API is viable.
    *   Integrate with at least one real DA network to make the DA story compelling for rollups.
3.  **Add "Teeth" to the Policy Engine:** Move from simply "binding" policies to actively "interpreting" and enforcing them in the verifier.


---

# docs/archive/BACKEND_READINESS_GAPS.md

<a name="docs-archive-BACKEND_READINESS_GAPSmd"></a>

Good, this is concrete enough that we can actually audit it instead of just vibing.

You basically have:

* A real STC/HSSA implementation (GPU + Rust verifier),
* A formal model and security analysis in the HSSA–STC notes, including DA + AoK,
* Benchmarks vs a KZG baseline,
* And some IVC-facing docs.

So the question is: what’s missing between this and “this is a backend I can show Bünz / StarkWare / an L2 team”?

I’ll break gaps into 5 buckets: crypto theory, IVC/zk backend, systems + benchmarks, DA/protocol, and story/docs.

---

## 1. Crypto / theory gaps

Stuff your own formal notes already hint at but the repo doesn’t fully “operationalize” yet:

1. Parameters → concrete guarantees

   You state soundness like:
   Pr[sketch collision] ≤ ((n_max−1)/(p−1))^m
   and DA cheating prob:
   (1−δ)^k + ((n_max−1)/(p−1))^m + hash term.

   But you don’t yet have, inside the repo:

   * A “parameter table” that says:
     * For n_max = 2^k, p = 2^61−1, m = 4/6/8, here is actual numerical soundness.
     * For DA: given δ, k, m, N, here are example cheating probabilities.
   * A clear recommendation: “For rollup-like workloads with up to X steps, we recommend m = … for total failure prob ≤ 2⁻¹²⁸.”

   Right now it’s all “formula exists” but not “this is the profile you should actually use”.

2. AoK integration with code

   The notes sketch an AoK experiment and FS transform, with an STC–AoK “non-interactive proof” on top of the commitment.

   What’s missing:

   * A real interface in code that corresponds to:
     * GenPrf(pp, v) → (C, π)
     * VrfyPrf(pp, C, π).
   * Even a minimal toy prover that:
     * runs Init/Update/Finalize,
     * produces meta + sketches + some random openings,
     * and a verifier that recomputes the challenge and checks them.

   Right now you effectively have Commit + GlobalCheck, but not the full “STC-AoK protocol” as an object.

3. Trace vs arbitrary vector

   The theory is written in terms of a trace v ∈ F^n, but your current pipeline is really:

   * “generic vector of values per chunk”, not a semantically meaningful step trace of a VM.
   * That’s fine mathematically, but if you want this to be a backend for execution, you need at least one worked example where:
     * v[i] is derived from real step-structured data (e.g. (pc, opcode, gas, state root)),
     * you show how that maps into the STC model.

   Right now the connection “this is not just a random vector, it’s an execution trace” is only conceptual.

---

## 2. IVC / zk backend gaps

This is the big hole between “sick primitive” and “actual zk backend”.

You say:

> Intended IVC embedding: the public accumulator (n, root, r⃗, s⃗) becomes the public state; per-step circuit enforces update_with_chunk.

But you don’t yet have:

1. An actual step circuit

   Missing:

   * A concrete SNARK/IVC-friendly circuit that takes:
     * (st_in, chunk) and outputs (st_out),
     * enforces:
       * hash-chain update,
       * sketch update (s_j, pow_j),
       * plus some toy VM transition.
   * Even a tiny algebraic circuit (Plonky2/Halo2/Nova gadget) implementing your Update algorithm.

   Right now docs/ivc_state_R.md is spec, but nothing compiles in a zk framework.

2. A working IVC loop

   No real:

   * Recursive accumulator that folds chunk proofs,
   * Demonstration that:
     * state and st are carried through recursion,
     * final proof attests to a full trace + STC commitment.

   One concrete “toy backend” you’re missing:

   * A simple 1D state machine (like s_{t+1} = A·s_t + b),
   * Chunk trace of length K,
   * STC update inside circuit,
   * Nova/Protostar-style recursion over, say, 100 chunks,
   * Final proof + final (n, root, r⃗, s⃗).

   That’s the minimal end-to-end example that says: “STC actually runs under IVC, not just on bare metal.”

3. Constraint-level cost analysis

   If you want to argue “this backend is viable for zkEVM,” you need:

   * Estimated/actual constraints for:
     * a single Update (one chunk),
     * hash-chain step,
     * sketch update over ℓ elements.
   * A comparison to:
     * “KZG-in-circuit” cost (pairings/MSMs encoded as constraints),
     * or “FRI-in-circuit” if you position STC + external FRI.

   Right now you only benchmark GPU commit vs CPU KZG commit, not “circuit cost of my backend vs alternatives”.

---

## 3. Systems + benchmarks gaps

The current bench story is:

* HSSA GPU commit throughput (GB/s),
* Fast Rust verify,
* KZG baseline on CPU (MB/s),
* Simple speed ratios (~40–45×).

Which is nice, but incomplete if you’re claiming “backend”.

1. Apples-to-apples KZG baseline

   You flag this yourself, but it’s important:

   * KZG baseline is CPU-only, not GPU/MSM-optimized,
   * different field/curve, different implementation maturity.

   A serious comparison needs at least one of:

   * GPU-accelerated KZG (or IPA) on similar hardware,
   * Or a clear “this is intentionally unfair, this is only a directional systems sanity check”.

   And then ideally a second benchmark: STC+FRI vs Merkle+FRI, since that’s the actual PC-level competition.

2. End-to-end zk stack benchmarks

   Missing: a mode where you actually measure:

   * VM execution → chunk traces → STC → zk proofs → verification.

   For even a toy VM, you want numbers like:

   * “Per million steps, this backend gives:
     * X ms proving time,
     * Y ms verification time,
     * Z bytes proof size.”

   Right now you measure commit/verify of the STC in isolation, which is good but not the whole backend.

3. Large-N, realistic configs

   You have:

   * N = 1,048,576 and N = 16,777,216 in the combined CSV.
   * README references a benchmarks.csv that’s missing.

   Gaps:

   * No visible runs at:
     * max intended n_max (e.g. 2³⁰ or 2³²),
     * wide range of m (2, 4, 8, 16),
     * chunk_len tradeoff (cache vs kernel occupancy).
   * No graphs:
     * throughput vs N,
     * throughput vs m,
     * verifier time vs N/K.

   You have all the CSV plumbing; what’s missing is the “paper-style” plot layer and a single markdown summarizing what the data actually says.

---

## 4. DA / protocol-level gaps

Your formal note already lays out a DA protocol and security bound. In the repo summary you capture:

* DA protocol via STC,
* docs/hssa_da_protocol.md.

What’s missing if you want this as “real rollup DA story”:

1. Network roles / messages

   There’s no concrete:
   * message formats (what gets posted on-chain, what’s gossiped),
   * sampling APIs: how a light client chooses indices and queries full nodes,
   * handling byzantine responders (e.g. equivocation, no response).

2. Parameter-picker / DA calculator

   You have formulas; you don’t have a tool that:

   * takes (δ, N, k, m, target_failure_prob),
   * tells you: “Here’s detection probability,” or “you need k ≥ … for your target”.

   It’s the same gap I mentioned in theory: no “DA parameter chooser” to make this plug-and-play.

3. Simulations

   You could easily:
   * Monte Carlo simulate DA cheating probabilities,
   * plot observed detection vs theory,
   * show that your DA layer behaves as predicted.

   That would turn hssa_da_protocol.md from “nice theory” into “we simulated the rollout behavior of this DA regime”.

---

## 5. Story / docs gaps (how this becomes a backend, not a bag of pieces)

Your writeup already does a lot of work, but some story gaps remain:

1. Backend diagram

   You and I just talked about the backend as:

   * execution engine (VM/rollup),
   * STC accumulator,
   * chunk zk proofs,
   * recursive aggregator,
   * DA samplers.

   That’s not currently written down as a single diagram + 1–2 page narrative. You have:

   * STC formal doc,
   * IVC state_R,
   * DA doc,
   * benches doc.

   Missing: “How these plug together into one backend architecture”.

2. Positioning vs alternatives

   Your HSSA notes compare Merkle, KZG, STC at a high level. The repo doesn’t yet have a crisp “Why this backend?” section that says:

   * When you should pick STC-based backend over:
     * pure Merkle+FRI,
     * KZG (Groth16-ish),
     * IPA-based PCs.
   * In which dimensions you explicitly don’t compete (proof size, on-chain verifier).

3. One worked “paper-style” example

   For Bünz/Microsoft/StarkWare, the killer artifact would be:

   * A short markdown or PDF that:
     * defines a tiny VM,
     * shows how its step trace is committed by STC,
     * shows an IVC prototype layout,
     * cites your benchmark tables,
     * and ends with something like:
       “For traces of length up to 2²⁴ on an A100, this backend reaches X GB/s proving throughput and Y ms verify, while providing DA guarantees of Z under chosen parameters.”

   You have all pieces to almost write that; you just haven’t glued them into one narrative.

---

### TL;DR gaps

If I compress all of that:

* Mathematically, you’re fine: primitive is well-defined, binding bounds and DA/AoK are written.
* Implementation-wise, STC as a standalone commitment + fast verifier is solid and benchmarked.
* Backend-wise, what’s missing is:
  1) A real SNARK/IVC integration (circuits + recursion).
  2) End-to-end zk+DA benchmarks instead of primitive-only throughput.
  3) A parameterization layer (soundness/DA calculators, recommended configs).
  4) A clean backend architecture doc that shows VM → STC → zk → IVC → DA as one pipeline.

If you knock out even a toy IVC integration + a parameter table + a short “backend architecture” doc, you go from “cool primitive + repo” to “this is a coherent alternative backend someone can actually evaluate.”




---

# docs/archive/BEF-Stream-Accum-summary.md

<a name="docs-archive-BEF-Stream-Accum-summarymd"></a>

# BEF-Stream-Accum Summary

BEF-Stream-Accum is a streaming accumulation scheme for execution traces. The
accumulator state carries a Merkle-style hash root over chunk commitments plus a
small vector of random linear sketches. Each update consumes one chunk of the
trace and mutates the state via a fixed transition rule (`update_with_chunk`).
This state is what we export from the GPU accumulator and what we plan to treat
as the public accumulator inside an accumulation-based IVC/PCD framework.

## Algorithms

- **Setup** picks the field `F_p`, hash `H`, chunk size, and number of
  challenges `m`.
- **Init** produces the empty state `(len=0, root=H("bef-init"), {r_j}, {s_j=0},
  {pow_j=1})` with challenges derived from the initial root.
- **Accumulate** takes `(state_t, chunk)` and applies the deterministic update
  rule: hash the new chunk root into `root`, update every sketch via
  `s_j += Σ chunk[k] · r_j^{offset+k}`, update `pow_j`, and bump `len`.
- **Output** projects the state to the public trace commitment
  `(len, root, {r_j}, {s_j})`.
- **Verify (reference)** replays the entire sequence, calling `Accumulate` for
  each chunk. This is linear in the total trace length and serves as the ground
  truth for correctness.

## Fast verification (`Verify_fast`)

Given the final commitment `com_T` and metadata about each chunk
`(chunk_index, offset, length, root, sketch_vec)`, we can verify consistency
without reprocessing raw values:

1. Sort chunks by offset and ensure they cover `[0, len)` with no gaps.
2. Recompute the commitment root by iteratively applying `hash_root_update` over
   the chunk roots and check it equals `com_T.root`.
3. Sum the per-chunk sketch vectors and ensure the result matches
   `com_T.sketches`.

This runs in `O(K · m)` time for `K` chunks and `m` challenges. It detects any
chunk-level tampering (bad root, bad sketch vector, broken coverage) while the
reference verifier remains available if we need to replay the trace.

## Implementation status

- The GPU accumulator emits the per-chunk metadata and the final commitment in
  JSON (`sketch_trace.py`).
- `TraceCommitState` and `bef_verify_fast` are implemented in the `bicep-crypto`
  crate.
- Automated tests load the sample JSON (`code/sketches/trace_demo_sketch.json`)
  and ensure the fast verifier accepts the honest transcript and rejects
  tampered chunk roots/sketches.
- Benchmarking hooks are ready: `demo_cuda(...)` reports accumulate timings,
  while `bef_verify_fast` and the reference replay give CPU-side verification
  costs. These metrics will feed into the accumulation-based IVC/PCD story.

The next step is to expose the BEF state/transition as the public accumulator in
an IVC/folding backend, so that the heavy trace processing stays in the GPU
layer while a generic accumulation/PCD framework handles succinct proofs.


## Soundness experiment (informal)

We view BEF-Stream-Accum as an accumulation scheme for the step predicate
“`state_out` is the result of applying `update_with_chunk(state_in, chunk)`.”
Let `Verify` denote the reference verifier that replays every step.

Define the experiment `Exp_BEFSound(A)`:

1. Challenger runs `pp ← Setup(1^λ)`.
2. Adversary outputs a final state `a_T` and a sequence of instances
   `{x_t} = {(state_in_t, chunk_t, state_out_t)}`.
3. Challenger computes `b = Verify(pp, a_T, {x_t})`.

A wins if `b = 1` but there exists some `t` such that
`state_out_t ≠ update_with_chunk(state_in_t, chunk_t)`.

We require that every PPT adversary wins with at most negligible probability in λ,
assuming collision resistance of H (SHA-256) and the usual large-field behavior
for the sketches. `Verify_fast` is a derived verifier operating on chunk metadata;
its correctness is relative to the honesty of those chunk summaries.


## Reference vs Fast Verification

- `Verify(pp, a_T, {x_t})`: the reference verifier that replays all steps (linear in the
  total trace length) and defines the canonical soundness notion.
- `Verify_fast(pp, com_T, {chunk_summaries})`: a metadata-only checker that ensures chunks
  cover `[0, len)`, the recomputed root matches `com_T.root`, and the sum of per-chunk
  sketches matches `com_T.sketches`. It runs in `O(K·m)` for `K` chunks and `m` challenges.

Soundness of BEF-Stream-Accum is defined with respect to `Verify`; `Verify_fast`
provides a practical consistency check when chunk summaries are available.


## Benchmark plan

We benchmark three operations on an NVIDIA A100 (40GB) over the field `F_p` with
`p = 2^61 - 1`:

| Trace length (N) | # chunks (K) | # challenges (m) | GPU accumulate (ms) | Verify (replay, ms) | Verify_fast (ms) |
|-----------------:|-------------:|-----------------:|--------------------:|--------------------:|-----------------:|
| 1e4              | 10           | 2                | TBD                 | TBD                 | TBD             |
| 1e6              | 100          | 4                | TBD                 | TBD                 | TBD             |
| 1e7              | 100          | 4                | TBD                 | TBD                 | TBD             |

- **GPU accumulate**: streaming accumulator on CUDA (r^i build + chunk sketches).
- **Verify (replay)**: CPU replay of `update_with_chunk` for every chunk.
- **Verify_fast**: metadata-only check using chunk roots and sketch vectors.

`Verify_fast` scales with `K·m` rather than `N`, while GPU accumulate leverages
massive parallelism. Actual numbers will be filled in once the A100 runs are
recorded.



---

# docs/archive/BEF_CURRENT_STATUS.md

<a name="docs-archive-BEF_CURRENT_STATUSmd"></a>

BEF Current Status (Streaming Accumulator, ENN, Fusion Alpha)

1) Implemented and working now
- GPU streaming accumulator (Python + CUDA): `gpu_accumulator/stream_accumulator.py`, `cuda/stream_sketch_kernel.cu`, `stream_sketch_extension.cpp`.
  - Fused multi‑challenge kernel (`fused_sketch_kernel`) and tiled r^i builder (`fill_rpow_blocks_kernel`).
  - Python API: `StreamingAccumulatorCUDA.prove()`, `build_rpow_gpu`, `chunk_dot_cuda`, CLI: `sketch_trace.py`.
  - Sample output with timings: `code/sketches/trace_demo_sketch.json` (timing_ms recorded).
- Fast verification (Rust): `BICEPsrc/BICEPrust/bicep/crates/bicep-crypto/src/lib.rs`.
  - `TraceCommitState::update_with_chunk`, `TraceCommitment`, `bef_verify_fast(...)` (chunk coverage, root recompute, sketch sum).
  - Unit tests load `code/sketches/trace_demo_sketch.json` and check accept/reject cases.
- ENN core (C++): `enn-cpp/src/*.cpp` with tests in `enn-cpp/tests/`.
  - Entangled cell (E = L L^T), collapse, gradient checks, PSD checks.
- Fusion Alpha (Rust + Python bindings): `FusionAlpha/crates/*`, `FusionAlpha/python/*`.
  - Graph, priors, propagation, action selection; integration test `FusionAlpha/integration_test.py` (BICEP→ENN→Fusion).
- Specs/docs: `docs/bef_trace_commitment.md`, `docs/ivc_state_R.md`, `docs/BEF-Stream-Accum-summary.md`.

2) Spec’d on paper but not fully coded
- IVC/PCD embedding: step relation R is specified; no in‑repo circuit/backend wiring yet.
- Formal soundness/Sketch collision analysis: marked TBD in docs.
- Full performance tables (A100/large N): placeholders in `BEF-Stream-Accum-summary.md`.

3) Missing / future work
- Folding/IVC glue: expose `(len, root, {r_j}, {s_j})` as public state in a concrete backend; add small per‑step circuit enforcing `update_with_chunk`.
- DA/rollup integration plan (protocol, data formats).
- Bench harness and CI for GPU kernels across SM architectures; parity tests vs CPU for large N.
- Python bindings for `bef_verify_fast` and a reference replay verifier.
- BICEP GPU backend and PyO3 bindings (not in this repo yet; placeholders exist in `BICEPsrc`).

4) Performance numbers we actually have
- Small demo timings are embedded in `code/sketches/trace_demo_sketch.json` (cuda_rpow, cuda_chunks, fused variants).
- Not present in repo: large‑N/A100 timing tables referenced in docs (`benchmarks.csv`).

5) Immediate next steps
- Record end‑to‑end timings on A100 for N∈{1e6,1e7}, m∈{2,4}, K variable; fill doc tables.
- Publish a `bef_verify_fast` Python shim + reference replay helper; add CLI to validate `.json` sketches.
- Add SM‑aware build/test matrix and correctness benchmarks for the fused kernel (multi‑challenge parity vs unfused path).
- Stand up a minimal folding/IVC example instantiating relation R with `(len, root, {r_j}, {s_j}, {pow_j})` as state.
- Optional: expose node‑/chunk‑level features from sketches into Fusion Alpha demos.




---

# docs/archive/bef_compact_v1_spec.md

<a name="docs-archive-bef_compact_v1_specmd"></a>

# BEF Compact Serialization (`bef_compact_v1`)

This document sketches the Phase‑0 specification for a production-grade binary
serialization format that supersedes the current DAG‑CBOR encodings. It focuses
on the core file layout, data types, and security invariants; subsequent phases
will provide the full byte-level schema for every proof/capsule structure.

## 1. Design Goals

1. **Unambiguous by construction** – every valid byte stream corresponds to a
   single semantic object. No optional keys, no duplicate entries, no slack.
2. **Performance-oriented** – optimized for zero-copy reads and minimal heap
   churn in verifiers. No key names, no self-describing overhead.
3. **Security hardened** – parsing must be strictly bounded, rejecting any
   malformed/crafted payload before it can trigger DoS or verification bypass.
4. **Layered abstraction** – canonical encoding is uncompressed; higher layers
   may apply zstd or similar without changing the canonical form.
5. **Backwards compatible** – legacy JSON / CBOR artifacts continue to decode via
   dispatcher logic keyed on magic bytes / tags.

## 2. Threat Model

* **DoS via parsing** – attacker supplies oversized lengths or malformed varints
  to exhaust CPU/RAM. Countermeasure: strict length-prefix validation and
  canonical varint enforcement.
* **Verification bypass** – malformed payload attempts to skip fields or violate
  field ordering. Countermeasure: schema-driven parsing, zero tolerance for
  missing/extra data.
* **Canonicality attacks** – attacker attempts to produce multiple encodings of
  the same proof. Countermeasure: canonical varints, canonical field ordering,
  no optional whitespace/keys.

## 3. File Layout

```
+------------+-------------------+---------------------------+
| Bytes 0-3  | Magic             | 0xBEF0C0DE                |
| Bytes 4-5  | Version           | 0x0100 (major=1, minor=0) |
| Bytes 6-9  | Metadata length N | uint32 LE                 |
| Bytes 10…  | Metadata JSON     | N bytes UTF-8             |
| …          | Proof payload     | schema-defined            |
+------------+-------------------+---------------------------+
```

* **Magic bytes** – allow instant rejection of unrelated files.
* **Version** – enables evolution via `major.minor`. Minor upgrades remain
  backward compatible; major upgrades signal breaking changes.
* **Metadata** – small, human-readable JSON (trace id, air id, timestamp, etc.)
  that is *not* part of the canonical proof hash. This allows operators to store
  auxiliary context without affecting verification.
* **Payload** – canonical proof bytes described in Section 4.

## 4. Payload Encoding Rules

### 4.1 Primitive Types

* **Varint** – unsigned LEB128-like encoding with canonicality checks (no leading
  zero continuation bytes, no overflow above 2^64-1). Used for lengths, counters,
  Goldilocks field elements, etc.
* **Bytes** – raw byte arrays. Used for hashes, Pallas/Vesta field elements,
  Merkle roots, etc.
* **Fixed arrays** – concatenation of child fields in schema order.

### 4.2 Field Elements (`Fp`)

| Field            | Encoding                             |
|------------------|--------------------------------------|
| Goldilocks (64b) | canonical varint (little endian)     |
| Pallas/Vesta     | 32-byte little-endian fixed array    |
| Generic big Fp   | ceil(bits/8) bytes little-endian     |

### 4.3 Structs

Each struct is encoded as a fixed-order sequence of its fields, with no key
names. Example for `FriLayer { commitment: Hash, beta: Fp, length: u32 }`:

```
[32 bytes commitment] [beta (varint or 32-byte)] [length varint]
```

### 4.4 Lists

All lists are `varint(count)` followed by `count` encoded elements in order.

## 5. Security Invariants

* **Length checking** – `BinaryDecoder.read_bytes(n)` fails if `offset+n > len`.
* **Canonical varints** – decoder rejects encodings with redundant leading
  zeros/continuations.
* **No recursion** – decoder iterates with explicit stacks to avoid stack
  exhaustion.
* **Full consumption** – after parsing the payload, `offset` must equal
  `len(buffer)` or decoder fails (prevents trailing garbage).

## 6. Backward Compatibility

* Legacy artifacts are detected as:
  * JSON – leading `{` or `[`.
  * CBOR – self-describe tag 0xD9D9F7 or other CBOR lead bytes.
* New dispatcher reads magic bytes; if `0xBEF0C0DE`, it routes to the
  `BinaryDecoder`; otherwise falls back to legacy decoders.

## 7. Next Steps

1. Complete the byte-level schema for proofs/capsules (field-by-field order).
2. Implement `BinaryEncoder` / `BinaryDecoder` with canonical varint, field-
   specific writers/readers, and hardened bounds checks.
3. Wire encoder into pipeline tooling, gating behind `--encoding-id
   bef_compact_v1`.

This document intentionally stops short of a full implementation; it establishes
the format and threatmodel so Phase 1/2 can proceed with confidence.



---

# docs/archive/bef_trace_commitment.md

<a name="docs-archive-bef_trace_commitmentmd"></a>

# BEF Trace Commitment

This note specifies the streaming trace/vector commitment that the BEF stack
exposes to the rest of the system.  The commitment is deliberately
GPU-friendly: all heavy work reduces to hashing chunk roots and evaluating
large dot products Σ v[i]·r^i mod p, which we accelerate with the CUDA
accumulator.

- **Trace:** sequence `v ∈ F_p^n` (field element per time step)
- **State:** `(n, root, {r_j}, {s_j}, {pow_j})`
  - `root`: hash/Merkle commitment to the chunk structure
  - `r_j`: hash-derived challenges (Fiat–Shamir over transcript)
  - `s_j = Σ v[i]·r_j^i`: linear sketches for integrity sampling
  - `pow_j = r_j^n`: carried only to support streaming updates
- **Commitment output:** `(n, root, {r_j}, {s_j})`

## Algorithms

### Setup
Choose:
- Field `F_p` (target: SNARK-friendly prime; prototype uses 2^61-1)
- Hash `H : {0,1}* -> {0,1}^{256}` (BLAKE3/SHA-256) with domain separators
- Chunk length `L`, number of challenges `m`
- Merkle layout (per-chunk trees + top tree or hash chain)

Publish `pp` with these parameters.

### Init
```
state.len   = 0
state.root  = H("bef-init" || context)
for j in 0..m-1:
    r_j        = FE( H(state.root || "bef-challenge" || j) )   // reject 0
    s_j        = 0
    pow_j      = 1
```
Return `(len, root, {r_j}, {s_j}, {pow_j})`.

### Update (append chunk values)
Input chunk `(v_len, …, v_len+ℓ-1)` with ℓ ≤ L.
1. Build chunk Merkle root `chunk_root` (or hash of `(len || values)`).
2. `root' = H(root || "bef-chunk" || encode(len) || chunk_root)`.
3. For each challenge j:
   ```
   s_j'   = s_j
   pow_j' = pow_j          // currently r_j^len
   for each value x in chunk:
       s_j'   = s_j'   + x * pow_j'
       pow_j' = pow_j' * r_j
   ```
4. `len' = len + ℓ`.
Return updated state.

### Commit
Stream the trace through Init+Update, then output
`Commitment = (len, root, {r_j}, {s_j})`.
This is what gets persisted / fed into IVC.

### Open & Verify
To open position i:
- Reveal `value = v[i]` plus Merkle paths
  * `proof_chunk`: leaf→chunk_root
  * `proof_top`: chunk_root→root

Verifier:
1. Check Merkle paths vs `root`.
2. (Optional) recompute sampled contributions and compare with `s_j` for global checks.

The sketches give probabilistic detection of bulk tampering:
with m challenges over a 128-bit field the chance of hiding a
non-zero error vector is ≤ (deg/|F|)^m by Schwartz–Zippel.

## Intended Usage
1. **Trace/DA commitment:** replace KZG/FRI for raw blobs with GPU hashes + sketches.
2. **IVC state:** `(len, root, {r_j}, {s_j})` becomes the public state inside a
   folding/Nova/Protostar relation `R` that enforces `Update`.
3. **PCD payload:** nodes exchange `(state, proof)` pairs where `state` embeds
   this commitment and `proof` is the IVC witness of honest evolution.

## Security Notes
- Binding reduces to hash binding (Merkle) + sketch collision bounds.
- Not hiding; all values can be opened.
- Formal reduction/probability analysis TBD once the field and sampling
  distributions are finalized.

## Chunk Summary Format

The GPU accumulator exports per-chunk metadata used by `Verify_fast` and by the
JSON sketch files:

```
{
  "chunk_index": k,
  "offset": start_index,
  "length": chunk_length,
  "root_hex": "...32-byte hex...",
  "sketch_vec": [s_{k,0}, s_{k,1}, ...]
}
```

Each entry contains:

- `chunk_index`: sequential identifier starting at 0.
- `offset` / `length`: describe the covered range in the global trace.
- `root_hex`: Merkle root for the chunk's raw values.
- `sketch_vec`: the chunk's contribution to each sketch after multiplying by
  `r_j^{offset}`. Summing all vectors reproduces the global `s_j` values.

The JSON also includes a `trace_commitment` object with `(len, root_hex,
challenges, sketches)` plus the `commitment_root`. These fields map directly to
`TraceCommitment` and `ChunkSketch` in the code and are the inputs for
`bef_verify_fast`.



---

# docs/archive/da_profile_router.md

<a name="docs-archive-da_profile_routermd"></a>

# DA Profile + Router Layer

To keep STC/HSSA agnostic to where the raw trace lives, we bind a tiny **data
availability profile** to every STC commitment and provide a backend-agnostic
router. This adds a DA “menu” without changing the accumulator.

## Profile schema

```
DAProfile = {
  "version": 1,
  "mode": "L1_BLOB" | "EXTERNAL_DA" | "COMMITTEE_SIGS" | "LIGHT_SAMPLING" | "ARCHIVE_ONLY",
  "sampling": {
    "delta_min": 0.1,          # minimum withholding fraction we aim to detect
    "epsilon_max": 1e-6,       # failure probability budget
    "k_min": 96                # required number of chunk samples (computed via scripts/stc_param_table.py)
  },
  "provider": {
    "chain_id": 1,
    "blob_tx": "0x..."
  }
}
```

The block header commitment becomes `H(STC_root || encode(DAProfile))`, so the
da policy is cryptographically bound to the trace it claims to cover.

## Router interface

```
trait DABackend {
    fn publish(stc_root: Hash, profile: DAProfile, data_ref: DataRef) -> DAHandle;
    fn check_available(handle: DAHandle, chunks: &[ChunkId]) -> Result<(), DAError>;
}
```

Examples:

- `L1_BLOB`: data_ref = `{chain_id, blob_tx}`, check = download blob + verify bucket set.
- `EXTERNAL_DA`: data_ref = `{namespace, height}`, check = call DA API + run sampling.
- `COMMITTEE_SIGS`: data_ref = `URI list`, publish collects threshold sigs attesting to `stc_root`.

Profiles can be swapped over time using a `SwapDA` message:

```
SwapDA {
  stc_root,
  old_profile, old_handle,
  new_profile, new_handle,
  attestation: signature or SNARK proving equality of data.
}
```

## Sample profiles

See `da_profiles/minimal.json` and `da_profiles/strong.json` for concrete
configurations. Clients can always run stricter sampling than `k_min`, but the
profile sets a minimum bar (“if you accept this commitment, you must at least
run k_min samples targeting δ and ε”).



---

# docs/archive/encoding.md

<a name="docs-archive-encodingmd"></a>

# Canonical Encoding

This repository ships deterministic binary artifacts for capsules and proofs using
DAG-CBOR based encodings. Two variants exist today:

* `dag_cbor_canonical_v1` – original format (ints encoded as CBOR integers).
* `dag_cbor_compact_fields_v1` – new default; field elements are serialized as
  compact little-endian byte strings whose width matches the actual field size
  (8 bytes for Goldilocks, 32 bytes for larger curves).

Both encodings share the same goals:

1. Stable hashes that do not depend on incidental formatting
2. Compact binary representations (no hex strings) that enable content-addressed
   storage
3. Data that can be round-tripped into JSON for debugging without loss

## Canonicalization rules

* **Encoding** – The encoding is a strict subset of DAG-CBOR (deterministic
  CBOR).  Only the following CBOR types are used: unsigned/signed integers,
  byte strings, UTF‑8 text strings, arrays, maps, booleans, null, and 64‑bit
  IEEE floats.  Indefinite-length items are forbidden.
* **Map ordering** – Map keys are encoded independently, and entries are sorted
  by the lexicographic order of their encoded byte representation before being
  emitted.  This matches the DAG-CBOR canonical ordering rule.
* **Byte formats** – Cryptographic data that is naturally a byte string (hashes,
  commitments, Merkle siblings, etc.) MUST be encoded as raw byte strings
  (`major type 2`).  JSON views may still present these values as lowercase hex,
  but binary artifacts never store textual encodings.
* **Integers** – Integers are encoded using the minimal CBOR representation
  (major types 0/1).  In the compact-fields encoding, integers that are tagged as
  field elements are instead emitted as byte strings whose width equals the
  field’s native bit width (little-endian).
  The serializer tags every field element (including nested FRI openings), so
  this rule applies recursively throughout the proof/capsule structure.
* **Floats** – Floating point values are encoded as IEEE754 binary64 (`0xfb`).
* **Null/default values** – Optional fields MAY be omitted; when present they
  are encoded as CBOR `null` (0xf6).  The Capsule hash is computed over whatever
  explicit representation is emitted, so either omission or explicit null is
  acceptable as long as it is deterministic for the generator.
* **Domain separation** – Every hash derived from canonical bytes uses an
  explicit ASCII prefix, e.g. `BEF_CAPSULE_V1 || canonical_bytes`.

## Sanitization helper

A lightweight sanitization pass (`sanitize_for_encoding`) canonicalizes host
objects before encoding:

* Paths become strings
* Tuples become arrays
* Dictionaries drop keys whose value is `None` only when the calling site
  requests pruning (currently capsules preserve explicit `None`s)

## API

`bef_zk.codec` exposes:

* `ENCODING_ID` – currently `dag_cbor_compact_fields_v1`
* `canonical_encode(obj, encoding_id=ENCODING_ID) -> bytes`
* `canonical_decode(data: bytes) -> object`
* `FieldElement(value, bits)` helper used when tagging field values for compact
  emission.

Higher-level helpers (`encode_capsule`, `encode_proof`, etc.) sanitize
structures before calling these primitives.

## Capsule hash and audit seed

* `capsule_hash = SHA256("BEF_CAPSULE_V1" || canonical_bytes_without_hash)`
* Default audit seed uses the capsule hash plus unbiasable anchors:
  `seed = first_u64( SHA256("BEF_AUDIT_SEED_V1" || capsule_hash || anchor_ref || policy_id) )`

Any change to the capsule contents now deterministically updates the hash,
allowing content-addressed storage and reproducible audit sampling.



---

# docs/archive/fri_parameter_guidance.md

<a name="docs-archive-fri_parameter_guidancemd"></a>

# FRI / STC parameter guidance

This backend exposes two sets of knobs:

1. **FRI / AIR parameters** – determine the soundness error of the zk argument.
2. **STC parameters** – determine binding / DA guarantees of the streaming accumulator.

Both must be configured together for a concrete deployment. The table below gives sample settings for the toy geometry AIR; adapt them to your own relation.

| Target steps (T) | Domain size (N) | AIR degree bound (d_max) | FRI rounds | Queries (q) | Estimated FRI error ((d_max / N)^q) |
|------------------|-----------------|--------------------------|-----------|-------------|-------------------------------------|
| 2^10 (≈1k)       | 2^10            | 2^10                     | 4         | 16          | ≈ 2^{-40}                           |
| 2^14 (≈16k)      | 2^15            | 2^14                     | 5         | 24          | ≈ 2^{-60}                           |
| 2^18 (≈260k)     | 2^19            | 2^18                     | 6         | 32          | ≈ 2^{-80}                           |
| 2^22 (≈4M)       | 2^23            | 2^22                     | 7         | 48          | ≈ 2^{-110}                          |

**How to read the table**

- Choose ((N)) as the next power of two above your padded trace length.
- Compute a conservative degree bound ((d_max)) for the composition polynomial (max AIR degree plus masking degree). In this prototype, `build_composition_vector` never exceeds the raw trace degree, so you can over-approximate with the trace length.
- Pick a number of FRI rounds and queries ((q)) so that ((d_max / N)^q) is below your target failure probability.
- Masking polynomials must have degree strictly less than the FRI degree bound—keep mask degree < (d_max).

**STC parameters**

Reuse `docs/stc_parameter_guidance.md` to pick the number of sketch challenges (m) and DA sampling rate. For convenience the existing helper script can be run as:

```bash
python scripts/stc_param_table.py --n-max 16777216 --p 2305843009213693951 --challenges 4 6 8
```

**FRI helper script**

`scripts/fri_param_table.py` prints the FRI soundness estimate for a grid of domain sizes and query counts. Example:

```bash
python scripts/fri_param_table.py --domain 1048576 --degree 262144 --queries 20 30 40
```

This reports ((degree / domain)^q). Use it to sanity-check new AIRs before collecting full benchmarks.

Remember that the final argument’s soundness is the sum of:

- FRI error (as above),
- STC binding failure (from the sketch bound in §3), and
- collision resistance of SHA-256 (≈2^{-128}).

Pick parameters so that each term is comfortably below your overall security target (e.g., 128 bits).



---

# docs/archive/geom_backend_analysis.md

<a name="docs-archive-geom_backend_analysismd"></a>

| steps | queries | backend | proof KB | prove (s) | verify (s) | zk/tr trace |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 16 | geom_plain_fri | 330.6 | 0.370 | 0.019 | 26.3 |
| 4096 | 16 | geom_stc_fri | 352.1 | 0.470 | 0.019 | 36.6 |
| 4096 | 32 | geom_plain_fri | 658.1 | 0.458 | 0.027 | 33.0 |
| 4096 | 32 | geom_stc_fri | 701.0 | 0.581 | 0.023 | 47.9 |
| 8192 | 16 | geom_plain_fri | 351.2 | 0.653 | 0.037 | 21.3 |
| 8192 | 16 | geom_stc_fri | 372.7 | 0.877 | 0.041 | 28.6 |
| 8192 | 32 | geom_plain_fri | 699.3 | 0.743 | 0.041 | 24.4 |
| 8192 | 32 | geom_stc_fri | 742.3 | 0.978 | 0.038 | 37.7 |
| 16384 | 16 | geom_plain_fri | 371.8 | 1.217 | 0.068 | 20.5 |
| 16384 | 16 | geom_stc_fri | 393.3 | 1.661 | 0.071 | 28.3 |
| 16384 | 32 | geom_plain_fri | 740.8 | 1.324 | 0.074 | 21.8 |
| 16384 | 32 | geom_stc_fri | 783.8 | 1.808 | 0.076 | 31.3 |
| 32768 | 16 | geom_plain_fri | 392.4 | 2.359 | 0.131 | 19.7 |
| 32768 | 16 | geom_stc_fri | 414.0 | 3.253 | 0.150 | 24.3 |
| 32768 | 32 | geom_plain_fri | 781.7 | 2.457 | 0.143 | 19.8 |
| 32768 | 32 | geom_stc_fri | 824.7 | 3.368 | 0.149 | 26.5 |


---

# docs/archive/hssa_vs_kzg_bench.md

<a name="docs-archive-hssa_vs_kzg_benchmd"></a>

HSSA vs KZG — Benchmark Plan and Artifacts

1) Goals
- Measure HSSA commit throughput (elements/s, GB/s) for realistic trace sizes.
- Measure HSSA fast verification (CPU) vs trace size, #chunks, m.
- Compare with a basic KZG commitment baseline (same N) to provide directional evidence.

2) Parameter Grid
- Trace sizes N ∈ {2^20, 2^24, 2^26} (≈ 1M, 16M, 64M elements).
- Challenges m ∈ {2, 4, 8}.
- Chunk length L ∈ {2^10, 2^13} (1024, 8192 elements).
- Field p = 2^61 − 1 (prototype), note CPU/GPU models used.

3) HSSA Benchmark Harness (bench_hssa.py)
- For each (N, m, L):
  - Generate random v ∈ F_p^N.
  - Partition into chunks.
 - Run StreamingAccumulatorCUDA.prove() to commit (r^i build or fused kernel; per‑chunk sketches).
 - Record commit wall‑clock time t_commit_gpu and compute throughput: elems/s and GB/s.
 - Run bef_verify_fast on the emitted sketch JSON to get t_verify_fast (CPU).
 - Write CSV: N, m, L, t_commit_gpu_ms, t_verify_fast_ms, elems_per_s, GB_per_s, hardware.
- Script: `python bench/bench_hssa.py --Ns 1048576,16777216 --challenges 2,4,8 --chunk-lens 1024,8192 --repeats 3`
  - Expects `bef_verify_fast_cli` at `BICEPsrc/BICEPrust/bicep/target/release`. Build via `cargo build --release -p bicep-crypto --bin bef_verify_fast_cli`.

4) KZG Baseline Harness (bench/kzg-bench)
- Located at `bench/kzg-bench` (standalone Cargo crate using arkworks KZG10).
- For each N: sample random v over BLS12-381 Fr and measure time to compute Commit(v).
- Compute throughput: elems/s and MB/s (Fr is 32 bytes).
- Run on the same hardware; note HSSA commit uses GPU while KZG runs on CPU.
- Command:
  ```bash
  cd bench/kzg-bench
  cargo run --release > ../kzg_results.csv
  ```

5) Report Template
- Produce markdown table such as:

| N (elems) | m | L (chunk) | HSSA commit (ms) | HSSA thr. (GB/s) | HSSA verify_fast (ms) | KZG commit (ms) | KZG thr. (MB/s) | Hardware |
|----------:|---|-----------:|-----------------:|------------------:|----------------------:|----------------:|------------------:|----------|
| 2^20      | 4 | 2^10       | …                | …                 | …                     | …               | …                 | A100 + CPU XYZ |
| 2^24      | 4 | 2^13       | …                | …                 | …                     | …               | …                 | … |
| 2^26      | 8 | 2^13       | …                | …                 | …                     | …               | …                 | … |

6) Narrative (short)
- On an A100, HSSA achieves X–Y GB/s commit throughput for N ∈ [1M, 64M] and m ∈ [2,8]; CPU bef_verify_fast stays ≤ Z ms.
- A naive KZG baseline on the same machine yields U–V MB/s commitment throughput. Preliminary results illustrate that HSSA is GPU‑native and offers high‑throughput commitment suitable for DA/IVC scenarios.



---

# docs/archive/ivc_state_R.md

<a name="docs-archive-ivc_state_Rmd"></a>

# Step Relation R for BEF-Stream-Accum

This document describes the step relation `R` used when embedding
BEF-Stream-Accum into an incrementally verifiable computation (IVC) or
PCD framework.

## State

At step `t`, the public state is the BEF trace accumulator:

```
S_t = (n_t, root_t, {r_j}_{j < m}, {s_{t,j}}_{j < m}, {pow_{t,j}}_{j < m})
```

- `n_t`: number of elements processed so far.
- `root_t`: SHA-256 hash/commitment root chaining over chunk roots up to `t`.
- `r_j`: constant challenges derived once from the initial root.
- `s_{t,j}`: sketches `Σ_{i < n_t} v[i] · r_j^i`.
- `pow_{t,j}`: `r_j^{n_t}`, used to stream updates; may be kept as auxiliary
  witness data inside the circuit.

## Step input

Each step consumes a chunk of values `chunk_t = (v_{n_t}, ..., v_{n_t+ℓ-1})`
with `ℓ ≤ L`. The implementation also computes a chunk Merkle root
`chunk_root_t` over these values.

## Relation R

`R(S_t, chunk_t, S_{t+1}) = 1` iff `S_{t+1}` is exactly what
`TraceCommitState::update_with_chunk(S_t, chunk_root_t, chunk_t)` produces:

1. `n_{t+1} = n_t + ℓ`.
2. `root_{t+1} = Poseidon2(root_t, n_t, chunk_root_t)` where `root_t` and
   `chunk_root_t` are interpreted as field elements modulo `2^61-1`.
3. For each challenge `r_j`:
   ```
   s' = s_{t,j}
   pow' = pow_{t,j}
   for value in chunk_t:
       s'   = s'   + value · pow'
       pow' = pow' · r_j
   s_{t+1,j}   = s'
   pow_{t+1,j} = pow'
   ```

In circuit form, this relation costs `O(ℓ · m)` field mul/add gates plus the
hash constraints. Implementing R in an IVC framework means treating this as the
tiny per-step circuit; the heavy data (all v[i]) stays in the BEF accumulator,
which we compute on GPU.



---

# docs/archive/operator_flow.md

<a name="docs-archive-operator_flowmd"></a>

# Operator Flow (Geom demo)

The geometry demo now has one blessed workflow that produces a full Capsule:

```
python scripts/run_pipeline.py --backend geom \
    --steps 64 \
    --num-challenges 2 \
    --output-dir out/demo_geom_
```

It performs:

1. **Trace + STC log**: run `simulate_trace` on `GEOM_PROGRAM`, flatten each row to
   `[pc, opcode, gas, acc, x1, x2, cnt, m11, m12, m22, s[0..m-1], pow[0..m-1]]`, and
generate a `bef_trace_v1` (`stc_trace.json`).

2. **Geom AIR proof**: run `zk_prove_geom`, serialize to `geom_proof.json`, verify
   immediately for sanity.

3. **Nova-over-STC**: call `cargo run -p nova_stc -- prove --chunks stc_trace.json ...`
   which produces both timings and the final STC state. `--stats-out` writes
   `nova_stats.json` (final `(n, root, s, pow)` plus timings and proof sizes).

4. **Capsule construction**: aggregate paths + metadata into
   `strategy_capsule.json` (schema `bef_capsule_v1`). The capsule now embeds the
   STC commitment `(n, root, s, pow)`, the Geom proof summary, and file pointers
   for every artifact.

5. **Pipeline stats**: `pipeline_stats.json` captures the plain-trace runtime,
   Geom proof profile, and Nova timings in one place. Sweeps can consume this via
   `bench/geom_pipeline_sweep.py` to produce reproducible scaling studies.

Operators follow this flow per epoch:

```
# produce trace + Geom proof + Capsule for 2048 steps
python scripts/run_pipeline.py --backend geom --steps 2048 --num-challenges 4 --output-dir out/epoch_42
```

Compression is still optional: Nova state is maintained continuously; the
compressed SNARK (~10 kB, ~20–40 s) is produced on demand (per epoch) via the
`--compressed` flag.

### Capsule verification

To re-check everything later (or share the artifact), run:

```
python scripts/verify_capsule.py out/epoch_42/strategy_capsule.json
```

The command replays the Geom verifier and ensures the capsule's STC commitment
matches the Nova stats. This is the same entry point we use when handing a
Capsule to another team.



---

# docs/archive/stc_parameter_guidance.md

<a name="docs-archive-stc_parameter_guidancemd"></a>

## STC parameter guidance

Assuming δ = 0.1, k = 64 DA samples, and a 128-bit hash binding term, the
combined failure probability is `(1-δ)^k + sketch_collision + 2^-128`.

| n | m | sketch collision | DA failure |
|---:|---:|----------------:|------------:|
| 2^20 (1,048,576) | 2 | 2.068e-25 | 1.179e-03 |
| 2^20 (1,048,576) | 4 | 4.276e-50 | 1.179e-03 |
| 2^20 (1,048,576) | 8 | 1.829e-99 | 1.179e-03 |
| 2^24 (16,777,216) | 2 | 5.294e-23 | 1.179e-03 |
| 2^24 (16,777,216) | 4 | 2.803e-45 | 1.179e-03 |
| 2^24 (16,777,216) | 8 | 7.855e-90 | 1.179e-03 |
| 2^32 (4,294,967,296) | 2 | 3.469e-18 | 1.179e-03 |
| 2^32 (4,294,967,296) | 4 | 1.204e-35 | 1.179e-03 |
| 2^32 (4,294,967,296) | 8 | 1.449e-70 | 1.179e-03 |



---

# docs/archive/stc_pc_backend.md

<a name="docs-archive-stc_pc_backendmd"></a>

# STC Polynomial Commitment Backend

This backend exposes STC as a vector commitment suitable for FRI/Σ-style PCs.
It introduces two JSON schemas:

```
// bef_pc_commit_v1
{
  "schema": "bef_pc_commit_v1",
  "length": n,
  "chunk_len": L,
  "num_chunks": K,
  "global_root": "...hex..."
}

// bef_pc_open_v1
{
  "schema": "bef_pc_open_v1",
  "index": i,
  "value": v_i,
  "chunk_index": k,
  "chunk_offset": k*L,
  "leaf_pos": j,
  "leaf_path": ["...", ...],
  "chunk_root": "...",
  "chunk_pos": k,
  "chunk_root_path": ["...", ...]
}
```

The commitment is a two-level Merkle tree:

1. Each chunk (values[k*L : (k+1)*L]) has its own Merkle tree, with leaves hashed as
   `H(offset || local_idx || value)`.
2. Chunk roots are hashed into a top-level Merkle tree whose root is `global_root`.

`stc_pc_cli.py` demonstrates the flow:

```
# Commit a trace
python scripts/stc_pc_cli.py commit-trace code/traces/vm_demo.json pc_commit.json

# Produce an opening using the original trace data
python scripts/stc_pc_cli.py open code/traces/vm_demo.json pc_commit.json 42 pc_open.json

# Verify using only the commitment + opening
python scripts/stc_pc_cli.py verify pc_commit.json pc_open.json
```

The opening size is `O(log L + log K)` hashes, and verification requires only the
commitment + proof. These APIs are the building blocks for the STC+FRI PC used
in the Σ+FS backend.



---

# docs/archive/stc_vm_mapping.md

<a name="docs-archive-stc_vm_mappingmd"></a>

# VM Trace → STC Encoding

`scripts/build_vm_trace.py` emits a toy VM trace following `bef_trace_v1`. Each
step packs `(pc, opcode, gas, accumulator)` into one field element via

```
value = (pc << 45) | (opcode << 37) | (gas << 25) | acc
```

The script writes `code/traces/vm_demo.json` with `chunk_length=4`. Running

```
python scripts/stc_aok.py prove-trace code/traces/vm_demo.json code/sketches/vm_demo_sketch.json --num-challenges 4 --chunk-len 4
```

produces a CPU-only sketch that matches the GPU layout. `python scripts/stc_aok.py verify code/sketches/vm_demo_sketch.json` runs the pure-Python `verify_fast` to
demonstrate end-to-end trace → STC → verification without CUDA.

## Geometry AIR trace mapping

The real geometry VM rows include more state. Each row is converted into a
length `10 + 2m` vector before feeding the STC accumulator:

```
[pc, opcode, gas, acc, x1, x2, cnt, m11, m12, m22, s[0..m-1], pow[0..m-1]]
```

`air/geom_trace_export.py` exposes helpers to flatten rows and emit a
`bef_trace_v1` JSON with `chunk_length = 10 + 2m`. Running

```
python scripts/export_geom_trace.py --steps 64 --num-challenges 2 --output nova_stc/examples/geom_trace.json
```

produces a real trace chunk file aligned with the STC row backend (each chunk =
one row). This is the format consumed by the `nova_stc` crate so the Nova proof
alibis the exact rows used in the GeomAIR proof.



---

# docs/archive/streaming_trace_commitment_formal.md

<a name="docs-archive-streaming_trace_commitment_formalmd"></a>

Streaming Trace Commitment (HSSA) — Formal Definition and Error Bounds

0) Streaming Trace Commitment (STC) — Abstract Definition
- Algorithms: (Setup, Init, Update, Commit, Open, VerifyOpen, GlobalCheck) over traces v ∈ F^n.
- Setup(1^λ): outputs pp with field F (|F|=p), CRH H, chunk bound L, sketch count m.
- Init(pp): returns initial state st0.
- Update(pp, st, chunk): deterministically updates state with chunk values; returns st′.
- Commit(pp, st): returns commitment C and public metadata (e.g., n, r⃗, s⃗).
- Open(pp, st, i): returns (val_i, π_i) proving v[i] under C.
- VerifyOpen(pp, C, i, val_i, π_i): deterministic check of opening.
- GlobalCheck(pp, C, meta): probabilistic fast check over per‑chunk metadata (offset/length/root, pre‑shifted sketches) ensuring consistency with a unique underlying trace.

1) Model and Parameters
- Field: a prime field Fp with p = 2^61 − 1 (prototype), extendable to a SNARK‑friendly prime.
- Trace: a length‑N vector v ∈ Fp^N indexed from 0.
- Chunking: values are processed in variable‑length chunks (≤ L). The chunker exposes, per chunk t, a Merkle root root_chunk_t over the chunk values (or H(len || values)).
- Hash: H is a collision‑resistant hash (CRH) with domain separation; we use SHA‑256 in the prototype.
- Challenges: m ≥ 1 Fiat–Shamir challenges r0,…,r{m−1} ∈ Fp\{0}, derived from the transcript (random oracle/ROM idealization).

2) Algorithms (HSSA construction)
Inputs shared across algorithms: public params pp = (p, H, m, L, domain tags).

Init()
- state.len ← 0
- state.root ← H("bef-init" || context)
- For j∈[0..m): r_j ← FE(H(state.root || "bef-challenge" || j)) with rejection sampling for 0
- state.s_j ← 0, state.pow_j ← 1 for all j
- Output state S0 = (len, root, {r_j}, {s_j}, {pow_j})

Update(S, chunk)
- Let ℓ = |chunk|, offset = S.len
- Compute root_chunk = MerkleRoot(chunk) (or H(offset || chunk_bytes))
- S.root ← H(S.root || "bef-chunk" || encode(offset) || root_chunk)
- For j in 0..m−1:
  - s ← S.s_j; pow ← S.pow_j; r ← S.r_j
  - For x in chunk (in order): s ← s + x·pow; pow ← pow·r
  - S.s_j ← s; S.pow_j ← pow
- S.len ← offset + ℓ
- Output updated S′

Commit(S)
- Output commitment com = (len=S.len, root=S.root, challenges={r_j}, sketches={s_j})

Open(v, i, proofs)
- Provide opening at index i: value = v[i], chunk index t containing i, Merkle path leaf→root_chunk_t, and path root_chunk_t→root using the hashed update chain (or a top tree).

VerifyOpen(com, i, value, paths)
- Check Merkle inclusion of value at position i inside its chunk to root_chunk_t.
- Recompute the hash chain root′ by iterating Update’s root accumulation across chunk roots up to t.
- Accept if root′ equals com.root and the path checks succeed.

GlobalCheck(com, summaries)
- summaries = list of per‑chunk metadata {offset_t, length_t, root_chunk_t, sketch_vec_t[0..m)}
- 1) Coverage: sort by offset; ensure they are contiguous, start at 0, sum of lengths equals com.len, and have canonical indices.
- 2) Root recompute: root′ ← H("bef-init"); for each t in order: root′ ← H(root′ || encode(offset_t) || root_chunk_t). Require root′ = com.root.
- 3) Sketch aggregate: for each j, set Ŝ_j ← Σ_t sketch_vec_t[j] and require Ŝ = com.sketches.
- Accept iff all hold. Note: per‑chunk sketches are expected to be pre‑shifted by r_j^{offset_t}. If not, the verifier multiplies each per‑chunk sketch by r_j^{offset_t} before summing.

Correctness
- Honest Init/Update/Commit/Open/VerifyOpen/GlobalCheck accept with probability 1.

3) Security Goals

Binding / Trace Integrity (informal)
- Assuming H is collision‑resistant and the transcript‑derived r_j behave as independent random elements in Fp (ROM), the adversary cannot produce two different traces v≠v′ together with proofs that both (a) pass VerifyOpen for all queried indices and (b) pass GlobalCheck for the same commitment com, except with negligible probability. Intuition: Merkle binding fixes chunk contents and order; linear sketches give a global equality check keyed by r⃗.

Global Sketch Soundness (schwartz–zippel bound)
- Let e = v′ − v ∈ Fp^N be the error vector between two traces with the same length and chunk structure. Define the error polynomial in r:
  E(r) = Σ_{i=0}^{N−1} e[i]·r^i ∈ Fp[r]
- If e ≠ 0, then deg(E) ≤ N−1 and the Schwartz–Zippel lemma yields:
  Pr_{r←Fp\{0}}[E(r) = 0] ≤ deg(E)/(p−1) ≤ (N−1)/(p−1)
- With m independent challenges r0,…,r_{m−1} (via ROM/Fiat‑Shamir with domain separation), the failure probability is at most:
  Pr[∀j, E(r_j)=0] ≤ ((N−1)/(p−1))^m
- If the adversary modifies only a δ fraction of coordinates (|supp(e)| = δN), then deg(E) ≤ N−1 still, and the bound remains ≤ ((N−1)/(p−1))^m. Using a coarser but interpretable bound: ≤ (δN/(p−1))^m when the error support is sized δN.

Instantiation Numbers (prototype)
- p ≈ 2.3058×10^18 (2^61−1). For N = 10^7 and m = 4:
  - (N−1)/(p−1) ≈ 1.0×10^7 / 2.3×10^18 ≈ 4.3×10^−12
  - Failure ≤ (4.3×10^−12)^4 ≈ 3.4×10^−46
- Thus even small m provides extremely strong global integrity detection at large N.

4) Remarks on Fiat–Shamir and Independence
- The prototype derives r_j by hashing the initial root with a per‑index domain tag. In the ROM/QROM model, this is modeled as independent uniform samples in Fp\{0}. If a standard model instantiation is required, sample r_j from an explicit PRF keyed by a public seed or from on‑chain randomness.

5) Notes on Verify_fast vs Full Replay
- Verify_fast (GlobalCheck) guarantees internal consistency of (len, root, {r_j}, {s_j}) with the provided chunk summaries without touching raw values. If chunk summaries themselves are adversarial, Verify_fast can accept only if they collude to match both the hash chain and the sketch sums; this is prevented when summaries are computed by trusted GPU workers or are cross‑checked via sampling and Merkle openings (Open/VerifyOpen) against the raw data.

6) Parameter Guidance
- m (sketches): 2–8 in practice; increases detection confidence exponentially.
- Chunk size L: balances Merkle overhead vs streaming overlap; 1–8k values typical; does not affect bounds.
- Field p: any large prime; 64‑bit Mersenne prime is fast for GPUs; SNARK primes align with IVC backends.

7) API Summary Snapshot (for implementers)
- Commit(v) → (p, len, root, r⃗, s⃗) and optional per‑chunk sketch vectors (pre‑shifted by r^offset).
- Open(i) → (v[i], Merkle paths) to root; independent of sketches.
- GlobalCheck(commit, summaries) → {True, False} (O(K·m) time for K chunks).

8) Proof Sketches (binding)
- If an adversary outputs com for two different traces v≠v′ such that both pass GlobalCheck with their own (consistent) chunk roots and both sets of chunk roots hash to the same com.root, then either (a) H is broken (collision on the root chain), or (b) the sketch vectors match on all m challenges, implying E(r_j)=0 for all j which, by the bound above, occurs with probability ≤ ((N−1)/(p−1))^m. Therefore the scheme is binding except with negligible probability under CRH+ROM.

9) Alternative Security Plug-In (n_max bound and combined advantage)
- For a system-level maximum trace length n_max, the sketch soundness advantage obeys
  Adv_sketch(λ) ≤ ((n_max−1)/p)^m.
- With a b-bit CRH (e.g., SHA-256 truncated to 128 bits), the overall binding advantage satisfies
  Adv_bind(λ) ≤ 2^{−b} + ((n_max−1)/p)^m.
- Example (coarse, round-number framing): p=2^61−1, n_max=2^32, m=4 ⇒ ((n_max−1)/p)^m ≈ (2^{−29})^4 = 2^{−116}.
  With a 128-bit CRH this yields ≈ 2^{−116} + 2^{−128} total binding failure probability.

10) Derived Zero-Knowledge Protocols Using STC
- HSSA–STC itself is intentionally transparent: (C,π) encodes sketch vectors and chunk metadata and is therefore distinguishable for different traces. This is the desired behavior for data-availability and auditability.
- To obtain privacy, treat the STC state as a *public input* to a separate zkSNARK/zkSTARK for a richer relation R_geom (e.g., “there exists an execution trace of an AIR that yields Σ and updates STC state S_old→S_new”).
- The zk layer commits to masked polynomial/codeword evaluations only. In our prototype this means:
  * Deriving per-column masks and a composition-polynomial mask via Fiat–Shamir.
  * Committing (via STC) to the masked vectors and running an FRI-style IOP on top.
  * During verification, subtracting the same deterministic masks before checking that the openings satisfy the AIR constraints.
- This composition yields: STC as a post-quantum streaming VC / AoK primitive, plus an outer transparent zk argument establishing that the committed trace satisfies the intended transition system while leaking only the public invariants (Σ, STC endpoints, parameters). The simulator for the zk layer never needs to break STC; it only programs the random oracle for the masking polynomials.

