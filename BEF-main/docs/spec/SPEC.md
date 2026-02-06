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
