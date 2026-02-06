# Critical Security Action Items

**Source**: INSTRUCTIONS.txt security audit  
**Framework**: Katz-Lindell cryptographic security methodology  
**Priority**: High - These are "will get killed in review" issues

---

## 🔴 CRITICAL Priority (Fix Immediately)

### 1. Data Availability Challenge Must Be Relay-Issued

**Current Problem**: Prover generates `verifier_nonce` locally  
**Attack**: Prover can grind nonces until sample indices avoid bad chunks  
**Security Impact**: DA audit is defeated, cheating probability ≈ 1

**Required Fix**:
```python
# WRONG (current):
nonce = secrets.token_bytes(32)  # prover-controlled
challenge = {"verifier_nonce": nonce.hex(), ...}

# CORRECT (required):
# 1. Prover commits capsule_commit_hash to relay
# 2. Relay generates nonce and signs challenge
# 3. Prover includes signed challenge in capsule
# 4. Verifier validates relay signature
```

**Implementation Checklist**:
- [ ] Create relay endpoint: `POST /v1/da/challenge`
- [ ] Relay returns signed `DAChallengeV1` with relay-generated nonce
- [ ] Update `run_pipeline.py` to call relay after computing commit hash
- [ ] Verifier must reject self-issued challenges
- [ ] Store relay public keys in `config/trusted_relays.json`

**Files to Modify**:
- `scripts/run_pipeline.py` - Add relay challenge fetch
- `bef_zk/capsule/cli/verify.py` - Add relay signature verification
- `server/` - Add relay challenge endpoint
- `bef_zk/da/` - Add relay challenge protocol

---

### 2. Verification Status Semantics Are Broken

**Current Problem**: System shows "VERIFIED" when only proof passes (no DA/policy checks)  
**User Impact**: Dashboards treat partial verification as full verification  
**Security Impact**: Users over-trust capsules that passed only basic checks

**Required Fix**:
```python
# Define explicit verification levels
class VerificationStatus(Enum):
    REJECTED = "rejected"               # Proof or header failed
    PROOF_ONLY = "proof_only"           # Proof passed, but no DA/policy
    POLICY_ENFORCED = "policy_enforced" # + policy + ACL checked
    FULLY_VERIFIED = "fully_verified"   # + DA audit + events

# NEVER show "VERIFIED" unless status == FULLY_VERIFIED
```

**Implementation Checklist**:
- [ ] Add `VerificationStatus` enum to `bef_zk/verifier_errors.py`
- [ ] Compute status from boolean predicates: `header_ok && proof_ok && policy_ok && da_ok && events_ok`
- [ ] Update CLI output to show precise status
- [ ] Update UI to only show green check for FULLY_VERIFIED
- [ ] Add `--require-level fully_verified` flag (default for CI)

**Files to Modify**:
- `bef_zk/capsule/cli/verify.py` - Status computation
- `bef_zk/verifier_errors.py` - Status enum
- `ui/src/App.jsx` - UI display logic
- `scripts/verify_capsule.py` - CLI output

---

### 3. Capsule Hash Excludes DA Challenge (Binding Failure)

**Current Problem**: `da_challenge` is excluded from capsule_hash computation  
**Attack**: Prover can attach different challenges to same capsule (equivocation)  
**Security Impact**: Same capsule_hash can have different DA audit results

**Required Fix** (Two Options):

**Option A - Two-Phase Hash (Recommended)**:
```python
# Phase 1: Commit (before challenge exists)
header_without_da = {...}  # All fields except da_ref
capsule_commit_hash = hash_with_prefix("CAPSULE_COMMIT_V1", header_without_da)

# Phase 2: Relay issues challenge bound to capsule_commit_hash
da_challenge = relay.issue_challenge(capsule_commit_hash)

# Phase 3: Final hash
header_with_da = {**header_without_da, "da_ref": {"challenge_hash": hash(da_challenge)}}
capsule_hash = hash_with_prefix("CAPSULE_FINAL_V1", header_with_da)
```

**Option B - Include DA Challenge Directly**:
```python
# Fetch relay challenge BEFORE computing capsule_hash
# Then include da_challenge in the hashed object
capsule_for_hash = {
    # ... all fields INCLUDING da_challenge
}
capsule_hash = hash_capsule(capsule_for_hash)
```

**Implementation Checklist**:
- [ ] Choose Option A or B based on relay timing constraints
- [ ] Update `hash_capsule()` in `bef_zk/capsule/header.py`
- [ ] If Option A: add `capsule_commit_hash` field
- [ ] Update pipeline to compute hashes in correct order
- [ ] Verifier must check da_ref.challenge_hash matches actual challenge

**Files to Modify**:
- `bef_zk/capsule/header.py` - Hash computation
- `scripts/run_pipeline.py` - Hash sequencing
- `bef_zk/capsule/cli/verify.py` - Validation

---

## 🟡 HIGH Priority (Fix This Sprint)

### 4. Chunk Handles Are Location-Based, Not Content-Based

**Current Problem**: `chunk_handles_root` commits to URI strings like `"r2://..."`  
**Attack**: Storage provider swaps content at same URI (equivocation)  
**Security Impact**: Different verifiers see different data for same capsule

**Required Fix**:
```python
# WRONG (current):
chunk_handles = [str(handle) for handle in handles]
handles_root = merkle_root(chunk_handles)

# CORRECT (required):
chunk_manifest = [
    {
        "id": i,
        "uri": "r2://bucket/chunk_42.bin",
        "sha256": "abc123...",  # MUST be included
        "size": 1024,
        "content_type": "application/octet-stream"
    }
    for i, handle in enumerate(handles)
]
chunk_manifest_hash = hash_with_prefix("CHUNK_MANIFEST_V1", canonical_encode(chunk_manifest))
```

**Verifier Rule**:
```python
# Before parsing any chunk:
downloaded_data = fetch(uri)
assert len(downloaded_data) <= declared_size
assert sha256(downloaded_data) == declared_sha256
# THEN proceed with Merkle verification
```

**Implementation Checklist**:
- [ ] Define `ChunkManifestV1` schema with required fields
- [ ] Update packer to compute content hashes for all chunks
- [ ] Store manifest as `chunk_manifest.json` in capsulepack
- [ ] Update `hash_chunk_handles()` to hash manifest, not strings
- [ ] Verifier must validate content hash before parsing

**Files to Modify**:
- `bef_zk/capsule/payload.py` - Chunk handle structure
- `capsule_bench/packing.py` - Manifest generation
- `bef_zk/capsule/cli/verify.py` - Content verification

---

### 5. Missing Verifier Key Binding (Instance Confusion)

**Current Problem**: No `vk_hash`, `air_params_hash`, `fri_params_hash` in capsule  
**Attack**: Verifier uses wrong parameters; prover targets weakest backend  
**Security Impact**: Proof verifies but for different security level or instance

**Required Fix**:
```python
# Add to CapsuleHeader:
{
    "proof_system": {
        "scheme_id": "geom_stc_fri_v1",
        "vk_hash": "sha256:...",           # Hash of verifier key bytes
        "air_params_hash": "sha256:...",    # Hash of AIR configuration
        "fri_params_hash": "sha256:...",    # Hash of FRI parameters
        "field_id": "goldilocks",
        "hash_fn_id": "sha256"
    }
}
```

**Canonical Encoding Example** (air_params):
```python
def hash_air_params(params: GeomAIRParams) -> str:
    canonical = {
        "steps": params.steps,
        "num_challenges": params.num_challenges,
        "field_modulus": params.field_modulus,
        "air_version": 1,
        # ... all security-relevant params
    }
    return hash_with_prefix("GEOM_AIR_PARAMS_V1", dag_cbor.encode(canonical))
```

**Implementation Checklist**:
- [ ] Define canonical encoding for `GeomAIRParams`, `FRIConfig`, verifier key
- [ ] Add `proof_system` section to `CapsuleHeaderV2`
- [ ] Compute hashes during pipeline setup
- [ ] Include hashes in capsule_hash preimage
- [ ] Verifier reconstructs hashes and validates match

**Files to Modify**:
- `bef_zk/capsule/header.py` - Add proof_system fields
- `backends/geom_adapter.py` - Compute canonical param hashes
- `bef_zk/fri/` - FRI config canonicalization

---

### 6. Absolute Paths in Committed Data

**Current Problem**: `provider_root` may contain absolute filesystem paths  
**Attack**: Verifier file read gadget; non-portable commitments  
**Security Impact**: Capsule binds to specific machine; potential path traversal

**Required Fix**:
```python
# REMOVE from security-critical structures:
# - row_index_ref.pointer.provider_root
# - da_policy.provider.archive_root
# - Any absolute paths in manifests

# REPLACE with:
# - Pack-relative paths only: "chunk_archive/chunk_0.bin"
# - Content addressing: reference by SHA256
# - Sandbox enforcement at artifact resolution time

def resolve_artifact(path: str, pack_root: Path) -> Path:
    if path.is_absolute():
        raise SecurityError("Absolute paths forbidden")
    if ".." in path.parts:
        raise SecurityError("Path traversal forbidden")
    resolved = (pack_root / path).resolve()
    if not resolved.is_relative_to(pack_root):
        raise SecurityError("Path escapes pack root")
    return resolved
```

**Implementation Checklist**:
- [ ] Audit all path fields in CapsuleHeader, RowIndexRef, DAPolicy
- [ ] Remove or make non-security-critical any absolute path fields
- [ ] Implement strict path sandboxing in `_resolve()` and `_ensure_local_artifact()`
- [ ] Add tests for path traversal attempts
- [ ] Document that provider_root is verifier-local config only

**Files to Modify**:
- `bef_zk/capsule/payload.py` - Remove provider_root from committed structures
- `bef_zk/capsule/cli/verify.py` - Path sandboxing
- `capsule_bench/artifacts.py` - Artifact resolution

---

## 🟢 MEDIUM Priority (Next Sprint)

### 7. Canonicalization Must Be Fully Specified

**Current**: Using dag-cbor (good!), but schema discipline is loose  
**Risk**: Unknown fields, optional field ambiguity, alternate representations

**Fix**:
- [ ] Strict schema validation: reject unknown fields
- [ ] Define presence rules for optional fields (omit vs null)
- [ ] Include schema_version in all hashed objects
- [ ] Enforce minimum acceptable versions in verifier

---

### 8. Domain Separation Tags

**Current**: Some tags exist (`"bef-init"`, `"bef-chunk"`), but not systematic  
**Fix**: Use tags everywhere:
- `"CAPSULE_COMMIT_V1"`, `"CAPSULE_FINAL_V1"`
- `"DA-SEED"`, `"DA-CHALv1"`
- `"STATEMENT-HASH"`, `"VK_V1"`

---

### 9. Event Log Binding

**Current**: `event_chain_head` and `events_log_hash` are optional  
**Fix**: If verification level requires events, enforce:
- [ ] `events_log_hash` present in header
- [ ] `event_chain_head` computed from canonical events
- [ ] Both included in capsule_hash
- [ ] Verifier recomputes and validates

---

## Implementation Timeline

**Week 1** (Immediate):
- [ ] Item #1: Relay-issued DA challenges (design + mock endpoint)
- [ ] Item #2: Verification status levels (enum + UI changes)

**Week 2** (Critical Path):
- [ ] Item #1: Complete relay challenge integration + testing
- [ ] Item #3: Two-phase capsule hash (commit → challenge → final)

**Week 3** (High Priority):
- [ ] Item #4: Content-addressed chunk handles
- [ ] Item #5: Verifier key binding

**Week 4** (Cleanup):
- [ ] Item #6: Path sandboxing
- [ ] Items #7-9: Canonicalization, domain separation, event binding

---

## Testing Requirements

For each fix, test:
1. **Attack prevented**: Original vulnerability no longer exploitable
2. **Backward compat**: Legacy capsules handled gracefully (with warnings)
3. **Error messages**: Clear, actionable error codes
4. **Performance**: No significant regression

---

## Review Criteria (Will This Pass Crypto Review?)

Before claiming "fixed", ensure:
- [ ] Can write formal security game for each property
- [ ] Assumptions are explicitly stated
- [ ] No circular dependencies in hash computation
- [ ] Verifier has no ambient authority (no file system escapes)
- [ ] Randomness sources are clearly identified and justified
- [ ] "VERIFIED" status has precise, documented meaning

---

## Reference

See `INSTRUCTIONS.txt` for detailed cryptographic analysis and `MERGE_RESOLUTION_GUIDE.md` for full context.
