# Transition Spec Contract v1

**Status**: FROZEN
**Version**: `transition_spec_v1`
**Date**: 2026-01-27

This document defines the determinism contract for BICEP trace verification.
Once frozen, changes require a new version ID.

---

## 1. Determinism Contract

### 1.1 PRG/PRF Algorithm

| Field | Value |
|-------|-------|
| `rng_id` | `hmac_sha256_v1` |
| Algorithm | HMAC-SHA256 |
| Input | `HMAC(seed, address)` where `address = tag_bytes || t_be64 || i_be64` |
| Output | First 8 bytes as big-endian uint64, normalized to [0, 1) by dividing by 2^64 |

**Byte Order**: Big-endian for all integer serialization in PRG addresses.

**Domain Separation**: Tag is UTF-8 encoded, followed by timestep `t` and index `i` as 64-bit big-endian unsigned integers.

```
address = encode_utf8(tag) || be64(t) || be64(i)
rand(tag, t, i) = unpack_be64(HMAC_SHA256(seed, address)[:8]) / 2^64
```

### 1.2 Domain Separation Scheme

| Field | Value |
|-------|-------|
| `domain_sep_scheme_id` | `tag_t_i_v1` |
| Format | `{tag: string, t: uint64, i: uint64}` |

Reserved tags:
- `"input"` - input feature generation
- `"noise"` - noise injection
- `"dropout"` - dropout mask
- `"sample"` - sampling operations
- `"state"` - state-dependent randomness

### 1.3 Numeric Model

| Field | Value |
|-------|-------|
| `numeric_model` | `float64_ieee754_v1` |
| Representation | IEEE 754 double precision |
| Rounding | Round-to-nearest-ties-to-even (default) |
| Special values | NaN and Inf are FORBIDDEN in trace values |

**Quantization for commitment** (integrity layer only):
- Scale: 2^24 (24 bits of precision)
- Quantized value: `int(round(value * 2^24))`

**Note**: For future STARK verification, we will define a fixed-point model.
The float model is acceptable for replay verification but not for algebraic proofs.

---

## 2. State View Schema

### 2.1 What `view_pre` and `view_post` mean

| Field | Value |
|-------|-------|
| `state_view_schema_id` | `minimal_projection_v1` |
| Definition | The **minimal projection** of internal state needed to verify the transition |

**Semantic contract**:
- `view_pre[t]` contains the state BEFORE step t executes
- `view_post[t]` contains the state AFTER step t executes
- `view_post[t]` MUST equal `view_pre[t+1]` (continuity invariant)

**Required fields** (for `minimal_projection_v1`):
```json
{
  "view_pre": {
    "state": <int>,           // discrete state index
    "value": <float>          // continuous state value (optional)
  },
  "view_post": {
    "state": <int>,
    "value": <float>
  }
}
```

### 2.2 Transition Function Contract

| Field | Value |
|-------|-------|
| `transition_fn_id` | `identity_v1` |
| Definition | `s_{t+1} = s_t + 1` (trivial for now) |

For `identity_v1`:
```
view_post.state = view_pre.state + 1
```

**Future versions** will define actual BICEP transition functions.

### 2.3 Output Function Contract

| Field | Value |
|-------|-------|
| `output_fn_id` | `rng_features_v1` |
| Definition | `x_t[i] = rng.rand("input", t, i)` |

For `rng_features_v1`:
```
x_t = [rng.rand("input", t, i) for i in range(dim)]
```

---

## 3. Canonicalization Boundary

### 3.1 Semantic vs Metadata Fields

**Semantic fields** (used in constraint verification):
- `t` - timestep
- `x_t` - output features
- `view_pre` - state before
- `view_post` - state after
- `rand_addrs` - RNG addresses consumed

**Metadata fields** (not used in constraints):
- `schema` - version identifier
- `aux` - auxiliary data for debugging

### 3.2 Canonical Encoding for Constraint Values

| Type | Encoding |
|------|----------|
| Integer | JSON number (no quotes) |
| Float | JSON number, max 15 significant digits |
| Array | JSON array, elements in order |
| String | JSON string, UTF-8 |

**Hash computation**: Always use canonical JSON (sorted keys, no whitespace).

---

## 4. Verification Levels

### Level 0: Integrity (IMPLEMENTED)
- Trace commitment (hash chain)
- Checkpoint Merkle roots
- Sidecar binding
- Independent verifier passes

### Level 1: Local Validity (THIS SPEC)
- Transition constraint: `view_post[t] = f(view_pre[t], r_t, params)`
- Output constraint: `x_t = g(view_pre[t], r_t)`
- RNG consistency: addresses match manifest PRG spec
- Continuity: `view_post[t] == view_pre[t+1]`

### Level 2: Global Soundness (FUTURE)
- STARK/IVC proof that Level 1 holds for entire trace
- Succinct verification without replay

---

## 5. Manifest Fields Required

```json
{
  "transition_spec_id": "transition_spec_v1",
  "rng_id": "hmac_sha256_v1",
  "domain_sep_scheme_id": "tag_t_i_v1",
  "numeric_model": "float64_ieee754_v1",
  "state_view_schema_id": "minimal_projection_v1",
  "transition_fn_id": "identity_v1",
  "output_fn_id": "rng_features_v1"
}
```

---

## 6. Version History

| Version | Date | Changes |
|---------|------|---------|
| `transition_spec_v1` | 2026-01-27 | Initial frozen spec |
