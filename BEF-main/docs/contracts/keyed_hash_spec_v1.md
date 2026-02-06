# Keyed Hash Transition Spec v1

**Status**: FROZEN
**Version**: `keyed_hash_v1`
**Date**: 2026-01-27

This is the first "real" transition spec beyond the toy `identity_v1`.
It demonstrates a verifiable state machine with cryptographic state evolution.

---

## 1. Overview

The keyed hash spec models a simple accumulator:
- State is a 256-bit hash that evolves by absorbing outputs
- Outputs are RNG-derived (same as `rng_features_v1`)
- Transition is deterministic and independently verifiable

This is the minimal spec that exercises:
- Non-trivial state evolution
- Cryptographic binding between state and outputs
- Deterministic verification without hidden dependencies

---

## 2. State Schema

### 2.1 State View Definition

| Field | Value |
|-------|-------|
| `state_view_schema_id` | `keyed_hash_state_v1` |

```json
{
  "view_pre": {
    "t": <int>,           // timestep (redundant but useful)
    "hash": <hex_string>  // 64-char hex (256 bits)
  },
  "view_post": {
    "t": <int>,
    "hash": <hex_string>
  }
}
```

### 2.2 Initial State

For `t=0`:
```
view_pre.hash = H("keyed_hash_v1:genesis:" || seed_commitment)
view_pre.t = 0
```

Where `seed_commitment` is from the manifest.

---

## 3. Transition Function

| Field | Value |
|-------|-------|
| `transition_fn_id` | `keyed_hash_absorb_v1` |

**Definition**:
```
state_{t+1}.hash = SHA256(state_t.hash || canonical_json(x_t))
state_{t+1}.t = t + 1
```

**Constraint** (what the verifier checks):
```python
expected_hash = sha256(view_pre["hash"] + canonical_json(x_t)).hexdigest()
assert view_post["hash"] == expected_hash
assert view_post["t"] == view_pre["t"] + 1
```

---

## 4. Output Function

| Field | Value |
|-------|-------|
| `output_fn_id` | `rng_features_v1` |

Same as `transition_spec_v1`:
```
x_t[i] = rng.rand("input", t, i)
```

---

## 5. Continuity Invariant

```
view_post[t].hash == view_pre[t+1].hash
view_post[t].t == view_pre[t+1].t
```

---

## 6. RNG Addresses

Same as `transition_spec_v1`:
```json
{
  "rand_addrs": [
    {"tag": "input", "t": <t>, "i": 0},
    {"tag": "input", "t": <t>, "i": 1},
    ...
  ]
}
```

---

## 7. Manifest Fields Required

```json
{
  "transition_spec_id": "keyed_hash_v1",
  "rng_id": "hmac_sha256_v1",
  "domain_sep_scheme_id": "tag_t_i_v1",
  "numeric_model": "float64_ieee754_v1",
  "state_view_schema_id": "keyed_hash_state_v1",
  "transition_fn_id": "keyed_hash_absorb_v1",
  "output_fn_id": "rng_features_v1",
  "seed_commitment": "<H(seed)>"
}
```

---

## 8. Why This Spec Matters

1. **Cryptographic state binding**: State hash chains outputs, making forgery hard
2. **No hidden dependencies**: Verifier needs only manifest + trace
3. **Deterministic**: Same inputs always produce same state evolution
4. **AIR-ready**: `view_post.hash = H(view_pre.hash || x_t)` is a local constraint

---

## 9. Verification Levels

### Level 0: Integrity
- Row digests chain correctly
- Checkpoint Merkle roots match

### Level 1: Local Validity (this spec)
- `view_post.hash == H(view_pre.hash || canonical_json(x_t))`
- `view_post.t == view_pre.t + 1`
- `x_t[i] == rng.rand("input", t, i)`
- Continuity: `view_post[t] == view_pre[t+1]`

### Level 2: Global Soundness (future)
- STARK proof that Level 1 holds for entire trace

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| `keyed_hash_v1` | 2026-01-27 | Initial frozen spec |
