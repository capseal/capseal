# BICEP Transition Spec v1

**Status**: FROZEN
**Version**: `bicep_v1`
**Date**: 2026-01-27

This is the production transition spec for BICEP traces.
It enforces determinism, quantization, and AIR-ready structure.

---

## 1. Design Principles

1. **Quantize at commitment boundary**: Everything hashed/compared is integer/fixed-point
2. **State as root + optional openings**: Per-partition digests rolled into single root
3. **RNG addresses, not tensors**: Log ranges + tags, verifier expands deterministically
4. **Sampling scheme in manifest**: Antithetic/standard is frozen, not runtime choice
5. **No unverifiable floats**: `x_t_f` is convenience only, derived from `x_t_q`

---

## 2. Row Schema

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | `"bicep_trace_v1"` |
| `t` | int | Timestep index (0-based) |
| `view_pre` | object | State before step |
| `view_post` | object | State after step |
| `x_t_q` | list[int] | **Canonical** quantized outputs (committed) |
| `x_t_f` | list[float] \| null | Optional float decode (NOT committed) |
| `rand_addrs` | list[object] | RNG addresses consumed |
| `rng_use_hash` | string | `H(canonical(rand_addrs))` |
| `aux_q` | object | Optional quantized auxiliary witness |

### 2.1 State View Schema

| Field | Value |
|-------|-------|
| `state_view_schema_id` | `bicep_state_v1` |

```json
{
  "view_pre": {
    "state_root": "<64 hex chars>",
    "output_chain": "<64 hex chars>"
  },
  "view_post": {
    "state_root": "<64 hex chars>",
    "output_chain": "<64 hex chars>"
  }
}
```

**state_root**: Root digest over partitioned state (channels, etc.)
- For simple case: `H(concat(partition_digests))`
- For selective opening: Merkle root over partition digests

**output_chain**: Running hash chain over quantized outputs
- Genesis: `H("bicep_v1:output_chain:genesis:" || seed_commitment)`
- Update: `H(prev_chain || canonical(x_t_q))`

---

## 3. Quantization Scheme

| Field | Value |
|-------|-------|
| `x_quant_scheme_id` | `fixed_point_v1` |

### 3.1 Parameters (frozen in manifest)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precision_bits` | 24 | Bits of precision |
| `scale` | 2^24 | Multiplier (derived from precision_bits) |
| `rounding` | `nearest_even` | IEEE 754 round-to-nearest-ties-to-even |
| `clamp_min` | -2^31 | Saturation floor |
| `clamp_max` | 2^31 - 1 | Saturation ceiling |

### 3.2 Encoding

```python
def quantize(value: float, precision_bits: int = 24) -> int:
    scale = 2 ** precision_bits
    raw = round(value * scale)  # nearest-even by default
    return max(-2**31, min(2**31 - 1, raw))

def dequantize(q_value: int, precision_bits: int = 24) -> float:
    scale = 2 ** precision_bits
    return q_value / scale
```

### 3.3 Canonical Encoding for Commitment

Quantized values are encoded as JSON integers (no quotes).
Array order is preserved. Canonical JSON (sorted keys, no whitespace).

```python
canonical(x_t_q) = json.dumps(x_t_q, separators=(',', ':')).encode('utf-8')
```

---

## 4. RNG Specification

| Field | Value |
|-------|-------|
| `rng_id` | `hmac_sha256_v1` |
| `domain_sep_scheme_id` | `tag_t_i_v1` |

### 4.1 Address Format

```json
{
  "tag": "sde_noise",
  "t": 5,
  "i_start": 0,
  "i_count": 256
}
```

Reserved tags:
- `"sde_noise"` - Euler-Maruyama diffusion noise
- `"jump_mask"` - Poisson jump indicators
- `"jump_mag"` - Jump magnitudes
- `"jump_time"` - Jump timing

### 4.2 rng_use_hash

```python
rng_use_hash = sha256(canonical_json(rand_addrs)).hexdigest()
```

This makes "what RNG was supposed to be used" tamper-evident.

### 4.3 Sampling Scheme

| Field | Value |
|-------|-------|
| `sampling_scheme_id` | `standard_v1` or `antithetic_v1` |

For `antithetic_v1`:
- Each base epsilon `e` expands to `(e, -e)` in paired order
- Halves PRG draws, doubles effective samples
- Verifier must know scheme to validate address counts

---

## 5. Transition Function

| Field | Value |
|-------|-------|
| `transition_fn_id` | `bicep_sde_v1` |

### 5.1 State Root Update

For `bicep_sde_v1` with simple concat root:

```python
# Compute new SDE state (deterministic given pre-state, params, RNG)
sde_state_new = step_sde(sde_state_pre, params, rng_values)

# Digest the new state
state_root_new = sha256(canonical_json(quantize_state(sde_state_new))).hexdigest()
```

**Constraint**:
```
view_post.state_root == expected_state_root(view_pre, params, rng)
```

### 5.2 Output Chain Update

```python
chain_new = sha256(
    view_pre.output_chain.encode() + canonical_json(x_t_q)
).hexdigest()
```

**Constraint**:
```
view_post.output_chain == H(view_pre.output_chain || canonical(x_t_q))
```

---

## 6. Output Function

| Field | Value |
|-------|-------|
| `output_fn_id` | `bicep_features_v1` |

### 6.1 Feature Extraction

From SDE paths, compute statistics:
- mean, std, q10, q90, aleatoric, epistemic (per channel or aggregated)

### 6.2 Quantization

All features are quantized before commitment:

```python
x_t_q = [quantize(f, precision_bits) for f in raw_features]
```

**Constraint**:
```
x_t_q == quantize(compute_features(sde_state_post, params, rng))
```

---

## 7. Genesis State

### 7.1 Initial State Root

```python
genesis_state_root = sha256(
    f"bicep_v1:state:genesis:{seed_commitment}".encode()
).hexdigest()
```

### 7.2 Initial Output Chain

```python
genesis_output_chain = sha256(
    f"bicep_v1:output_chain:genesis:{seed_commitment}".encode()
).hexdigest()
```

---

## 8. Verification Checklist

For `verify_transition_bicep_v1(row_t, manifest, rng)`:

1. **Timestep**: `view_post.t == view_pre.t + 1` (if t stored in view)
2. **rng_use_hash**: `row.rng_use_hash == H(canonical(row.rand_addrs))`
3. **Output chain**: `view_post.output_chain == H(view_pre.output_chain || canonical(x_t_q))`
4. **State root**: `view_post.state_root == expected_root(view_pre, params, rng)`
5. **Outputs**: `x_t_q == quantize(compute_features(...))`
6. **Continuity**: `row[t].view_post == row[t+1].view_pre`

---

## 9. Manifest Fields Required

```json
{
  "transition_spec_id": "bicep_v1",
  "state_view_schema_id": "bicep_state_v1",
  "transition_fn_id": "bicep_sde_v1",
  "output_fn_id": "bicep_features_v1",

  "rng_id": "hmac_sha256_v1",
  "domain_sep_scheme_id": "tag_t_i_v1",
  "sampling_scheme_id": "standard_v1",

  "x_quant_scheme_id": "fixed_point_v1",
  "x_quant_precision_bits": 24,

  "seed_commitment": "<64 hex>"
}
```

---

## 10. Auxiliary Witness (aux_q)

Optional quantized intermediate values for:
- Reducing verifier recomputation
- AIR witness columns

If present, verifier MUST check:
```python
aux_q["drift_q"] == quantize(compute_drift(view_pre, params))
aux_q["diff_q"] == quantize(compute_diffusion(params, rng))
```

Never trust unverified aux data.

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| `bicep_v1` | 2026-01-27 | Initial frozen spec |

---

## 12. Migration from keyed_hash_v1

| keyed_hash_v1 | bicep_v1 |
|---------------|----------|
| `view_pre.hash` | `view_pre.state_root` |
| `view_pre.t` | Removed (redundant with row.t) |
| `x_t` (floats) | `x_t_q` (quantized ints) |
| — | `rng_use_hash` (new) |
| — | `x_quant_scheme_id` (new) |
| — | `sampling_scheme_id` (new) |

The key upgrade: floats → quantized ints at commitment boundary.
