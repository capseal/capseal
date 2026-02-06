# BICEP Transition Spec v2 (Audited State)

**Status**: FROZEN
**Version**: `bicep_v2`
**Date**: 2026-01-27

Upgrades from `bicep_v1`:
- `state_root` = Merkle root over quantized state tensor (not toy hash)
- Deterministic audit challenges (Fiat-Shamir-lite)
- Audit openings sidecar for verifiable state transitions
- Range-based RNG addressing for scalable randomness

---

## 1. What This Spec Adds

| Capability | v1 | v2 |
|------------|----|----|
| Output correctness | x_t_q verified | x_t_q verified |
| Output chain | H(chain \|\| x_t_q) | H(chain \|\| x_t_q) |
| State binding | Opaque hash | **Merkle root over quantized state** |
| State transition | Trusted producer | **Audited on sampled leaves** |
| RNG addresses | Per-element list | **Range records with layout** |
| Challenge model | None | **Deterministic (Fiat-Shamir-lite)** |

---

## 2. State Tensor and Merkle Root

### 2.1 Quantized State Tensor

The SDE state at timestep t is a tensor: `state_t[path, channel]`.

For commitment, this tensor is:
1. Flattened to a 1D array using a **canonical layout**
2. Each element quantized to fixed-point integer
3. Committed via Merkle tree

```python
state_q = [quantize(state[i]) for i in canonical_order(state)]
```

### 2.2 Canonical Layout

| Field | Value |
|-------|-------|
| `state_layout_id` | `flat_path_channel_v1` |

Leaf index formula:
```
leaf_index(path_id, channel_id) = path_id * num_channels + channel_id
```

Total leaves = `num_paths * num_channels`

Layout is frozen in manifest. Verifier can compute any leaf's semantic meaning from its index.

### 2.3 Merkle Root

```
state_root_t = MerkleRoot(state_q_t)
```

Merkle tree:
- Leaves: SHA256 of canonical encoding of each quantized element
- Internal: `H(left || ":" || right)` (same as checkpoint Merkle)
- Padded to next power of 2 with zero-digest leaves

---

## 3. Deterministic Audit Challenges

### 3.1 Challenge Seed

For each step t, derive challenge seed from committed data:

```python
challenge_seed_t = SHA256(
    manifest_hash || head_t || state_root_t || rng_use_hash_t
)
```

Where:
- `manifest_hash`: from manifest
- `head_t`: hash chain head after step t-1 (or genesis head for t=0)
- `state_root_t`: pre-state root at step t
- `rng_use_hash_t`: from row t

### 3.2 Index Sampling

```python
def sample_audit_indices(challenge_seed: bytes, num_leaves: int, k: int) -> list[int]:
    """Sample k distinct leaf indices from challenge seed."""
    indices = set()
    counter = 0
    while len(indices) < k and len(indices) < num_leaves:
        addr = challenge_seed + counter.to_bytes(4, 'big')
        h = SHA256(addr)
        idx = int.from_bytes(h[:4], 'big') % num_leaves
        indices.add(idx)
        counter += 1
    return sorted(indices)
```

This is Fiat-Shamir-lite: the executor can't predict which leaves
get checked without knowing the committed head.

### 3.3 Audit Count

| Field | Value |
|-------|-------|
| `audit_k` | Frozen in manifest (default: 16) |

Higher k = more confidence, more proof data.

---

## 4. Audit Openings Sidecar

### 4.1 Structure

Per-step audit file: `audit_step_{t:04d}.json`

```json
{
  "schema": "bicep_audit_v1",
  "t": 0,
  "challenge_seed": "<64 hex>",
  "audit_indices": [3, 17, 42, ...],
  "openings_pre": [
    {
      "leaf_index": 3,
      "value_q": 1234567,
      "merkle_proof": [["<sibling_hash>", "L"], ...]
    },
    ...
  ],
  "openings_post": [
    {
      "leaf_index": 3,
      "value_q": 1234890,
      "merkle_proof": [["<sibling_hash>", "L"], ...]
    },
    ...
  ]
}
```

### 4.2 What the Verifier Checks

For each audited leaf index `idx`:

1. **Merkle opening**: Verify `openings_pre[idx]` against `view_pre.state_root`
2. **Merkle opening**: Verify `openings_post[idx]` against `view_post.state_root`
3. **Transition**: Recompute the SDE step for that leaf:
   ```
   expected_post_q = quantize(step_sde(
       dequantize(pre_value_q),
       params,
       rng.rand(tag, t, leaf_rng_index(idx))
   ))
   assert openings_post[idx].value_q == expected_post_q
   ```
4. **Challenge correctness**: Verify `audit_indices == sample_audit_indices(challenge_seed, ...)`
5. **Challenge seed**: Verify `challenge_seed == H(manifest_hash || head_t || state_root_t || rng_use_hash_t)`

---

## 5. SDE Step Function

### 5.1 Euler-Maruyama Step

| Field | Value |
|-------|-------|
| `transition_fn_id` | `bicep_sde_em_v1` |

For each state element at leaf index `idx`:

```python
def sde_step(state_pre_q: int, params: dict, eps: float,
             precision_bits: int) -> int:
    """Single Euler-Maruyama step, quantized."""
    state_pre = dequantize(state_pre_q, precision_bits)

    theta = params["theta"]
    mu0 = params["mu0"]
    sigma = params["sigma"]
    dt = params["dt"]

    # OU drift + diffusion
    drift = theta * (mu0 - state_pre) * dt
    diffusion = sigma * math.sqrt(dt) * eps

    state_post = state_pre + drift + diffusion
    return quantize(state_post, precision_bits)
```

### 5.2 Jump Process (Optional)

If `jump_rate > 0` in manifest params:

```python
def apply_jump(state_post_q: int, jump_flag: bool, jump_mag: float,
               precision_bits: int) -> int:
    if not jump_flag:
        return state_post_q
    state = dequantize(state_post_q, precision_bits)
    state += jump_mag
    return quantize(state, precision_bits)
```

---

## 6. RNG Range Addressing

### 6.1 Range Record

```json
{
  "tag": "sde_noise",
  "t": 5,
  "i_start": 0,
  "i_count": 256,
  "layout_id": "flat_path_channel_v1"
}
```

### 6.2 Leaf-to-RNG Index Mapping

For a given leaf index `idx`, the corresponding RNG address is:

```python
def leaf_rng_index(tag: str, t: int, idx: int) -> tuple[str, int, int]:
    """Map leaf index to RNG address (tag, t, i)."""
    return (tag, t, idx)
```

For antithetic sampling (`antithetic_v1`):
```python
def leaf_rng_antithetic(tag: str, t: int, idx: int, num_base: int):
    """Antithetic: first half are base, second half are negated."""
    if idx < num_base:
        return rng.rand(tag, t, idx), 1.0   # base epsilon
    else:
        return rng.rand(tag, t, idx - num_base), -1.0  # negated
```

### 6.3 rng_use_hash

Same as v1: `H(canonical(rand_addrs))`, but `rand_addrs` now contains range records.

---

## 7. Row Schema (Updated)

```json
{
  "schema": "bicep_trace_v2",
  "t": 0,
  "view_pre": {
    "state_root": "<64 hex>",
    "output_chain": "<64 hex>"
  },
  "view_post": {
    "state_root": "<64 hex>",
    "output_chain": "<64 hex>"
  },
  "x_t_q": [16549339, 9195638, ...],
  "x_t_f": [0.986, 0.548, ...],
  "rand_addrs": [
    {"tag": "sde_noise", "t": 0, "i_start": 0, "i_count": 256, "layout_id": "flat_path_channel_v1"},
    {"tag": "jump_mask", "t": 0, "i_start": 0, "i_count": 256, "layout_id": "flat_path_channel_v1"}
  ],
  "rng_use_hash": "<64 hex>"
}
```

Note: Audit openings are in a **sidecar**, not in the row.

---

## 8. Manifest Fields

```json
{
  "transition_spec_id": "bicep_v2",
  "state_view_schema_id": "bicep_state_v2",
  "transition_fn_id": "bicep_sde_em_v1",
  "output_fn_id": "bicep_features_v1",

  "rng_id": "hmac_sha256_v1",
  "domain_sep_scheme_id": "tag_t_i_v1",
  "sampling_scheme_id": "standard_v1",

  "x_quant_scheme_id": "fixed_point_v1",
  "x_quant_precision_bits": 24,

  "state_layout_id": "flat_path_channel_v1",
  "state_num_paths": 64,
  "state_num_channels": 4,

  "sde_params": {
    "theta": 2.0,
    "mu0": 0.0,
    "sigma": 0.5,
    "dt": 0.01,
    "jump_rate": 0.0,
    "jump_scale": 0.0
  },

  "audit_k": 16,

  "seed_commitment": "<64 hex>"
}
```

---

## 9. Verification Checklist

### Level 0: Integrity (unchanged from v1)
- Row digest chain
- Checkpoint Merkle roots

### Level 1: Output Correctness (unchanged from v1)
- `rng_use_hash == H(canonical(rand_addrs))`
- `output_chain == H(prev_chain || canonical(x_t_q))`
- `x_t_q == quantize(compute_features(...))`
- Continuity: `view_post[t] == view_pre[t+1]`

### Level 1.5: State Audit (NEW)
- `challenge_seed == H(manifest_hash || head_t || state_root_t || rng_use_hash_t)`
- `audit_indices == sample(challenge_seed, num_leaves, k)`
- For each audited index:
  - Merkle opening verified against `state_root_t`
  - Merkle opening verified against `state_root_{t+1}`
  - `post_value_q == sde_step(pre_value_q, params, rng[idx])`

### Level 2: Global Soundness (future)
- STARK proof that Level 1.5 holds for ALL leaves (not just k)

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| `bicep_v1` | 2026-01-27 | Quantized outputs, output_chain, rng_use_hash |
| `bicep_v2` | 2026-01-27 | Merkle state_root, audit openings, SDE transition check |
