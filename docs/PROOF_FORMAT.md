# CapSeal Proof Format

Version: 0.4.0

## Overview

CapSeal generates cryptographic proofs that an agent's execution trace
satisfies a set of algebraic constraints. The proof system uses an
Algebraic Intermediate Representation (AIR) over the Goldilocks field.

Source of truth: `capseal/agent_air.py`, `capseal/agent_constraints.py`

## AgentAIR Row Encoding

Each action is encoded as a row of 14 Goldilocks field elements
(p = 2^64 - 2^32 + 1):

| Element | Field | Encoding |
|---------|-------|----------|
| 0 | `action_index` | Sequential integer (0, 1, 2, ...) |
| 1 | `action_type_hash_lo` | SHA-256(action_type) bytes 0-7, little-endian u64 mod p |
| 2 | `action_type_hash_hi` | SHA-256(action_type) bytes 8-15, little-endian u64 mod p |
| 3 | `instruction_hash_lo` | Instruction hash lo |
| 4 | `instruction_hash_hi` | Instruction hash hi |
| 5 | `input_hash_lo` | Input hash lo |
| 6 | `input_hash_hi` | Input hash hi |
| 7 | `output_hash_lo` | Output hash lo |
| 8 | `output_hash_hi` | Output hash hi |
| 9 | `prev_receipt_hash_lo` | Parent receipt hash lo (0 for first action) |
| 10 | `prev_receipt_hash_hi` | Parent receipt hash hi (0 for first action) |
| 11 | `receipt_hash_lo` | This action's receipt hash lo |
| 12 | `receipt_hash_hi` | This action's receipt hash hi |
| 13 | `status_flags` | Bitfield (see below) |

### Hash-to-Field Conversion

SHA-256 hex strings are converted to field element pairs:

```python
raw = bytes.fromhex(hex_hash)
lo = int.from_bytes(raw[0:8], 'little') % GOLDILOCKS_P
hi = int.from_bytes(raw[8:16], 'little') % GOLDILOCKS_P
```

This retains 128 bits of collision resistance per hash.

### Status Flags Bitfield

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `SUCCESS` | 1 = action succeeded, 0 = failed |
| 1 | `GATE_PASSED` | 1 = gate decision != "skip" |
| 2 | `GATE_EVALUATED` | 1 = gate_score is not None |
| 3 | `POLICY_MET` | 1 = policy_verdict is not None |
| 4 | `IS_TOOL_CALL` | 1 = action_type == "tool_call" |
| 5 | `IS_CODE_GEN` | 1 = action_type == "code_gen" |

## Transition Constraints (3)

Applied between consecutive rows (row_curr, row_next):

### 1. Chain Continuity (lo)

```
row_next[9] - row_curr[11] == 0
```

The next row's `prev_receipt_hash_lo` must equal the current row's
`receipt_hash_lo`. This proves the chain is unbroken.

### 2. Chain Continuity (hi)

```
row_next[10] - row_curr[12] == 0
```

Same constraint for the high field element.

### 3. Sequential Ordering

```
row_next[0] - row_curr[0] - 1 == 0
```

Action indices must be strictly sequential.

## Boundary Constraints (4)

### First Row

```
row_first[9] == 0    # prev_receipt_hash_lo
row_first[10] == 0   # prev_receipt_hash_hi
```

The first action has no parent, so its previous receipt hash is zero.

### Last Row

```
row_last[11] == declared_final_receipt_lo
row_last[12] == declared_final_receipt_hi
```

The last row's receipt hash must match the declared final receipt hash.

## Composition Vector

All constraints are combined into a single composition polynomial:

```python
composition = build_composition_vector(trace_matrix, alphas, final_receipt_hash)
```

Where `alphas` are Fiat-Shamir challenge values derived from the trace commitment:

```python
alphas = derive_constraint_alphas(row_root_hex, num_constraints=7)
```

The composition vector should be all zeros for a valid trace.

## Proof Types

### constraint_check (default)

Verifies all constraints hold over the trace matrix. No cryptographic
proof is generated — the verifier re-evaluates the constraints directly.

```json
{
    "proof_type": "constraint_check",
    "status": "proved",
    "verified": true,
    "trace_length": 5,
    "row_root": "hex...",
    "composition_all_zero": true
}
```

### fri (optional upgrade)

When `bef_zk.fri` is installed, generates a FRI (Fast Reed-Solomon IOP)
proof over the composition polynomial. This allows verification without
access to the full trace.

```json
{
    "proof_type": "fri",
    "status": "proved",
    "verified": true,
    "trace_length": 5,
    "row_root": "hex...",
    "fri_proof": {
        "domain_size": 32,
        "max_degree": 7,
        "num_rounds": 3,
        "num_queries": 8,
        "base_commitment_root": "hex...",
        "round_commitments": ["hex...", ...],
        "query_proofs": [...]
    }
}
```

## agent_capsule.json

The proof capsule stored in each run directory:

```json
{
    "schema": "agent_capsule_v1",
    "air_id": "agent_air_v1",
    "trace_length": 5,
    "row_width": 14,
    "row_root": "hex...",
    "final_receipt_hash": "hex...",
    "verification": {
        "constraints_satisfied": true,
        "num_transition_constraints": 3,
        "num_boundary_constraints": 4,
        "composition_all_zero": true,
        "proof_type": "constraint_check"
    },
    "timing": {
        "simulate_ms": 1.2,
        "commit_ms": 0.5,
        "prove_ms": 2.3,
        "verify_ms": 0.8,
        "total_ms": 4.8
    }
}
```

## Verification Flow

```
1. Load agent_capsule.json
2. Check schema == "agent_capsule_v1"
3. Check verification.constraints_satisfied == true
4. If proof_type == "fri":
   a. Load FRI proof
   b. Verify FRI proof against commitment
5. If proof_type == "constraint_check":
   a. Rebuild trace matrix from actions
   b. Re-evaluate all 7 constraints
   c. Check composition vector is all zeros
```

## Goldilocks Field

All arithmetic is performed modulo the Goldilocks prime:

```
p = 2^64 - 2^32 + 1 = 18446744069414584321
```

This prime supports efficient modular arithmetic on 64-bit platforms
and is widely used in modern proof systems (Plonky2, Winterfell, etc.).
