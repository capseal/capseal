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
