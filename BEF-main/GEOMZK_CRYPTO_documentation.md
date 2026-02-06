# Cryptographic Architecture of BEF-ZK & GEOMZK: A Formal Construction

## 1. Foundational Primitives: Collision-Resistant Hashing
As established in **Katz & Lindell (KL) Chapter 6**, the security of the entire verification stack rests upon the existence of collision-resistant hash functions.

### 1.1. Deterministic Integrity
We employ **SHA-256** for standard artifact fingerprinting. Per **KL Definition 6.2**, this ensures that finding two distinct execution traces that yield the same commitment root is computationally infeasible. For algebraic circuits (the "Math" layer), we utilize **Blake3** and **Poseidon**, which offer superior performance within the Goldilocks Field $\mathbb{F}_{p}$ where $p = 2^{61}-1$.

### 1.2. The Random Oracle Model
Following the **Random Oracle Methodology (KL Section 6.5)**, we utilize the **Fiat-Shamir Heuristic** to transform interactive arguments into the non-interactive "Receipts" (Capsules) seen in `bef_zk`. This is implemented via the `Transcript` class (`bef_zk/transcript.py`), which absorbs all public data to derive cryptographically secure challenges.

---

## 2. Vector Commitments: State Transition Commitments (STC)
The system implements a **Vector Commitment (VC)** scheme, which **Boneh & Shoup (BS) Section 10.2** defines as a method to commit to a sequence of values such that one can later open any $v_i$ with a succinct proof.

### 2.1. The $k$-ary Merkle Construction
Instead of standard binary trees, `bef_zk/stc/` implements a $k$-ary **STC**.
*   **Construction**: The execution trace matrix $T$ is partitioned into chunks. A Merkle Tree (**KL Section 6.6.2**) commits to these chunks.
*   **Optimization**: By increasing the arity $k$, we reduce the tree depth, thereby minimizing the number of "sibling" hashes required in the "Middle Part" of the FRI query phase.

---

## 3. Algebraic Intermediate Representation (AIR)
To prove the correctness of a simulation (e.g., BICEP paths), we translate the trace into **Algebraic Constraints**.

### 3.1. Transition and Boundary Constraints
In `bef_zk/zk_geom/verifier.py`, the verifier checks:
*   **Boundary Constraints**: $P(0) = \text{Initial State}$.
*   **Transition Constraints**: $C(T_t, T_{t+1}) = 0$, ensuring the "Geometry Program" logic was followed at every step.
*   **Consistency**: As discussed in **BS Section 20.6 (SNARKs)**, these are combined into a **Composition Polynomial** $H(x)$ using random linear combinations $(\alpha_i)$ derived from the transcript.

---

## 4. The Proximity Argument: FRI Protocol
The core of the "Canonical Instance" is the **FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity)** protocol. Its purpose is to prove that the committed trace values are "close" to a low-degree polynomial.

### 4.1. Folding and Query Logic (`bef_zk/fri/`)
The Prover (`prover.py`) performs a series of **Folding** steps (akin to an FFT) to reduce the degree of the polynomial.
The Verifier (`verifier.py` and `pc_verify`) performs:
1.  **Commitment Consistency**: Matches roots of folded polynomials to the transcript.
2.  **Collinearity Check**: Verifies that for a random query index $i$, the values at $x$ and $-x$ satisfy the linear interpolation required by the folding factor $\zeta$.
3.  **Final Degree Test**: Directly inspects the base case polynomial to ensure its degree is within the tolerated bounds.

---

## 5. Mapping the Computational Chain

| Level | Component | Folder Location | Theory Reference |
| :--- | :--- | :--- | :--- |
| **L0: Hash** | SHA256 / Blake3 | `geomzk/src/hash.cpp` | KL Chapter 6 |
| **L1: Commitment** | Merkle / STC | `bef_zk/stc/` | KL 6.6.2 / BS 10.2 |
| **L2: Constraint** | AIR Definitions | `bef_zk/zk_geom/` | BS SNARKs/IOPs |
| **L3: Proximity** | FRI Prover/Verifier | `bef_zk/fri/` | RS Codes / FRI IOP |
| **L4: Receipt** | Capsule / Receipt | `bef_zk/capsule/` | "The Proof" |

---

## 6. Formal Audit: Discrepancies and Integrity

### 6.1. The Verification Gap
A critical finding of this audit is that **the C++ engine (`geomzk/src/verify.cpp`) does not yet verify the low-degree test.** The function `verify_fri_stub` is an empty implementation. The **Python implementation (`bef_zk/zk_geom/verifier.py`)** remains the only complete and canonical verifier.

### 6.2. Masking and Zero-Knowledge
The `masking.py` module implements **Algebraic Masking**. This ensures that the opened rows during the FRI query phase do not leak sensitive information about the remainder of the trace, maintaining the **Zero-Knowledge** property required for secure delegated computation.

### 6.3. Junk and Redundancy
*   `bef_zk/zk_geom/simple_lib.rs`: Redundant duplication of `lib.rs`.
*   `geomzk/include/geomzk/geom_air_example.hpp`: Contains hardcoded constraints that should be abstracted if used as a general-purpose verifier.

---

---

## 7. The Policy Engine: Predicate Private Computation

The **Policy Engine** (`policy/policy_v1.py`) serves as the "Predicate" in our proof system. It defines the deterministic logic that transforms stochastic inputs (from ENN and FusionAlpha) into auditable actions.

### 7.1. Semantic Binding and Commitment
Following the principles of **Commitment Schemes (KL Section 6.6.5)**, the system ensures that the Prover cannot deviate from the agreed-upon strategy.
*   **The Policy Contract**: The `benchmark_policy_v1.json` defines the "rules of the game" (e.g., `forbid_gpu`).
*   **Binding**: The `policy_id` and a hash of the policy rules are included in the **Statement (Public Inputs)**. Per **KL Definition 6.2**, the collision resistance of the statement hash binds the entire proof to this specific policy instance.

### 7.2. Algebraic Encoding of Gating Logic
In the **AIR** (Algebraic Intermediate Representation), the deterministic `decide` function is transformed into a set of polynomial constraints.
*   **The Break-Even Gate**: The code calculates `edge = (2p - 1) * m - costs`. In the field $\mathbb{F}_p$, this is a linear operation.
*   **Thresholding as Constraints**: Logical gates (e.g., `p_trade >= cfg.t_pred_long`) are represented using **Arithmetic Circuits (BS Section 20.1)**. A boolean condition $b \in \{0,1\}$ is enforced by the constraint $b(b-1) = 0$.
*   **Action Selection**: The "LONG/SHORT/FLAT" side is encoded as a numerical register. The Verifier ensures that the value in this register at any time $t$ is the unique solution to the polynomial representation of the `decide` function.

### 7.3. Risk-Adjusted Kelly Sizing
The policy implements a modified **Kelly Criterion** for position sizing:
$$ \text{size} = \text{clamp}(0, \text{cap}, \text{shrink} \cdot \text{reliability} \cdot \frac{\text{edge}}{\text{variance}}) $$
*   **Mathematical Grounding**: This calculation is part of the execution trace $T$. The **FRI Protocol (Section 4)** verifies that this complex non-linear calculation was performed correctly across thousands of path steps.
*   **Reliability Haircut**: The integration of the `obs_reliability` from the ENN ensures that the sizing is inversely proportional to the epistemic uncertainty, a feature aligned with **BS Discussion on Probabilistic IOPs**.

### 7.4. Execution Model and Realized PnL
The `ExecutionModel` (`policy/execution_model.py`) provides the "Final Reward" constraint. The **Composition Polynomial (Section 3.2)** includes a constraint that checks the recursive update of the total profit:
$$ \text{PnL}_{t+1} = \text{PnL}_t + \text{Execution}(\text{Action}_t, \text{Returns}_t) $$
The **Boundary Constraint** at $t=L$ proves the final realized performance relative to the initial dataset.

---

## 8. Summary of the Cryptographic Chain

| Module | Cryptographic Role | Academic Primitive |
| :--- | :--- | :--- |
| `bef_zk` | Protocol Orchestrator | Fiat-Shamir (KL 6.5) |
| `stc/` | Vector Commitment | Merkle Tree (KL 6.6.2) |
| `fri/` | Proximity Proof | Low-Degree Testing (BS 20.6) |
| `policy/` | Verified Predicate | Arithmetic Circuit (BS 20.1) |
| `capsule/` | Receipt Management | Collision-Resistant ID (KL 6.2) |

---

**Canonical Verifier**: `bef_zk/zk_geom/verifier.py`
**Academic Verdict**: The BEF-ZK system is a complete **STARK** implementation where the "Program" is the **Policy Engine**. The security of the system ensures that the reported performance metrics are the unique result of applying the committed policy to the verified datasets.
