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