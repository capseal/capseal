# BEF-ZK / CapSeal Source Documentation

## 1. System Overview

**BEF-ZK** (also referred to as **CapSeal**) is a framework for generating, verifying, and managing cryptographic receipts ("Capsules") of execution traces. It serves as the verification layer for the broader BEF (Brownian-Inspired/Fusion) ecosystem.

Its primary goal is to provide a unified, backend-agnostic interface for proving that a specific computation (defined by a policy and input datasets) yielded a specific result, without necessarily re-executing the full computation.

## 2. Core Abstractions

### 2.1. Trace Adapter (`adapter.py`)
The plugin interface for proof backends.
*   **`TraceAdapter`**: Abstract base class.
    *   `simulate_trace()`: Runs the computation to produce raw trace artifacts.
    *   `commit_to_trace()`: Generates Merkle commitments to the trace rows (STC - State Transition Commitment).
    *   `generate_proof()`: Calls the backend prover (e.g., FRI-based, RISC0) to generate a cryptographic proof.
    *   `verify()`: Verifies the proof against the statement hash.
    *   `replay_trace()`: (Optional) Re-executes trace for semantic verification/debugging.

### 2.2. Capsules & Receipts (`capsule/contracts.py`)
Defines the immutable data structures exchanged by the system.
*   **Receipt**: A JSON document containing the Proof, Public Inputs (Statement), and Commitments.
*   **Capsule**: A self-contained archive (JSON or ZIP) including the Receipt, Policy, and metadata.
*   **Contracts**: Strict schema definitions for `ReceiptContract`, `PolicyContract`, and `WorkspaceContract` to ensure backward compatibility across versions.

## 3. CLI & Tooling (`bef_zk/capsule/cli/`)

A rich CLI (`capsule`) provides workflows for developers and CI systems.

*   **`run`**: Orchestrates the full pipeline: Trace $\to$ Commit $\to$ Prove. Supports sandboxing (Bubblewrap/Firejail).
*   **`verify`**: Verifies a capsule with stable exit codes (0=Success, 10=Invalid Proof, 11=Policy Mismatch, etc.).
*   **`audit`**: Inspects the audit trail/event logs within a capsule.
*   **`replay`**: Re-runs the computation to verify semantic equivalence of the trace.
*   **`doctor`**: Diagnostic tool for environment and capsule health.

## 4. MCP Server (`capsule/mcp_server.py`)

Implements the **Model Context Protocol (MCP)** to expose CapSeal tools to AI agents (like Claude/Cline).
*   **Security**: Enforces strict path validation (workspace-only), allowlisted commands, and output truncation.
*   **Auditability**: Logs every tool invocation into a hash-chained event log (`mcp_events.jsonl`) for cryptographic accountability of agent actions.
*   **Tools**: Exposes `verify`, `audit`, `diff_bundle`, `spawn_agent`, etc.

## 5. Orchestration (`capsule/orchestrator.py`)

A multi-agent workflow engine.
*   **Pipelines**: Defines multi-step tasks (e.g., "Diff Review" pipeline: `diff_bundle` $\to$ `review` $\to$ `proposal`).
*   **Receipts**: Every step in the orchestration generates a cryptographically linked receipt, ensuring the entire agentic workflow is auditable.

## 6. STC / Archiving (`stc/`)

Handles the storage and retrieval of trace data.
*   **`ChunkArchive`**: Manages large execution traces split into binary chunks.
*   **Security**: Includes path traversal protections (`test_archive_security.py`) to ensure untrusted archives cannot access the host filesystem.

## 7. Usage Flow

1.  **Define Policy**: Create a `policy.json` defining allowed computations and constraints.
2.  **Run Pipeline**:
    ```bash
    capsule run -p policy.json -d data/ --trace-id my_run
    ```
3.  **Verify**:
    ```bash
    capsule verify out/my_run/strategy_capsule.json
    ```
4.  **Audit (Agentic)**:
    An IDE agent connects via MCP and calls `verify` or `doctor` to validate the run autonomously.
