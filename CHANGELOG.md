# Changelog

## v0.1.0 — Proof-Carrying Agent Execution

Three AIR proof systems (WorkflowAIR, EvalAIR, AgentAIR) sharing 14-element Goldilocks row encoding with FRI verification. Committor gating via epistemic eval loop with Beta posteriors. AgentRuntime as 5-line integration surface. 61 tests. Receipt-bound artifact pipeline with .cap packaging.

### Added
- **WorkflowAIR**: FRI proofs over workflow DAG execution (`capseal workflow --prove`)
- **EvalAIR**: FRI proofs over epistemic learning rounds (`capseal eval --prove`)
- **AgentAIR**: General proof-carrying agent protocol via `AgentRuntime`
- Unified 14-element Goldilocks field encoding across all three AIRs
- STC (Sumcheck Tensor Commitment) for trace commitment
- Committor gating with Beta posterior updates
- Framework adapters (generic decorator, LangChain, OpenAI stubs)
- `capseal verify-capsule` for verifying any capsule type
- Public API: `from capseal import AgentRuntime, AgentAction, wrap_function`

### Technical Details
- Goldilocks field: p = 2^64 - 2^32 + 1
- Chain constraint: receipt_hash[i] == prev_receipt_hash[i+1]
- Boundary constraints: first row prev=0, last row receipt=declared_root
- 128-bit collision resistance via SHA256 → (lo, hi) field pair encoding
