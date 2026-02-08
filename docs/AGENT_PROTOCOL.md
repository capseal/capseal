# CapSeal Agent Protocol Specification

Version: 0.4.0

## Overview

The Agent Protocol defines the standard record format (`AgentAction`) that any
external agent framework emits. The CapSeal runtime handles receipt computation,
trace encoding, constraint verification, and proof generation.

Source of truth: `capseal/agent_protocol.py`

## AgentAction Schema

Each action in the chain is an `AgentAction` with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_id` | string | yes | Unique ID for this action (e.g. `act_0001`) |
| `action_type` | string | yes | Category of action (see Action Types) |
| `instruction_hash` | string | yes | SHA-256 of the prompt/instruction that triggered this action |
| `input_hash` | string | yes | SHA-256 of inputs (tool args, code context, API params) |
| `output_hash` | string | yes | SHA-256 of outputs (tool result, generated code, API response) |
| `parent_action_id` | string? | no | ID of the preceding action (`null` for first) |
| `parent_receipt_hash` | string? | no | Receipt hash of the parent action (`null` for first) |
| `gate_score` | float? | no | Committor q(x) if the gate was evaluated |
| `gate_decision` | string? | no | Gate decision (see Gate Decisions) |
| `policy_verdict` | string? | no | Policy enforcement level (see Policy Verdicts) |
| `success` | boolean | yes | Whether the action achieved its intended result |
| `duration_ms` | integer | yes | Execution time in milliseconds |
| `timestamp` | string | yes | ISO 8601 timestamp |
| `metadata` | object? | no | Extensible agent-specific data |

## Action Types

Actions are categorized into these types:

| Type | Description |
|------|-------------|
| `tool_call` | Agent invokes an external tool |
| `code_gen` | Agent generates code or text |
| `code_edit` | Agent modifies existing code |
| `api_request` | Agent makes an HTTP/API request |
| `decision` | Agent makes a routing/control flow decision |
| `observation` | Agent records an environment observation |
| `file_read` | Agent reads a file |
| `file_write` | Agent writes a file |
| `file_delete` | Agent deletes a file |
| `shell_exec` | Agent executes a shell command |
| `command` | Agent executes a command |
| `browser_action` | Agent performs a browser action |
| `web_request` | Agent makes a web request |
| `action` | Generic fallback |
| `unknown` | Unknown action type |

## Gate Decisions

The committor gate evaluates each action against the learned risk model:

| Decision | Meaning | Agent Behavior |
|----------|---------|----------------|
| `pass` | Low risk, approved | Proceed with the action |
| `skip` | High risk, denied | Do NOT proceed; inform the user |
| `human_review` | Moderate risk, flagged | Ask user for explicit permission |
| `human_approved` | Human reviewed and approved | Proceed (human override) |
| `skipped` | Skipped after max retries | Action was abandoned |

## Policy Verdicts

| Verdict | Meaning |
|---------|---------|
| `PROOF_ONLY` | Cryptographic proof generated, no policy enforcement |
| `POLICY_ENFORCED` | Policy rules applied (gate + constraints) |
| `FULLY_VERIFIED` | Full verification: proof + policy + gate |

## Receipt Hash Computation

Each action's receipt hash is computed as:

```
canonical_fields = {
    "action_id": ...,
    "action_type": ...,
    "instruction_hash": ...,
    "input_hash": ...,
    "output_hash": ...,
    "parent_action_id": ...,
    "parent_receipt_hash": ...,
    "gate_score": ...,
    "gate_decision": ...,
    "success": ...,
    "duration_ms": ...,
    "timestamp": ...
}
canonical = json.dumps(canonical_fields, sort_keys=True, separators=(",", ":"))
receipt_hash = sha256(canonical.encode()).hexdigest()
```

The `canonical_fields` dict is stored alongside each action in `actions.jsonl`
for third-party recomputation.

## Action Chain Construction

Actions form a linked chain via `parent_receipt_hash`:

```
Action 0: parent_receipt_hash = null
           receipt_hash = H(action_0)

Action 1: parent_receipt_hash = H(action_0)
           receipt_hash = H(action_1)

Action 2: parent_receipt_hash = H(action_1)
           receipt_hash = H(action_2)
```

The `AgentRuntime` automatically sets `parent_action_id` and
`parent_receipt_hash` when `auto_chain=True` (the default).

## Chain Integrity Verification

To verify a chain:

1. For each action, recompute `receipt_hash` from `canonical_fields`
2. Check that `receipt_hash` matches the stored value
3. Check that `parent_receipt_hash` of action N+1 equals `receipt_hash` of action N
4. Check that action 0 has `parent_receipt_hash = null`

## MCP Integration

The CapSeal MCP server exposes five tools that implement this protocol:

| Tool | Purpose | Protocol Step |
|------|---------|---------------|
| `capseal_gate` | Evaluate risk before an action | Pre-action gate |
| `capseal_record` | Record an executed action | Action recording |
| `capseal_seal` | Seal the session into a .cap receipt | Session finalization |
| `capseal_status` | Get session state + history | Session management |
| `capseal_context` | Get file change history | Context retrieval |

### Typical MCP Session Flow

```
1. capseal_status      → Check session state
2. capseal_gate        → Gate each file change (one per file)
3. (make the change)
4. capseal_record      → Record each file change (one per file)
5. (repeat 2-4 for each change)
6. capseal_seal        → Seal the session
```

## Live Events

The MCP server emits events to `.capseal/events.jsonl` for live status
consumers (e.g., the PTY shell status bar):

```json
{"type": "gate",   "timestamp": 1707000000.0, "summary": "approve: Fix shell injection (p_fail=0.12)"}
{"type": "record", "timestamp": 1707000001.0, "summary": "code_edit: Replaced subprocess.call with subprocess.run"}
{"type": "seal",   "timestamp": 1707000002.0, "summary": "Sealed 3 actions"}
```
