# CapSeal Receipt Format (.cap)

Version: 0.4.0

## Overview

A `.cap` file is a gzip-compressed tarball containing the complete session
record: actions, proofs, commitments, and metadata. It serves as a
portable, verifiable receipt of agent execution.

Source of truth: `capseal/cli/cap_format.py`

## .cap File Structure

```
<session-id>.cap (tar.gz)
├── manifest.json           # Capsule metadata
├── actions.jsonl           # Action log (one JSON object per line)
├── agent_capsule.json      # Proof capsule with verification data
├── proof.bin.zst           # Compressed binary proof (optional)
├── commitments.json        # Merkle commitment data
├── capsule.json            # Full capsule data (compatibility)
├── run_metadata.json       # Runtime metadata (agent name, etc.)
├── archive/                # Binary row archive (optional, for DA audit)
└── signatures/             # Detached Ed25519 signatures (optional)
```

## manifest.json Schema

```json
{
    "schema": "cap_manifest_v1",
    "capsule_id": "uuid",
    "trace_id": "uuid",
    "policy_id": "string",
    "policy_hash": "hex",
    "backend": "agent",
    "verification_profile": "proof_only",
    "root_hex": "hex",
    "num_chunks": 1,
    "proof_size": 1234,
    "archive_format": "json",
    "created_at": "2024-01-15T10:30:00Z",
    "extras": {
        "session_name": "my-session",
        "actions_count": 5,
        "agent": "Claude Code",
        "proof_generated": true,
        "proof_verified": true,
        "proof_type": "constraint_check",
        "chain_hash": "abc123..."
    }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `schema` | Always `cap_manifest_v1` |
| `capsule_id` | UUID identifying this capsule |
| `trace_id` | UUID identifying the execution trace |
| `policy_id` | Policy identifier (if policy enforcement was used) |
| `policy_hash` | SHA-256 of the policy document |
| `backend` | Proof backend: `"agent"` for agent sessions |
| `verification_profile` | `"proof_only"`, `"policy_enforced"`, or `"fully_verified"` |
| `root_hex` | Merkle root of the committed trace |
| `num_chunks` | Number of chunks in the commitment |
| `proof_size` | Size of the proof in bytes |
| `archive_format` | `"json"` or `"binary"` |
| `created_at` | ISO 8601 creation timestamp |
| `extras` | Session-specific metadata |

## actions.jsonl Format

Each line is a JSON object representing one `AgentAction`:

```json
{
    "action_id": "act_0001",
    "action_type": "code_edit",
    "instruction_hash": "sha256hex...",
    "input_hash": "sha256hex...",
    "output_hash": "sha256hex...",
    "parent_action_id": null,
    "parent_receipt_hash": null,
    "gate_score": 0.12,
    "gate_decision": "pass",
    "policy_verdict": "FULLY_VERIFIED",
    "success": true,
    "duration_ms": 45,
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
        "description": "Fix command injection in app.py",
        "tool_name": "edit",
        "files_affected": ["app.py"]
    },
    "canonical_fields": {
        "action_id": "act_0001",
        "action_type": "code_edit",
        "instruction_hash": "sha256hex...",
        "input_hash": "sha256hex...",
        "output_hash": "sha256hex...",
        "parent_action_id": null,
        "parent_receipt_hash": null,
        "gate_score": 0.12,
        "gate_decision": "pass",
        "success": true,
        "duration_ms": 45,
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "receipt_hash": "sha256hex..."
}
```

The `canonical_fields` and `receipt_hash` enable third-party recomputation
of the hash chain without trusting CapSeal's implementation.

## Verification Steps

To verify a `.cap` receipt:

1. **Extract the tarball** and read `manifest.json` and `actions.jsonl`
2. **Verify chain integrity**: For each action, recompute `receipt_hash`
   from `canonical_fields` and check parent linkage
3. **Verify proof**: Load `agent_capsule.json` and check constraint
   satisfaction or FRI proof validity
4. **Verify signature** (optional): If `signatures/` contains a `.sig`
   file, verify the Ed25519 signature

### Quick Verification

```bash
capseal verify path/to/session.cap
capseal verify path/to/session.cap --json
```

### Programmatic Verification

```python
from capseal.cli.cap_format import read_cap_manifest
from capseal.agent_adapter import verify_agent_capsule

manifest = read_cap_manifest(Path("session.cap"))
# ... load actions and verify chain
```

## Export Receipt

The `export-receipt` command produces a self-contained JSON receipt:

```bash
capseal export-receipt session.cap --print
```

Output includes:
- `schema`: `"capseal_receipt_v1"`
- `actions`: Array of actions with `canonical_fields` and `receipt_hash`
- `chain_hash`: Final receipt hash
- `integrity`: Algorithm info and `fully_recomputable` flag

## Signature Format

When signed with `capseal sign`, a `.cap.sig` file is created:

```json
{
    "schema": "capseal_signature_v1",
    "algorithm": "Ed25519",
    "public_key": "base64...",
    "signature": "base64...",
    "cap_hash": "sha256hex...",
    "signed_at": "2024-01-15T10:30:00Z"
}
```
