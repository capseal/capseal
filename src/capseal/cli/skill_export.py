"""Export CapSeal as an OpenClaw skill."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


SKILL_MD = '''# CapSeal - Verified AI Agent Execution

CapSeal is a trust layer that provides cryptographic receipts for AI agent actions.

## Tools Available

When CapSeal is connected via MCP, you have access to:

### capseal_gate
Gate a proposed action before execution. Returns approve/deny/flag based on the learned risk model.

**Use before**: Any tool call, code edit, file write, or API request.

```json
{
  "action_type": "code_edit",
  "description": "Refactor the login function to use async/await",
  "files_affected": ["src/auth/login.py"],
  "diff_text": "..." // optional unified diff
}
```

**Response**:
```json
{
  "decision": "approve",  // or "deny" or "flag"
  "predicted_failure": 0.12,
  "confidence": 0.87,
  "reason": ""
}
```

### capseal_record
Record an executed action. Call this after every tool execution to build the audit trail.

```json
{
  "action_type": "code_edit",
  "description": "Refactored login function",
  "tool_name": "Edit",
  "success": true,
  "files_affected": ["src/auth/login.py"],
  "duration_ms": 150
}
```

### capseal_seal
Seal all recorded actions into a .cap receipt file. Call at the end of a session.

```json
{
  "session_name": "refactor-auth-module"
}
```

**Response**:
```json
{
  "sealed": true,
  "cap_file": "/path/to/.capseal/runs/20240115T103045-mcp.cap",
  "chain_hash": "a287f7e44a75...",
  "actions_sealed": 12
}
```

## Workflow

1. Before each action: Call `capseal_gate` to check if it should proceed
2. After each action: Call `capseal_record` to log what happened
3. At session end: Call `capseal_seal` to create the cryptographic receipt

## Verification

After a session, verify the receipt:
```bash
capseal verify .capseal/runs/latest.cap
```

Output:
```
✓ Capsule verified: a287f7e44a75...
  Actions:  12
  Chain:    intact (12/12 hashes valid)
  Session:  refactor-auth-module
  Proof:    constraints_valid=true
```
'''


def export_skill(destination: Path) -> None:
    """Export the CapSeal skill to the given directory.

    Args:
        destination: Directory to write SKILL.md to
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    skill_path = destination / "SKILL.md"
    skill_path.write_text(SKILL_MD)

    # Also write a manifest.json for tool discovery
    manifest = {
        "name": "capseal",
        "version": "0.1.0",
        "description": "Cryptographic verification for AI agent actions",
        "mcp_server": {
            "command": "capseal",
            "args": ["mcp-serve"],
            "transport": "stdio"
        },
        "tools": [
            {
                "name": "capseal_gate",
                "description": "Gate an action before execution"
            },
            {
                "name": "capseal_record",
                "description": "Record an executed action"
            },
            {
                "name": "capseal_seal",
                "description": "Seal session into .cap receipt"
            }
        ]
    }

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
