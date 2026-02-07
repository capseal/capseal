"""MCP Server for CapSeal - Expose AgentRuntime as MCP tools.

This allows any agent framework that speaks MCP (OpenClaw, Claude Code, Cursor,
LangChain, etc.) to use CapSeal as a trust layer.

Three tools:
    capseal_gate   - Gate a proposed action before execution
    capseal_record - Record what happened after execution
    capseal_seal   - Seal the session into a .cap receipt

Usage:
    # From project directory (uses .capseal/ in cwd):
    cd ~/projects/my-project
    capseal mcp-serve

    # Or with explicit workspace:
    capseal mcp-serve --workspace ~/projects/my-project

    # Programmatically:
    from capseal.mcp_server import run_mcp_server
    run_mcp_server(workspace="/path/to/project")

MCP clients can then call:
    tools/list                     # See available tools
    tools/call capseal_gate {...}  # Gate an action
    tools/call capseal_record {...} # Record an action
    tools/call capseal_seal        # Seal the session
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, PromptMessage, PromptArgument

# Global runtime instance (initialized on first use)
_runtime = None
_runtime_path = None
_workspace = None  # Set via --workspace or defaults to cwd


def _get_runtime():
    """Get or create the AgentRuntime instance."""
    global _runtime, _runtime_path, _workspace

    if _runtime is not None:
        return _runtime

    from .agent_runtime import AgentRuntime

    # Use workspace if set, otherwise cwd
    workspace = Path(_workspace) if _workspace else Path.cwd()
    workspace = workspace.resolve()

    # Determine output directory
    capseal_dir = workspace / ".capseal"
    runs_dir = capseal_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = runs_dir / f"{timestamp}-mcp"
    run_dir.mkdir(parents=True, exist_ok=True)
    _runtime_path = run_dir

    # Check for learned posteriors
    posteriors_path = capseal_dir / "models" / "beta_posteriors.npz"

    # Log where we're looking for posteriors (to stderr so it doesn't interfere with MCP)
    if posteriors_path.exists():
        print(f"[capseal] Loaded posteriors from {posteriors_path}", file=sys.stderr)
    else:
        print(f"[capseal] No posteriors at {posteriors_path} (gate will approve all)", file=sys.stderr)

    _runtime = AgentRuntime(
        output_dir=run_dir,
        gate_posteriors=posteriors_path if posteriors_path.exists() else None,
    )

    return _runtime


def _load_capseal_env():
    """Load API keys from .capseal/.env if available."""
    global _workspace

    workspace = Path(_workspace) if _workspace else Path.cwd()
    env_file = workspace / ".capseal" / ".env"
    if not env_file.exists():
        return

    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and not os.environ.get(key):
                        os.environ[key] = value
    except Exception:
        pass


# Create the MCP server
server = Server("capseal")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available CapSeal tools."""
    return [
        Tool(
            name="capseal_gate",
            description="⚠️ REQUIRED before any code change. Gate a proposed action through the learned risk model. You MUST call this before editing any file. Returns approve/deny/flag. If 'deny': DO NOT proceed, tell user it was blocked.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Type of action (tool_call, code_edit, file_write, api_request, etc.)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what the action does",
                    },
                    "files_affected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths that will be affected",
                    },
                    "diff_text": {
                        "type": "string",
                        "description": "Optional unified diff for code changes",
                    },
                },
                "required": ["action_type", "description"],
            },
        ),
        Tool(
            name="capseal_record",
            description="⚠️ REQUIRED after any code change. Record what you did for the cryptographic audit trail. Call this after each file edit with action type, tool used, and success status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Type of action that was executed",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what happened",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool that was called",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the action succeeded",
                    },
                    "files_affected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths that were affected",
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "How long the action took in milliseconds",
                    },
                    "output": {
                        "type": "string",
                        "description": "Output or result of the action",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if the action failed",
                    },
                },
                "required": ["action_type", "description", "tool_name", "success"],
            },
        ),
        Tool(
            name="capseal_seal",
            description="⚠️ REQUIRED at end of session. Seal all recorded actions into a verifiable .cap receipt. You MUST call this before ending any session where you made code changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Optional name for this session (used in receipt metadata)",
                    },
                },
            },
        ),
        Tool(
            name="capseal_status",
            description="⚠️ Call this at the START of every session and after any conversation compaction/summary. Returns current CapSeal session state: how many actions recorded, last gate decision, whether sealed. Use this to recover context if you lose track of what has been gated and recorded.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ── Prompts ──────────────────────────────────────────────────────────────────

CAPSEAL_SYSTEM_PROMPT = """You have CapSeal tools available for verified code execution.

**At the START of every session (or after any conversation compaction/summary):**
Call `capseal_status` to check if there is an existing session with recorded actions. This recovers your place if context was lost.

**Before making any code change:**
Call `capseal_gate` with a description of what you plan to change and which files will be affected.
- If decision is "approve": proceed with the change
- If decision is "deny": do NOT proceed — explain to the user why it was blocked (high predicted failure rate)
- If decision is "flag": proceed with caution, mention the uncertainty to the user

**After each code change:**
Call `capseal_record` to log what happened (action type, tool used, success/failure).

**When your session is complete:**
Call `capseal_seal` to create a cryptographic receipt proving exactly what you did.

This creates a verifiable audit trail. The user can run `capseal verify` to cryptographically verify your session."""


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """Return available prompts."""
    return [
        Prompt(
            name="capseal-instructions",
            description="Instructions for using CapSeal tools to gate, record, and seal code changes",
            arguments=[],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> list[PromptMessage]:
    """Return prompt content."""
    if name == "capseal-instructions":
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=CAPSEAL_SYSTEM_PROMPT),
            ),
        ]
    raise ValueError(f"Unknown prompt: {name}")


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "capseal_gate":
        return await _handle_gate(arguments)
    elif name == "capseal_record":
        return await _handle_record(arguments)
    elif name == "capseal_seal":
        return await _handle_seal(arguments)
    elif name == "capseal_status":
        return await _handle_status(arguments)
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}",
        }))]


async def _handle_gate(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_gate tool call."""
    runtime = _get_runtime()

    action_type = args.get("action_type", "unknown")
    description = args.get("description", "")
    files_affected = args.get("files_affected", [])
    diff_text = args.get("diff_text", "")

    # Build findings from files_affected for feature extraction
    findings = []
    for f in files_affected:
        findings.append({
            "path": f,
            "extra": {"severity": "warning"},
        })

    try:
        # Call the gate
        gate_result = runtime.gate(diff_text=diff_text, findings=findings)

        # Map internal decision to simpler output
        decision_map = {
            "pass": "approve",
            "skip": "deny",
            "human_review": "flag",
        }

        decision = gate_result.get("decision", "pass")
        mapped_decision = decision_map.get(decision, decision)

        result = {
            "decision": mapped_decision,
            "predicted_failure": gate_result.get("q", 0.0),
            "confidence": 1.0 - gate_result.get("uncertainty", 0.0),
            "reason": gate_result.get("reason", ""),
            "grid_idx": gate_result.get("grid_idx", 0),
        }

        # Store for status reporting
        runtime._last_gate_result = {"decision": mapped_decision, "q": gate_result.get("q", 0.0)}

    except Exception as e:
        # If gating fails (no model, etc.), approve by default
        result = {
            "decision": "approve",
            "predicted_failure": 0.0,
            "confidence": 0.0,
            "reason": f"No risk model available: {e}",
        }

    return [TextContent(type="text", text=json.dumps(result))]


async def _handle_record(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_record tool call."""
    runtime = _get_runtime()

    action_type = args.get("action_type", "unknown")
    description = args.get("description", "")
    tool_name = args.get("tool_name", "unknown")
    success = args.get("success", True)
    files_affected = args.get("files_affected", [])
    duration_ms = args.get("duration_ms", 0)
    output = args.get("output", "")
    error = args.get("error")

    try:
        # Build inputs dict
        inputs = {
            "tool_name": tool_name,
            "files_affected": files_affected,
        }

        # Build outputs dict
        outputs = {
            "output": output,
            "duration_ms": duration_ms,
        }
        if error:
            outputs["error"] = error

        # Record the action
        # Note: record_simple takes "instruction" not "description"
        receipt_hash = runtime.record_simple(
            action_type=action_type,
            instruction=description,  # MCP calls it "description", runtime calls it "instruction"
            inputs=inputs,
            outputs=outputs,
            success=success,
            duration_ms=duration_ms,
        )

        result = {
            "recorded": True,
            "receipt_hash": receipt_hash,
            "actions_count": len(runtime.actions),
        }

    except Exception as e:
        result = {
            "recorded": False,
            "error": str(e),
            "actions_count": len(runtime.actions) if hasattr(runtime, 'actions') else 0,
        }

    return [TextContent(type="text", text=json.dumps(result))]


async def _handle_seal(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_seal tool call."""
    global _runtime, _runtime_path

    runtime = _get_runtime()
    session_name = args.get("session_name", "mcp-session")

    try:
        # Finalize the runtime (generates proof)
        capsule = runtime.finalize(prove=True)

        # Create .cap file
        from .cli.cap_format import create_run_cap_file

        cap_path = _runtime_path.parent / f"{_runtime_path.name}.cap"
        create_run_cap_file(
            run_dir=_runtime_path,
            output_path=cap_path,
            run_type="mcp",
            extras={
                "session_name": session_name,
                "actions_count": len(runtime.actions),
            },
        )

        # Update latest.cap symlink
        latest_cap_link = _runtime_path.parent / "latest.cap"
        if latest_cap_link.is_symlink() or latest_cap_link.exists():
            latest_cap_link.unlink()
        latest_cap_link.symlink_to(cap_path.name)

        # Update latest symlink
        latest_link = _runtime_path.parent / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(_runtime_path.name)

        result = {
            "sealed": True,
            "cap_file": str(cap_path),
            "chain_hash": capsule.get("capsule_hash", capsule.get("run_hash", ""))[:16] + "...",
            "actions_sealed": len(runtime.actions),
        }

        # Reset runtime for next session
        _runtime = None
        _runtime_path = None

    except Exception as e:
        result = {
            "sealed": False,
            "error": str(e),
            "actions_sealed": 0,
        }

    return [TextContent(type="text", text=json.dumps(result))]


async def _handle_status(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_status tool call."""
    global _runtime, _runtime_path, _workspace

    workspace = Path(_workspace) if _workspace else Path.cwd()

    # Check if we have an active session
    if _runtime is None:
        result = {
            "session_active": False,
            "actions_recorded": 0,
            "actions": [],
            "last_gate_decision": None,
            "last_gate_predicted_failure": None,
            "sealed": False,
            "workspace": str(workspace),
            "posteriors_loaded": False,
        }
        return [TextContent(type="text", text=json.dumps(result))]

    runtime = _runtime

    # Build actions list from recorded actions
    actions_list = []
    for i, action in enumerate(runtime.actions):
        actions_list.append({
            "index": i,
            "type": action.get("action_type", "unknown"),
            "description": action.get("instruction", action.get("description", "")),
            "receipt_hash": action.get("receipt_hash", "")[:16] + "..." if action.get("receipt_hash") else "",
        })

    # Get last gate info if available
    last_gate_decision = None
    last_gate_failure = None
    if hasattr(runtime, '_last_gate_result') and runtime._last_gate_result:
        last_gate_decision = runtime._last_gate_result.get("decision", "approve")
        last_gate_failure = runtime._last_gate_result.get("q", 0.0)

    # Check if posteriors are loaded
    posteriors_loaded = runtime.gate_posteriors is not None if hasattr(runtime, 'gate_posteriors') else False

    result = {
        "session_active": True,
        "actions_recorded": len(runtime.actions),
        "actions": actions_list,
        "last_gate_decision": last_gate_decision,
        "last_gate_predicted_failure": last_gate_failure,
        "sealed": False,  # If we're here, not sealed yet
        "workspace": str(workspace),
        "posteriors_loaded": posteriors_loaded,
    }

    return [TextContent(type="text", text=json.dumps(result))]


async def run_server():
    """Run the MCP server with stdio transport."""
    global _workspace

    _load_capseal_env()

    # Log startup info to stderr
    workspace = Path(_workspace) if _workspace else Path.cwd()
    print(f"[capseal] MCP server starting (workspace: {workspace})", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_mcp_server(workspace: str | Path | None = None):
    """Entry point for running the MCP server.

    Args:
        workspace: Project directory containing .capseal/. Defaults to cwd.
    """
    global _workspace
    _workspace = str(workspace) if workspace else None
    asyncio.run(run_server())


if __name__ == "__main__":
    # Simple CLI for direct invocation
    import argparse
    parser = argparse.ArgumentParser(description="CapSeal MCP Server")
    parser.add_argument("--workspace", "-w", help="Project directory containing .capseal/")
    args = parser.parse_args()
    run_mcp_server(workspace=args.workspace)
