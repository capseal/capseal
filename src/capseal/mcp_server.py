"""MCP Server for CapSeal - Expose AgentRuntime as MCP tools.

This allows any agent framework that speaks MCP (OpenClaw, Claude Code, Cursor,
LangChain, etc.) to use CapSeal as a trust layer.

Five tools:
    capseal_gate    - Gate a proposed action before execution
    capseal_record  - Record what happened after execution
    capseal_seal    - Seal the session into a .cap receipt
    capseal_status  - Get session state + cross-session history
    capseal_context - Get file change history across all sessions

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
import time as _time
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, PromptMessage, PromptArgument

from .mcp_responses import (
    format_context_response,
    format_gate_response,
    format_record_response,
    format_seal_response,
    format_status_response,
)
from .risk_engine import evaluate_action_risk, to_internal_decision

# Global runtime instance (initialized on first use)
_runtime = None
_runtime_path = None
_workspace = None  # Set via --workspace or defaults to cwd

# Session history cache (invalidated when new .cap files appear)
_session_cache: dict | None = None
_session_cache_count: int = 0  # Number of .cap files at last cache build

# Pending gate decisions — keyed by first file in files_affected
_gate_decisions: dict[str, dict] = {}


def _read_intervention(gate_key: str) -> dict | None:
    """Read and consume a matching intervention from intervention.json."""
    global _workspace
    workspace = Path(_workspace) if _workspace else Path.cwd()
    intervention_path = workspace / ".capseal" / "intervention.json"

    if not intervention_path.exists():
        return None

    try:
        with open(intervention_path) as f:
            queue = json.load(f)

        if not isinstance(queue, list) or not queue:
            return None

        # Find matching intervention (by target file or first in queue)
        match = None
        remaining = []
        for item in queue:
            if match is None:
                target = item.get("target")
                if target is None or target == gate_key:
                    match = item
                    continue
            remaining.append(item)

        # Write remaining back (or delete file if empty)
        if remaining:
            with open(intervention_path, "w") as f:
                json.dump(remaining, f, indent=2)
        else:
            intervention_path.unlink(missing_ok=True)

        return match

    except (json.JSONDecodeError, OSError):
        return None


def _emit_event(event_type: str, summary: str, data: dict | None = None) -> None:
    """Append event to .capseal/events.jsonl for live status consumers."""
    global _workspace
    workspace = Path(_workspace) if _workspace else Path.cwd()
    events_path = workspace / ".capseal" / "events.jsonl"
    event = {"type": event_type, "timestamp": _time.time(), "summary": summary}
    if data:
        event["data"] = data
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


def _get_similar_patches(grid_idx: int, max_results: int = 3) -> list[dict]:
    """Find past episodes with matching risk profile for instant preview.

    Matches by grid_idx first. If no exact matches, returns the most recent
    episodes as context regardless of grid_idx.
    """
    global _workspace
    workspace = Path(_workspace) if _workspace else Path.cwd()
    history_path = workspace / ".capseal" / "models" / "episode_history.jsonl"
    if not history_path.exists():
        return []

    exact: list[dict] = []
    recent: list[dict] = []
    try:
        with open(history_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                item = {
                    "description": entry.get("description", "unknown"),
                    "success": entry.get("success", True),
                    "round": entry.get("round_num", "?"),
                }
                if entry.get("grid_idx") == grid_idx:
                    exact.append(item)
                recent.append(item)
    except OSError:
        return []

    # Prefer exact grid_idx matches; fall back to recent episodes
    if exact:
        return exact[-max_results:]
    return recent[-max_results:]


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

    # Emit session_start event for operator/TUI
    agent_name = os.environ.get("CAPSEAL_AGENT", "unknown")
    _emit_event("session_start", f"Session started: {agent_name}", data={
        "session_id": run_dir.name,
        "agent": agent_name,
        "workspace": str(workspace),
        "model": os.environ.get("CAPSEAL_MODEL", "unknown"),
    })

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


def _auto_learn_from_session(runtime) -> tuple[bool, int | None]:
    """Update Beta posteriors based on this session's gate decisions.

    Returns (model_updated, total_episodes).
    """
    global _workspace
    workspace = Path(_workspace) if _workspace else Path.cwd()
    model_path = workspace / ".capseal" / "models" / "beta_posteriors.npz"
    if not model_path.exists():
        return False, None  # No model — need initial training first

    try:
        import numpy as np

        data = np.load(model_path, allow_pickle=True)
        alphas = data["alpha"].copy() if "alpha" in data else data["alphas"].copy()
        betas = data["beta"].copy() if "beta" in data else data["betas"].copy()

        actions = runtime.actions
        updated = False
        session_episodes = 0

        for action in actions:
            # The gate produces grid_idx; stored in runtime's gate result cache.
            # We can reconstruct from the action's features or use stored grid_idx.
            # For simplicity, look up gate_score and decide: if gated, update.
            if action.gate_score is None:
                continue  # Not gated

            # Extract grid_idx from action features (re-compute)
            try:
                from capseal.shared.features import (
                    extract_patch_features,
                    discretize_features,
                    features_to_grid_idx,
                )
                # Build a minimal diff from metadata
                metadata = action.metadata or {}
                files = metadata.get("files_affected", [])
                diff_preview = ""
                for f in files:
                    diff_preview += f"diff --git a/{f} b/{f}\n+++ b/{f}\n@@ -1,5 @@\n"

                features = extract_patch_features(diff_preview, [{"severity": "warning"}])
                discrete = discretize_features(features)
                grid_idx = features_to_grid_idx(discrete)
            except Exception:
                grid_idx = 0

            if grid_idx < 0 or grid_idx >= len(alphas):
                continue

            if action.success:
                betas[grid_idx] += 1
            else:
                alphas[grid_idx] += 1
            updated = True
            session_episodes += 1

        if updated:
            n_episodes = int(data.get("n_episodes", 0)) if "n_episodes" in data else 0
            np.savez(
                model_path,
                alpha=alphas,
                beta=betas,
                n_episodes=n_episodes + session_episodes,
                **{k: data[k] for k in data.files if k not in ("alpha", "beta", "alphas", "betas", "n_episodes")},
            )
            return True, n_episodes + session_episodes

    except Exception:
        pass

    return False, None


def _get_session_history() -> dict:
    """Scan .capseal/runs/ for past sessions and build a summary.

    Returns a dict with 'recent_sessions' and 'project_stats'.
    Uses a cache invalidated when new .cap files appear.
    """
    global _session_cache, _session_cache_count, _workspace

    workspace = Path(_workspace) if _workspace else Path.cwd()
    runs_dir = workspace / ".capseal" / "runs"

    if not runs_dir.exists():
        return {"recent_sessions": [], "project_stats": {}}

    cap_files = sorted(
        [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
        key=lambda f: f.name,
        reverse=True,
    )

    # Return cache if still valid
    if _session_cache is not None and len(cap_files) == _session_cache_count:
        return _session_cache

    recent_sessions = []
    total_actions = 0
    total_denied = 0

    for cap_file in cap_files[:10]:  # Last 10 sessions
        session_info = _extract_session_info(cap_file, runs_dir)
        if session_info:
            recent_sessions.append(session_info)
            total_actions += session_info.get("actions", 0)
            total_denied += session_info.get("denied", 0)

    # Model stats
    posteriors = workspace / ".capseal" / "models" / "beta_posteriors.npz"
    model_trained = posteriors.exists()

    result = {
        "recent_sessions": recent_sessions,
        "project_stats": {
            "total_sessions": len(cap_files),
            "total_actions": total_actions,
            "total_denied": total_denied,
            "model_trained": model_trained,
        },
    }

    _session_cache = result
    _session_cache_count = len(cap_files)
    return result


def _extract_session_info(cap_file: Path, runs_dir: Path) -> dict | None:
    """Extract session summary from a .cap file and its run directory."""
    name = cap_file.stem
    run_dir = runs_dir / name

    # Parse date
    try:
        ts_part = name[:15]
        dt = datetime.datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
        date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    except (ValueError, IndexError):
        date_str = ""

    # Read actions from run directory or .cap file
    actions = _load_actions_for_session(cap_file, run_dir)

    if not actions:
        # Try manifest for basic info
        try:
            from .cli.cap_format import read_cap_manifest
            manifest = read_cap_manifest(cap_file)
            return {
                "date": date_str,
                "agent": manifest.extras.get("agent", "unknown"),
                "actions": manifest.extras.get("actions_count", 0),
                "approved": 0,
                "denied": 0,
                "files_changed": [],
                "summary": "",
                "receipt": cap_file.name,
            }
        except Exception:
            return None

    # Build summary from actions
    files_changed = set()
    descriptions = []
    approved = 0
    denied = 0
    agent = "unknown"

    # Check run metadata for agent name
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            agent = meta.get("agent", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    for action in actions:
        metadata = action.get("metadata") or {}
        desc = metadata.get("description", "")
        files = metadata.get("files_affected", [])
        gate = action.get("gate_decision")

        if desc:
            descriptions.append(desc[:60])
        files_changed.update(files)

        if gate in ("skip", "deny"):
            denied += 1
        else:
            approved += 1

    return {
        "date": date_str,
        "agent": agent,
        "actions": len(actions),
        "approved": approved,
        "denied": denied,
        "files_changed": sorted(files_changed),
        "summary": ", ".join(descriptions[:8]),
        "receipt": cap_file.name,
    }


def _load_actions_for_session(
    cap_file: Path, run_dir: Path,
) -> list[dict]:
    """Load actions from a run directory or .cap file."""
    # Try run directory first (faster)
    actions_file = run_dir / "actions.jsonl"
    if actions_file.exists():
        return _parse_actions_jsonl(actions_file.read_text())

    # Fall back to extracting from .cap tarball
    import tarfile
    try:
        with tarfile.open(cap_file, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        return _parse_actions_jsonl(f.read().decode("utf-8"))
    except Exception:
        pass

    return []


def _parse_actions_jsonl(content: str) -> list[dict]:
    """Parse JSONL content into a list of action dicts."""
    actions = []
    for line in content.strip().split("\n"):
        if line.strip():
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return actions


# Create the MCP server
server = Server("capseal")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available CapSeal tools."""
    return [
        Tool(
            name="capseal_gate",
            description="⚠️ REQUIRED before EACH code change. Gate a proposed action through the learned risk model. Call this ONCE PER FILE before editing — do not batch multiple files into one gate call. Returns approve/deny/flag. If 'deny': DO NOT proceed, tell user it was blocked.",
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
            description="⚠️ REQUIRED after EACH code change. Record what you did for the cryptographic audit trail. Call this ONCE PER FILE — one record per file changed, not one summary record for all changes. Include action type, tool used, and success status.",
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
            description="⚠️ Call this at the START of every session and after any conversation compaction/summary. Returns current session state PLUS recent_sessions showing what other agents/sessions have done on this project. Use recent_sessions to avoid re-fixing issues and to understand project history.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="capseal_context",
            description="Get the full change history for a specific file across all CapSeal sessions. Call this before modifying a file to see what previous agents have done to it and avoid duplicate work or conflicting changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "File path to look up (relative to project root)",
                    },
                },
                "required": ["file"],
            },
        ),
    ]


# ── Prompts ──────────────────────────────────────────────────────────────────

CAPSEAL_SYSTEM_PROMPT = """You have CapSeal tools available for verified code execution.

**At the START of every session (or after any conversation compaction/summary):**
Call `capseal_status` to check session state and see what other agents have done on this project. The response includes `recent_sessions` showing past sessions from ALL agents, and `project_stats` with overall project metrics. Use this to:
- Avoid re-fixing issues that were already fixed in a previous session
- Be cautious about changes that were denied in previous sessions
- Reference previous work naturally: "I see a previous session already fixed the SQL injection in database.py"

**Before modifying a file that may have been changed before:**
Call `capseal_context` with the file path to see its full change history across all sessions.

**Before making any code change (ONCE PER FILE):**
Call `capseal_gate` with a description of what you plan to change. Gate each file INDIVIDUALLY — do not batch multiple files into one gate call.
- If decision is "approve": proceed with the change
- If decision is "deny": do NOT proceed — explain to the user why it was blocked (high predicted failure rate)
- If decision is "flag": proceed with caution, mention the uncertainty to the user

**After each code change (ONCE PER FILE):**
Call `capseal_record` to log what happened. Record each file change individually — one record per file, not one summary for all changes. Include action type, tool used, and success/failure.

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
    elif name == "capseal_context":
        return await _handle_context(arguments)
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}",
        }))]


async def _handle_gate(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_gate tool call."""
    global _gate_decisions
    runtime = _get_runtime()

    action_type = args.get("action_type", "unknown")
    description = args.get("description", "")
    files_affected = args.get("files_affected", [])
    diff_text = args.get("diff_text", "")

    # Key for linking this gate to its record
    gate_key = files_affected[0] if files_affected else "_default"

    workspace = Path(_workspace) if _workspace else Path.cwd()
    risk = evaluate_action_risk(
        action_type=action_type,
        description=description,
        files_affected=files_affected,
        diff_text=diff_text,
        workspace=workspace,
    )

    session_id = _runtime_path.name if _runtime_path else "unknown"
    result = format_gate_response(
        risk,
        session_id=session_id,
        action_type=action_type,
        description=description,
        files_affected=files_affected,
    )
    result["similar_patches"] = _get_similar_patches(risk.grid_cell)

    # Check for operator intervention (approve/deny override from Telegram etc.)
    intervention = _read_intervention(gate_key)
    if intervention:
        action = intervention.get("action", "")
        if action in ("approve", "override_approve") and result["decision"] != "approve":
            result["decision"] = "approve"
            result["reason"] = "Operator override: approved"
            print(f"[capseal] Operator override: approved {gate_key}", file=sys.stderr)
        elif action in ("deny", "override_deny") and result["decision"] != "deny":
            result["decision"] = "deny"
            result["reason"] = "Operator override: denied"
            print(f"[capseal] Operator override: denied {gate_key}", file=sys.stderr)
        decision_text = {
            "approve": "APPROVED",
            "deny": "DENIED",
            "flag": "FLAGGED",
        }.get(result["decision"], str(result["decision"]).upper())
        result["human_summary"] = (
            f"CAPSEAL GATE: {decision_text} | p_fail={result.get('p_fail', 0.0):.2f} | "
            f"{result.get('label', 'unclassified')}"
        )

    # Store for status reporting and for linking to the matching record
    internal = to_internal_decision(result["decision"])
    runtime._last_gate_result = {"decision": result["decision"], "q": result.get("p_fail", 0.0)}
    _gate_decisions[gate_key] = {
        "decision": internal,  # internal form: pass/skip/human_review
        "q": result.get("p_fail", 0.0),
        "confidence": result.get("confidence", 0.0),
        "label": result.get("label", ""),
        "grid_cell": result.get("grid_cell", 0),
    }

    _emit_event("gate", f"{result['decision']}: {description[:60]} (p_fail={result['p_fail']:.2f})", data={
        "decision": result["decision"],
        "p_fail": round(result.get("p_fail", 0.0), 4),
        "files": files_affected,
        "action_type": action_type,
        "description": description,
        "diff": diff_text[:500] if diff_text else None,
        "reason": result.get("reason"),
        "label": result.get("label"),
        "features": result.get("features"),
    })

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_record(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_record tool call."""
    global _gate_decisions
    runtime = _get_runtime()

    action_type = args.get("action_type", "unknown")
    description = args.get("description", "")
    tool_name = args.get("tool_name", "unknown")
    success = args.get("success", True)
    files_affected = args.get("files_affected", [])
    duration_ms = args.get("duration_ms", 0)
    output = args.get("output", "")
    error = args.get("error")

    # Consume the matching gate decision (keyed by first file)
    gate_key = files_affected[0] if files_affected else "_default"
    gate_decision = None
    gate_score = None
    receipt_hash = None
    gate_info = _gate_decisions.pop(gate_key, None)
    if gate_info:
        gate_decision = gate_info.get("decision")  # pass/skip/human_review
        gate_score = gate_info.get("q")
    gate_label = gate_info.get("label") if gate_info else None
    gate_cell = gate_info.get("grid_cell") if gate_info else None

    session_id = _runtime_path.name if _runtime_path else "unknown"
    response: dict[str, Any]

    try:
        inputs = {
            "tool_name": tool_name,
            "files_affected": files_affected,
        }
        outputs = {
            "output": output,
            "duration_ms": duration_ms,
        }
        if error:
            outputs["error"] = error

        receipt_hash = runtime.record_simple(
            action_type=action_type,
            instruction=description,
            inputs=inputs,
            outputs=outputs,
            success=success,
            duration_ms=duration_ms,
            gate_score=gate_score,
            gate_decision=gate_decision,
            metadata={
                "description": description,
                "tool_name": tool_name,
                "files_affected": files_affected,
                "risk_label": gate_label,
                "grid_cell": gate_cell,
            },
        )

        count = len(runtime.actions)
        response = format_record_response(
            recorded=True,
            session_id=session_id,
            action_type=action_type,
            files_affected=files_affected,
            action_id=runtime.actions[-1].action_id if runtime.actions else None,
            receipt_hash=receipt_hash,
            receipt_chain_length=count,
        )

    except Exception as e:
        response = format_record_response(
            recorded=False,
            session_id=session_id,
            action_type=action_type,
            files_affected=files_affected,
            receipt_chain_length=len(runtime.actions),
            error=str(e),
        )

    record_data = {
        "action_type": action_type,
        "files": files_affected,
    }
    if receipt_hash:
        record_data["receipt_hash"] = receipt_hash
    _emit_event("record", f"{action_type}: {description[:60]} sha:{(receipt_hash or '')[:16]}", data=record_data)

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_seal(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_seal tool call."""
    global _runtime, _runtime_path

    runtime = _get_runtime()
    session_name = args.get("session_name", "mcp-session")
    response: dict[str, Any]

    try:
        capsule = runtime.finalize(prove=True)

        from .cli.cap_format import create_run_cap_file

        agent_name = os.environ.get("CAPSEAL_AGENT", "unknown")

        meta_path = _runtime_path / "run_metadata.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        meta["agent"] = agent_name
        meta_path.write_text(json.dumps(meta, indent=2))

        # Extract proof status from capsule
        proof_generated = capsule.get("status") == "proved"
        proof_verified = capsule.get("verified", False)
        proof_type = capsule.get("proof_type", "constraint_check")
        capsule_hash = capsule.get("capsule", {}).get("capsule_hash", "")

        cap_path = _runtime_path.parent / f"{_runtime_path.name}.cap"
        create_run_cap_file(
            run_dir=_runtime_path,
            output_path=cap_path,
            run_type="mcp",
            extras={
                "session_name": session_name,
                "actions_count": len(runtime.actions),
                "agent": agent_name,
                "proof_generated": proof_generated,
                "proof_verified": proof_verified,
                "proof_type": proof_type,
            },
        )

        # Auto-learn from this session's gate outcomes
        model_updated, model_episodes = _auto_learn_from_session(runtime)

        global _session_cache
        _session_cache = None

        latest_cap_link = _runtime_path.parent / "latest.cap"
        if latest_cap_link.is_symlink() or latest_cap_link.exists():
            latest_cap_link.unlink()
        latest_cap_link.symlink_to(cap_path.name)

        latest_link = _runtime_path.parent / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(_runtime_path.name)

        actions_sealed = len(runtime.actions)
        chain_hash_full = capsule.get("final_receipt_hash", "")
        cap_rel = f".capseal/runs/{cap_path.name}"

        _emit_event("seal", f"Sealed {actions_sealed} actions", data={
            "session_id": _runtime_path.name if _runtime_path else None,
            "total_actions": actions_sealed,
            "chain_intact": True,
            "receipt_hash": capsule.get("final_receipt_hash", "")[:32],
        })

        session_id = _runtime_path.name if _runtime_path else None
        _runtime = None
        _runtime_path = None

        response = format_seal_response(
            sealed=True,
            session_id=session_id,
            session_name=session_name,
            cap_file=cap_rel,
            chain_hash=chain_hash_full,
            actions_count=actions_sealed,
            proof_generated=proof_generated,
            proof_verified=proof_verified,
            model_updated=model_updated,
            model_episodes=model_episodes,
        )
        response["proof_type"] = proof_type
        response["capsule_hash"] = capsule_hash
        response["verify_command"] = "capseal verify .capseal/runs/latest.cap"

    except Exception as e:
        _runtime = None
        _runtime_path = None
        response = format_seal_response(
            sealed=False,
            session_id=None,
            session_name=session_name,
            error=str(e),
        )

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_status(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_status tool call."""
    global _runtime, _runtime_path, _workspace

    workspace = Path(_workspace) if _workspace else Path.cwd()
    history = _get_session_history()
    stats = history.get("project_stats", {})
    recent = history.get("recent_sessions", [])

    if _runtime is None:
        session_active = False
        session_id = None
        actions_count = 0
        denials_count = 0
        uptime_seconds = 0
        model_loaded = (workspace / ".capseal" / "models" / "beta_posteriors.npz").exists()
    else:
        runtime = _runtime
        session_active = True
        session_id = _runtime_path.name if _runtime_path else None
        actions_count = len(runtime.actions)
        denials_count = sum(
            1
            for action in runtime.actions
            if action.gate_decision in ("skip", "deny")
        )
        uptime_seconds = int(_time.time() - getattr(runtime, "_start_time", _time.time()))
        model_loaded = runtime.gate_posteriors is not None if hasattr(runtime, "gate_posteriors") else False

    response = format_status_response(
        session_id=session_id,
        session_active=session_active,
        workspace=str(workspace),
        actions_count=actions_count,
        gates_count=actions_count,
        denials_count=denials_count,
        model_loaded=model_loaded,
        uptime_seconds=uptime_seconds,
        recent_sessions=recent[:10],
        project_stats=stats,
    )
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_context(args: dict[str, Any]) -> list[TextContent]:
    """Handle capseal_context tool call."""
    global _workspace

    workspace = Path(_workspace) if _workspace else Path.cwd()
    target_file = args.get("file", "")

    if not target_file:
        return [TextContent(type="text", text=json.dumps({
            "error": "file parameter is required",
            "schema_version": "1.0",
        }))]

    target_file = target_file.lstrip("./")

    runs_dir = workspace / ".capseal" / "runs"
    if not runs_dir.exists():
        response = format_context_response(
            file_path=target_file,
            total_changes=0,
            total_sessions=0,
            sessions=[],
        )
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    # Scan all sessions for changes to this file
    cap_files = sorted(
        [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
        key=lambda f: f.name,
    )

    # Group changes by session
    sessions_with_changes: dict[str, list[dict]] = {}
    session_meta: dict[str, dict] = {}

    for cap_file in cap_files:
        run_dir = runs_dir / cap_file.stem
        actions = _load_actions_for_session(cap_file, run_dir)

        agent = "unknown"
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                agent = meta.get("agent", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        try:
            ts_part = cap_file.stem[:15]
            dt = datetime.datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
            session_date = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, IndexError):
            session_date = "unknown"

        for action in actions:
            metadata = action.get("metadata") or {}
            files = metadata.get("files_affected", [])

            if not any(target_file in f or f in target_file for f in files):
                continue

            desc = metadata.get("description", action.get("action_type", "unknown"))
            gate_decision = action.get("gate_decision")
            gate_score = action.get("gate_score")

            decision_map = {"pass": "approved", "skip": "denied", "human_review": "flagged"}
            gate_str = decision_map.get(gate_decision, "ungated")
            if gate_score is not None:
                gate_str += f", p_fail={gate_score:.2f}"

            key = cap_file.stem
            if key not in sessions_with_changes:
                sessions_with_changes[key] = []
                session_meta[key] = {"date": session_date, "agent": agent}
            sessions_with_changes[key].append(f"{desc} ({gate_str})")

    total_changes = sum(len(v) for v in sessions_with_changes.values())
    total_sessions = len(sessions_with_changes)
    sessions: list[dict[str, Any]] = []
    for sess_key, changes in sessions_with_changes.items():
        meta = session_meta.get(sess_key, {})
        sessions.append({
            "session_id": sess_key,
            "date": meta.get("date", ""),
            "agent": meta.get("agent", "unknown"),
            "changes": changes,
        })

    response = format_context_response(
        file_path=target_file,
        total_changes=total_changes,
        total_sessions=total_sessions,
        sessions=sessions,
    )
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


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
