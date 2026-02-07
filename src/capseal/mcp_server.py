"""CapSeal MCP Server - expose deterministic CLI tools to IDE agents.

Security model:
- Strict allowlist of subcommands + flags
- Workspace-only path validation
- Capped output sizes
- Hash-chained append-only event log

Usage:
    python -m bef_zk.capsule.mcp_server

Configure in Cline (~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json):
    {
      "mcpServers": {
        "capseal": {
          "command": "python",
          "args": ["-m", "bef_zk.capsule.mcp_server"],
          "env": {
            "CAPSEAL_WORKSPACE_ROOT": "/home/ryan/BEF-main"
          }
        }
      }
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# ============================================================
# Policy knobs
# ============================================================
CAPSEAL_BIN = os.environ.get("CAPSEAL_BIN", str(Path(__file__).parent.parent.parent / "capseal"))
MAX_OUTPUT_BYTES = 2_000_000  # 2MB
WORKSPACE_ROOT = os.environ.get("CAPSEAL_WORKSPACE_ROOT", os.getcwd())
EVENT_LOG_PATH = os.environ.get(
    "CAPSEAL_MCP_EVENT_LOG",
    os.path.join(WORKSPACE_ROOT, ".capseal", "mcp_events.jsonl")
)

ALLOWED_TOOLS = {"doctor", "verify", "audit", "row", "diff_bundle", "spawn_agent", "load_context", "save_result", "collect_results", "greptile_review"}


# ============================================================
# Utilities
# ============================================================
def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_within_workspace(path: str) -> bool:
    """Check path is within workspace (no escape via symlinks)."""
    try:
        rp = os.path.realpath(path)
        wr = os.path.realpath(WORKSPACE_ROOT)
        return rp == wr or rp.startswith(wr + os.sep)
    except (OSError, ValueError):
        return False


def _run_checked(args: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Run a command with safety + bounded output."""
    start = _now_ms()
    try:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # Get bytes, decode with error handling
            check=False,
            cwd=cwd,
            timeout=120,  # 2 minute timeout
        )
        dur = _now_ms() - start

        # Decode bytes with error handling (handles binary in diff output)
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")

        # Cap output
        if len(out) > MAX_OUTPUT_BYTES:
            out = out[:MAX_OUTPUT_BYTES] + "\n…(truncated)…"
        if len(err) > MAX_OUTPUT_BYTES:
            err = err[:MAX_OUTPUT_BYTES] + "\n…(truncated)…"

        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "duration_ms": dur,
            "stdout": out,
            "stderr": err,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -1,
            "duration_ms": _now_ms() - start,
            "stdout": "",
            "stderr": "Command timed out after 120s",
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "duration_ms": _now_ms() - start,
            "stdout": "",
            "stderr": str(e),
        }


# ============================================================
# Hash-chained event log (receipts for tool invocations)
# ============================================================
_last_hash: str | None = None


def _log_event(tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    """Append event to hash-chained log. Returns event hash."""
    global _last_hash

    os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)

    # Load last hash from file if we don't have it
    if _last_hash is None:
        try:
            with open(EVENT_LOG_PATH, "r") as f:
                lines = f.readlines()
                if lines:
                    last_event = json.loads(lines[-1])
                    _last_hash = last_event.get("event_hash", "")
        except (FileNotFoundError, json.JSONDecodeError):
            _last_hash = ""

    event = {
        "ts_ms": _now_ms(),
        "tool": tool,
        "args": args,
        "result_ok": result.get("ok", False),
        "result_returncode": result.get("returncode"),
        "duration_ms": result.get("duration_ms"),
        "prev_hash": _last_hash or "",
    }

    # Compute hash of this event (without event_hash field)
    event_bytes = json.dumps(event, sort_keys=True, ensure_ascii=False).encode()
    event_hash = hashlib.sha256(event_bytes).hexdigest()[:32]
    event["event_hash"] = event_hash

    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    _last_hash = event_hash
    return event_hash


# ============================================================
# Tool implementations
# ============================================================
def tool_verify(
    capsule_path: str,
    dataset: list[str] | None = None,
    json_out: bool = True,
) -> dict[str, Any]:
    """Verify a capsule cryptographic proof.

    Args:
        capsule_path: Path to capsule JSON file
        dataset: Optional list of dataset bindings (format: "id=path")
        json_out: Return JSON output (default True)

    Returns:
        Verification result with proof status
    """
    if not _is_within_workspace(capsule_path):
        return {"ok": False, "error": "capsule_path outside workspace"}

    argv = [CAPSEAL_BIN, "verify", capsule_path]
    if dataset:
        for d in dataset:
            argv += ["--dataset", d]
    if json_out:
        argv += ["--json"]

    result = _run_checked(argv)
    _log_event("verify", {"capsule_path": capsule_path, "dataset": dataset}, result)
    return result


def tool_audit(
    capsule_path: str,
    json_out: bool = True,
) -> dict[str, Any]:
    """Export and inspect audit trail from a capsule.

    Args:
        capsule_path: Path to capsule JSON file
        json_out: Return JSON output (default True)

    Returns:
        Audit trail with hash chain verification
    """
    if not _is_within_workspace(capsule_path):
        return {"ok": False, "error": "capsule_path outside workspace"}

    argv = [CAPSEAL_BIN, "audit", capsule_path]
    if json_out:
        argv += ["--json"]

    result = _run_checked(argv)
    _log_event("audit", {"capsule_path": capsule_path}, result)
    return result


def tool_row(
    capsule_path: str,
    row: int,
    dataset: list[str] | None = None,
    policy: str | None = None,
    ticket: str | None = None,
    opening_state_dir: str | None = None,
    allow_opening_reset: bool = False,
    json_out: bool = True,
) -> dict[str, Any]:
    """Open a specific trace row with STC membership proof.

    Args:
        capsule_path: Path to capsule JSON file
        row: Row index to open
        dataset: Optional dataset bindings
        policy: Optional policy file path
        ticket: Optional ticket file path
        opening_state_dir: Directory for opening state persistence
        allow_opening_reset: Allow resetting opening state
        json_out: Return JSON output (default True)

    Returns:
        Row data with membership proof
    """
    if not _is_within_workspace(capsule_path):
        return {"ok": False, "error": "capsule_path outside workspace"}

    argv = [CAPSEAL_BIN, "row", capsule_path, "--row", str(row)]

    if dataset:
        for d in dataset:
            argv += ["--dataset", d]
    if policy:
        if not _is_within_workspace(policy):
            return {"ok": False, "error": "policy path outside workspace"}
        argv += ["--policy", policy]
    if ticket:
        if not _is_within_workspace(ticket):
            return {"ok": False, "error": "ticket path outside workspace"}
        argv += ["--ticket", ticket]
    if opening_state_dir:
        if not _is_within_workspace(opening_state_dir):
            return {"ok": False, "error": "opening_state_dir outside workspace"}
        argv += ["--opening-state-dir", opening_state_dir]
    if allow_opening_reset:
        argv += ["--allow-opening-reset"]
    if json_out:
        argv += ["--json"]

    result = _run_checked(argv)
    _log_event("row", {"capsule_path": capsule_path, "row": row}, result)
    return result


def tool_doctor(
    capsule_path: str,
    out_dir: str | None = None,
    sample_rows: int = 1,
    json_out: bool = True,
) -> dict[str, Any]:
    """One-click pipeline verification with derived reports.

    Args:
        capsule_path: Path to capsule JSON file
        out_dir: Output directory for reports (default: alongside capsule)
        sample_rows: Number of rows to sample for verification (default: 1)
        json_out: Return JSON output (default True)

    Returns:
        Verification report with all checks
    """
    if not _is_within_workspace(capsule_path):
        return {"ok": False, "error": "capsule_path outside workspace"}

    argv = [CAPSEAL_BIN, "doctor", capsule_path]

    if out_dir:
        if not _is_within_workspace(out_dir):
            return {"ok": False, "error": "out_dir outside workspace"}
        argv += ["-o", out_dir]
    if sample_rows is not None:
        argv += ["--sample-rows", str(sample_rows)]
    if json_out:
        argv += ["--json"]

    result = _run_checked(argv)
    _log_event("doctor", {"capsule_path": capsule_path, "sample_rows": sample_rows}, result)
    return result


def _call_anthropic_api(prompt: str, timeout: int = 120) -> tuple[bool, str]:
    """Call Anthropic Claude API directly."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return False, "ANTHROPIC_API_KEY not set"

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    data = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            content = result.get("content", [])
            if content and len(content) > 0:
                return True, content[0].get("text", "")
            return False, "Empty response"
    except urllib.error.HTTPError as e:
        return False, f"API error {e.code}: {e.read().decode()[:500]}"
    except Exception as e:
        return False, f"Error: {e}"


def _call_gemini_api(prompt: str, timeout: int = 120) -> tuple[bool, str]:
    """Call Google Gemini API directly."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY not set"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"content-type": "application/json"}

    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 4096},
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return True, parts[0].get("text", "")
            return False, "Empty response"
    except urllib.error.HTTPError as e:
        return False, f"API error {e.code}: {e.read().decode()[:500]}"
    except Exception as e:
        return False, f"Error: {e}"


def _call_openai_api(prompt: str, timeout: int = 120) -> tuple[bool, str]:
    """Call OpenAI API directly."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }

    data = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            if choices:
                return True, choices[0].get("message", {}).get("content", "")
            return False, "Empty response"
    except urllib.error.HTTPError as e:
        return False, f"API error {e.code}: {e.read().decode()[:500]}"
    except Exception as e:
        return False, f"Error: {e}"


def tool_spawn_agent(
    task: str,
    agent_id: str | None = None,
    context_name: str = "latest",
    timeout: int = 120,
    backend: str = "auto",
) -> dict[str, Any]:
    """Spawn a subagent to work on a specific task.

    Use this to parallelize work. Each subagent gets the context checkpoint
    and works on its assigned task independently.

    Args:
        task: The specific task for this subagent (be detailed!)
        agent_id: Optional ID for this agent (auto-generated if not provided)
        context_name: Context checkpoint to load (default: latest)
        timeout: Max seconds for subagent to run (default: 120)
        backend: Which API to use: "anthropic", "gemini", "openai", or "auto" (tries in order)

    Returns:
        Subagent result with findings
    """
    import uuid

    agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"

    # Load context for the subagent
    context_prompt = ""
    try:
        from capseal.cli.context import load_context
        ctx = load_context(context_name)
        if ctx:
            summary = ctx.get("summary", {})
            context_prompt = f"""
CONTEXT: Comparing {summary.get('comparison', 'unknown')}
Repository: {summary.get('repo', 'unknown')}
Committed changes: {summary.get('total_files', 0)} files
Uncommitted changes: {summary.get('uncommitted_files', 0)} files

You are a code review subagent. Analyze and return findings.
"""
    except ImportError:
        pass

    # Build subagent prompt with strict JSON output requirement
    subagent_prompt = f"""{context_prompt}
YOUR TASK: {task}

CRITICAL: You MUST output ONLY a JSON block in this exact format, nothing else:

```json
{{
  "scope": "your_scope_name",
  "summary": "one sentence assessment",
  "findings": [
    {{
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "description",
      "file": "path/to/file",
      "line": 123,
      "recommendation": "how to fix"
    }}
  ]
}}
```

If you find no issues, return: {{"scope": "your_scope", "summary": "No issues found", "findings": []}}
"""

    start = _now_ms()

    # Try backends in order based on preference
    backends_to_try = []
    if backend == "auto":
        # Try in order: anthropic, gemini, openai
        if os.environ.get("ANTHROPIC_API_KEY"):
            backends_to_try.append(("anthropic", _call_anthropic_api))
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            backends_to_try.append(("gemini", _call_gemini_api))
        if os.environ.get("OPENAI_API_KEY"):
            backends_to_try.append(("openai", _call_openai_api))
    elif backend == "anthropic":
        backends_to_try.append(("anthropic", _call_anthropic_api))
    elif backend == "gemini":
        backends_to_try.append(("gemini", _call_gemini_api))
    elif backend == "openai":
        backends_to_try.append(("openai", _call_openai_api))

    if not backends_to_try:
        response = {
            "ok": False,
            "agent_id": agent_id,
            "task": task[:100],
            "duration_ms": 0,
            "output": "No API keys configured. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY",
            "backend": None,
        }
        _log_event("spawn_agent", {"agent_id": agent_id, "task": task[:100]}, response)
        return response

    # Try each backend until one succeeds
    output = ""
    ok = False
    used_backend = None

    for backend_name, backend_fn in backends_to_try:
        ok, output = backend_fn(subagent_prompt, timeout)
        used_backend = backend_name
        if ok:
            break

    duration = _now_ms() - start

    # Truncate if too long
    if len(output) > 50000:
        output = output[:50000] + "\n...(truncated)..."

    response = {
        "ok": ok,
        "agent_id": agent_id,
        "task": task[:100],
        "duration_ms": duration,
        "output": output,
        "backend": used_backend,
    }

    _log_event("spawn_agent", {"agent_id": agent_id, "task": task[:100]}, response)
    return response


def tool_load_context(
    name: str = "latest",
    format: str = "summary",
) -> dict[str, Any]:
    """Load a diff context checkpoint.

    Use this to understand what changed between repos/refs.

    Args:
        name: Checkpoint name (default: latest)
        format: 'summary' for overview, 'full' for complete diffs

    Returns:
        Context checkpoint data
    """
    try:
        from capseal.cli.context import load_context, format_context_for_agent
        ctx = load_context(name)
        if not ctx:
            return {"ok": False, "error": f"No checkpoint found: {name}"}

        if format == "full":
            return {
                "ok": True,
                "checkpoint_id": ctx.get("checkpoint_id"),
                "prompt": format_context_for_agent(ctx)[:30000],
            }
        else:
            return {
                "ok": True,
                "checkpoint_id": ctx.get("checkpoint_id"),
                "name": ctx.get("name"),
                "created_at": ctx.get("created_at"),
                "summary": ctx.get("summary"),
                "files": ctx.get("files", [])[:30],
                "uncommitted": [f["path"] for f in ctx.get("working_tree", {}).get("files", [])[:30]] if ctx.get("working_tree") else [],
            }
    except ImportError:
        return {"ok": False, "error": "Context module not available"}


def tool_save_result(
    agent_id: str,
    result_type: str,
    content: str,
) -> dict[str, Any]:
    """Save a result from a subagent for the lead agent to collect.

    Args:
        agent_id: ID of the agent saving the result
        result_type: Type of result (e.g., 'findings', 'recommendation', 'diff_analysis')
        content: The result content

    Returns:
        Confirmation with result ID
    """
    results_dir = Path(WORKSPACE_ROOT) / ".capseal" / "agent_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_id = f"{agent_id}_{int(time.time())}"
    result_file = results_dir / f"{result_id}.json"

    result_data = {
        "result_id": result_id,
        "agent_id": agent_id,
        "result_type": result_type,
        "content": content,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    result_file.write_text(json.dumps(result_data, indent=2))

    _log_event("save_result", {"agent_id": agent_id, "result_type": result_type}, {"ok": True})

    return {
        "ok": True,
        "result_id": result_id,
        "path": str(result_file),
    }


def tool_collect_results(
    agent_ids: list[str] | None = None,
    result_type: str | None = None,
    since_minutes: int = 30,
) -> dict[str, Any]:
    """Collect results from subagents.

    Use this to gather all findings from spawned subagents.

    Args:
        agent_ids: Filter to specific agent IDs (default: all)
        result_type: Filter to specific result type (default: all)
        since_minutes: Only include results from last N minutes (default: 30)

    Returns:
        All matching subagent results
    """
    results_dir = Path(WORKSPACE_ROOT) / ".capseal" / "agent_results"

    if not results_dir.exists():
        return {"ok": True, "results": [], "count": 0}

    cutoff = time.time() - (since_minutes * 60)
    collected = []

    for result_file in results_dir.glob("*.json"):
        try:
            result = json.loads(result_file.read_text())

            # Apply filters
            if agent_ids and result.get("agent_id") not in agent_ids:
                continue
            if result_type and result.get("result_type") != result_type:
                continue

            # Check timestamp
            file_mtime = result_file.stat().st_mtime
            if file_mtime < cutoff:
                continue

            collected.append(result)
        except (json.JSONDecodeError, OSError):
            continue

    # Sort by timestamp
    collected.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    _log_event("collect_results", {"agent_ids": agent_ids, "result_type": result_type}, {"ok": True, "count": len(collected)})

    return {
        "ok": True,
        "results": collected,
        "count": len(collected),
    }


def tool_greptile_review(
    repo: str,
    context_name: str = "latest",
    focus: str | None = None,
) -> dict[str, Any]:
    """Review code changes using Greptile's codebase understanding.

    Loads a context checkpoint and asks Greptile to review the diff
    against the indexed codebase for bugs, security issues, and suggestions.

    Args:
        repo: Repository to review against (format: owner/repo or github:owner/repo:branch)
        context_name: Context checkpoint to review (default: latest)
        focus: Optional focus area (e.g., 'security', 'performance', 'style')

    Returns:
        Review with summary, concerns, suggestions, and verdict
    """
    try:
        from capseal.cli.greptile import greptile_review_api
        result = greptile_review_api(repo=repo, context_name=context_name, focus=focus)
        _log_event("greptile_review", {"repo": repo, "context": context_name, "focus": focus}, result)
        return result
    except ImportError:
        return {"ok": False, "error": "Greptile module not available"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_diff_bundle(
    repo_path: str,
    base_ref: str,
    head_ref: str,
    paths: list[str] | None = None,
    context_lines: int = 5,
) -> dict[str, Any]:
    """Generate a diff artifact between two git refs.

    Args:
        repo_path: Path to git repository
        base_ref: Base git reference (commit, branch, tag)
        head_ref: Head git reference
        paths: Optional list of paths to limit diff
        context_lines: Lines of context around changes (default: 5)

    Returns:
        Diff artifact with patch text, file list, and metadata
    """
    if not _is_within_workspace(repo_path):
        return {"ok": False, "error": "repo_path outside workspace"}

    # Get the diff
    diff_argv = [
        "git", "-C", repo_path, "diff",
        f"{base_ref}..{head_ref}",
        f"--unified={context_lines}",
    ]
    if paths:
        diff_argv += ["--"] + paths

    diff_result = _run_checked(diff_argv)
    if not diff_result.get("ok"):
        _log_event("diff_bundle", {"repo_path": repo_path, "base_ref": base_ref, "head_ref": head_ref}, diff_result)
        return diff_result

    # Get file list
    files_argv = [
        "git", "-C", repo_path, "diff",
        "--name-only",
        f"{base_ref}..{head_ref}",
    ]
    if paths:
        files_argv += ["--"] + paths

    files_result = _run_checked(files_argv)
    file_list = [ln for ln in files_result.get("stdout", "").splitlines() if ln.strip()]

    # Get commit info
    base_info = _run_checked(["git", "-C", repo_path, "rev-parse", base_ref])
    head_info = _run_checked(["git", "-C", repo_path, "rev-parse", head_ref])

    result = {
        "ok": True,
        "repo_path": repo_path,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "base_sha": base_info.get("stdout", "").strip()[:12],
        "head_sha": head_info.get("stdout", "").strip()[:12],
        "files": file_list,
        "file_count": len(file_list),
        "patch": diff_result.get("stdout", ""),
        "patch_lines": len(diff_result.get("stdout", "").splitlines()),
    }

    _log_event("diff_bundle", {"repo_path": repo_path, "base_ref": base_ref, "head_ref": head_ref}, result)
    return result


# ============================================================
# MCP Server implementation
# ============================================================
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import asyncio

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def create_mcp_server() -> "Server":
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP package not installed. Run: pip install mcp")

    server = Server("capseal")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="verify",
                description="Verify a capsule cryptographic proof",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capsule_path": {
                            "type": "string",
                            "description": "Path to capsule JSON file",
                        },
                        "dataset": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dataset bindings (format: id=path)",
                        },
                        "json_out": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return JSON output",
                        },
                    },
                    "required": ["capsule_path"],
                },
            ),
            Tool(
                name="audit",
                description="Export and inspect audit trail from a capsule",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capsule_path": {
                            "type": "string",
                            "description": "Path to capsule JSON file",
                        },
                        "json_out": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return JSON output",
                        },
                    },
                    "required": ["capsule_path"],
                },
            ),
            Tool(
                name="row",
                description="Open a specific trace row with STC membership proof",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capsule_path": {
                            "type": "string",
                            "description": "Path to capsule JSON file",
                        },
                        "row": {
                            "type": "integer",
                            "description": "Row index to open",
                        },
                        "dataset": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dataset bindings (format: id=path)",
                        },
                        "policy": {
                            "type": "string",
                            "description": "Policy file path",
                        },
                        "json_out": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return JSON output",
                        },
                    },
                    "required": ["capsule_path", "row"],
                },
            ),
            Tool(
                name="doctor",
                description="One-click pipeline verification with derived reports",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capsule_path": {
                            "type": "string",
                            "description": "Path to capsule JSON file",
                        },
                        "out_dir": {
                            "type": "string",
                            "description": "Output directory for reports",
                        },
                        "sample_rows": {
                            "type": "integer",
                            "default": 1,
                            "description": "Number of rows to sample",
                        },
                        "json_out": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return JSON output",
                        },
                    },
                    "required": ["capsule_path"],
                },
            ),
            Tool(
                name="diff_bundle",
                description="Generate a diff artifact between two git refs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to git repository",
                        },
                        "base_ref": {
                            "type": "string",
                            "description": "Base git reference (commit, branch, tag)",
                        },
                        "head_ref": {
                            "type": "string",
                            "description": "Head git reference",
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths to limit diff scope",
                        },
                        "context_lines": {
                            "type": "integer",
                            "default": 5,
                            "description": "Lines of context around changes",
                        },
                    },
                    "required": ["repo_path", "base_ref", "head_ref"],
                },
            ),
            # Multi-agent orchestration tools
            Tool(
                name="spawn_agent",
                description="Spawn a subagent to work on a specific task in parallel. Use this to delegate work to specialized agents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The specific task for this subagent (be detailed!)",
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Optional ID for this agent (auto-generated if not provided)",
                        },
                        "context_name": {
                            "type": "string",
                            "default": "latest",
                            "description": "Context checkpoint to load",
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 120,
                            "description": "Max seconds for subagent to run",
                        },
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="load_context",
                description="Load a diff context checkpoint to understand what changed between repos/refs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "default": "latest",
                            "description": "Checkpoint name (default: latest)",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["summary", "full"],
                            "default": "summary",
                            "description": "'summary' for overview, 'full' for complete diffs",
                        },
                    },
                },
            ),
            Tool(
                name="save_result",
                description="Save a result from a subagent for the lead agent to collect",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "ID of the agent saving the result",
                        },
                        "result_type": {
                            "type": "string",
                            "description": "Type of result (e.g., 'findings', 'recommendation', 'diff_analysis')",
                        },
                        "content": {
                            "type": "string",
                            "description": "The result content",
                        },
                    },
                    "required": ["agent_id", "result_type", "content"],
                },
            ),
            Tool(
                name="collect_results",
                description="Collect results from all spawned subagents. Use this after spawning agents to gather their findings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to specific agent IDs (default: all)",
                        },
                        "result_type": {
                            "type": "string",
                            "description": "Filter to specific result type (default: all)",
                        },
                        "since_minutes": {
                            "type": "integer",
                            "default": 30,
                            "description": "Only include results from last N minutes",
                        },
                    },
                },
            ),
            Tool(
                name="greptile_review",
                description="Review code changes using Greptile's codebase understanding. Analyzes diffs for bugs, security issues, and improvements.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository to review against (format: owner/repo)",
                        },
                        "context_name": {
                            "type": "string",
                            "default": "latest",
                            "description": "Context checkpoint to review",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus area: 'security', 'performance', 'style', etc.",
                        },
                    },
                    "required": ["repo"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name not in ALLOWED_TOOLS:
            return [TextContent(type="text", text=json.dumps({"ok": False, "error": f"Tool '{name}' not allowed"}))]

        if name == "verify":
            result = tool_verify(**arguments)
        elif name == "audit":
            result = tool_audit(**arguments)
        elif name == "row":
            result = tool_row(**arguments)
        elif name == "doctor":
            result = tool_doctor(**arguments)
        elif name == "diff_bundle":
            result = tool_diff_bundle(**arguments)
        elif name == "spawn_agent":
            result = tool_spawn_agent(**arguments)
        elif name == "load_context":
            result = tool_load_context(**arguments)
        elif name == "save_result":
            result = tool_save_result(**arguments)
        elif name == "collect_results":
            result = tool_collect_results(**arguments)
        elif name == "greptile_review":
            result = tool_greptile_review(**arguments)
        else:
            result = {"ok": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def run_server():
    """Run the MCP server."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point."""
    if not MCP_AVAILABLE:
        print("MCP package not installed. Install with:")
        print("  pip install mcp")
        raise SystemExit(1)

    print(f"Starting CapSeal MCP server...", file=__import__("sys").stderr)
    print(f"  Workspace: {WORKSPACE_ROOT}", file=__import__("sys").stderr)
    print(f"  Event log: {EVENT_LOG_PATH}", file=__import__("sys").stderr)
    print(f"  Capseal bin: {CAPSEAL_BIN}", file=__import__("sys").stderr)

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
