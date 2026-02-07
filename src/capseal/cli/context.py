"""Context checkpoints for agent continuity.

Stores full diff context so any fresh agent can:
1. Load the checkpoint
2. Understand what changed
3. Continue working on resolution

Usage:
    capseal context save                    # Save current diff as checkpoint
    capseal context load                    # Load latest checkpoint (for agents)
    capseal context list                    # List checkpoints
    capseal context resolve                 # Ask agent to resolve diffs
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import click

from .utils import get_workspace_root

WORKSPACE = str(get_workspace_root())
CONTEXT_DIR = Path(WORKSPACE) / ".capseal" / "contexts"
RECEIPT_LOG = Path(WORKSPACE) / ".capseal" / "mcp_events.jsonl"


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Binary file extensions to skip in diffs
BINARY_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm",
    ".pyc", ".pyo", ".class", ".o", ".a",
    ".db", ".sqlite", ".sqlite3",
    ".lock", ".sum",  # package locks often huge
}


def _is_binary_file(path: str) -> bool:
    """Check if file is likely binary based on extension."""
    ext = Path(path).suffix.lower()
    return ext in BINARY_EXTENSIONS


def _get_working_tree_changes(repo_path: str) -> dict[str, Any]:
    """Get uncommitted working tree changes."""
    # Staged changes
    staged_result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-status", "--cached"],
        capture_output=True, text=True
    )

    # Unstaged changes
    unstaged_result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-status"],
        capture_output=True, text=True
    )

    files = []

    # Parse staged
    for line in staged_result.stdout.strip().split("\n"):
        if line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                files.append({"status": parts[0], "path": parts[1], "staged": True})

    # Parse unstaged (only tracked files that are modified)
    for line in unstaged_result.stdout.strip().split("\n"):
        if line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                files.append({"status": parts[0], "path": parts[1], "staged": False})

    # Skip untracked files by default - they clutter the context
    # If you need them, add --include-untracked flag later

    # Get diffs for modified files (skip binaries)
    file_diffs = []
    binary_files = []
    seen_paths = set()

    for f in files[:50]:
        if f["path"] in seen_paths:
            continue
        seen_paths.add(f["path"])

        # Skip binary files
        if _is_binary_file(f["path"]):
            binary_files.append(f["path"])
            continue

        # Get actual diff
        diff_result = subprocess.run(
            ["git", "-C", repo_path, "diff", "HEAD", "--", f["path"]],
            capture_output=True, text=False
        )
        patch = diff_result.stdout.decode("utf-8", errors="replace")

        # Also skip if git detected it as binary
        if "Binary files" in patch[:200]:
            binary_files.append(f["path"])
            continue

        if len(patch) < 50000:
            file_diffs.append({
                "path": f["path"],
                "status": f["status"],
                "staged": f.get("staged", False),
                "patch": patch,
                "lines_added": patch.count("\n+") - patch.count("\n+++"),
                "lines_removed": patch.count("\n-") - patch.count("\n---"),
            })

    return {
        "files": files,
        "file_count": len(files),
        "file_diffs": file_diffs,
        "binary_files": binary_files,
    }


def _get_diff(repo_path: str, base_ref: str, head_ref: str) -> dict[str, Any]:
    """Get full diff with content."""
    # Get file list
    result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-status", f"{base_ref}..{head_ref}"],
        capture_output=True, text=True
    )

    files = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                status, path = parts
                files.append({"status": status, "path": path})

    # Get full patch
    result = subprocess.run(
        ["git", "-C", repo_path, "diff", f"{base_ref}..{head_ref}"],
        capture_output=True, text=False  # bytes for binary safety
    )
    patch = result.stdout.decode("utf-8", errors="replace")

    # Get per-file diffs with context (skip binaries)
    file_diffs = []
    binary_files = []
    for f in files[:50]:  # Limit to 50 files
        # Skip binary files
        if _is_binary_file(f["path"]):
            binary_files.append(f["path"])
            continue

        file_result = subprocess.run(
            ["git", "-C", repo_path, "diff", f"{base_ref}..{head_ref}", "--", f["path"]],
            capture_output=True, text=False
        )
        file_patch = file_result.stdout.decode("utf-8", errors="replace")

        # Also skip if git detected it as binary
        if "Binary files" in file_patch[:200]:
            binary_files.append(f["path"])
            continue

        if len(file_patch) < 50000:  # Skip huge files
            file_diffs.append({
                "path": f["path"],
                "status": f["status"],
                "patch": file_patch,
                "lines_added": file_patch.count("\n+") - file_patch.count("\n+++"),
                "lines_removed": file_patch.count("\n-") - file_patch.count("\n---"),
            })

    # Get commit info
    base_info = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", base_ref],
        capture_output=True, text=True
    )
    head_info = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", head_ref],
        capture_output=True, text=True
    )

    return {
        "repo_path": repo_path,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "base_sha": base_info.stdout.strip()[:12],
        "head_sha": head_info.stdout.strip()[:12],
        "files": files,
        "file_count": len(files),
        "file_diffs": file_diffs,
        "binary_files": binary_files,
        "full_patch_lines": len(patch.split("\n")),
    }


def save_context(
    repo_path: str,
    base_ref: str,
    head_ref: str,
    name: str | None = None,
    description: str | None = None,
    include_working: bool = False,
) -> dict[str, Any]:
    """Save a diff context checkpoint.

    Args:
        include_working: Also capture uncommitted working tree changes
    """
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Get full diff data
    diff_data = _get_diff(repo_path, base_ref, head_ref)

    # Optionally get working tree changes
    working_data = None
    if include_working:
        working_data = _get_working_tree_changes(repo_path)

    # Build agent prompt
    working_section = ""
    if working_data and working_data["file_count"] > 0:
        working_files = "\n".join(
            f"  - {f['path']} ({f['status']}){' [staged]' if f.get('staged') else ''}"
            for f in working_data['files'][:20]
        )
        working_section = f"""

UNCOMMITTED CHANGES: {working_data['file_count']} files
These are local modifications not yet committed:
{working_files}
{'  ... and more' if len(working_data['files']) > 20 else ''}
"""

    # Create checkpoint
    checkpoint = {
        "version": "1.1",
        "type": "diff_context",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_ts": int(time.time() * 1000),
        "name": name or f"diff_{int(time.time())}",
        "description": description or f"Diff between {base_ref} and {head_ref}",

        # Context for agents
        "summary": {
            "repo": repo_path,
            "comparison": f"{base_ref}..{head_ref}",
            "base_sha": diff_data["base_sha"],
            "head_sha": diff_data["head_sha"],
            "total_files": diff_data["file_count"],
            "files_with_diffs": len(diff_data["file_diffs"]),
            "uncommitted_files": working_data["file_count"] if working_data else 0,
        },

        # File manifest (committed diff)
        "files": diff_data["files"],

        # Actual diffs (what agents need to understand changes)
        "diffs": diff_data["file_diffs"],

        # Working tree changes (uncommitted)
        "working_tree": working_data if working_data else None,

        # Agent instructions
        "agent_prompt": f"""You are reviewing a diff between two codebases.

COMPARISON: {base_ref} → {head_ref}
REPOSITORY: {repo_path}
FILES CHANGED: {diff_data['file_count']}

The diffs are provided below. Your task is to:
1. Understand what changed in each file
2. Identify potential conflicts or issues
3. Suggest how to merge/resolve the differences
4. Note any improvements that could be made

COMMITTED CHANGES ({base_ref} → {head_ref}):
{chr(10).join(f"  - {f['path']} ({f['status']})" for f in diff_data['files'][:20])}
{'  ... and more' if len(diff_data['files']) > 20 else ''}
{working_section}""",
    }

    # Compute checkpoint hash
    checkpoint_hash = _hash_content(json.dumps(checkpoint, sort_keys=True))
    checkpoint["checkpoint_id"] = checkpoint_hash

    # Save to file
    checkpoint_path = CONTEXT_DIR / f"{checkpoint['name']}.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    # Also save as "latest"
    latest_path = CONTEXT_DIR / "latest.json"
    latest_path.write_text(json.dumps(checkpoint, indent=2))

    # Log receipt
    _log_context_receipt(checkpoint)

    return {
        "checkpoint_id": checkpoint_hash,
        "path": str(checkpoint_path),
        "files": diff_data["file_count"],
        "diffs": len(diff_data["file_diffs"]),
    }


def _log_context_receipt(checkpoint: dict) -> None:
    """Log context save to receipt chain."""
    try:
        from capseal.mcp_server import _log_event
        _log_event("context_save", {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "name": checkpoint["name"],
            "files": checkpoint["summary"]["total_files"],
        }, {"ok": True})
    except ImportError:
        pass


def load_context(name: str = "latest") -> dict[str, Any] | None:
    """Load a context checkpoint."""
    path = CONTEXT_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_contexts() -> list[dict[str, Any]]:
    """List all saved contexts."""
    if not CONTEXT_DIR.exists():
        return []

    contexts = []
    for f in CONTEXT_DIR.glob("*.json"):
        if f.name == "latest.json":
            continue
        try:
            ctx = json.loads(f.read_text())
            contexts.append({
                "name": ctx.get("name"),
                "created_at": ctx.get("created_at"),
                "checkpoint_id": ctx.get("checkpoint_id"),
                "files": ctx.get("summary", {}).get("total_files", 0),
                "path": str(f),
            })
        except (json.JSONDecodeError, KeyError):
            pass

    return sorted(contexts, key=lambda x: x.get("created_at", ""), reverse=True)


def format_context_for_agent(checkpoint: dict) -> str:
    """Format checkpoint as agent-readable prompt."""
    lines = [
        "=" * 70,
        "DIFF CONTEXT CHECKPOINT",
        "=" * 70,
        "",
        checkpoint.get("agent_prompt", ""),
        "",
        "=" * 70,
        "COMMITTED FILE DIFFS",
        "=" * 70,
    ]

    for diff in checkpoint.get("diffs", [])[:30]:  # Limit output
        lines.append("")
        lines.append(f"--- {diff['path']} ({diff['status']}) ---")
        lines.append(f"+{diff.get('lines_added', 0)} / -{diff.get('lines_removed', 0)} lines")

        # Include patch preview
        patch = diff.get("patch", "")
        patch_lines = patch.split("\n")
        if len(patch_lines) > 100:
            lines.append("\n".join(patch_lines[:50]))
            lines.append(f"... ({len(patch_lines) - 100} lines omitted) ...")
            lines.append("\n".join(patch_lines[-50:]))
        else:
            lines.append(patch)

    # Include working tree changes if present
    working = checkpoint.get("working_tree")
    if working and working.get("file_diffs"):
        lines.append("")
        lines.append("=" * 70)
        lines.append("UNCOMMITTED WORKING TREE CHANGES")
        lines.append("=" * 70)
        lines.append("")
        lines.append("These are local modifications not yet committed:")
        lines.append("")

        for diff in working.get("file_diffs", [])[:30]:
            lines.append("")
            staged = " [STAGED]" if diff.get("staged") else ""
            lines.append(f"--- {diff['path']} ({diff['status']}){staged} ---")
            lines.append(f"+{diff.get('lines_added', 0)} / -{diff.get('lines_removed', 0)} lines")

            patch = diff.get("patch", "")
            patch_lines = patch.split("\n")
            if len(patch_lines) > 50:
                lines.append("\n".join(patch_lines[:25]))
                lines.append(f"... ({len(patch_lines) - 50} lines omitted) ...")
                lines.append("\n".join(patch_lines[-25:]))
            else:
                lines.append(patch)

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Checkpoint ID: {checkpoint.get('checkpoint_id', 'unknown')}")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# CLI Commands
# ============================================================

@click.group("context")
def context_group():
    """Manage diff context checkpoints for agent continuity."""
    pass


@context_group.command("save")
@click.argument("source", required=False)
@click.argument("target", required=False)
@click.option("--ref", "-r", default="HEAD~5..HEAD", help="Git ref range")
@click.option("--name", "-n", help="Checkpoint name")
@click.option("--description", "-d", help="Description")
@click.option("--working", "-w", is_flag=True, help="Include uncommitted working tree changes")
def context_save(source: str | None, target: str | None, ref: str, name: str | None, description: str | None, working: bool):
    """Save current diff as a context checkpoint.

    \b
    Examples:
        capseal context save                          # Diff HEAD~5..HEAD
        capseal context save ~/Repo1 . --name merge  # Cross-repo diff
        capseal context save --working               # Include uncommitted changes
    """
    # Determine repos
    if source and target:
        source_path = str(Path(source).expanduser().resolve())
        target_path = str(Path(target).expanduser().resolve())
        if target_path == str(Path(".").resolve()):
            target_path = WORKSPACE

        # Setup remote
        subprocess.run(
            ["git", "-C", target_path, "remote", "add", "ctx_compare", source_path],
            capture_output=True
        )
        subprocess.run(
            ["git", "-C", target_path, "fetch", "ctx_compare"],
            capture_output=True
        )
        base_ref = "ctx_compare/main"
        head_ref = "HEAD"
        repo_path = target_path
    else:
        repo_path = WORKSPACE
        if ".." in ref:
            base_ref, head_ref = ref.split("..", 1)
        else:
            base_ref, head_ref = f"{ref}~5", ref

    click.echo(f"Saving context: {base_ref}..{head_ref}")
    if working:
        click.echo("  (including uncommitted changes)")
    result = save_context(repo_path, base_ref, head_ref, name=name, description=description, include_working=working)

    click.echo(f"✓ Checkpoint saved: {result['checkpoint_id']}")
    click.echo(f"  Files: {result['files']}")
    click.echo(f"  Diffs: {result['diffs']}")
    click.echo(f"  Path:  {result['path']}")
    click.echo()
    click.echo("Load with: capseal context load")
    click.echo("Or ask agent: 'Load the diff context and resolve the changes'")


@context_group.command("load")
@click.argument("name", default="latest")
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
@click.option("--prompt", is_flag=True, help="Output agent-formatted prompt")
def context_load(name: str, json_out: bool, prompt: bool):
    """Load a context checkpoint.

    \b
    Examples:
        capseal context load              # Load latest
        capseal context load merge        # Load named checkpoint
        capseal context load --prompt     # Format for agent
    """
    ctx = load_context(name)
    if not ctx:
        click.echo(f"No checkpoint found: {name}")
        click.echo("Save one with: capseal context save")
        return

    if json_out:
        click.echo(json.dumps(ctx, indent=2))
    elif prompt:
        click.echo(format_context_for_agent(ctx))
    else:
        # Summary view
        click.echo(f"\nCheckpoint: {ctx.get('name')}")
        click.echo(f"Created:    {ctx.get('created_at')}")
        click.echo(f"ID:         {ctx.get('checkpoint_id')}")
        click.echo()

        summary = ctx.get("summary", {})
        click.echo(f"Comparison: {summary.get('comparison')}")
        click.echo(f"Repository: {summary.get('repo')}")
        click.echo(f"Committed changes: {summary.get('total_files')} files")

        uncommitted = summary.get('uncommitted_files', 0)
        if uncommitted:
            click.echo(f"Uncommitted changes: {uncommitted} files")
        click.echo()

        click.echo("Committed files:")
        for f in ctx.get("files", [])[:10]:
            click.echo(f"  {f['status']} {f['path']}")
        if len(ctx.get("files", [])) > 10:
            click.echo(f"  ... and {len(ctx['files']) - 10} more")

        # Show working tree if present
        working = ctx.get("working_tree")
        if working and working.get("files"):
            click.echo()
            click.echo("Uncommitted files:")
            for f in working.get("files", [])[:10]:
                staged = " [staged]" if f.get("staged") else ""
                click.echo(f"  {f['status']} {f['path']}{staged}")
            if len(working.get("files", [])) > 10:
                click.echo(f"  ... and {len(working['files']) - 10} more")

        click.echo()
        click.echo("Use --prompt to get agent-formatted output")


@context_group.command("list")
def context_list():
    """List all saved checkpoints."""
    contexts = list_contexts()
    if not contexts:
        click.echo("No checkpoints saved yet.")
        click.echo("Save one with: capseal context save")
        return

    click.echo(f"\nCheckpoints ({len(contexts)}):")
    click.echo("─" * 50)
    for ctx in contexts:
        click.echo(f"  {ctx['name']:<20} {ctx['files']:>3} files  {ctx['created_at']}")
    click.echo()


@context_group.command("resolve")
@click.argument("name", default="latest")
@click.option("--agent", default="cline", help="Agent to use (cline)")
def context_resolve(name: str, agent: str):
    """Ask an agent to resolve the diff.

    Loads the checkpoint and spawns an agent to analyze and resolve.
    """
    ctx = load_context(name)
    if not ctx:
        click.echo(f"No checkpoint found: {name}")
        return

    prompt = format_context_for_agent(ctx)

    # Add resolution instructions
    resolve_prompt = f"""Load this diff context and help me resolve/merge the changes:

{prompt}

Please:
1. Summarize what changed in each file
2. Identify any conflicts or issues
3. Suggest the best way to merge these changes
4. Note any code improvements that could be made during the merge
"""

    click.echo(f"Spawning {agent} to resolve diff...")
    click.echo(f"Checkpoint: {ctx.get('checkpoint_id')}")
    click.echo()

    if agent == "cline":
        import subprocess
        cline_bin = os.path.expanduser("~/.local/node_modules/.bin/cline")
        subprocess.run([cline_bin, "--yolo", resolve_prompt])
    else:
        # Just print the prompt for manual use
        click.echo(resolve_prompt)
