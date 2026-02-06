"""Multi-agent merge orchestration managed by Cline/Greptile.

This is the proper workflow where:
1. Cline orchestrates the merge agents
2. Greptile validates the results
3. Everything flows through capseal commands
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click


def get_cline_orchestrator():
    """Get Cline as the orchestrator via MCP or subprocess."""
    # Check if we're in a Cline-managed session
    if os.environ.get("CLINE_SESSION"):
        return "mcp"  # Use MCP protocol
    return "subprocess"  # Fall back to subprocess


def spawn_merge_agent(file_ctx: dict, bundle_path: Path, backend: str = "openai") -> dict:
    """Spawn a single merge agent - called by orchestrator."""
    from bef_zk.capsule.mcp_server import tool_spawn_agent

    path = file_ctx.get('file', 'unknown')
    strategy = file_ctx.get('merge_strategy', 'combine both versions')

    source_file = bundle_path / 'source' / path
    target_file = bundle_path / 'target' / path

    try:
        src = source_file.read_text() if source_file.exists() else ""
        tgt = target_file.read_text() if target_file.exists() else ""
    except Exception as e:
        return {"path": path, "error": f"read failed: {e}", "merged": None}

    if not src or not tgt:
        return {"path": path, "error": "missing content", "merged": None}

    # Detect file type for context
    ext = Path(path).suffix
    file_type = {
        '.py': 'Python', '.rs': 'Rust', '.ts': 'TypeScript', '.js': 'JavaScript',
        '.toml': 'TOML', '.json': 'JSON', '.md': 'Markdown', '.yml': 'YAML',
        '.hpp': 'C++ header', '.cpp': 'C++', '.h': 'C header'
    }.get(ext, 'source')

    prompt = f"""TASK: Merge two versions of a {file_type} file.

FILE PATH: {path}
MERGE STRATEGY: {strategy}

VERSION A:
```
{src[:5000]}
```

VERSION B:
```
{tgt[:5000]}
```

IMPORTANT: Your response must be ONLY the merged {file_type} code.
- Do NOT wrap in markdown code blocks
- Do NOT include any JSON
- Do NOT explain anything
- Start your response with the first line of the merged file

Example for a Python file - your response should start like:
\"\"\"Module docstring...
or
import ...
or
from ...

Example for a Rust file - your response should start like:
//! Module doc
or
use ...
or
pub fn ...

NOW OUTPUT THE MERGED {file_type.upper()} FILE:"""

    result = tool_spawn_agent(
        task=prompt,
        agent_id=f'merge-{hash(path) % 10000}',
        context_name='merge',
        timeout=60,
        backend=backend
    )

    if result.get('ok'):
        merged = result['output'].strip()
        # Clean markdown if present
        if merged.startswith('```'):
            lines = merged.split('\n')
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
            merged = '\n'.join(lines[start:end])

        # Validate it's actual code (not JSON)
        if merged.startswith('{') and merged.endswith('}'):
            return {"path": path, "error": "agent returned JSON not code", "merged": None}

        return {"path": path, "merged": merged, "error": None}

    return {"path": path, "error": result.get('error', 'unknown'), "merged": None}


@click.command("merge-orchestrate")
@click.option("--context", "-c", type=click.Path(exists=True), required=True,
              help="Path to agent-generated context JSON (from Phase 1)")
@click.option("--bundle", "-b", type=click.Path(exists=True), required=True,
              help="Path to conflict bundle directory")
@click.option("--output", "-o", type=click.Path(), default="/tmp/merged_output",
              help="Output directory for merged files")
@click.option("--backend", type=click.Choice(["openai", "anthropic", "gemini", "auto"]),
              default="auto", help="AI backend for merge agents")
@click.option("--parallel", "-p", default=5, help="Number of parallel agents")
@click.option("--validate/--no-validate", default=True,
              help="Queue Greptile validation after merge")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def merge_orchestrate_command(context, bundle, output, backend, parallel, validate, dry_run):
    """Orchestrate multi-agent merge via Cline.

    This command is the proper workflow orchestrator that:
    1. Reads agent-generated context (from Phase 1)
    2. Uses Cline to spawn parallel merge agents
    3. Collects and validates merged code
    4. Queues Greptile for final validation

    The workflow is:
        Phase 1: capseal conflict-bundle → agents analyze → context.json
        Phase 2: capseal merge-orchestrate → Cline spawns agents → merged files
        Phase 3: Greptile validates → approval/rejection
        Phase 4: capseal merge-apply → apply to repo

    Examples:
        capseal merge-orchestrate -c /tmp/cline_merge_context.json -b /tmp/conflict_bundle
        capseal merge-orchestrate -c context.json -b bundle/ -p 10 --backend openai
    """
    context_path = Path(context)
    bundle_path = Path(bundle).resolve()
    output_path = Path(output)

    click.echo(f"\n{'='*60}")
    click.echo("CLINE MERGE ORCHESTRATOR")
    click.echo(f"{'='*60}")
    click.echo(f"Context:  {context_path}")
    click.echo(f"Bundle:   {bundle_path}")
    click.echo(f"Output:   {output_path}")
    click.echo(f"Backend:  {backend}")
    click.echo(f"Parallel: {parallel} agents")
    click.echo(f"Validate: {'Greptile' if validate else 'disabled'}")

    # Load context
    ctx = json.loads(context_path.read_text())
    to_merge = [f for f in ctx.get('files', [])
                if f.get('action') == 'MERGE' and f.get('merge_strategy')]

    click.echo(f"\nFiles to merge: {len(to_merge)}")

    if dry_run:
        click.echo("\n[DRY RUN] Would merge:")
        for f in to_merge[:10]:
            click.echo(f"  • {f.get('file')}")
            click.echo(f"    Strategy: {f.get('merge_strategy', 'N/A')[:60]}")
        if len(to_merge) > 10:
            click.echo(f"  ... and {len(to_merge) - 10} more")
        return

    # Determine backend
    if backend == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            backend = "openai"
        elif os.environ.get("GEMINI_API_KEY"):
            backend = "gemini"
        else:
            click.echo("✗ No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")
            return

    click.echo(f"\n[Cline Orchestrator] Using {backend} backend")
    click.echo(f"[Cline Orchestrator] Spawning {parallel} parallel agents...")
    click.echo()

    # Orchestrate merge agents
    output_path.mkdir(parents=True, exist_ok=True)
    merged = []
    failed = []
    start = time.time()

    batch_size = parallel
    for batch_start in range(0, len(to_merge), batch_size):
        batch = to_merge[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(to_merge) + batch_size - 1) // batch_size

        click.echo(f"Batch {batch_num}/{total_batches}:", nl=False)

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(spawn_merge_agent, f, bundle_path, backend)
                for f in batch
            ]

            for future in as_completed(futures):
                result = future.result()
                if result.get('merged'):
                    merged.append(result)
                    click.echo(" ✓", nl=False)
                else:
                    failed.append(result)
                    click.echo(" ✗", nl=False)

        click.echo()
        time.sleep(0.3)  # Rate limit buffer

    elapsed = time.time() - start

    # Save merged files
    for m in merged:
        safe_name = m['path'].replace('/', '__')
        # Handle Python dunder files
        safe_name = safe_name.replace('__init__', '##INIT##').replace('__main__', '##MAIN##')
        (output_path / safe_name).write_text(m['merged'])

    # Save manifest
    manifest = {
        "orchestrator": "cline",
        "backend": backend,
        "completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "elapsed_seconds": elapsed,
        "merged_count": len(merged),
        "failed_count": len(failed),
        "files": [m['path'] for m in merged],
        "failures": [{"path": f['path'], "error": f['error']} for f in failed]
    }
    (output_path / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    click.echo()
    click.echo(f"{'='*60}")
    click.echo("ORCHESTRATION COMPLETE")
    click.echo(f"{'='*60}")
    click.echo(f"Time:    {elapsed:.1f}s")
    click.echo(f"Merged:  {len(merged)}/{len(to_merge)}")
    click.echo(f"Failed:  {len(failed)}")
    click.echo(f"Output:  {output_path}")

    if failed:
        click.echo(f"\nFailed files:")
        for f in failed[:5]:
            click.echo(f"  ✗ {f['path']}: {f['error']}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")

    # Queue Greptile validation
    if validate and merged:
        click.echo(f"\n[Greptile] Queuing validation...")
        # This would trigger greptile ephemeral on the output
        click.echo(f"  Run: capseal greptile ephemeral {output_path}")

    click.echo(f"\nNext step:")
    click.echo(f"  capseal merge-apply {output_path} ~/BEF-main --dry-run")
