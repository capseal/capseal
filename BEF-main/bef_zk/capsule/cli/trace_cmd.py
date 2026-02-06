"""CLI commands for project_trace_v1: trace, verify, open, review."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Optional, Any

import click

# Ensure canonical/ is importable
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical.project_trace import (
    canonical_json_bytes,
    sha256_bytes,
    verify_project_trace,
    DEFAULT_CHUNK_SIZE,
)
from bef_zk.capsule.explain_pipeline import (
    run_explain_pipeline,
    verify_explain_receipt,
)
from bef_zk.capsule.finding_utils import (
    FINDING_NORM_VERSION,
    LEGACY_FINDING_NORM_VERSION,
    compute_finding_fingerprint,
    policy_severity_order,
    severity_rank,
    snippet_hash as finding_snippet_hash,
)
from bef_zk.capsule.review_agent import (
    LLM_MAX_TOKENS,
    MAX_AGENT_FINDINGS,
    build_llm_backend_id,
    build_shard_context,
    cache_lookup,
    cache_store,
    compute_agent_input_hash,
    load_trace_index,
    resolve_llm_model,
    run_llm_agent,
    run_llm_explain,
    SEVERITY_ORDER,
    write_prompt_bundle,
    verify_review_packet,
)


def _snippet_hash(text: str) -> str:
    return finding_snippet_hash(text)


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _resolve_run_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _finding_sort_key(f: dict[str, Any]) -> tuple:
    rule = f.get("rule_id", "") or ""
    fingerprint = f.get("finding_fingerprint", "") or ""
    path = f.get("file_path", "") or ""
    line_range = f.get("line_range", [0, 0]) or [0, 0]
    try:
        line_start = int(line_range[0]) if isinstance(line_range, list) else int(line_range)
    except (TypeError, ValueError):
        line_start = 0
    severity = f.get("severity", "") or ""
    message = f.get("message", "") or ""
    return (rule, fingerprint, path, line_start, severity, message)


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(findings, key=_finding_sort_key)


# =============================================================================
# capsule trace <project_dir>
# =============================================================================

@click.command("trace")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--out", "-o", required=True, type=click.Path(), help="Output run directory")
@click.option("--policy", default="default", help="Policy preset name")
@click.option("--store-blobs", is_flag=True, help="Store content-addressed chunks in blobs/")
@click.option("--num-shards", default=4, type=int, help="Number of work shards for fold")
@click.option("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int, help="Chunk size in bytes")
@click.option("--incremental-from", type=click.Path(exists=True, file_okay=False),
              help="Reuse unchanged files from an existing run directory")
def trace_command(project_dir, out, policy, store_blobs, num_shards, chunk_size, incremental_from):
    """Trace a project directory into a verifiable run.

    Produces: trace.jsonl, manifest.json, commitments.json, fold/
    """
    from bef_zk.capsule.project_trace_emitter import emit_project_trace

    project_path = Path(project_dir).resolve()
    run_dir = Path(out)

    click.echo(f"Tracing: {project_path} (policy={policy})")
    emit_stats: dict[str, int] = {}
    manifest = emit_project_trace(
        project_path, run_dir,
        store_blobs=store_blobs,
        num_shards=num_shards,
        chunk_size=chunk_size,
        policy_id=policy,
        incremental_from=Path(incremental_from) if incremental_from else None,
        stats_out=emit_stats,
    )

    # Read back commitments for display
    with open(run_dir / "commitments.json") as f:
        commitments = json.load(f)

    trace_root = commitments["head_T"]
    total_rows = commitments["total_rows"]

    click.echo()
    click.echo(f"Run directory: {run_dir}")
    click.echo(f"  trace_root: {trace_root[:32]}...")
    click.echo(f"  total_rows: {total_rows}")
    click.echo(f"  spec:       {manifest['spec_id']}")
    if manifest.get("git_sha"):
        click.echo(f"  git:        {manifest['git_sha'][:12]} ({manifest.get('git_branch', '?')})")
    if emit_stats.get("reused_files"):
        click.echo(
            f"  Incremental reuse: {emit_stats['reused_files']} files, {emit_stats['reused_bytes']:,} bytes"
        )
    click.echo()
    click.echo(f"  Fold:       {run_dir / 'fold' / 'repo_outline.json'}")
    click.echo(f"  Shards:     {run_dir / 'fold' / 'shards.json'}")
    if store_blobs:
        click.echo(f"  Blobs:      {run_dir / 'blobs'}")


# =============================================================================
# capsule verify project-trace <run_dir>
# =============================================================================

@click.command("verify-trace")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False),
              help="Override project directory (default: from manifest)")
def verify_trace_command(run_dir, project_dir):
    """Verify a project trace run against the filesystem."""
    run_path = Path(run_dir)

    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)
    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)

    rows = []
    with open(run_path / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    proj = Path(project_dir) if project_dir else Path(manifest["root_path"])

    if not proj.exists():
        click.echo(f"FAIL: Project directory not found: {proj}", err=True)
        raise SystemExit(1)

    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))

    click.echo(f"Verifying: {run_path}")
    click.echo(f"  Against:  {proj}")
    click.echo(f"  Rows:     {len(rows)}")

    ok, msg = verify_project_trace(
        proj, rows, manifest,
        manifest_hash, commitments["head_T"],
    )

    if ok:
        click.echo(f"  PASS: {msg}")
    else:
        click.echo(f"  FAIL: {msg}", err=True)
        raise SystemExit(1)


# =============================================================================
# capsule open <run_dir> (project trace variant)
# =============================================================================

@click.command("trace-open")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def trace_open_command(run_dir):
    """Display summary of a project trace run."""
    run_path = Path(run_dir)

    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)
    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)

    click.echo("=" * 60)
    click.echo("PROJECT TRACE RUN")
    click.echo("=" * 60)
    click.echo(f"  Root:       {manifest['root_path']}")
    click.echo(f"  Spec:       {manifest['spec_id']}")
    click.echo(f"  Policy:     {manifest['policy_id']}")
    click.echo(f"  Trace root: {commitments['head_T'][:32]}...")
    click.echo(f"  Total rows: {commitments['total_rows']}")
    if manifest.get("git_sha"):
        click.echo(f"  Git:        {manifest['git_sha'][:12]} ({manifest.get('git_branch', '?')})")

    # Show outline summary if available
    outline_path = run_path / "fold" / "repo_outline.json"
    if outline_path.exists():
        with open(outline_path) as f:
            outline = json.load(f)
        s = outline.get("summary", {})
        click.echo()
        click.echo("  Outline:")
        click.echo(f"    Dirs:     {s.get('total_dirs', '?')}")
        click.echo(f"    Files:    {s.get('total_files', '?')} ({s.get('included_files', '?')} included, {s.get('opaque_files', '?')} opaque)")
        click.echo(f"    Size:     {s.get('total_size', 0):,} bytes")

    # Show shards if available
    shards_path = run_path / "fold" / "shards.json"
    if shards_path.exists():
        with open(shards_path) as f:
            shards = json.load(f)
        click.echo()
        click.echo(f"  Shards: {len(shards.get('shards', []))}")
        for shard in shards.get("shards", []):
            click.echo(f"    [{shard['shard_id']}] {len(shard['files'])} files, {shard['total_size']:,} bytes")

    # Show reviews if any
    reviews_dir = run_path / "reviews"
    if reviews_dir.exists():
        review_files = sorted(reviews_dir.glob("*.json"))
        if review_files:
            click.echo()
            click.echo(f"  Reviews: {len(review_files)}")
            for rf in review_files:
                with open(rf) as f:
                    r = json.load(f)
                n = len(r.get("findings", []))
                click.echo(f"    {rf.name}: {n} findings")

    click.echo("=" * 60)


@click.command("prompt-open")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--shard", "shard_id", required=True, type=int)
def prompt_open_command(run_dir, shard_id):
    """Show the prompt bundle a shard's agent received."""

    run_path = Path(run_dir)
    prompt_path = run_path / "reviews" / "prompts" / f"prompt_shard_{shard_id}.json"
    if not prompt_path.exists():
        click.echo(f"No prompt bundle found for shard {shard_id}: {prompt_path}", err=True)
        raise SystemExit(1)

    with open(prompt_path) as f:
        bundle = json.load(f)

    files = bundle.get("files", [])
    click.echo(f"Prompt bundle: {prompt_path}")
    click.echo(f"  trace_root: {bundle.get('trace_root')}")
    click.echo(f"  policy:     {bundle.get('policy_id')}")
    click.echo(f"  shard_id:   {bundle.get('shard_id')}")
    click.echo(f"  files:      {len(files)}")
    for entry in files[:10]:
        click.echo(
            f"    - {entry['path']} ({entry['file_hash'][:12]}..., {entry['num_chunks']} chunks)"
        )
    if len(files) > 10:
        click.echo(f"    ... {len(files) - 10} more")

    review_path = run_path / "reviews" / f"review_shard_{shard_id}.json"
    if review_path.exists():
        with open(review_path) as f:
            review = json.load(f)
        agent_hash = review.get("agent_input_hash")
        if agent_hash:
            click.echo(f"  agent_input_hash: {agent_hash}")
    else:
        click.echo("  agent_input_hash: (review packet not yet generated)")


# =============================================================================
# capsule verify review <review_files> --run <run_dir>
# =============================================================================

@click.command("verify-review")
@click.argument("review_files", nargs=-1, type=click.Path(exists=True))
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Run directory the reviews reference")
def verify_review_command(review_files, run_dir):
    """Verify review packets against a project trace run.

    Checks that every finding references valid file and chunk hashes
    from the trace.
    """
    run_path = Path(run_dir)

    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)

    trace_root = commitments["head_T"]
    trace_index = load_trace_index(run_path)

    # Verify each review file
    total_findings = 0
    total_valid = 0
    all_ok = True

    for rf_path in review_files:
        rf = Path(rf_path)
        with open(rf) as f:
            review = json.load(f)

        click.echo(f"Verifying: {rf.name}")
        ok, errors, findings_count = verify_review_packet(review, trace_root, trace_index)
        total_findings += findings_count
        if ok:
            total_valid += findings_count
            click.echo(f"  PASS: {findings_count} findings verified")
        else:
            all_ok = False
            for err in errors[:5]:
                click.echo(f"  FAIL: {err}", err=True)

    click.echo()
    click.echo(f"Total: {total_valid}/{total_findings} findings valid across {len(review_files)} packets")

    if not all_ok:
        raise SystemExit(1)


# =============================================================================
# capsule agent review --run <run_dir> --shard <k>
# =============================================================================

@click.command("agent-review")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--shard", "shard_id", required=True, type=int, help="Shard index to review")
@click.option("--out", type=click.Path(), help="Output review packet path")
@click.option("--backend", default="stub", help="Agent backend: stub, semgrep, llm")
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False),
              help="Project dir for file access (default: from manifest)")
@click.option("--llm-provider", default="anthropic",
              type=click.Choice(["anthropic", "openai", "gemini", "mock"]),
              show_default=True,
              help="Provider to use when --backend llm")
@click.option("--llm-model", default=None, help="Override model id for --backend llm")
@click.option("--llm-temperature", default=0.0, type=float, show_default=True,
              help="Sampling temperature for --backend llm")
@click.option("--llm-max-findings", default=MAX_AGENT_FINDINGS, type=int,
              show_default=True, help="Max findings to request from the LLM backend")
@click.option("--llm-replay", type=click.Path(exists=True),
              help="Replay raw LLM outputs from a file or directory instead of calling the API")
@click.option("--llm-include-ext", default=None,
              help="Comma-separated extensions to include for LLM prompts (e.g., py,js,ts). If set, only these are included.")
@click.option("--llm-skip-ext", default=None,
              help="Comma-separated extensions to exclude from LLM prompts (e.g., csv,tsv)")
@click.option("--llm-lenient", is_flag=True, default=False,
              help="Enable prefix matching for chunk hashes (LLMs often truncate 64-char hex)")
@click.option("--llm-redaction-mode", default="allow",
              type=click.Choice(["allow", "fail", "public"]), show_default=True,
              help="Redaction policy: allow (default), fail on redact, or public receipts (omit snippets)")
def agent_review_command(run_dir, shard_id, out, backend, project_dir,
                         llm_provider, llm_model, llm_temperature, llm_max_findings,
                         llm_replay, llm_include_ext, llm_skip_ext, llm_lenient,
                         llm_redaction_mode):
    """Run an agent review on a single shard.

    Reads shard metadata, feeds files to agent, writes review_packet_v1.
    """
    run_path = Path(run_dir)
    if llm_max_findings <= 0:
        llm_max_findings = MAX_AGENT_FINDINGS

    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)
    with open(run_path / "fold" / "shards.json") as f:
        shards_data = json.load(f)

    trace_root = commitments["head_T"]
    proj = Path(project_dir) if project_dir else Path(manifest["root_path"])
    if not proj.exists():
        click.echo(f"FAIL: project directory not found: {proj}", err=True)
        raise SystemExit(1)
    policy_id = manifest.get("policy_id", "unknown")
    policy_version = manifest.get("policy_version", "unknown")
    trace_index = load_trace_index(run_path)

    # Find shard
    shard = None
    for s in shards_data.get("shards", []):
        if s["shard_id"] == shard_id:
            shard = s
            break

    if shard is None:
        click.echo(f"FAIL: shard {shard_id} not found", err=True)
        raise SystemExit(1)

    click.echo(f"Agent review: shard {shard_id} ({len(shard['files'])} files, {shard['total_size']:,} bytes)")
    click.echo(f"  Backend: {backend}")
    click.echo(f"  Project: {proj}")

    def _filter_files(sh: dict) -> dict:
        if not (llm_include_ext or llm_skip_ext):
            return sh
        include = set(x.strip().lower().lstrip('.') for x in (llm_include_ext or '').split(',') if x.strip())
        skip = set(x.strip().lower().lstrip('.') for x in (llm_skip_ext or '').split(',') if x.strip())
        files = []
        for f in sh.get("files", []):
            ext = Path(f.get("path", "")).suffix.lower().lstrip('.')
            if include and ext not in include:
                continue
            if skip and ext in skip:
                continue
            files.append(f)
        return {**sh, "files": files}

    try:
        shard_eff = _filter_files(shard)
        shard_ctx = build_shard_context(
            shard_eff,
            trace_index,
            proj,
            run_path,
            trace_root,
            policy_id,
            redaction_mode=llm_redaction_mode,
        )
    except Exception as exc:  # pragma: no cover - CLI surface area
        click.echo(f"FAIL: {exc}", err=True)
        raise SystemExit(1)

    backend_lower, llm_mode, llm_backend_provider, llm_model_override = _classify_backend(
        backend, llm_provider, llm_model
    )
    semgrep_by_file = None
    backend_input = backend
    if backend_lower == "semgrep":
        semgrep_by_file, _, _, sg_backend_id = _run_semgrep_global(proj, run_path)
        backend_input = sg_backend_id

    replay_path = _resolve_replay_path(llm_replay, shard_id)

    try:
        findings, backend_label, backend_id, prompt_path, raw_path, agent_input_hash, cache_hit = _run_backend_for_shard(
            shard,
            shard_ctx,
            backend_input,
            backend_lower,
            llm_mode,
            llm_backend_provider,
            llm_model_override,
            run_path,
            trace_root,
            policy_id,
            llm_temperature,
            llm_max_findings,
            semgrep_by_file=semgrep_by_file,
            llm_replay_path=replay_path,
            llm_lenient=llm_lenient,
        )
    except Exception as exc:
        click.echo(f"FAIL: {exc}", err=True)
        raise SystemExit(1)
    backend = backend_label

    review = {
        "schema": "review_packet_v1",
        "trace_root": trace_root,
        "shard_id": shard_id,
        "backend": backend,
        "backend_id": backend_id,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "findings": findings,
    }
    if agent_input_hash:
        review["agent_input_hash"] = agent_input_hash
    if prompt_path:
        try:
            review["prompt_bundle_path"] = str(prompt_path.relative_to(run_path))
        except ValueError:
            review["prompt_bundle_path"] = str(prompt_path)
        review["prompt_bundle_hash"] = _file_hash(prompt_path)
    if raw_path:
        try:
            review["llm_raw_path"] = str(raw_path.relative_to(run_path))
        except ValueError:
            review["llm_raw_path"] = str(raw_path)
        review["llm_raw_hash"] = _file_hash(raw_path)

    # Write output
    if out is None:
        reviews_dir = run_path / "reviews"
        reviews_dir.mkdir(exist_ok=True)
        out = str(reviews_dir / f"review_shard_{shard_id}.json")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(review, f, indent=2, sort_keys=True)

    ok, errors, total = verify_review_packet(review, trace_root, trace_index)
    if not ok:
        click.echo("  FAIL: verification errors detected:", err=True)
        for err in errors[:10]:
            click.echo(f"    - {err}", err=True)
        raise SystemExit(1)

    if agent_input_hash and not cache_hit:
        cache_store(run_path, agent_input_hash, review)

    click.echo(f"  Findings:  {len(findings)} (verified {total})")
    if shard_ctx.redactions:
        click.echo(f"  Redactions: {len(shard_ctx.redactions)} substrings replaced")
    if cache_hit:
        click.echo("  Cache:     reused cached review packet")
    if prompt_path:
        click.echo(f"  Prompt:    {prompt_path}")
    if raw_path:
        click.echo(f"  LLM raw:   {raw_path}")
    if agent_input_hash:
        click.echo(f"  Input hash: {agent_input_hash[:32]}...")
    click.echo(f"  Output:    {out_path}")


STUB_BACKEND_ID = "stub:rules@1.0.0"


def _stamp_findings(findings: list[dict], backend_id: str) -> list[dict]:
    """Add snippet_hash, norm version, and fingerprint to each finding."""
    for f in findings:
        if "snippet_hash" not in f:
            # Use message as snippet proxy when no raw snippet available
            f["snippet_hash"] = _snippet_hash(f.get("snippet", f.get("message", "")))
        f["finding_norm_version"] = FINDING_NORM_VERSION
        f["finding_fingerprint"] = compute_finding_fingerprint(
            f, backend_id, norm_version=FINDING_NORM_VERSION
        )
    return findings


def _write_llm_raw(run_path: Path, shard_id: int, content: str, suffix: str) -> Path:
    prompts_dir = run_path / "reviews" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / f"{suffix}_shard_{shard_id}.txt"
    out_path.write_text(content)
    return out_path


def _resolve_replay_path(replay_option: Optional[str], shard_id: int) -> Optional[Path]:
    if not replay_option:
        return None
    base = Path(replay_option)
    if base.is_file():
        return base
    candidate = base / f"raw_shard_{shard_id}.txt"
    if candidate.exists():
        return candidate
    raise click.BadParameter(
        f"Replay file not found for shard {shard_id}: {candidate}",
        param_hint="--llm-replay",
    )


def _classify_backend(backend: str, llm_provider: str, llm_model: str | None) -> tuple[str, bool, str, Optional[str]]:
    """Normalize backend string and detect llm provider/model overrides."""

    backend_lower = backend.lower()
    llm_mode = False
    provider = llm_provider
    model_override = llm_model

    if backend_lower.startswith("llm"):
        llm_mode = True
        parts = backend.split(":", 2)
        if len(parts) == 2:
            model_override = model_override or parts[1]
        elif len(parts) == 3:
            provider = parts[1] or provider
            model_override = model_override or parts[2]
    elif backend_lower in {"claude", "anthropic"}:
        llm_mode = True
        provider = "anthropic"
    elif backend_lower in {"openai"}:
        llm_mode = True
        provider = "openai"
    elif backend_lower in {"gemini"}:
        llm_mode = True
        provider = "gemini"

    return backend_lower, llm_mode, provider, model_override


def _run_backend_for_shard(
    shard: dict,
    shard_ctx,
    backend_input: str,
    backend_lower: str,
    llm_mode: bool,
    llm_backend_provider: str,
    llm_model_override: Optional[str],
    run_path: Path,
    trace_root: str,
    policy_id: str,
    llm_temperature: float,
    llm_max_findings: int,
    semgrep_by_file: Optional[dict[str, list[dict]]] = None,
    llm_replay_path: Optional[Path] = None,
    llm_lenient: bool = False,
) -> tuple[list[dict], str, str, Optional[Path], Optional[Path], Optional[str], bool]:
    """Execute the chosen backend and return findings + metadata."""

    backend_id = STUB_BACKEND_ID if backend_lower == "stub" else backend_input
    backend_label = backend_input
    prompt_path: Optional[Path] = None
    raw_path: Optional[Path] = None
    agent_input_hash: Optional[str] = None
    cache_hit = False

    if backend_lower == "stub":
        file_contents = {path: f.text for path, f in shard_ctx.files.items()}
        findings = _stub_agent_review(shard, file_contents, trace_root)
    elif backend_lower == "semgrep":
        if semgrep_by_file is None:
            raise ValueError("semgrep findings not precomputed")
        findings = _distribute_semgrep_to_shard(shard, semgrep_by_file)
        backend_id = backend_input
    elif llm_mode:
        model_name = resolve_llm_model(llm_backend_provider, llm_model_override)
        backend_label = f"llm:{llm_backend_provider}:{model_name}"
        backend_id = build_llm_backend_id(
            llm_backend_provider,
            model_name,
            temperature=llm_temperature,
            max_tokens=LLM_MAX_TOKENS,
        )
        agent_input_hash = compute_agent_input_hash(
            trace_root,
            shard["shard_id"],
            shard_ctx.chunk_hashes,
            policy_id,
            backend_id,
        )
        prompt_path = write_prompt_bundle(run_path, shard["shard_id"], shard_ctx.prompt_bundle)

        cached_review = None
        if agent_input_hash:
            cached_review = cache_lookup(run_path, agent_input_hash)
        replay_text = None
        if llm_replay_path:
            replay_text = Path(llm_replay_path).read_text()
        elif cached_review and cached_review.get("backend_id") == backend_id:
            findings = cached_review.get("findings", [])
            cache_hit = True
            cached_raw_path = _resolve_run_path(run_path, cached_review.get("llm_raw_path"))
            return (
                findings,
                backend_label,
                backend_id,
                prompt_path,
                cached_raw_path,
                agent_input_hash,
                cache_hit,
            )

        findings, raw_output, included_chunks, total_chunks = run_llm_agent(
            shard_ctx,
            llm_backend_provider,
            model_name,
            backend_id,
            temperature=llm_temperature,
            max_tokens=LLM_MAX_TOKENS,
            max_findings=llm_max_findings,
            raw_override=replay_text,
            lenient=llm_lenient,
        )
        if included_chunks < total_chunks:
            # Log truncation but continue - prompt budget enforcement worked
            pass  # Info logged in prompt itself
        findings = _stamp_findings(findings, backend_id)
        raw_path = _write_llm_raw(run_path, shard["shard_id"], raw_output, "raw")
    else:
        raise ValueError(f"Unknown backend: {backend_input}")

    return findings, backend_label, backend_id, prompt_path, raw_path, agent_input_hash, cache_hit


def _stub_agent_review(shard: dict, file_contents: dict[str, str],
                       trace_root: str) -> list[dict]:
    """Stub agent that produces deterministic findings for testing."""
    findings = []
    for finfo in shard["files"]:
        path = finfo["path"]
        content = file_contents.get(path, "")
        lines = content.split("\n") if content else []

        # Simple rule: flag files with no docstring/comment in first 5 lines
        has_comment = any(
            line.strip().startswith(("#", "//", "/*", '"""', "'''"))
            for line in lines[:5]
        )

        if not has_comment and lines:
            snippet = "\n".join(lines[:5])
            findings.append({
                "file_path": path,
                "file_hash": finfo.get("content_hash", ""),
                "chunk_hashes": [],
                "rule_id": "missing_header_comment",
                "severity": "info",
                "message": f"File {path} has no header comment or docstring.",
                "line_range": [1, min(5, len(lines))],
                "snippet": snippet,
                "snippet_hash": _snippet_hash(snippet),
            })

        # Flag TODO/FIXME
        for i, line in enumerate(lines):
            for keyword in ("TODO", "FIXME", "HACK", "XXX"):
                if keyword in line:
                    findings.append({
                        "file_path": path,
                        "file_hash": finfo.get("content_hash", ""),
                        "chunk_hashes": [],
                        "rule_id": f"found_{keyword.lower()}",
                        "severity": "warning",
                        "message": f"{keyword} found: {line.strip()[:100]}",
                        "line_range": [i + 1, i + 1],
                        "snippet": line.strip(),
                        "snippet_hash": _snippet_hash(line.strip()),
                    })

    return _stamp_findings(findings, STUB_BACKEND_ID)


# =============================================================================
# capsule review --run <run_dir> --agents N (orchestrator)
# =============================================================================

@click.command("review")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--agents", "num_agents", default=None, type=int,
              help="Number of parallel agents (default: one per shard)")
@click.option("--backend", default="stub", help="Agent backend: stub, semgrep, llm")
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--llm-provider", default="anthropic",
              type=click.Choice(["anthropic", "openai", "gemini", "mock"]),
              show_default=True,
              help="Provider to use when --backend llm")
@click.option("--llm-model", default=None, help="Override model id for --backend llm")
@click.option("--llm-temperature", default=0.0, type=float, show_default=True)
@click.option("--llm-max-findings", default=MAX_AGENT_FINDINGS, type=int, show_default=True)
@click.option("--llm-replay", type=click.Path(exists=True),
              help="Replay raw outputs from a file or directory per shard")
@click.option("--llm-include-ext", default=None,
              help="Comma-separated extensions to include for LLM prompts (e.g., py,js,ts). If set, only these are included.")
@click.option("--llm-skip-ext", default=None,
              help="Comma-separated extensions to exclude from LLM prompts (e.g., csv,tsv)")
@click.option("--llm-lenient", is_flag=True, default=False,
              help="Enable prefix matching for chunk hashes (LLMs often truncate 64-char hex)")
@click.option("--llm-redaction-mode", default="allow",
              type=click.Choice(["allow", "fail", "public"]), show_default=True,
              help="Redaction policy: allow (default), fail on redact, or public receipts (omit snippets)")
def review_orchestrator_command(run_dir, num_agents, backend, project_dir,
                                llm_provider, llm_model, llm_temperature, llm_max_findings,
                                llm_replay, llm_include_ext, llm_skip_ext, llm_lenient,
                                llm_redaction_mode):
    """Run parallel agent reviews across all shards, then verify.

    Reads fold/shards.json, dispatches one agent per shard,
    verifies all review packets, produces aggregate report.
    """
    import concurrent.futures

    run_path = Path(run_dir)
    if llm_max_findings <= 0:
        llm_max_findings = MAX_AGENT_FINDINGS

    with open(run_path / "fold" / "shards.json") as f:
        shards_data = json.load(f)
    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)

    shards = shards_data.get("shards", [])
    trace_root = commitments["head_T"]
    proj = Path(project_dir) if project_dir else Path(manifest["root_path"])
    if not proj.exists():
        click.echo(f"FAIL: project directory not found: {proj}", err=True)
        raise SystemExit(1)
    policy_id = manifest.get("policy_id", "unknown")
    policy_version = manifest.get("policy_version", "unknown")
    review_rules = manifest.get("review_rules", {})
    trace_index = load_trace_index(run_path)

    if num_agents is None:
        num_agents = len(shards)

    reviews_dir = run_path / "reviews"
    reviews_dir.mkdir(exist_ok=True)

    click.echo(f"Orchestrating review: {len(shards)} shards, {num_agents} agents, backend={backend}")
    click.echo(f"  Trace root: {trace_root[:32]}...")
    click.echo()

    backend_lower, llm_mode, llm_backend_provider, llm_model_override = _classify_backend(
        backend, llm_provider, llm_model
    )
    semgrep_by_file = None
    backend_input = backend
    if backend_lower == "semgrep":
        semgrep_by_file, _, _, sg_backend_id = _run_semgrep_global(proj, run_path)
        backend_input = sg_backend_id

    if llm_replay and Path(llm_replay).is_file() and len(shards) > 1:
        raise click.BadParameter(
            "Provide a directory with raw_shard_<id>.txt when replaying multiple shards",
            param_hint="--llm-replay",
        )

    # Dispatch agents (parallel)
    review_paths = []

    def run_shard(shard):
        sid = shard["shard_id"]
        out_path = reviews_dir / f"review_shard_{sid}.json"
        # Optionally filter by extension to avoid huge prompts
        def _filter_files(sh: dict) -> dict:
            if not (llm_include_ext or llm_skip_ext):
                return sh
            include = set(x.strip().lower().lstrip('.') for x in (llm_include_ext or '').split(',') if x.strip())
            skip = set(x.strip().lower().lstrip('.') for x in (llm_skip_ext or '').split(',') if x.strip())
            files = []
            for f in sh.get("files", []):
                ext = Path(f.get("path", "")).suffix.lower().lstrip('.')
                if include and ext not in include:
                    continue
                if skip and ext in skip:
                    continue
                files.append(f)
            return {**sh, "files": files}

        shard_eff = _filter_files(shard)
        shard_ctx = build_shard_context(
            shard_eff,
            trace_index,
            proj,
            run_path,
            trace_root,
            policy_id,
            redaction_mode=llm_redaction_mode,
        )
        replay_path = _resolve_replay_path(llm_replay, sid)
        (
            findings,
            backend_label,
            backend_id,
            prompt_path,
            raw_path,
            agent_input_hash,
            cache_hit,
        ) = _run_backend_for_shard(
            shard,
            shard_ctx,
            backend_input,
            backend_lower,
            llm_mode,
            llm_backend_provider,
            llm_model_override,
            run_path,
            trace_root,
            policy_id,
            llm_temperature,
            llm_max_findings,
            semgrep_by_file=semgrep_by_file,
            llm_replay_path=replay_path,
            llm_lenient=llm_lenient,
        )

        review = {
            "schema": "review_packet_v1",
            "trace_root": trace_root,
            "shard_id": sid,
            "backend": backend_label,
            "backend_id": backend_id,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "findings": findings,
        }
        if agent_input_hash:
            review["agent_input_hash"] = agent_input_hash
        if prompt_path:
            try:
                review["prompt_bundle_path"] = str(prompt_path.relative_to(run_path))
            except ValueError:
                review["prompt_bundle_path"] = str(prompt_path)
            review["prompt_bundle_hash"] = _file_hash(prompt_path)
        if raw_path:
            try:
                review["llm_raw_path"] = str(raw_path.relative_to(run_path))
            except ValueError:
                review["llm_raw_path"] = str(raw_path)
            review["llm_raw_hash"] = _file_hash(raw_path)

        with open(out_path, "w") as f:
            json.dump(review, f, indent=2, sort_keys=True)

        ok, errors, _ = verify_review_packet(review, trace_root, trace_index)
        if not ok:
            raise RuntimeError(f"verification failed for shard {sid}: {errors[:3]}")

        if agent_input_hash and not cache_hit:
            cache_store(run_path, agent_input_hash, review)

        return out_path, len(findings), len(shard_ctx.redactions), cache_hit

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as pool:
        futures = {pool.submit(run_shard, s): s for s in shards}
        for fut in concurrent.futures.as_completed(futures):
            shard_meta = futures[fut]
            try:
                path, n_findings, redactions, cache_hit = fut.result()
            except Exception as exc:  # pragma: no cover - concurrency
                click.echo(f"  Shard {shard_meta['shard_id']} FAILED: {exc}", err=True)
                raise
            review_paths.append(path)
            msg = f"  Shard {shard_meta['shard_id']}: {n_findings} findings"
            # Only surface redactions when using LLM backend (since only LLM sends text externally)
            if redactions and backend_lower == "llm":
                msg += f", {redactions} redactions"
            if cache_hit:
                msg += " (cache)"
            msg += f" -> {path.name}"
            click.echo(msg)

    click.echo()

    # Verify all review packets
    click.echo("Verifying review packets...")
    all_ok = True
    total_findings = 0
    all_findings = []

    for rp in sorted(review_paths):
        with open(rp) as f:
            review = json.load(f)

        ok, errors, _ = verify_review_packet(review, trace_root, trace_index)
        if not ok:
            click.echo(f"  FAIL: {rp.name}:", err=True)
            for err in errors[:5]:
                click.echo(f"    - {err}", err=True)
            all_ok = False
            continue

        findings = review.get("findings", [])
        total_findings += len(findings)
        all_findings.extend(findings)

    if all_ok:
        click.echo(f"  PASS: {total_findings} findings across {len(review_paths)} packets verified")
    else:
        click.echo(f"  FAIL: some findings reference invalid hashes")
        raise SystemExit(1)

    agg_backend_id = ""
    agg_policy_id = policy_id
    agg_norm_version = FINDING_NORM_VERSION
    sample_finding = None
    if review_paths:
        with open(review_paths[0]) as f:
            sample_packet = json.load(f)
        agg_backend_id = sample_packet.get("backend_id", sample_packet.get("backend", ""))
        agg_policy_id = sample_packet.get("policy_id", policy_id)
        for fnd in sample_packet.get("findings", []):
            if fnd.get("finding_norm_version"):
                sample_finding = fnd
                break
    if sample_finding:
        agg_norm_version = sample_finding.get("finding_norm_version", FINDING_NORM_VERSION)

    # Aggregate report
    _emit_aggregate(
        reviews_dir,
        all_findings,
        trace_root,
        agg_backend_id,
        agg_policy_id,
        manifest.get("policy_version", "unknown"),
        manifest.get("review_rules", {}),
        agg_norm_version,
    )

    click.echo()
    click.echo(f"Aggregate: {reviews_dir / 'aggregate.json'}")


def _emit_aggregate(
    reviews_dir: Path,
    findings: list[dict],
    trace_root: str,
    backend_id: str,
    policy_id: str,
    policy_version: str,
    review_rules: dict,
    norm_version: str,
) -> None:
    """Emit aggregate.json from all findings along with identity metadata."""
    # Group by severity
    by_severity: dict[str, list] = {}
    by_file: dict[str, list] = {}
    by_rule: dict[str, int] = {}

    for f in findings:
        sev = f.get("severity", "info")
        by_severity.setdefault(sev, []).append(f)
        by_file.setdefault(f.get("file_path", "?"), []).append(f)
        rule = f.get("rule_id", "unknown")
        by_rule[rule] = by_rule.get(rule, 0) + 1

    # Hotspots: files with most findings
    hotspots = sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    aggregate = {
        "schema": "review_aggregate_v1",
        "trace_root": trace_root,
        "backend_id": backend_id,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "review_rules": review_rules,
        "finding_norm_version": norm_version,
        "total_findings": len(findings),
        "by_severity": {k: len(v) for k, v in by_severity.items()},
        "by_rule": by_rule,
        "hotspots": [
            {"path": path, "count": len(flist)}
            for path, flist in hotspots
        ],
        "findings": findings,
    }

    with open(reviews_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)


# =============================================================================
# capsule dag --run <run_dir>
# =============================================================================

def _file_hash(path: Path) -> str:
    """SHA256 of a file's bytes."""
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_merkle_root(leaves: list[str]) -> str:
    """Simple binary Merkle root over a list of hex hashes."""
    import hashlib
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()
    layer = list(leaves)
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            combined = hashlib.sha256(
                bytes.fromhex(left) + bytes.fromhex(right)
            ).hexdigest()
            next_layer.append(combined)
        layer = next_layer
    return layer[0]


@click.command("dag")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False),
              help="Project dir for verify-trace re-check")
def dag_command(run_dir, project_dir):
    """Build workflow DAG and emit rollup receipt.

    Produces workflow/dag.json and workflow/rollup.json.
    """
    import datetime

    run_path = Path(run_dir)

    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)

    trace_root = commitments["head_T"]
    proj = Path(project_dir) if project_dir else Path(manifest["root_path"])

    # ── Build vertices ──────────────────────────────────────────────
    vertices = []

    # Vertex 0: the trace run
    vertices.append({
        "id": "trace",
        "type": "project_trace_v1",
        "hash": trace_root,
        "artifact": "commitments.json",
    })

    # Vertex 1..N: review packets
    reviews_dir = run_path / "reviews"
    review_vertex_ids = []
    if reviews_dir.exists():
        for rf in sorted(reviews_dir.glob("review_shard_*.json")):
            vid = rf.stem  # e.g. review_shard_0
            fh = _file_hash(rf)
            vertices.append({
                "id": vid,
                "type": "review_packet_v1",
                "hash": fh,
                "artifact": f"reviews/{rf.name}",
            })
            review_vertex_ids.append(vid)

    # Vertex N+1: aggregate
    agg_path = reviews_dir / "aggregate.json" if reviews_dir.exists() else None
    if agg_path and agg_path.exists():
        vertices.append({
            "id": "aggregate",
            "type": "review_aggregate_v1",
            "hash": _file_hash(agg_path),
            "artifact": "reviews/aggregate.json",
        })

    # Explain receipts (optional)
    explain_root = reviews_dir / "explain_llm"
    explain_vertex_ids = []
    if explain_root.exists():
        for receipt in sorted(explain_root.glob("*/receipt.json")):
            rel = receipt.relative_to(run_path)
            vid = f"explain::{receipt.parent.name}"
            vertices.append({
                "id": vid,
                "type": "llm_explain_receipt_v1",
                "hash": _file_hash(receipt),
                "artifact": str(rel),
            })
            explain_vertex_ids.append(vid)

    # Diff receipt (optional)
    diff_receipt = run_path / "diff" / "receipt.json"
    diff_vertex_id = None
    if diff_receipt.exists():
        vertices.append({
            "id": "review_diff",
            "type": "review_diff_receipt_v1",
            "hash": _file_hash(diff_receipt),
            "artifact": "diff/receipt.json",
        })
        diff_vertex_id = "review_diff"

    # ── Build edges ─────────────────────────────────────────────────
    edges = []
    for vid in review_vertex_ids:
        edges.append({"from": "trace", "to": vid, "relation": "reviewed_by"})
    if agg_path and agg_path.exists():
        for vid in review_vertex_ids:
            edges.append({"from": vid, "to": "aggregate", "relation": "aggregated_into"})
        for evid in explain_vertex_ids:
            edges.append({"from": "aggregate", "to": evid, "relation": "explained_by"})
    if diff_vertex_id:
        edges.append({"from": "trace", "to": diff_vertex_id, "relation": "diff_receipt"})

    dag = {
        "schema": "workflow_dag_v1",
        "vertices": vertices,
        "edges": edges,
        "root": "trace",
    }

    workflow_dir = run_path / "workflow"
    workflow_dir.mkdir(exist_ok=True)
    with open(workflow_dir / "dag.json", "w") as f:
        json.dump(dag, f, indent=2, sort_keys=True)

    # ── Run verification checks ─────────────────────────────────────
    checks = []

    # Check 1: verify-trace
    rows = []
    with open(run_path / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    ok, msg = verify_project_trace(
        proj, rows, manifest, manifest_hash, trace_root,
    )
    checks.append({"check": "verify-trace", "pass": ok, "message": msg})

    # Check 2: verify-review (all packets)
    trace_index = load_trace_index(run_path)
    review_ok = True
    total_findings = 0
    for vid in review_vertex_ids:
        rp = reviews_dir / f"{vid}.json"
        with open(rp) as f:
            review = json.load(f)
        ok, errors, count = verify_review_packet(review, trace_root, trace_index)
        total_findings += count
        if not ok:
            review_ok = False

    checks.append({
        "check": "verify-review",
        "pass": review_ok,
        "message": f"{total_findings} findings across {len(review_vertex_ids)} packets",
    })

    # ── Build rollup ────────────────────────────────────────────────
    vertex_hashes = [v["hash"] for v in vertices]
    dag_hash = sha256_bytes(canonical_json_bytes(dag))
    merkle_root = _build_merkle_root(vertex_hashes)

    rollup = {
        "schema": "workflow_rollup_v1",
        "dag_hash": dag_hash,
        "merkle_root": merkle_root,
        "vertex_hashes": vertex_hashes,
        "num_vertices": len(vertices),
        "num_edges": len(edges),
        "trace_root": trace_root,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tool_version": "capseal-0.2.0",
    }

    with open(workflow_dir / "rollup.json", "w") as f:
        json.dump(rollup, f, indent=2, sort_keys=True)

    click.echo(f"Workflow DAG: {len(vertices)} vertices, {len(edges)} edges")
    click.echo(f"  dag_hash:    {dag_hash[:32]}...")
    click.echo(f"  merkle_root: {merkle_root[:32]}...")
    click.echo(f"  trace_root:  {trace_root[:32]}...")
    click.echo()
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        click.echo(f"  [{status}] {c['check']}: {c['message']}")
    click.echo()
    click.echo(f"  Rollup: {workflow_dir / 'rollup.json'}")
    if rollup["all_pass"]:
        click.echo(f"  Status: ALL PASS")
    else:
        click.echo(f"  Status: SOME CHECKS FAILED")
        raise SystemExit(1)


# =============================================================================
# capsule verify-rollup <rollup.json>
# =============================================================================

@click.command("verify-rollup")
@click.argument("rollup_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False),
              help="Project dir for verify-trace re-check")
def verify_rollup_command(rollup_path, project_dir):
    """Verify a workflow rollup receipt.

    Re-runs all checks and confirms committed hashes match.
    """
    rollup_file = Path(rollup_path)
    with open(rollup_file) as f:
        rollup = json.load(f)

    # Locate run dir (rollup is at workflow/rollup.json)
    workflow_dir = rollup_file.parent
    run_path = workflow_dir.parent

    dag_path = workflow_dir / "dag.json"
    if not dag_path.exists():
        click.echo("FAIL: dag.json not found next to rollup.json", err=True)
        raise SystemExit(1)

    with open(dag_path) as f:
        dag = json.load(f)

    # Check 1: dag_hash matches
    dag_hash = sha256_bytes(canonical_json_bytes(dag))
    if dag_hash != rollup["dag_hash"]:
        click.echo(f"FAIL: dag_hash mismatch: {dag_hash[:16]}... != {rollup['dag_hash'][:16]}...")
        raise SystemExit(1)
    click.echo(f"  [PASS] dag_hash matches")

    # Check 2: merkle_root over vertex hashes
    vertex_hashes = [v["hash"] for v in dag["vertices"]]
    merkle_root = _build_merkle_root(vertex_hashes)
    if merkle_root != rollup["merkle_root"]:
        click.echo(f"FAIL: merkle_root mismatch")
        raise SystemExit(1)
    click.echo(f"  [PASS] merkle_root matches ({len(vertex_hashes)} vertices)")

    # Check 3: verify each vertex artifact hash
    all_ok = True
    explain_receipts_ok = True
    diff_receipt_ok = True
    for v in dag["vertices"]:
        artifact = run_path / v["artifact"]
        if v["type"] == "project_trace_v1":
            # trace vertex hash is trace_root from commitments, not file hash
            with open(run_path / "commitments.json") as f:
                c = json.load(f)
            if c["head_T"] != v["hash"]:
                click.echo(f"  [FAIL] vertex {v['id']}: trace_root mismatch")
                all_ok = False
            else:
                click.echo(f"  [PASS] vertex {v['id']}: trace_root={v['hash'][:16]}...")
        elif v["type"] == "llm_explain_receipt_v1":
            if artifact.exists():
                fh = _file_hash(artifact)
                if fh != v["hash"]:
                    click.echo(f"  [FAIL] vertex {v['id']}: explain receipt hash mismatch")
                    all_ok = False
                    explain_receipts_ok = False
                else:
                    click.echo(f"  [PASS] vertex {v['id']}: explain receipt hash={v['hash'][:16]}...")
            else:
                click.echo(f"  [FAIL] vertex {v['id']}: explain receipt missing ({artifact})")
                all_ok = False
                explain_receipts_ok = False
        elif v["type"] == "review_diff_receipt_v1":
            if artifact.exists():
                fh = _file_hash(artifact)
                if fh != v["hash"]:
                    click.echo(f"  [FAIL] vertex {v['id']}: diff receipt hash mismatch")
                    all_ok = False
                    diff_receipt_ok = False
                else:
                    click.echo(f"  [PASS] vertex {v['id']}: diff receipt hash={v['hash'][:16]}...")
            else:
                click.echo(f"  [FAIL] vertex {v['id']}: diff receipt missing ({artifact})")
                all_ok = False
                diff_receipt_ok = False
        elif artifact.exists():
            fh = _file_hash(artifact)
            if fh != v["hash"]:
                click.echo(f"  [FAIL] vertex {v['id']}: file hash mismatch")
                all_ok = False
            else:
                click.echo(f"  [PASS] vertex {v['id']}: hash={v['hash'][:16]}...")
        else:
            click.echo(f"  [FAIL] vertex {v['id']}: artifact not found: {artifact}")
            all_ok = False

    # Check 4: re-run verify-trace
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)
    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)

    proj = Path(project_dir) if project_dir else Path(manifest["root_path"])
    trace_root = commitments["head_T"]

    rows = []
    with open(run_path / "trace.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    ok, msg = verify_project_trace(proj, rows, manifest, manifest_hash, trace_root)
    if ok:
        click.echo(f"  [PASS] verify-trace: {msg}")
    else:
        click.echo(f"  [FAIL] verify-trace: {msg}")
        all_ok = False

    # Check 5: re-run verify-review
    file_hashes_set = {r["content_hash"] for r in rows
                       if r.get("row_type") == "file_entry" and r.get("content_hash")}
    chunk_hashes_set = {r["chunk_hash"] for r in rows if r.get("row_type") == "chunk_entry"}

    reviews_dir = run_path / "reviews"
    review_count = 0
    finding_count = 0
    for rf in sorted(reviews_dir.glob("review_shard_*.json")):
        with open(rf) as f:
            review = json.load(f)
        review_count += 1
        if review.get("trace_root") != trace_root:
            click.echo(f"  [FAIL] {rf.name}: trace_root mismatch")
            all_ok = False
            continue
        for finding in review.get("findings", []):
            finding_count += 1
            fh = finding.get("file_hash")
            if fh and fh not in file_hashes_set:
                all_ok = False
            for ch in finding.get("chunk_hashes", []):
                if ch not in chunk_hashes_set:
                    all_ok = False

    click.echo(f"  [PASS] verify-review: {finding_count} findings across {review_count} packets")

    click.echo()
    if all_ok:
        click.echo(f"ROLLUP VERIFIED: {rollup_file}")
    else:
        click.echo(f"ROLLUP FAILED: some checks did not pass")
        raise SystemExit(1)


# =============================================================================
# Semgrep backend
# =============================================================================

def _get_semgrep_backend_id() -> str:
    """Get semgrep version for backend_id."""
    import subprocess
    try:
        ver = subprocess.check_output(
            ["semgrep", "--version"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"semgrep:auto@{ver}"
    except Exception:
        return "semgrep:auto@unknown"


def _run_semgrep_global(project_dir: Path, run_path: Path) -> tuple:
    """Run Semgrep once on the whole project.

    Returns (by_file, chunk_map, file_hash_map, backend_id).
    """
    import subprocess

    backend_id = _get_semgrep_backend_id()

    # Build chunk lookup from trace
    chunk_map: dict[str, list[dict]] = {}
    file_hash_map: dict[str, str] = {}
    with open(run_path / "trace.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") == "chunk_entry":
                chunk_map.setdefault(row["path"], []).append({
                    "offset": row["offset"],
                    "length": row["length"],
                    "chunk_hash": row["chunk_hash"],
                })
            elif row.get("row_type") == "file_entry" and row.get("content_hash"):
                file_hash_map[row["path"]] = row["content_hash"]

    # Run semgrep once
    try:
        result = subprocess.run(
            ["semgrep", "scan", "--json", "--config", "auto",
             "--no-git-ignore", "--quiet",
             "--include", "*.py", "--include", "*.js", "--include", "*.ts",
             "--include", "*.jsx", "--include", "*.tsx", "--include", "*.go",
             "--include", "*.rs", "--include", "*.java", "--include", "*.rb",
             "--include", "*.c", "--include", "*.cpp", "--include", "*.h",
             "--include", "*.hpp",
             str(project_dir)],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        click.echo("  WARNING: semgrep not found", err=True)
        return {}, chunk_map, file_hash_map
    except subprocess.TimeoutExpired:
        click.echo("  WARNING: semgrep timed out", err=True)
        return {}, chunk_map, file_hash_map

    try:
        semgrep_out = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        click.echo("  WARNING: semgrep output parse error", err=True)
        return {}, chunk_map, file_hash_map

    # Parse and group by relative path
    by_file: dict[str, list[dict]] = {}
    for sg in semgrep_out.get("results", []):
        abs_path = sg.get("path", "")
        try:
            rel = str(Path(abs_path).relative_to(project_dir))
        except ValueError:
            continue

        start_line = sg.get("start", {}).get("line", 1)
        end_line = sg.get("end", {}).get("line", start_line)
        start_offset = sg.get("start", {}).get("offset", 0)
        end_offset = sg.get("end", {}).get("offset", start_offset)

        # Map to chunk hashes
        matched_chunks = []
        for ch in chunk_map.get(rel, []):
            co = ch["offset"]
            ce = co + ch["length"]
            if co < end_offset and ce > start_offset:
                matched_chunks.append(ch["chunk_hash"])

        check_id = sg.get("check_id", "unknown")
        severity = sg.get("extra", {}).get("severity", "WARNING").lower()
        message = sg.get("extra", {}).get("message", check_id)
        # Semgrep provides matched snippet in extra.lines or we reconstruct
        snippet = sg.get("extra", {}).get("lines", "").strip()

        finding = {
            "file_path": rel,
            "file_hash": file_hash_map.get(rel, ""),
            "chunk_hashes": matched_chunks,
            "rule_id": check_id,
            "severity": severity,
            "message": message[:500],
            "line_range": [start_line, end_line],
            "snippet": snippet[:500] if snippet else "",
            "snippet_hash": _snippet_hash(snippet) if snippet else _snippet_hash(message),
        }
        by_file.setdefault(rel, []).append(finding)

    # Stamp all findings with fingerprints
    for file_findings in by_file.values():
        _stamp_findings(file_findings, backend_id)

    total = sum(len(v) for v in by_file.values())
    click.echo(f"  Semgrep ({backend_id}): {total} findings across {len(by_file)} files")
    return by_file, chunk_map, file_hash_map, backend_id


def _distribute_semgrep_to_shard(
    shard: dict,
    semgrep_by_file: dict[str, list[dict]],
) -> list[dict]:
    """Pick findings that belong to this shard's files."""
    findings = []
    shard_paths = {fi["path"] for fi in shard["files"]}
    for path in shard_paths:
        findings.extend(semgrep_by_file.get(path, []))
    return findings


# =============================================================================
# capsule explain-review --run <run_dir>
# =============================================================================

@click.command("explain-review")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json"]),
              help="Output format")
@click.option("--max-actions", default=10, type=int, help="Number of top actions to list")
def explain_review_command(run_dir, fmt, max_actions):
    """Generate a human-readable review report.

    Produces reviews/report.md with actionable findings tied to the rollup receipt.
    """
    run_path = Path(run_dir)

    with open(run_path / "commitments.json") as f:
        commitments = json.load(f)
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)

    trace_root = commitments["head_T"]

    # Load rollup if available
    rollup_hash = ""
    rollup_path = run_path / "workflow" / "rollup.json"
    if rollup_path.exists():
        with open(rollup_path) as f:
            rollup = json.load(f)
        rollup_hash = rollup.get("dag_hash", "")

    # Load aggregate
    agg_path = run_path / "reviews" / "aggregate.json"
    if not agg_path.exists():
        click.echo("No aggregate.json found. Run `capsule review` first.", err=True)
        raise SystemExit(1)

    with open(agg_path) as f:
        agg = json.load(f)

    findings = agg.get("findings", [])
    by_severity = agg.get("by_severity", {})
    by_rule = agg.get("by_rule", {})
    hotspots = agg.get("hotspots", [])

    # Sort findings: error first, then warning, then info; within each by file
    sev_order = {"error": 0, "warning": 1, "info": 2}
    sorted_findings = sorted(findings,
        key=lambda f: (sev_order.get(f.get("severity", "info"), 9), f.get("file_path", "")))

    # Pick top actions: highest severity first, deduplicate by file+rule
    seen = set()
    actions = []
    for f in sorted_findings:
        key = (f.get("file_path"), f.get("rule_id"))
        if key in seen:
            continue
        seen.add(key)
        actions.append(f)
        if len(actions) >= max_actions:
            break

    if fmt == "json":
        report = {
            "trace_root": trace_root,
            "rollup_hash": rollup_hash,
            "total_findings": len(findings),
            "by_severity": by_severity,
            "by_rule": by_rule,
            "hotspots": hotspots[:10],
            "actions": actions,
        }
        out_path = run_path / "reviews" / "report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        click.echo(f"Report: {out_path}")
        return

    # Markdown report
    lines = []
    lines.append("# Review Report")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| trace_root | `{trace_root[:32]}...` |")
    if rollup_hash:
        lines.append(f"| rollup_hash | `{rollup_hash[:32]}...` |")
    lines.append(f"| policy | {manifest.get('policy_id', '?')} |")
    if manifest.get("git_sha"):
        lines.append(f"| git | `{manifest['git_sha'][:12]}` ({manifest.get('git_branch', '?')}) |")
    lines.append(f"| total findings | {len(findings)} |")
    lines.append("")

    # Severity breakdown
    lines.append("## Findings by Severity")
    lines.append("")
    for sev in ["error", "warning", "info"]:
        count = by_severity.get(sev, 0)
        if count:
            lines.append(f"- **{sev}**: {count}")
    lines.append("")

    # Top rules
    lines.append("## Top Rules")
    lines.append("")
    lines.append("| Count | Rule |")
    lines.append("|------:|------|")
    for rule, count in sorted(by_rule.items(), key=lambda x: -x[1])[:10]:
        # Shorten rule_id for readability
        short = rule.split(".")[-1] if "." in rule else rule
        lines.append(f"| {count} | `{short}` |")
    lines.append("")

    # Hotspot files
    lines.append("## Hotspot Files")
    lines.append("")
    lines.append("| Findings | File |")
    lines.append("|---------:|------|")
    for h in hotspots[:10]:
        lines.append(f"| {h['count']} | `{h['path']}` |")
    lines.append("")

    # Actions
    lines.append(f"## Top {len(actions)} Actions")
    lines.append("")
    for i, a in enumerate(actions, 1):
        sev = a.get("severity", "info").upper()
        rule = a.get("rule_id", "unknown")
        short_rule = rule.split(".")[-1] if "." in rule else rule
        fp = a.get("file_path", "?")
        lr = a.get("line_range", [0, 0])
        fh = a.get("file_hash", "")
        chunks = a.get("chunk_hashes", [])

        lines.append(f"### {i}. [{sev}] `{short_rule}`")
        lines.append("")
        lines.append(f"**File:** `{fp}` (lines {lr[0]}-{lr[1]})")
        lines.append("")
        if fh:
            lines.append(f"**file_hash:** `{fh[:16]}...`")
        if chunks:
            chunk_str = ", ".join(f"`{c[:12]}...`" for c in chunks[:3])
            if len(chunks) > 3:
                chunk_str += f" (+{len(chunks)-3} more)"
            lines.append(f"**chunks:** {chunk_str}")
        lines.append("")
        lines.append(f"> {a.get('message', '')}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by capseal-0.2.0. Trace root: `{trace_root[:16]}...`*")

    report_text = "\n".join(lines)
    out_path = run_path / "reviews" / "report.md"
    out_path.write_text(report_text)
    click.echo(f"Report: {out_path}")
    click.echo(f"  {len(findings)} findings, {len(actions)} actions")
@click.command("verify-explain")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--receipt", type=click.Path(exists=True, dir_okay=False),
              help="Path to explain receipt (default: derived from --hash)")
@click.option("--hash", "input_hash", default=None,
              help="Explain input hash (subdirectory under reviews/explain_llm)")
def verify_explain_command(run_dir, receipt, input_hash):
    """Verify an explain-llm receipt and its referenced artifacts."""

    if not receipt and not input_hash:
        click.echo("Provide --receipt or --hash", err=True)
        raise SystemExit(1)

    run_path = Path(run_dir)
    if receipt:
        receipt_path = Path(receipt)
    else:
        receipt_path = run_path / "reviews" / "explain_llm" / input_hash / "receipt.json"

    ok = verify_explain_receipt(run_path, receipt_path, quiet=False)
    if ok:
        click.echo(f"Explain receipt verified: {receipt_path}")
    else:
        click.echo("Explain receipt verification failed", err=True)
        raise SystemExit(1)





@click.command("explain-llm")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--llm-provider", default="anthropic",
              type=click.Choice(["anthropic", "openai", "gemini", "mock"]),
              show_default=True,
              help="Provider to use for explanations")
@click.option("--llm-model", default=None, help="Override model id for explanations")
@click.option("--temperature", default=0.0, type=float, show_default=True)
@click.option("--llm-max-tokens", default=1500, type=int, show_default=True,
              help="Max tokens for the explanation response")
@click.option("--max-findings", default=20, type=int, show_default=True,
              help="Max findings to include in the prompt")
@click.option("--min-severity", default="warning",
              type=click.Choice(["info", "warning", "error"]), show_default=True,
              help="Lowest severity to include")
@click.option("--diff", "diff_path", type=click.Path(exists=True, dir_okay=False),
              help="Optional review-diff receipt to summarize new findings")
@click.option("--format", "output_format", default="json",
              type=click.Choice(["json", "markdown"]), show_default=True,
              help="Summary output format (json always emitted; markdown adds stdout preview)")
@click.option("--report-top", default=5, type=int, show_default=True,
              help="How many finding groups to fully expand in the markdown report")
@click.option("--force", is_flag=True, help="Force re-generation even if cached")
@click.option("--out", type=click.Path(),
              help="Summary output path (default: reviews/explain_llm/<hash>/summary.json)")
def explain_llm_command(run_dir, llm_provider, llm_model, temperature,
                        llm_max_tokens, max_findings, min_severity,
                        diff_path, output_format, report_top, force, out):
    """Use an LLM to narrate recommendations for verified findings."""

    run_path = Path(run_dir)
    model_name = resolve_llm_model(llm_provider, llm_model)

    diff_obj = Path(diff_path) if diff_path else None
    out_path = Path(out) if out else None

    try:
        result = run_explain_pipeline(
            run_path,
            llm_provider,
            model_name,
            temperature,
            llm_max_tokens,
            max_findings,
            min_severity,
            diff_obj,
            output_format,
            report_top,
            force,
            out_path,
        )
    except RuntimeError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    summary_data = json.loads(result.summary_path.read_text())
    explanations = summary_data.get("explanations", [])

    label = "Explain output (cached)" if result.cached else "Explain output"
    click.echo(f"{label}: {result.summary_path}")
    click.echo(f"  Selected findings: {summary_data.get('selected_findings', len(explanations))}")
    if result.receipt_path.exists():
        click.echo(f"  Receipt: {result.receipt_path}")
    if result.report_path:
        click.echo(f"  Report: {result.report_path}")

    for item in explanations[:5]:
        preview = item.get("analysis", "")[:80]
        click.echo(f"  - {item.get('fingerprint', '')[:12]}... {preview}")

    if output_format == "markdown" and result.report_path and result.report_path.exists():
        click.echo(result.report_path.read_text())

@click.command("review-diff")
@click.option("--base", "base_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Base run directory (e.g. main branch)")
@click.option("--head", "head_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Head run directory (e.g. PR branch)")
@click.option("--fail-on", "fail_severity", default=None,
              help="Fail if new findings at this severity or above (error, warning, info)")
@click.option("--allow-backend-mismatch", is_flag=True,
              help="Allow comparing runs from different backends")
@click.option("--allow-policy-mismatch", is_flag=True,
              help="Allow comparing runs from different policies")
def review_diff_command(base_dir, head_dir, fail_severity,
                        allow_backend_mismatch, allow_policy_mismatch):
    """Compare two review runs and report new/resolved/unchanged findings.

    Diffs by finding_fingerprint (stable under line drift).
    Refuses apples-to-oranges comparisons unless --allow-* flags are set.
    Emits diff/receipt.json as a proof-carrying gate decision.
    """
    import datetime

    base_path = Path(base_dir)
    head_path = Path(head_dir)

    # Load aggregates
    for label, p in [("base", base_path), ("head", head_path)]:
        if not (p / "reviews" / "aggregate.json").exists():
            click.echo(f"No aggregate.json in {label} run. Run `capsule review` first.", err=True)
            raise SystemExit(1)

    with open(base_path / "reviews" / "aggregate.json") as f:
        base_agg = json.load(f)
    with open(head_path / "reviews" / "aggregate.json") as f:
        head_agg = json.load(f)

    # Load metadata for identity enforcement
    with open(base_path / "commitments.json") as f:
        base_commits = json.load(f)
    with open(head_path / "commitments.json") as f:
        head_commits = json.load(f)
    with open(base_path / "manifest.json") as f:
        base_manifest = json.load(f)
    with open(head_path / "manifest.json") as f:
        head_manifest = json.load(f)

    base_root = base_commits["head_T"]
    head_root = head_commits["head_T"]

    # Extract backend_id and policy_id from first review packet (legacy fallback)
    def _extract_review_meta(run_path: Path) -> tuple[str, str]:
        for rf in sorted((run_path / "reviews").glob("review_shard_*.json")):
            with open(rf) as f:
                pkt = json.load(f)
            return pkt.get("backend_id", pkt.get("backend", "unknown")), pkt.get("policy_id", "unknown")
        return "unknown", "unknown"

    base_backend_id = base_agg.get("backend_id") or ""
    base_policy_id = base_agg.get("policy_id") or ""
    head_backend_id = head_agg.get("backend_id") or ""
    head_policy_id = head_agg.get("policy_id") or ""
    base_policy_version = base_agg.get("policy_version") or base_manifest.get("policy_version", "unknown")
    head_policy_version = head_agg.get("policy_version") or head_manifest.get("policy_version", "unknown")
    severity_rules = head_agg.get("review_rules") or head_manifest.get("review_rules", {})
    base_backend_id = base_backend_id or _extract_review_meta(base_path)[0]
    base_policy_id = base_policy_id or _extract_review_meta(base_path)[1]
    head_backend_id = head_backend_id or _extract_review_meta(head_path)[0]
    head_policy_id = head_policy_id or _extract_review_meta(head_path)[1]
    base_norm_version = base_agg.get("finding_norm_version", FINDING_NORM_VERSION)
    head_norm_version = head_agg.get("finding_norm_version", FINDING_NORM_VERSION)
    severity_order = policy_severity_order(severity_rules)
    severity_rank_map = severity_rank(severity_order)
    if fail_severity is None:
        fail_severity = severity_rules.get("default_fail_on")

    # Enforce identity
    if base_backend_id != head_backend_id and not allow_backend_mismatch:
        click.echo(f"REFUSED: backend mismatch: base={base_backend_id}, head={head_backend_id}", err=True)
        click.echo("  Use --allow-backend-mismatch to override.", err=True)
        raise SystemExit(2)

    if base_policy_id != head_policy_id and not allow_policy_mismatch:
        click.echo(f"REFUSED: policy mismatch: base={base_policy_id}, head={head_policy_id}", err=True)
        click.echo("  Use --allow-policy-mismatch to override.", err=True)
        raise SystemExit(2)

    # Diff by finding_fingerprint (stable under line drift)
    base_findings = base_agg.get("findings", [])
    head_findings = head_agg.get("findings", [])

    def _fp_for(f: dict, backend_id: str, default_norm: str) -> str:
        """Get or compute finding fingerprint with backend/norm context."""
        if f.get("finding_fingerprint"):
            return f["finding_fingerprint"]
        norm = f.get("finding_norm_version") or default_norm
        return compute_finding_fingerprint(
            f,
            backend_id,
            norm_version=norm,
        )

    base_by_fp: dict[str, list[dict]] = {}
    for f in base_findings:
        base_by_fp.setdefault(_fp_for(f, base_backend_id, base_norm_version), []).append(f)

    head_by_fp: dict[str, list[dict]] = {}
    for f in head_findings:
        head_by_fp.setdefault(_fp_for(f, head_backend_id, head_norm_version), []).append(f)

    base_fps = set(base_by_fp.keys())
    head_fps = set(head_by_fp.keys())

    new_fps = head_fps - base_fps
    resolved_fps = base_fps - head_fps
    unchanged_fps = base_fps & head_fps

    new_findings = []
    for fp in sorted(new_fps):
        new_findings.extend(head_by_fp[fp])

    resolved_findings = []
    for fp in sorted(resolved_fps):
        resolved_findings.extend(base_by_fp[fp])

    unchanged_findings = []
    for fp in sorted(unchanged_fps):
        unchanged_findings.extend(head_by_fp[fp])

    # Write diff outputs
    diff_dir = head_path / "diff"
    diff_dir.mkdir(exist_ok=True)

    for name, data in [("new_findings", new_findings),
                       ("resolved_findings", resolved_findings),
                       ("unchanged_findings", unchanged_findings)]:
        with open(diff_dir / f"{name}.json", "w") as f:
            json.dump({"count": len(data), "findings": data}, f, indent=2, sort_keys=True)

    # CI gate evaluation
    gate_pass = True
    blocking = []
    if fail_severity:
        threshold = severity_rank_map.get(fail_severity)
        if threshold is None:
            click.echo(
                f"Unknown fail-on severity '{fail_severity}'. Available: {', '.join(severity_order)}",
                err=True,
            )
            threshold = None
        if threshold is not None:
            blocking = [
                f for f in new_findings
                if severity_rank_map.get(f.get("severity", "info"), len(severity_rank_map)) <= threshold
            ]
            gate_pass = len(blocking) == 0

    # Load rollup hashes if available
    base_rollup_hash = ""
    head_rollup_hash = ""
    for label, rp, dest in [("base", base_path, "base_rollup_hash"),
                             ("head", head_path, "head_rollup_hash")]:
        rpath = rp / "workflow" / "rollup.json"
        if rpath.exists():
            with open(rpath) as f:
                rdata = json.load(f)
            if dest == "base_rollup_hash":
                base_rollup_hash = rdata.get("dag_hash", "")
            else:
                head_rollup_hash = rdata.get("dag_hash", "")

    # Emit diff receipt
    receipt = {
        "schema": "diff_receipt_v1",
        "base_trace_root": base_root,
        "head_trace_root": head_root,
        "base_rollup_hash": base_rollup_hash,
        "head_rollup_hash": head_rollup_hash,
        "backend_id": head_backend_id,
        "policy_id": head_policy_id,
        "policy_version": head_policy_version,
        "review_rules": severity_rules,
        "finding_norm_version": head_norm_version,
        "comparator_version": "diff_v1",
        "base_finding_count": len(base_findings),
        "head_finding_count": len(head_findings),
        "new_count": len(new_findings),
        "resolved_count": len(resolved_findings),
        "unchanged_count": len(unchanged_findings),
        "new_findings_hash": sha256_bytes(canonical_json_bytes(
            {"count": len(new_findings), "findings": new_findings})),
        "resolved_findings_hash": sha256_bytes(canonical_json_bytes(
            {"count": len(resolved_findings), "findings": resolved_findings})),
        "unchanged_findings_hash": sha256_bytes(canonical_json_bytes(
            {"count": len(unchanged_findings), "findings": unchanged_findings})),
        "fail_on": fail_severity or "none",
        "gate_pass": gate_pass,
        "blocking_count": len(blocking),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tool_version": "capseal-0.2.0",
    }
    receipt["receipt_hash"] = sha256_bytes(canonical_json_bytes(receipt))

    with open(diff_dir / "receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    # Display
    click.echo(f"Review Diff ({receipt['comparator_version']})")
    click.echo(f"  base: {base_root[:16]}... ({len(base_findings)} findings)")
    click.echo(f"  head: {head_root[:16]}... ({len(head_findings)} findings)")
    click.echo(f"  backend: {head_backend_id}")
    click.echo(f"  policy:  {head_policy_id}")
    click.echo()
    click.echo(f"  New findings:       {len(new_findings)}")
    click.echo(f"  Resolved findings:  {len(resolved_findings)}")
    click.echo(f"  Unchanged findings: {len(unchanged_findings)}")
    click.echo()
    click.echo(f"  Receipt: {diff_dir / 'receipt.json'}")

    if new_findings:
        click.echo()
        click.echo("  New findings by severity:")
        new_sev: dict[str, int] = {}
        for f in new_findings:
            s = f.get("severity", "info")
            new_sev[s] = new_sev.get(s, 0) + 1
        for s in severity_order:
            if s in new_sev:
                click.echo(f"    {s}: {new_sev[s]}")

    # Auditable one-line summary
    click.echo()
    if fail_severity:
        status = "PASSED" if gate_pass else "FAILED"
        click.echo(
            f"GATE {status}: {len(blocking)} new findings at {fail_severity}+ "
            f"(base={len(base_findings)}, head={len(head_findings)}, "
            f"new={len(new_findings)}, resolved={len(resolved_findings)}, "
            f"backend={head_backend_id}, "
            f"base_rollup={base_rollup_hash[:16]}..., "
            f"head_rollup={head_rollup_hash[:16]}...)."
        )
    else:
        click.echo(
            f"DIFF COMPLETE: new={len(new_findings)}, resolved={len(resolved_findings)}, "
            f"unchanged={len(unchanged_findings)} "
            f"(backend={head_backend_id}, "
            f"base_rollup={base_rollup_hash[:16]}..., "
            f"head_rollup={head_rollup_hash[:16]}...)."
        )


# =============================================================================
# capsule pipeline --run <dir>
# =============================================================================

@click.command("pipeline")
@click.option("--project-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--run", "run_dir", required=True, type=click.Path())
@click.option("--policy", default="review_v1")
@click.option("--backend", default="llm")
@click.option("--agents", default=1, type=int)
@click.option("--llm-provider", default="openai")
@click.option("--llm-model", default=None)
@click.option("--llm-temperature", default=0.0, type=float)
@click.option("--llm-max-findings", default=MAX_AGENT_FINDINGS, type=int)
@click.option("--llm-include-ext", default=None)
@click.option("--llm-skip-ext", default=None)
@click.option("--llm-lenient", is_flag=True, default=False)
@click.option("--llm-redaction-mode", default="allow",
              type=click.Choice(["allow", "fail", "public"]), show_default=True)
@click.option("--explain", is_flag=True, help="Run explain-llm on new findings")
@click.option("--diff-base", type=click.Path(exists=True, file_okay=False),
              help="Existing run directory to diff against")
@click.option("--fail-on", default=None)
@click.option("--incremental-from", type=click.Path(exists=True, file_okay=False),
              help="Reuse trace artifacts from an existing run")
def pipeline_command(
    project_dir,
    run_dir,
    policy,
    backend,
    agents,
    llm_provider,
    llm_model,
    llm_temperature,
    llm_max_findings,
    llm_include_ext,
    llm_skip_ext,
    llm_lenient,
    llm_redaction_mode,
    explain,
    diff_base,
    fail_on,
    incremental_from,
):
    """Run trace -> review -> dag -> verify (and optional diff & explain)."""

    project_path = Path(project_dir).resolve()
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    click.echo("[1/5] Tracing...")
    trace_command.callback(
        project_dir=str(project_path),
        out=str(run_path),
        policy=policy,
        store_blobs=False,
        num_shards=agents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        incremental_from=incremental_from,
    )

    click.echo("[2/5] Review...")
    review_orchestrator_command.callback(
        run_dir=str(run_path),
        num_agents=agents,
        backend=backend,
        project_dir=str(project_path),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_findings=llm_max_findings,
        llm_replay=None,
        llm_include_ext=llm_include_ext,
        llm_skip_ext=llm_skip_ext,
        llm_lenient=llm_lenient,
        llm_redaction_mode=llm_redaction_mode,
    )

    click.echo("[3/5] DAG + rollup...")
    dag_command.callback(run_dir=str(run_path), project_dir=str(project_path))

    click.echo("[4/5] Verify rollup...")
    verify_rollup_command.callback(
        rollup_path=str(run_path / "workflow" / "rollup.json"),
        project_dir=str(project_path),
    )

    if diff_base:
        click.echo("[5/5] Review diff...")
        review_diff_command.callback(
            base_dir=str(Path(diff_base)),
            head_dir=str(run_path),
            fail_severity=fail_on,
            allow_backend_mismatch=False,
            allow_policy_mismatch=False,
        )

    if explain:
        explain_llm_command.callback(
            run_dir=str(run_path),
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=llm_temperature,
            llm_max_tokens=LLM_MAX_TOKENS,
            max_findings=llm_max_findings,
            min_severity="warning",
            diff_path=None,
            output_format="json",
            force=False,
            out=None,
        )

    click.echo("Pipeline complete.")


# =============================================================================
# capsule demo-review  (one-command end-to-end demo)
# =============================================================================

@click.command("demo-review")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False), default=".")
@click.option("--out", "-o", type=click.Path(), default=None,
              help="Output run directory (default: temp dir)")
@click.option("--backend", default="semgrep", help="Review backend: stub, semgrep")
@click.option("--num-shards", default=16, type=int)
@click.option("--agents", "num_agents", default=8, type=int)
def demo_review_command(project_dir, out, backend, num_shards, num_agents):
    """End-to-end review demo: trace + review + DAG + verify.

    Runs the full pipeline and prints a summary receipt.
    """
    import concurrent.futures
    import datetime
    import tempfile

    from bef_zk.capsule.project_trace_emitter import emit_project_trace

    proj = Path(project_dir).resolve()
    run_dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="capseal_demo_"))

    click.echo("=" * 60)
    click.echo("CAPSEAL REVIEW DEMO")
    click.echo("=" * 60)
    click.echo()

    # Step 1: Trace
    click.echo(f"[1/4] Tracing {proj} ...")
    manifest = emit_project_trace(
        proj, run_dir,
        policy_id="review_v1",
        num_shards=num_shards,
        chunk_size=65536,
    )

    with open(run_dir / "commitments.json") as f:
        commitments = json.load(f)
    trace_root = commitments["head_T"]
    click.echo(f"  trace_root: {trace_root[:32]}...")
    click.echo(f"  rows:       {commitments['total_rows']}")
    click.echo()

    # Step 2: Review
    click.echo(f"[2/4] Reviewing ({backend}, {num_shards} shards, {num_agents} agents) ...")
    with open(run_dir / "fold" / "shards.json") as f:
        shards_data = json.load(f)
    shards = shards_data.get("shards", [])
    reviews_dir = run_dir / "reviews"
    reviews_dir.mkdir(exist_ok=True)

    # Resolve backend
    semgrep_by_file = None
    backend_id = STUB_BACKEND_ID if backend == "stub" else backend
    if backend == "semgrep":
        semgrep_by_file, _, _, backend_id = _run_semgrep_global(proj, run_dir)

    review_paths = []
    all_findings = []

    def run_shard(shard):
        sid = shard["shard_id"]
        out_path = reviews_dir / f"review_shard_{sid}.json"

        if backend == "semgrep" and semgrep_by_file is not None:
            findings = _distribute_semgrep_to_shard(shard, semgrep_by_file)
        else:
            file_contents = {}
            for finfo in shard["files"]:
                fpath = proj / finfo["path"]
                if fpath.exists():
                    try:
                        file_contents[finfo["path"]] = fpath.read_text(errors="replace")
                    except Exception:
                        pass
            findings = _stub_agent_review(shard, file_contents, trace_root)

        sorted_findings = _sort_findings(findings)

        review = {
            "schema": "review_packet_v1",
            "trace_root": trace_root,
            "shard_id": sid,
            "backend": backend,
            "backend_id": backend_id,
            "policy_id": manifest.get("policy_id", "unknown"),
            "findings": sorted_findings,
        }
        with open(out_path, "w") as f:
            json.dump(review, f, indent=2, sort_keys=True)
        return out_path, sorted_findings

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as pool:
        futures = {pool.submit(run_shard, s): s for s in shards}
        for fut in concurrent.futures.as_completed(futures):
            path, findings = fut.result()
            review_paths.append(path)
            all_findings.extend(findings)

    agg_norm_version = FINDING_NORM_VERSION
    for f in all_findings:
        if f.get("finding_norm_version"):
            agg_norm_version = f["finding_norm_version"]
            break

    # Emit aggregate
    all_findings = _sort_findings(all_findings)

    _emit_aggregate(
        reviews_dir,
        all_findings,
        trace_root,
        backend_id,
        manifest.get("policy_id", "unknown"),
        manifest.get("policy_version", "unknown"),
        manifest.get("review_rules", {}),
        agg_norm_version,
    )
    click.echo(f"  findings:   {len(all_findings)}")

    # Count by severity
    sev_counts: dict[str, int] = {}
    for f in all_findings:
        s = f.get("severity", "info")
        sev_counts[s] = sev_counts.get(s, 0) + 1
    for s, c in sorted(sev_counts.items()):
        click.echo(f"    {s}: {c}")
    click.echo()

    # Step 3: DAG + Rollup
    click.echo("[3/4] Building DAG + rollup ...")
    # Inline the dag logic to avoid click context issues
    vertices = [{"id": "trace", "type": "project_trace_v1",
                 "hash": trace_root, "artifact": "commitments.json"}]
    review_vids = []
    for rf in sorted(reviews_dir.glob("review_shard_*.json")):
        vid = rf.stem
        vertices.append({"id": vid, "type": "review_packet_v1",
                         "hash": _file_hash(rf), "artifact": f"reviews/{rf.name}"})
        review_vids.append(vid)
    agg = reviews_dir / "aggregate.json"
    if agg.exists():
        vertices.append({"id": "aggregate", "type": "review_aggregate_v1",
                         "hash": _file_hash(agg), "artifact": "reviews/aggregate.json"})

    edges = [{"from": "trace", "to": v, "relation": "reviewed_by"} for v in review_vids]
    if agg.exists():
        edges += [{"from": v, "to": "aggregate", "relation": "aggregated_into"} for v in review_vids]

    dag = {"schema": "workflow_dag_v1", "vertices": vertices,
           "edges": edges, "root": "trace"}

    workflow_dir = run_dir / "workflow"
    workflow_dir.mkdir(exist_ok=True)
    with open(workflow_dir / "dag.json", "w") as f_:
        json.dump(dag, f_, indent=2, sort_keys=True)

    # Verification checks
    rows = []
    with open(run_dir / "trace.jsonl") as f_:
        for line in f_:
            if line.strip():
                rows.append(json.loads(line))
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    ok_trace, msg_trace = verify_project_trace(proj, rows, manifest, manifest_hash, trace_root)

    trace_index = load_trace_index(run_dir)
    ok_review = True
    for rp in review_paths:
        with open(rp) as f_:
            rv = json.load(f_)
        ok, _, _ = verify_review_packet(rv, trace_root, trace_index)
        if not ok:
            ok_review = False

    dag_hash = sha256_bytes(canonical_json_bytes(dag))
    merkle_root = _build_merkle_root([v["hash"] for v in vertices])

    rollup = {
        "schema": "workflow_rollup_v1",
        "dag_hash": dag_hash,
        "merkle_root": merkle_root,
        "vertex_hashes": [v["hash"] for v in vertices],
        "num_vertices": len(vertices),
        "num_edges": len(edges),
        "trace_root": trace_root,
        "checks": [
            {"check": "verify-trace", "pass": ok_trace, "message": msg_trace},
            {"check": "verify-review", "pass": ok_review,
             "message": f"{len(all_findings)} findings across {len(review_vids)} packets"},
        ],
        "all_pass": ok_trace and ok_review,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tool_version": "capseal-0.2.0",
    }
    with open(workflow_dir / "rollup.json", "w") as f_:
        json.dump(rollup, f_, indent=2, sort_keys=True)

    click.echo(f"  vertices:   {len(vertices)}")
    click.echo(f"  edges:      {len(edges)}")
    click.echo()

    # Step 4: Summary
    click.echo("[4/4] Verification ...")
    click.echo(f"  verify-trace:  {'PASS' if ok_trace else 'FAIL'}")
    click.echo(f"  verify-review: {'PASS' if ok_review else 'FAIL'}")
    click.echo()
    click.echo("=" * 60)
    click.echo(f"  trace_root:  {trace_root[:32]}...")
    click.echo(f"  merkle_root: {merkle_root[:32]}...")
    click.echo(f"  rollup_hash: {dag_hash[:32]}...")
    click.echo(f"  findings:    {len(all_findings)} ({backend})")
    click.echo(f"  shards:      {len(shards)}")
    click.echo(f"  run_dir:     {run_dir}")
    click.echo()
    if rollup["all_pass"]:
        click.echo("  STATUS: ALL PASS")
    else:
        click.echo("  STATUS: FAILED")
        raise SystemExit(1)
    click.echo("=" * 60)
