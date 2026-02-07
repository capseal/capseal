"""Helpers for deterministic shard prompt bundles and agent context."""
from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

from canonical.project_trace import sha256_bytes


@dataclass
class ChunkMeta:
    """Metadata for a chunk entry from trace.jsonl."""

    chunk_hash: str
    offset: int
    length: int


@dataclass
class FileMeta:
    """Metadata for an included file in the trace."""

    path: str
    file_hash: str
    size: int
    classification: str
    chunks: list[ChunkMeta] = field(default_factory=list)


@dataclass
class TraceIndex:
    """Fast lookup tables built from trace.jsonl for verification."""

    files_by_path: dict[str, FileMeta]
    file_hashes: set[str]
    chunk_hashes: set[str]
    file_chunks: dict[str, set[str]]  # file_hash -> chunk hashes
    chunk_to_path: dict[str, str]


@dataclass
class ChunkData:
    """Chunk data with decoded text + line range for prompting."""

    file_path: str
    chunk_hash: str
    offset: int
    length: int
    text: str
    prompt_text: str
    redactions: list[str]
    line_start: int
    line_end: int


@dataclass
class FileData:
    """Included file with decoded text and chunk data."""

    path: str
    file_hash: str
    size: int
    text: str
    lines: list[str]
    chunks: list[ChunkData]

    def snippet_for_range(self, start: int, end: int) -> str:
        """Return snippet covering [start, end] (1-based inclusive)."""
        if not self.lines:
            return ""
        lo = max(1, start)
        hi = max(lo, end)
        hi = min(hi, len(self.lines))
        return "\n".join(self.lines[lo - 1 : hi])


@dataclass
class ShardContext:
    """Fully materialized shard data used for prompting/verification."""

    shard_id: int
    files: dict[str, FileData]
    chunk_lookup: dict[str, ChunkData]
    chunk_hashes: list[str]
    outline_summary: dict[str, Any]
    prompt_bundle: dict[str, Any]
    redactions: list[str]


PROMPT_TEMPLATE_VERSION = "agent_prompt_v1"
NORMALIZER_VERSION = "finding_norm_v1"
MAX_AGENT_FINDINGS = 20
LLM_MAX_TOKENS = 2000
DEFAULT_MAX_INPUT_TOKENS = 80000
CACHE_SUBDIR = ("packets", "by_input_hash")
SEVERITY_ORDER = {"info": 1, "warning": 2, "error": 3}

DEFAULT_LLM_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "mock": "mock",
}

SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "aws_secret",
        re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+=]{30,}"),
    ),
    ("github_token", re.compile(r"gh[pous]_[A-Za-z0-9]{20,}")),
    (
        "dotenv",
        re.compile(r"(?i)(api[-_]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}"),
    ),
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
]


def load_trace_index(run_path: Path) -> TraceIndex:
    """Parse trace.jsonl once to build lookup tables."""

    trace_file = run_path / "trace.jsonl"
    if not trace_file.exists():
        raise FileNotFoundError(f"trace.jsonl not found under {run_path}")

    files_by_path: dict[str, FileMeta] = {}
    file_hashes: set[str] = set()
    chunk_hashes: set[str] = set()
    file_chunks: dict[str, set[str]] = {}
    chunk_to_path: dict[str, str] = {}

    current_path: Optional[str] = None
    current_hash: Optional[str] = None

    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rtype = row.get("row_type")

            if rtype == "file_entry":
                classification = row.get("classification")
                content_hash = row.get("content_hash")
                path = row.get("path")
                size = row.get("size", 0)

                if classification == "include" and content_hash and path:
                    meta = FileMeta(
                        path=path,
                        file_hash=content_hash,
                        size=size,
                        classification=classification,
                    )
                    files_by_path[path] = meta
                    file_hashes.add(content_hash)
                    file_chunks.setdefault(content_hash, set())
                    current_path = path
                    current_hash = content_hash
                else:
                    current_path = None
                    current_hash = None

            elif rtype == "chunk_entry" and current_path and current_hash:
                chunk_hash = row.get("chunk_hash")
                if not chunk_hash:
                    continue
                offset = int(row.get("offset", 0))
                length = int(row.get("length", 0))
                chunk_meta = ChunkMeta(chunk_hash=chunk_hash, offset=offset, length=length)
                files_by_path[current_path].chunks.append(chunk_meta)
                chunk_hashes.add(chunk_hash)
                file_chunks[current_hash].add(chunk_hash)
                chunk_to_path[chunk_hash] = current_path

    return TraceIndex(
        files_by_path=files_by_path,
        file_hashes=file_hashes,
        chunk_hashes=chunk_hashes,
        file_chunks=file_chunks,
        chunk_to_path=chunk_to_path,
    )


def _line_starts(file_bytes: bytes) -> list[int]:
    starts = [0]
    for idx, b in enumerate(file_bytes):
        if b == 0x0A:  # newline
            starts.append(idx + 1)
    return starts


def _offset_to_line(starts: list[int], offset: int) -> int:
    # bisect_right returns index representing line count; convert to 1-based line number
    return max(1, bisect.bisect_right(starts, max(0, offset)))


def _decode_chunk(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _load_file_bytes(
    path: Path,
    expected_hash: str,
) -> bytes:
    data = path.read_bytes()
    file_hash = sha256_bytes(data)
    if file_hash != expected_hash:
        raise ValueError(
            f"File {path} hash mismatch vs trace. Expected {expected_hash[:16]}..., got {file_hash[:16]}..."
        )
    return data


def _chunk_bytes_from_blobs(blobs_dir: Path, chunk_hash: str) -> Optional[bytes]:
    blob_path = blobs_dir / chunk_hash
    if blob_path.exists():
        data = blob_path.read_bytes()
        if sha256_bytes(data) != chunk_hash:
            raise ValueError(f"Blob {blob_path} hash mismatch")
        return data
    return None


def _redact_sensitive_text(text: str) -> tuple[str, list[str]]:
    """Redact obvious secret patterns while recording what changed."""

    notes: list[str] = []
    sanitized = text

    for name, pattern in SECRET_PATTERNS:
        if not pattern.search(sanitized):
            continue

        def _repl(match: re.Match[str], label: str = name) -> str:
            token = match.group(0)
            marker = f"[REDACTED_{sha256_bytes(token.encode('utf-8'))[:12]}]"
            notes.append(f"{label}:{marker}")
            return marker

        sanitized = pattern.sub(_repl, sanitized)

    return sanitized, notes


def build_shard_context(
    shard: dict,
    trace_index: TraceIndex,
    project_dir: Path,
    run_path: Path,
    trace_root: str,
    policy_id: str,
    *,
    redaction_mode: str = "allow",
) -> ShardContext:
    """Materialize shard files into prompt bundle + lookup tables."""

    blobs_dir = run_path / "blobs"
    blobs_available = blobs_dir.exists()

    files: dict[str, FileData] = {}
    chunk_lookup: dict[str, ChunkData] = {}
    chunk_hashes: list[str] = []
    redactions: list[str] = []

    for finfo in sorted(shard.get("files", []), key=lambda x: x.get("path", "")):
        path = finfo.get("path")
        if not path:
            continue

        meta = trace_index.files_by_path.get(path)
        if not meta:
            raise ValueError(f"Shard references path not present in trace: {path}")

        chunk_datas: list[ChunkData] = []

        # Attempt to gather chunk bytes either from blobs or filesystem
        chunk_bytes_list: list[bytes] = []
        use_blobs = blobs_available
        if use_blobs:
            for chunk_meta in meta.chunks:
                blob_bytes = _chunk_bytes_from_blobs(blobs_dir, chunk_meta.chunk_hash)
                if blob_bytes is None:
                    use_blobs = False
                    break
                chunk_bytes_list.append(blob_bytes)

        file_bytes: bytes
        if use_blobs:
            file_bytes = b"".join(chunk_bytes_list)
        else:
            full_path = project_dir / path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"File {path} missing under project dir {project_dir}. Re-run trace or keep blobs."
                )
            file_bytes = _load_file_bytes(full_path, meta.file_hash)
            chunk_bytes_list = []
            for chunk_meta in meta.chunks:
                start = chunk_meta.offset
                end = chunk_meta.offset + chunk_meta.length
                chunk_bytes_list.append(file_bytes[start:end])

        if not meta.chunks:
            if meta.size > 0:
                raise ValueError(f"Included file has no chunks recorded: {path}")
            chunk_bytes_list = []
            file_bytes = b""

        file_text = file_bytes.decode("utf-8", errors="replace")
        lines = file_text.splitlines()
        starts = _line_starts(file_bytes)

        for chunk_meta, chunk_bytes in zip(meta.chunks, chunk_bytes_list):
            # Sanity check sizes/source hash when using filesystem fallback
            if len(chunk_bytes) != chunk_meta.length:
                raise ValueError(
                    f"Chunk length mismatch for {path} offset {chunk_meta.offset}:"
                    f" expected {chunk_meta.length}, got {len(chunk_bytes)}"
                )
            if sha256_bytes(chunk_bytes) != chunk_meta.chunk_hash:
                raise ValueError(
                    f"Chunk hash mismatch for {path} offset {chunk_meta.offset}:"
                    f" expected {chunk_meta.chunk_hash[:16]}..."
                )

            line_start = _offset_to_line(starts, chunk_meta.offset)
            line_end = _offset_to_line(starts, chunk_meta.offset + max(chunk_meta.length - 1, 0))

            chunk_text = _decode_chunk(chunk_bytes)
            prompt_text, notes = _redact_sensitive_text(chunk_text)
            for note in notes:
                redactions.append(f"{path}:{chunk_meta.offset}-{chunk_meta.offset + chunk_meta.length}: {note}")
            chunk_data = ChunkData(
                file_path=path,
                chunk_hash=chunk_meta.chunk_hash,
                offset=chunk_meta.offset,
                length=chunk_meta.length,
                text=chunk_text,
                prompt_text=prompt_text,
                redactions=notes,
                line_start=line_start,
                line_end=line_end,
            )
            chunk_datas.append(chunk_data)
            chunk_lookup[chunk_meta.chunk_hash] = chunk_data
            chunk_hashes.append(chunk_meta.chunk_hash)

        file_data = FileData(
            path=path,
            file_hash=meta.file_hash,
            size=meta.size,
            text=file_text,
            lines=lines,
            chunks=chunk_datas,
        )
        files[path] = file_data

    outline_summary: dict[str, Any] = {}
    outline_path = run_path / "fold" / "repo_outline.json"
    if outline_path.exists():
        with open(outline_path) as f:
            outline = json.load(f)
        outline_summary = outline.get("summary", {})

    chunk_hashes_sorted = sorted(set(chunk_hashes))

    prompt_bundle = {
        "schema": "agent_prompt_bundle_v1",
        "trace_root": trace_root,
        "shard_id": shard.get("shard_id"),
        "policy_id": policy_id,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "chunk_hashes": chunk_hashes_sorted,
        "files": [
            {
                "path": path,
                "file_hash": f.file_hash,
                "size": f.size,
                "num_chunks": len(f.chunks),
                "chunks": [
                    {
                        "chunk_hash": c.chunk_hash,
                        "offset": c.offset,
                        "length": c.length,
                        "line_range": [c.line_start, c.line_end],
                        "text": c.prompt_text,
                        "redactions": c.redactions,
                    }
                    for c in f.chunks
                ],
            }
            for path, f in sorted(files.items())
        ],
        "outline_summary": outline_summary,
    }

    return ShardContext(
        shard_id=shard.get("shard_id"),
        files=files,
        chunk_lookup=chunk_lookup,
        chunk_hashes=chunk_hashes_sorted,
        outline_summary=outline_summary,
        prompt_bundle=prompt_bundle,
        redactions=redactions,
    )


def write_prompt_bundle(run_path: Path, shard_id: int, bundle: dict[str, Any]) -> Path:
    """Persist deterministic prompt bundle JSON for auditing."""
    prompts_dir = run_path / "reviews" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / f"prompt_shard_{shard_id}.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)
    return out_path


def compute_agent_input_hash(
    trace_root: str,
    shard_id: int,
    chunk_hashes: Iterable[str],
    policy_id: str,
    backend_id: str,
    prompt_version: str = PROMPT_TEMPLATE_VERSION,
) -> str:
    material = "||".join(
        [
            trace_root,
            str(shard_id),
            ",".join(sorted(chunk_hashes)),
            policy_id,
            backend_id,
            prompt_version,
        ]
    )
    return sha256_bytes(material.encode())


def cache_lookup(run_path: Path, agent_input_hash: str) -> Optional[dict[str, Any]]:
    cache_dir = run_path / "reviews"
    for part in CACHE_SUBDIR:
        cache_dir = cache_dir / part
    cache_path = cache_dir / f"{agent_input_hash}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def cache_store(run_path: Path, agent_input_hash: str, review: dict[str, Any]) -> None:
    cache_dir = run_path / "reviews"
    for part in CACHE_SUBDIR:
        cache_dir = cache_dir / part
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{agent_input_hash}.json"
    with open(cache_path, "w") as f:
        json.dump(review, f, indent=2, sort_keys=True)


def validate_finding_references(finding: dict[str, Any], trace_index: TraceIndex) -> list[str]:
    """Return a list of reference errors for a single finding."""

    errors: list[str] = []
    file_hash = finding.get("file_hash")
    file_path = finding.get("file_path")

    if file_hash and file_hash not in trace_index.file_hashes:
        errors.append(f"unknown file_hash {file_hash[:16]}...")

    chunk_hashes = finding.get("chunk_hashes", []) or []
    for ch in chunk_hashes:
        if ch not in trace_index.chunk_hashes:
            errors.append(f"unknown chunk_hash {ch[:16]}...")
            continue
        if file_hash:
            file_chunk_set = trace_index.file_chunks.get(file_hash, set())
            if ch not in file_chunk_set:
                errors.append(
                    f"chunk {ch[:16]}... not under file hash {file_hash[:16]}..."
                )
        elif file_path:
            expected_path = trace_index.chunk_to_path.get(ch)
            if expected_path and expected_path != file_path:
                errors.append(
                    f"chunk {ch[:16]}... does not belong to {file_path} (expected {expected_path})"
                )

    return errors


def verify_review_packet(
    review: dict[str, Any],
    trace_root: str,
    trace_index: TraceIndex,
) -> tuple[bool, list[str], int]:
    """Verify references inside a review packet. Returns (ok, errors, total_findings)."""

    errors: list[str] = []
    findings = review.get("findings", []) or []

    if review.get("trace_root") != trace_root:
        errors.append("trace_root mismatch")

    for idx, finding in enumerate(findings):
        refs = validate_finding_references(finding, trace_index)
        for err in refs:
            errors.append(f"finding[{idx}]: {err}")

    return not errors, errors, len(findings)


DEFAULT_MAX_INPUT_TOKENS = 60000  # Conservative default (~240KB prompt)


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~4 chars per token."""
    return len(text) // 4


def render_llm_prompt(
    shard_ctx: ShardContext,
    max_findings: int = MAX_AGENT_FINDINGS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> tuple[str, int, int]:
    """Render deterministic text prompt for the LLM backend.

    Returns (prompt_text, included_chunks, total_chunks).
    Truncates deterministically if over budget.
    """

    bundle = shard_ctx.prompt_bundle
    outline = bundle.get("outline_summary", {}) or {}
    files = bundle.get("files", [])

    instructions = textwrap.dedent(
        f"""
You are a senior software engineer performing an independent code review.
The repository has been traced into canonical chunks. Every claim MUST cite chunk hashes.

Rules:
1. Only reason about the provided files/chunks. No speculation beyond them.
2. Each finding must cite a `primary_chunk_hash` and optional `supporting_chunk_hashes`.
3. `severity` must be one of: error, warning, info.
4. `message` MUST be one sentence that states the issue and how to fix it.
5. Prefer real bugs/security issues over style nits.
6. Return at most {max_findings} findings.
7. Output STRICT JSON matching the schema shown below — no prose, code fences, or markdown.

JSON schema (single object):
{{
  "findings": [
    {{
      "rule_id": "category.identifier",
      "severity": "error|warning|info",
      "file_path": "relative/path.py",
      "primary_chunk_hash": "sha256...",
      "supporting_chunk_hashes": ["sha256..."],
      "line_range": [start_line, end_line],
      "message": "Issue summary + actionable remediation.",
      "snippet": "Optional short excerpt from those lines."
    }}
  ]
}}
"""
    ).strip()

    # Build header (always included)
    header_lines = [instructions, ""]
    header_lines.append(f"Trace root: {bundle.get('trace_root')}  |  Policy: {bundle.get('policy_id')}  |  Shard: {bundle.get('shard_id')}")
    if shard_ctx.redactions:
        header_lines.append(f"Redactions: {len(shard_ctx.redactions)} sensitive substrings replaced")
    header_lines.append(
        "Repo outline summary: "
        + json.dumps(outline, sort_keys=True)
    )
    header_lines.append("")
    header_lines.append("FILES AND CHUNKS:")
    header_text = "\n".join(header_lines)

    # Calculate budget for code content
    header_tokens = _estimate_tokens(header_text)
    budget_tokens = max_input_tokens - header_tokens - 500  # Reserve for truncation note

    # Build file/chunk content with budget enforcement
    content_lines: list[str] = []
    total_chunks = 0
    included_chunks = 0
    current_tokens = 0
    truncated = False

    for file_entry in files:
        file_header = f"FILE {file_entry['path']}  hash={file_entry['file_hash']}  size={file_entry['size']} bytes  chunks={file_entry['num_chunks']}"

        for chunk in file_entry.get("chunks", []):
            total_chunks += 1
            lr = chunk.get("line_range", [0, 0])
            chunk_text = chunk.get("text", "")

            chunk_block = (
                f"  CHUNK {chunk['chunk_hash']}  lines {lr[0]}-{lr[1]}  offset={chunk['offset']} len={chunk['length']}\n"
                f"  <<<CODE>>>\n{chunk_text}\n  <</CODE>>"
            )
            chunk_tokens = _estimate_tokens(chunk_block)

            # Check budget before adding
            if current_tokens + chunk_tokens > budget_tokens:
                truncated = True
                break

            # Add file header before first chunk of this file
            if not any(file_header in line for line in content_lines):
                content_lines.append(file_header)

            content_lines.append(chunk_block)
            current_tokens += chunk_tokens
            included_chunks += 1

        if truncated:
            break
        content_lines.append("")

    # Assemble final prompt
    lines = [header_text] + content_lines

    if truncated:
        lines.append("")
        lines.append(f"[TRUNCATED: {included_chunks}/{total_chunks} chunks included due to token budget ({max_input_tokens} max)]")
        lines.append("Focus on the chunks shown above. Findings referencing omitted chunks will be rejected.")

    return "\n".join(lines), included_chunks, total_chunks


def _call_anthropic(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": "You are a precise code review agent. Respond with strict JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API error {e.code}: {e.read().decode()[:200]}")

    content = payload.get("content", [])
    if content:
        return content[0].get("text", "")
    raise RuntimeError("Anthropic API returned empty content")


def _call_openai(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    data = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise code review agent. Output strict JSON matching the schema.",
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI API error {e.code}: {e.read().decode()[:200]}")

    choices = payload.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    raise RuntimeError("OpenAI API returned no choices")


def _call_gemini(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"content-type": "application/json"}
    data = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
    ).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:200]}")

    candidates = payload.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "")
    raise RuntimeError("Gemini API returned empty response")


def _call_mock(prompt: str, model: str, *_args, **_kwargs) -> str:
    if model and Path(model).exists():
        return Path(model).read_text()
    env_file = os.environ.get("CAPSEAL_MOCK_LLM_FILE")
    if env_file and Path(env_file).exists():
        return Path(env_file).read_text()
    inline = os.environ.get("CAPSEAL_MOCK_LLM_RAW")
    if inline:
        return inline
    raise RuntimeError(
        "Mock LLM provider requires --llm-model pointing to a file, "
        "CAPSEAL_MOCK_LLM_FILE, or CAPSEAL_MOCK_LLM_RAW"
    )


def call_llm_backend(
    provider: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: int = 120,
) -> str:
    provider = provider.lower()
    if provider == "anthropic":
        return _call_anthropic(prompt, model, temperature, max_tokens, timeout)
    if provider == "openai":
        return _call_openai(prompt, model, temperature, max_tokens, timeout)
    if provider == "gemini":
        return _call_gemini(prompt, model, temperature, max_tokens, timeout)
    if provider == "mock":
        return _call_mock(prompt, model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _json_candidates(raw_text: str) -> list[str]:
    stripped = _strip_code_fence(raw_text)
    candidates = [stripped]
    if stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            braces = stripped[start : end + 1]
            if braces not in candidates:
                candidates.append(braces)
    return [c for c in candidates if c]


def parse_llm_output(raw_text: str) -> list[dict[str, Any]]:
    for candidate in _json_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        findings = parsed.get("findings") if isinstance(parsed, dict) else None
        if findings is None or not isinstance(findings, list):
            continue
        return findings
    raise ValueError("LLM output was not valid JSON after repair attempts")


def render_explain_prompt(entries: list[dict[str, Any]]) -> str:
    instructions = textwrap.dedent(
        """
You are summarizing verified code review findings. For each issue, provide JSON
fields:
  - fingerprint (exact string provided)
  - analysis (one sentence describing the risk/bug)
  - recommendation (specific remediation step)
  - suggested_change (optional short patch or instructions)

Strict guidance for common categories to ensure safe, correct advice:
  - subprocess / command execution:
      - Prefer shell=False with argv list; do not recommend shell quoting as a primary mitigation.
      - Do not suggest shlex.escape/quote unless the shell is truly unavoidable; instead remove shell usage.
      - If user input influences behavior, enforce an allowlist of permitted commands/modes.
  - SQL / database access:
      - Use parameterized queries or ORM bindings; never string-concatenate untrusted input into SQL.
  - HTTP / server configuration:
      - Default dev servers to bind 127.0.0.1; avoid 0.0.0.0 unless explicitly required and gated by config.

Output strict JSON:
{
  "explanations": [
    {
      "fingerprint": "...",
      "analysis": "...",
      "recommendation": "...",
      "suggested_change": "..."
    }
  ]
}
"""
    ).strip()

    lines = [instructions, "", "FINDINGS:"]
    for idx, entry in enumerate(entries, 1):
        snippet = entry.get("snippet", "").strip()
        message = entry.get("message", "").strip()
        lines.append(
            textwrap.dedent(
                f"""
{idx}. fingerprint: {entry['fingerprint']}
   severity: {entry.get('severity')}
   file: {entry.get('file_path')} lines {entry.get('line_range')}
   message: {message}
   snippet:\n```\n{snippet}\n```
"""
            ).strip()
        )
    return "\n".join(lines)


def parse_explain_output(raw_text: str) -> list[dict[str, Any]]:
    for candidate in _json_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        explanations = parsed.get("explanations") if isinstance(parsed, dict) else None
        if explanations is None or not isinstance(explanations, list):
            continue
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(explanations):
            if not isinstance(item, dict):
                continue
            fingerprint = item.get("fingerprint")
            analysis = item.get("analysis")
            recommendation = item.get("recommendation")
            if not (fingerprint and analysis and recommendation):
                continue
            summary = {
                "fingerprint": str(fingerprint),
                "analysis": str(analysis).strip(),
                "recommendation": str(recommendation).strip(),
            }
            suggested = item.get("suggested_change")
            if suggested:
                summary["suggested_change"] = str(suggested).strip()
            normalized.append(summary)
        if normalized:
            return normalized
    raise ValueError("LLM explanation output was not valid JSON after repair attempts")


def run_llm_explain(
    entries: list[dict[str, Any]],
    provider: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> tuple[str, str, list[dict[str, Any]]]:
    prompt = render_explain_prompt(entries)
    raw_output = call_llm_backend(
        provider, model, prompt, temperature=temperature, max_tokens=max_tokens
    )
    explanations = parse_explain_output(raw_output)
    return prompt, raw_output, explanations


def _resolve_chunk_hash(
    candidate: str,
    shard_ctx: ShardContext,
    file_path: str,
    lenient: bool,
) -> str | None:
    """Resolve a chunk hash, optionally using prefix matching in lenient mode."""
    # Exact match first
    if candidate in shard_ctx.chunk_lookup:
        owner = shard_ctx.chunk_lookup[candidate].file_path
        if owner == file_path:
            return candidate
        return None  # Wrong file

    if not lenient:
        return None

    # Lenient: try prefix match (min 12 chars for uniqueness)
    if len(candidate) < 12:
        return None

    prefix = candidate[:16].lower()
    matches = []
    for ch in shard_ctx.chunk_hashes:
        if ch.lower().startswith(prefix):
            owner = shard_ctx.chunk_lookup[ch].file_path
            if owner == file_path:
                matches.append(ch)

    if len(matches) == 1:
        return matches[0]  # Unique prefix match
    return None  # Ambiguous or no match


def normalize_llm_findings(
    raw_findings: list[dict[str, Any]],
    shard_ctx: ShardContext,
    max_findings: int = MAX_AGENT_FINDINGS,
    lenient: bool = False,
) -> list[dict[str, Any]]:
    """Validate + normalize agent findings into review schema.

    If lenient=True, allows prefix matching for chunk hashes (LLMs often
    truncate/mangle 64-char hex strings).
    """

    normalized: list[dict[str, Any]] = []
    allowed_severity = {"error", "warning", "info"}

    for idx, entry in enumerate(raw_findings):
        if idx >= max_findings:
            break
        if not isinstance(entry, dict):
            raise ValueError(f"LLM finding[{idx}] is not an object")

        file_path = entry.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"LLM finding[{idx}] missing file_path")
        file_data = shard_ctx.files.get(file_path)
        if not file_data:
            raise ValueError(f"LLM finding[{idx}] references unknown file {file_path}")

        primary_chunk = entry.get("primary_chunk_hash")
        if not isinstance(primary_chunk, str) or not primary_chunk:
            raise ValueError(f"LLM finding[{idx}] missing primary_chunk_hash")

        chunk_hashes: list[str] = []
        extras = entry.get("supporting_chunk_hashes", [])
        if extras is None:
            extras = []
        if not isinstance(extras, list):
            raise ValueError(
                f"LLM finding[{idx}] supporting_chunk_hashes must be a list if provided"
            )
        for ch in [primary_chunk] + extras:
            if not ch:
                continue
            resolved = _resolve_chunk_hash(ch, shard_ctx, file_path, lenient)
            if resolved is None:
                if lenient:
                    continue  # Skip unresolvable in lenient mode
                raise ValueError(f"LLM finding[{idx}] references unknown chunk {ch[:16]}...")
            if resolved not in chunk_hashes:
                chunk_hashes.append(resolved)

        if not chunk_hashes:
            if lenient:
                continue  # Skip findings with no resolvable chunk hashes in lenient mode
            raise ValueError(f"LLM finding[{idx}] produced no valid chunk hashes")

        severity = str(entry.get("severity", "warning")).lower()
        if severity not in allowed_severity:
            raise ValueError(
                f"LLM finding[{idx}] has invalid severity '{entry.get('severity')}'."
                " Must be error|warning|info."
            )

        message = entry.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"LLM finding[{idx}] missing message")

        primary_chunk_data = shard_ctx.chunk_lookup[chunk_hashes[0]]
        line_range = entry.get("line_range")
        if isinstance(line_range, list) and len(line_range) == 2:
            try:
                start_line = int(line_range[0])
                end_line = int(line_range[1])
            except (TypeError, ValueError):
                start_line = primary_chunk_data.line_start
                end_line = primary_chunk_data.line_end
        else:
            start_line = primary_chunk_data.line_start
            end_line = primary_chunk_data.line_end
        normalized_line_range = [start_line, end_line]

        snippet = entry.get("snippet")
        snippet_text = ""
        if isinstance(snippet, str) and snippet.strip():
            snippet_text = snippet.strip()
        else:
            snippet_text = file_data.snippet_for_range(start_line, end_line)
            if not snippet_text:
                snippet_text = primary_chunk_data.text.strip()
        if snippet_text:
            snippet_lines = snippet_text.splitlines()
            if len(snippet_lines) > 40:
                snippet_text = "\n".join(snippet_lines[:40])

        rule_id = entry.get("rule_id")
        rule_id_str = rule_id if isinstance(rule_id, str) and rule_id else "llm.analysis"

        normalized.append(
            {
                "file_path": file_path,
                "file_hash": file_data.file_hash,
                "chunk_hashes": chunk_hashes,
                "rule_id": rule_id_str,
                "severity": severity,
                "message": message.strip(),
                "line_range": normalized_line_range,
                "snippet": snippet_text,
            }
        )

    return normalized


def run_llm_agent(
    shard_ctx: ShardContext,
    provider: str,
    model: str,
    backend_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = LLM_MAX_TOKENS,
    max_findings: int = MAX_AGENT_FINDINGS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    raw_override: str | None = None,
    lenient: bool = False,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Execute or replay the LLM backend and return (findings, raw_output, included_chunks, total_chunks).

    If lenient=True, chunk hash prefix matching is enabled (useful when LLMs
    truncate or mangle the 64-char hex strings).

    max_input_tokens enforces a hard cap on prompt size to prevent API errors.
    """

    prompt, included_chunks, total_chunks = render_llm_prompt(
        shard_ctx, max_findings=max_findings, max_input_tokens=max_input_tokens
    )
    if raw_override is None:
        raw_output = call_llm_backend(
            provider, model, prompt, temperature=temperature, max_tokens=max_tokens
        )
    else:
        raw_output = raw_override
    findings = normalize_llm_findings(
        parse_llm_output(raw_output), shard_ctx, max_findings=max_findings, lenient=lenient
    )
    return findings, raw_output, included_chunks, total_chunks


def build_llm_backend_id(
    provider: str,
    model: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    temp_str = ("{:.2f}".format(temperature)).rstrip("0").rstrip(".")
    if not temp_str:
        temp_str = "0"
    return (
        f"llm:{provider}:{model}|prompt={PROMPT_TEMPLATE_VERSION}|"
        f"norm={NORMALIZER_VERSION}|temp={temp_str}|max={max_tokens}"
    )


def resolve_llm_model(provider: str, override: Optional[str]) -> str:
    provider = provider.lower()
    if override:
        return override
    default = DEFAULT_LLM_MODELS.get(provider)
    if not default:
        raise ValueError(f"No default model configured for provider {provider}")
    return default
