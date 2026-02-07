"""Greptile integration - codebase indexing and semantic queries."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import click
import urllib.request
import urllib.error

from dataclasses import dataclass

from .redact import redact_secrets
from .git_utils import repo_fingerprint


GREPTILE_API_BASE = "https://api.greptile.com/v2"

# Performance: Use 75s deadline instead of 600s (fail fast, retry later)
GREPTILE_MAX_WAIT_SEC = int(os.environ.get("GREPTILE_MAX_WAIT_SEC", "75"))

# Cache directory for Greptile ephemeral repos
GREPTILE_CACHE_DIR = Path.home() / ".capseal" / "greptile_cache"

def _offline_mode() -> bool:
    """Return True if GREPTILE_OFFLINE mock mode is enabled."""
    val = os.environ.get("GREPTILE_OFFLINE", "").strip().lower()
    return val in {"1", "true", "yes", "on"}

def _mock_status_store_path(repo_id: str) -> Path:
    safe_id = repo_id.replace("/", "_").replace(":", "_")
    return GREPTILE_CACHE_DIR / f"mock_status_{safe_id}.json"

def _mock_api_request(endpoint: str, method: str = "GET", data: dict | None = None) -> dict[str, Any]:
    """Minimal offline stub for Greptile API.

    Simulates:
      - POST /repositories: returns submitted and a fake sha
      - GET  /repositories/{id}: progresses PROCESSING -> COMPLETED over a few reads
      - POST /query: returns a canned message and a couple of fake sources
    """
    GREPTILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ep = endpoint.strip()

    if method == "POST" and ep == "/repositories":
        repo = (data or {}).get("repository", "unknown/unknown")
        br = (data or {}).get("branch", "main")
        return {
            "status": "submitted",
            "sha": "deadbeefcafebabe0123456789abcdef",
            "filesProcessed": 0,
            "numFiles": 0,
            "repository": repo,
            "branch": br,
        }

    if method == "GET" and ep.startswith("/repositories/"):
        repo_id = ep.split("/repositories/")[-1]
        store = _mock_status_store_path(repo_id)
        try:
            state = json.loads(store.read_text())
        except Exception:
            state = {"reads": 0}
        reads = int(state.get("reads", 0)) + 1
        state["reads"] = reads

        if reads <= 2:
            result = {
                "status": "PROCESSING",
                "filesProcessed": 5 * reads,
                "numFiles": 15,
                "sha": "deadbeefcafebabe0123456789abcdef",
            }
        else:
            result = {
                "status": "COMPLETED",
                "filesProcessed": 15,
                "numFiles": 15,
                "sha": "deadbeefcafebabe0123456789abcdef",
            }
        try:
            store.write_text(json.dumps(state))
        except Exception:
            pass
        return result

    if method == "POST" and ep == "/query":
        messages = (data or {}).get("messages", [])
        content = messages[0]["content"] if messages else ""
        repos = (data or {}).get("repositories", [])
        repo_str = ", ".join([f"{r.get('remote','github')}:{r.get('repository','unknown')}:{r.get('branch','main')}" for r in repos])
        return {
            "message": f"[MOCK] Analyzed request against {repo_str}.\nPrompt excerpt: {content[:120]}...\nThis is a mock response for offline UX testing.",
            "sources": [
                {"filepath": "src/example.py", "lineNumbers": [10, 20], "repository": repos[0].get("repository", "repo") if repos else "repo"},
                {"filepath": "lib/utils.ts", "lineNumbers": [5, 12], "repository": repos[0].get("repository", "repo") if repos else "repo"},
            ],
        }

    return {"message": "[MOCK] Unsupported mock endpoint", "endpoint": ep, "method": method}


@dataclass
class IndexStatus:
    """Result of Greptile index polling with confidence signaling."""
    completed: bool           # Index finished (success or failure)
    success: bool             # Index succeeded
    partial: bool             # Only partial results available (timeout/stagnation)
    files_done: int           # Files processed
    files_total: int          # Total files expected
    elapsed_sec: float        # Time spent polling
    stagnant: bool = False    # Progress stalled for too long
    error: str | None = None  # Error message if failed

    @property
    def confidence(self) -> str:
        """Return confidence level: high, medium, low."""
        if self.success and not self.partial:
            return "high"
        elif self.partial and self.files_done > 0:
            ratio = self.files_done / max(self.files_total, 1)
            return "medium" if ratio > 0.5 else "low"
        return "low"


def _poll_index_status(
    repo_id: str,
    max_wait_sec: int = GREPTILE_MAX_WAIT_SEC,
    stagnation_threshold: int = 5,
    verbose: bool = True,
) -> IndexStatus:
    """Poll Greptile index status with stagnation detection and exponential backoff.

    Args:
        repo_id: Encoded repository ID (github:main:owner%2Frepo)
        max_wait_sec: Maximum time to wait (default 10 min)
        stagnation_threshold: Iterations without progress before declaring stagnant
        verbose: Print progress updates

    Returns:
        IndexStatus with completion state and confidence level
    """
    start_time = time.time()
    last_files_done = -1
    stagnant_count = 0
    poll_interval = 5  # Start with 5 seconds

    files_done = 0
    files_total = 0

    while (time.time() - start_time) < max_wait_sec:
        elapsed = time.time() - start_time

        try:
            status = _api_request(f"/repositories/{repo_id}", method="GET")
            state = (status.get("status") or "unknown").upper()
            files_done = status.get("filesProcessed", 0)
            files_total = status.get("numFiles", 0) or files_done or 1

            if state == "COMPLETED":
                if verbose:
                    click.echo(f"  ✓ Index complete ({files_done}/{files_total} files)")
                return IndexStatus(
                    completed=True,
                    success=True,
                    partial=False,
                    files_done=files_done,
                    files_total=files_total,
                    elapsed_sec=elapsed,
                )

            elif state == "FAILED":
                error_msg = status.get("message", "unknown error")
                if verbose:
                    click.echo(f"  ✗ Index failed: {error_msg}")
                return IndexStatus(
                    completed=True,
                    success=False,
                    partial=False,
                    files_done=files_done,
                    files_total=files_total,
                    elapsed_sec=elapsed,
                    error=error_msg,
                )

            else:
                # Check for stagnation (no progress)
                if files_done == last_files_done and files_done > 0:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                    last_files_done = files_done

                if stagnant_count >= stagnation_threshold:
                    if verbose:
                        click.echo(f"  ⚠ Index stagnant ({files_done}/{files_total} files, no progress)")
                    return IndexStatus(
                        completed=True,
                        success=True,  # Treat as partial success
                        partial=True,
                        files_done=files_done,
                        files_total=files_total,
                        elapsed_sec=elapsed,
                        stagnant=True,
                    )

                if verbose:
                    click.echo(f"  ... {state} {files_done}/{files_total} files ({int(elapsed)}s)")

        except Exception as e:
            if verbose:
                click.echo(f"  ... checking ({int(elapsed)}s)")

        # Exponential backoff: 5s → 10s → 15s → 20s (capped at 20s)
        time.sleep(poll_interval)
        poll_interval = min(poll_interval + 5, 20)

    # Timeout
    elapsed = time.time() - start_time
    if verbose:
        click.echo(f"  ⚠ Index timeout ({int(elapsed)}s) - proceeding with partial results")

    return IndexStatus(
        completed=True,
        success=True,  # Treat timeout as partial success
        partial=True,
        files_done=files_done,
        files_total=files_total,
        elapsed_sec=elapsed,
    )


def _compute_tree_hash(local_path: Path, exclude_patterns: list[str]) -> str:
    """Compute deterministic hash of filtered file tree.

    Returns a hash that changes only when file contents change.
    """
    import fnmatch

    file_hashes = []

    for root, dirs, files in os.walk(local_path):
        rel_root = os.path.relpath(root, local_path)

        # Skip excluded directories (same logic as _create_sparse_clone)
        skip_dir = False
        for pattern in exclude_patterns:
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                if fnmatch.fnmatch(rel_root, dir_pattern) or fnmatch.fnmatch(rel_root, f"*/{dir_pattern}"):
                    skip_dir = True
                    break
                for part in rel_root.split(os.sep):
                    if fnmatch.fnmatch(part, dir_pattern):
                        skip_dir = True
                        break

        if skip_dir or rel_root.startswith('.git'):
            dirs[:] = []
            continue

        # Filter directories: exclude patterns + symlinks (avoid traversal attacks)
        dirs[:] = [d for d in dirs if not any(
            fnmatch.fnmatch(d, p.rstrip('/')) for p in exclude_patterns if p.endswith('/')
        ) and not os.path.islink(os.path.join(root, d))]

        for fname in sorted(files):  # Sort for determinism
            skip_file = False
            for pattern in exclude_patterns:
                if not pattern.endswith('/') and fnmatch.fnmatch(fname, pattern):
                    skip_file = True
                    break

            if skip_file:
                continue

            src = Path(root) / fname

            # Skip symlinks (security: avoid following links outside tree)
            if src.is_symlink():
                continue

            try:
                if src.stat().st_size > 1_000_000:
                    continue
                # Hash file path + size + mtime for quick comparison
                stat = src.stat()
                file_key = f"{src.relative_to(local_path)}:{stat.st_size}:{int(stat.st_mtime)}"
                file_hashes.append(file_key)
            except OSError:
                continue

    # Combine all file hashes
    content = "\n".join(sorted(file_hashes))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_cache(cache_key: str) -> dict | None:
    """Load cached ephemeral repo info if still valid."""
    GREPTILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = GREPTILE_CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text())
        # Check if cache is stale (older than 24 hours)
        if time.time() - data.get("created_at", 0) > 86400:
            cache_file.unlink()
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(cache_key: str, repo_full: str, indexed_at: float) -> None:
    """Save ephemeral repo info to cache."""
    GREPTILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = GREPTILE_CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "cache_key": cache_key,
        "repo_full": repo_full,
        "created_at": time.time(),
        "indexed_at": indexed_at,
    }, indent=2))


def _clear_cache(cache_key: str) -> None:
    """Remove a cache entry."""
    cache_file = GREPTILE_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        cache_file.unlink()


def _load_env_file() -> None:
    """Load .env file if present."""
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",  # BEF-main/.env
        Path.home() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    if key not in os.environ:
                        os.environ[key] = value


def _get_api_key() -> str | None:
    """Get Greptile API key from environment."""
    _load_env_file()
    return os.environ.get("GREPTILE_API_KEY") or os.environ.get("GREPTILE_KEY")


def _get_github_token() -> str | None:
    """Get GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _api_request(
    endpoint: str,
    method: str = "GET",
    data: dict | None = None,
    api_key: str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Make Greptile API request."""
    # Offline mock mode: simulate responses without network or tokens
    if _offline_mode():
        return _mock_api_request(endpoint, method=method, data=data or {})
    key = api_key or _get_api_key()
    if not key:
        raise click.ClickException(
            "GREPTILE_API_KEY not set. Export it:\n"
            "  export GREPTILE_API_KEY='your-key'"
        )

    gh_token = github_token or _get_github_token()
    if not gh_token:
        raise click.ClickException(
            "GITHUB_TOKEN not set (needed for private repos). Export it:\n"
            "  export GITHUB_TOKEN='ghp_...'"
        )

    url = f"{GREPTILE_API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {key}",
        "X-GitHub-Token": gh_token,
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        try:
            error_json = json.loads(error_body)
            msg = error_json.get("message", error_body)
        except json.JSONDecodeError:
            msg = error_body
        raise click.ClickException(f"Greptile API error ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise click.ClickException(f"Network error: {e.reason}")


def _parse_repo_string(repo: str) -> tuple[str, str, str]:
    """Parse repo string like 'github:owner/repo:branch' or 'owner/repo'."""
    parts = repo.split(":")
    if len(parts) == 3:
        remote, repository, branch = parts
    elif len(parts) == 2:
        # Could be 'github:owner/repo' or 'owner/repo:branch'
        if parts[0] in ("github", "gitlab"):
            remote, repository = parts
            branch = "main"
        else:
            remote = "github"
            repository, branch = parts
    else:
        remote = "github"
        repository = parts[0]
        branch = "main"
    return remote, repository, branch


@click.group("greptile")
def greptile_group():
    """Greptile codebase intelligence commands.

    Index repositories and query them with natural language.

    Setup:
        export GREPTILE_API_KEY='your-key'
        export GITHUB_TOKEN='ghp_...'  # For private repos
    """
    pass


def _gh_api(endpoint: str, method: str = "GET", data: dict | None = None) -> dict:
    """Call GitHub API via gh CLI or curl fallback."""
    import subprocess
    import shutil

    # Try gh CLI first
    if shutil.which("gh"):
        cmd = ["gh", "api", endpoint, "-X", method]
        if data:
            for k, v in data.items():
                cmd.extend(["-f", f"{k}={v}"])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else {}

    # Fallback to curl with token
    token = _get_github_token()
    if not token:
        raise click.ClickException("No GitHub token. Set GITHUB_TOKEN or run: gh auth login")

    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else f"https://api.github.com/{endpoint}"
    headers = ["-H", f"Authorization: token {token}", "-H", "Accept: application/vnd.github+json"]

    if method == "GET":
        cmd = ["curl", "-s"] + headers + [url]
    elif method == "POST":
        cmd = ["curl", "-s", "-X", "POST"] + headers + ["-d", json.dumps(data or {}), url]
    elif method == "DELETE":
        cmd = ["curl", "-s", "-X", "DELETE"] + headers + [url]
    else:
        cmd = ["curl", "-s", "-X", method] + headers + [url]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


# Default patterns to exclude from ephemeral push (build artifacts, deps, etc.)
EPHEMERAL_EXCLUDE_PATTERNS = [
    # Dependencies
    "node_modules/", ".venv/", "venv/", "env/", ".env/",
    "__pycache__/", "*.pyc", "*.pyo", ".pytest_cache/",
    "vendor/", "bower_components/", ".bundle/",
    # Build artifacts
    "dist/", "build/", "out/", "target/", "_build/",
    "*.egg-info/", "*.egg", "*.whl", "*.so", "*.dylib",
    # IDE/editor
    ".idea/", ".vscode/", "*.swp", "*.swo", ".DS_Store",
    # Large files
    "*.pdf", "*.zip", "*.tar", "*.gz", "*.rar", "*.7z",
    "*.mp4", "*.mov", "*.avi", "*.mp3", "*.wav",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.webp",
    # Logs/temp
    "*.log", "logs/", "tmp/", "temp/", ".tmp/",
    # Package locks (huge, not useful for analysis)
    "package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock",
    # Coverage/test artifacts
    "coverage/", ".coverage", "htmlcov/", ".tox/",
    # Misc
    ".git/", ".hg/", ".svn/",
]


def _create_sparse_clone(local_path: Path, target_path: Path, exclude_patterns: list[str]) -> int:
    """Create a filtered copy of repo for ephemeral push.

    Returns number of files included.
    """
    import shutil
    import fnmatch

    target_path.mkdir(parents=True, exist_ok=True)
    file_count = 0

    for root, dirs, files in os.walk(local_path):
        # Skip excluded directories
        rel_root = os.path.relpath(root, local_path)
        skip_dir = False
        for pattern in exclude_patterns:
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                if fnmatch.fnmatch(rel_root, dir_pattern) or fnmatch.fnmatch(rel_root, f"*/{dir_pattern}"):
                    skip_dir = True
                    break
                # Also check if any parent dir matches
                for part in rel_root.split(os.sep):
                    if fnmatch.fnmatch(part, dir_pattern):
                        skip_dir = True
                        break

        if skip_dir or rel_root.startswith('.git'):
            dirs[:] = []  # Don't recurse
            continue

        # Filter subdirs: exclude patterns + symlinks (avoid traversal attacks)
        dirs[:] = [d for d in dirs if not any(
            fnmatch.fnmatch(d, p.rstrip('/')) for p in exclude_patterns if p.endswith('/')
        ) and not os.path.islink(os.path.join(root, d))]

        for fname in files:
            # Check file patterns
            skip_file = False
            for pattern in exclude_patterns:
                if not pattern.endswith('/') and fnmatch.fnmatch(fname, pattern):
                    skip_file = True
                    break

            if skip_file:
                continue

            src = Path(root) / fname

            # Skip symlinks (security: avoid following links outside tree)
            if src.is_symlink():
                continue

            rel_path = src.relative_to(local_path)
            dst = target_path / rel_path

            # Skip large files (>1MB)
            try:
                if src.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            file_count += 1

    return file_count


@greptile_group.command("ephemeral")
@click.argument("local_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--name", "-n", default=None, help="Temp repo name (auto-generated if not provided)")
@click.option("--keep", is_flag=True, help="Don't delete repo after (for debugging)")
@click.option("--no-filter", is_flag=True, help="Don't filter out build artifacts/deps (slower)")
@click.option("--include", "-i", multiple=True, help="Only include these paths (e.g., -i src -i lib)")
@click.option("--no-cache", is_flag=True, help="Ignore cache and force fresh indexing")
@click.pass_context
def ephemeral_review(ctx, local_path: Path, name: str | None, keep: bool, no_filter: bool, include: tuple, no_cache: bool):
    """Create ephemeral GitHub repo, index with Greptile, review, then nuke.

    This automates the full flow:
    1. Check cache - reuse existing indexed repo if content unchanged
    2. Create temporary private GitHub repo (if cache miss)
    3. Push your local code (filtered)
    4. Trigger Greptile indexing
    5. Wait for index to complete
    6. Run review
    7. Delete the temporary repo (unless --keep or cached)

    Caching: Computes a hash of your filtered file tree. If the same content
    was indexed recently, reuses that repo instead of re-indexing.

    Examples:
        capseal greptile ephemeral
        capseal greptile ephemeral ~/projects/mycode
        capseal greptile ephemeral --keep     # don't delete after
        capseal greptile ephemeral --no-cache # force fresh indexing

    The repo is automatically deleted after review (unless --keep or cached).
    """
    import subprocess
    import uuid

    local_path = local_path.resolve()

    # Fast repo fingerprint: use git HEAD instead of walking file tree
    # Falls back to path hash if not a git repo
    content_hash = repo_fingerprint(local_path)
    click.echo(f"Repo fingerprint: {content_hash}")

    # Check cache
    cached = None if no_cache else _load_cache(content_hash)
    if cached:
        click.echo(f"\n{'='*60}")
        click.echo(f"CACHE HIT - Reusing indexed repo")
        click.echo(f"{'='*60}")
        click.echo(f"Cached repo: {cached['repo_full']}")
        click.echo(f"Indexed at:  {time.strftime('%Y-%m-%d %H:%M', time.localtime(cached['indexed_at']))}")
        click.echo(f"{'='*60}\n")

        # Skip straight to review using cached repo
        repo_full = cached['repo_full']

        click.echo("[Review] Running review on cached index...")
        try:
            review_result = greptile_review_api(
                repo=f"github:{repo_full}:main",
                context_name="latest",
            )
            if review_result.get("ok"):
                click.echo("\n" + "="*60)
                click.echo("REVIEW RESULTS (cached)")
                click.echo("="*60)
                click.echo(review_result.get("review", "(no review content)"))
            else:
                click.echo(f"Review returned: {review_result}")
        except Exception as e:
            click.echo(f"  ⚠ Review failed: {e}")
            # Cache may be stale - clear it
            _clear_cache(content_hash)

        return

    # No cache hit - proceed with full flow
    # Generate temp repo name
    if not name:
        short_id = uuid.uuid4().hex[:8]
        name = f"greptile-tmp-{short_id}"

    # Determine target repo (offline mock vs online)
    if _offline_mode():
        repo_full = f"mock/{name}"
        repo_created = False
        push_succeeded = False
    else:
        # Get GitHub username
        try:
            user_info = _gh_api("/user")
            username = user_info.get("login")
            if not username:
                raise click.ClickException("Failed to get GitHub username from API")
        except Exception as e:
            raise click.ClickException(f"Failed to get GitHub username: {e}. Set GITHUB_TOKEN env var.")

        repo_full = f"{username}/{name}"
        repo_created = False
        push_succeeded = False

    click.echo(f"\n{'='*60}")
    click.echo(f"EPHEMERAL GREPTILE REVIEW")
    click.echo(f"{'='*60}")
    click.echo(f"Local path: {local_path}")
    click.echo(f"Temp repo:  {repo_full}")
    click.echo(f"Filter:     {'disabled' if no_filter else 'enabled (skipping artifacts/deps)'}")
    if include:
        click.echo(f"Include:    {', '.join(include)}")
    click.echo(f"{'='*60}\n")

    # Create temp staging directory for filtered push
    import tempfile
    import shutil
    staging_dir = None

    def cleanup():
        """Clean up the ephemeral repo and staging dir."""
        if repo_created and not keep:
            click.echo(f"\n[Cleanup] Deleting temporary repo {repo_full}...")
            try:
                _gh_api(f"/repos/{repo_full}", method="DELETE")
                click.echo(f"  ✓ Deleted {repo_full}")
            except Exception as e:
                click.echo(f"  ⚠ Failed to delete: {e}")
        # Clean up staging dir
        if staging_dir and Path(staging_dir).exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        # Remove remote from original repo if added
        subprocess.run(
            ["git", "-C", str(local_path), "remote", "remove", "ephemeral"],
            capture_output=True
        )

    try:
        if _offline_mode():
            # Offline mock: skip GitHub creation/push, just compute filtered file count
            click.echo("[1/5] (offline) Skipping GitHub repo creation")
            click.echo("\n[2/5] (offline) Preparing filtered snapshot...")
            import tempfile
            staging_dir = tempfile.mkdtemp(prefix="greptile_")
            if include:
                include_paths = [local_path / p for p in include]
                file_count = 0
                for inc_path in include_paths:
                    if inc_path.is_dir():
                        fc = _create_sparse_clone(inc_path, Path(staging_dir) / inc_path.name, EPHEMERAL_EXCLUDE_PATTERNS if not no_filter else [])
                        file_count += fc
                    elif inc_path.is_file():
                        dst = Path(staging_dir) / inc_path.name
                        shutil.copy2(inc_path, dst)
                        file_count += 1
            else:
                file_count = _create_sparse_clone(local_path, Path(staging_dir), EPHEMERAL_EXCLUDE_PATTERNS if not no_filter else [])
            click.echo(f"  Filtered to {file_count} source files")
        else:
            # Step 1: Create private repo
            click.echo("[1/5] Creating temporary private repo...")
            try:
                create_result = _gh_api("/user/repos", method="POST", data={
                    "name": name,
                    "private": "true",
                    "auto_init": "false",
                })
                if create_result.get("id"):
                    repo_created = True
                    click.echo(f"  ✓ Created {repo_full}")
                elif "already exists" in str(create_result):
                    repo_created = True  # We'll try to use it
                    click.echo(f"  ✓ Using existing {repo_full}")
                else:
                    raise click.ClickException(f"Failed to create repo: {create_result}")
            except click.ClickException:
                raise
            except Exception as e:
                if "already exists" in str(e):
                    repo_created = True
                else:
                    raise click.ClickException(f"Failed to create repo: {e}")

            # Step 2: Push local code (filtered)
            click.echo("\n[2/5] Preparing and pushing code...")
            token = _get_github_token()
            https_url = f"https://{token}@github.com/{repo_full}.git"

            if no_filter:
                # Push directly from local repo
                push_dir = str(local_path)
                subprocess.run(
                    ["git", "-C", push_dir, "remote", "remove", "ephemeral"],
                    capture_output=True
                )
                subprocess.run(
                    ["git", "-C", push_dir, "remote", "add", "ephemeral", https_url],
                    capture_output=True, check=True
                )
                result = subprocess.run(
                    ["git", "-C", push_dir, "push", "-f", "ephemeral", "HEAD:main"],
                    capture_output=True, text=True
                )
                file_count = "all"
            else:
                # Create filtered staging directory
                import tempfile
                staging_dir = tempfile.mkdtemp(prefix="greptile_")
                click.echo(f"  Filtering files (excluding artifacts/deps)...")

                # Determine what to include
                if include:
                    # Only include specified paths
                    include_paths = [local_path / p for p in include]
                    file_count = 0
                    for inc_path in include_paths:
                        if inc_path.is_dir():
                            fc = _create_sparse_clone(inc_path, Path(staging_dir) / inc_path.name, EPHEMERAL_EXCLUDE_PATTERNS)
                            file_count += fc
                        elif inc_path.is_file():
                            dst = Path(staging_dir) / inc_path.name
                            shutil.copy2(inc_path, dst)
                            file_count += 1
                else:
                    # Filter whole repo
                    file_count = _create_sparse_clone(local_path, Path(staging_dir), EPHEMERAL_EXCLUDE_PATTERNS)

                click.echo(f"  Filtered to {file_count} source files")

                # Initialize git in staging dir
                subprocess.run(["git", "-C", staging_dir, "init", "-q"], check=True)
                subprocess.run(["git", "-C", staging_dir, "add", "-A"], check=True)
                subprocess.run(
                    ["git", "-C", staging_dir, "commit", "-q", "-m", "ephemeral snapshot"],
                    capture_output=True
                )
                subprocess.run(
                    ["git", "-C", staging_dir, "remote", "add", "origin", https_url],
                    check=True
                )

                # Push
                result = subprocess.run(
                    ["git", "-C", staging_dir, "push", "-f", "origin", "HEAD:main"],
                    capture_output=True, text=True
                )
                push_dir = staging_dir

            if result.returncode != 0:
                click.echo(f"  ✗ Push failed: {redact_secrets(result.stderr)}")
                cleanup()
                raise click.ClickException("Push failed - see error above")
            push_succeeded = True
            click.echo(f"  ✓ Pushed {file_count} files to {repo_full}:main")

        # Step 3: Trigger Greptile index
        click.echo("\n[3/5] Triggering Greptile index...")
        try:
            index_result = _api_request(
                "/repositories",
                method="POST",
                data={
                    "remote": "github",
                    "repository": repo_full,
                    "branch": "main",
                    "reload": True,
                },
            )
            click.echo(f"  ✓ Index triggered: {index_result.get('status', 'submitted')}")
        except Exception as e:
            click.echo(f"  ⚠ Index request failed: {e}")

        # Step 4: Wait for indexing with stagnation detection
        click.echo("\n[4/5] Waiting for index to complete...")
        repo_id = f"github:main:{repo_full}".replace("/", "%2F")
        index_status = _poll_index_status(repo_id, max_wait_sec=GREPTILE_MAX_WAIT_SEC, stagnation_threshold=5)

        # Report confidence level for partial results
        if index_status.partial:
            click.echo(f"  ℹ Confidence: {index_status.confidence} ({index_status.files_done}/{index_status.files_total} files indexed)")
            if index_status.stagnant:
                click.echo(f"  ℹ Index stagnated - backend may be overloaded")

        # Save to cache if indexing succeeded (allows reuse)
        if index_status.success and not no_cache:
            _save_cache(content_hash, repo_full, time.time())
            click.echo(f"  ✓ Cached for reuse (hash: {content_hash})")
            # Don't delete repo if caching - it will be reused
            keep = True

        # Step 5: Run review
        click.echo("\n[5/5] Running Greptile review...")
        result = greptile_review_api(repo=repo_full, context_name="latest", focus=None)
        if result.get("ok"):
            click.echo("\n" + "="*60)
            confidence_note = f" [confidence: {index_status.confidence}]" if index_status.partial else ""
            click.echo(f"REVIEW RESULTS{confidence_note}")
            click.echo("="*60 + "\n")
            click.echo(result.get("review", "No review generated"))
        else:
            click.echo(f"  ⚠ Review failed: {result.get('error')}")
            # Clear cache on failure
            _clear_cache(content_hash)

    finally:
        # Cleanup: Delete temp repo only if not caching
        if push_succeeded and not keep:
            cleanup()
        if keep and repo_created:
            click.echo(f"\n[Note] Repo kept for cache reuse: {repo_full}")
            click.echo(f"  To delete: gh repo delete {repo_full} --yes")


@greptile_group.command("sync")
@click.argument("local_path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--repo", "-r", required=True, help="GitHub repo to sync to (owner/repo)")
@click.option("--branch", "-b", default="greptile-sync", help="Branch name for sync")
@click.option("--index", is_flag=True, help="Also trigger Greptile re-index after sync")
def sync_local(local_path: Path, repo: str, branch: str, index: bool):
    """Sync local repo to GitHub shadow repo for Greptile indexing.

    Push your divergent local state to a private GitHub repo so Greptile
    can index your ACTUAL code, not the stale main branch.

    Examples:
        capseal greptile sync -r myuser/shadow-repo
        capseal greptile sync ~/projects/mycode -r myuser/shadow-repo --index
        capseal greptile sync . -r myuser/bef-shadow -b local-dev --index

    Setup (one-time):
        1. Create a private GitHub repo (e.g., 'youruser/code-shadow')
        2. Add it as remote: git remote add shadow git@github.com:youruser/code-shadow.git
        3. Run: capseal greptile sync -r youruser/code-shadow --index
    """
    import subprocess

    local_path = local_path.resolve()
    click.echo(f"Syncing {local_path} → github:{repo}:{branch}")

    # Check if shadow remote exists
    result = subprocess.run(
        ["git", "-C", str(local_path), "remote", "get-url", "shadow"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        # Add shadow remote
        remote_url = f"git@github.com:{repo}.git"
        click.echo(f"Adding shadow remote: {remote_url}")
        subprocess.run(
            ["git", "-C", str(local_path), "remote", "add", "shadow", remote_url],
            check=True
        )

    # Force push current state to shadow branch
    click.echo(f"Force pushing to shadow/{branch}...")
    result = subprocess.run(
        ["git", "-C", str(local_path), "push", "-f", "shadow", f"HEAD:{branch}"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        click.echo(f"Push failed: {redact_secrets(result.stderr)}")
        raise click.ClickException("Failed to push to shadow repo. Check your GitHub auth.")

    click.echo(f"✓ Synced to github:{repo}:{branch}")

    if index:
        click.echo(f"\nTriggering Greptile re-index...")
        # Call the index function
        remote, repository, _ = _parse_repo_string(repo)
        try:
            result = _api_request(
                "/repositories",
                method="POST",
                data={
                    "remote": remote,
                    "repository": repository,
                    "branch": branch,
                    "reload": True,
                },
            )
            click.echo(f"✓ Index triggered: {result.get('status', 'submitted')}")
            click.echo(f"\nNow you can run:")
            click.echo(f"  capseal pr {repo}:{branch}")
        except Exception as e:
            click.echo(f"Index failed: {e}")


@greptile_group.command("index")
@click.argument("repo")
@click.option("--branch", "-b", default="main", help="Branch to index")
@click.option("--reload", is_flag=True, help="Force re-index")
def index_repo(repo: str, branch: str, reload: bool):
    """Index a repository for querying.

    REPO format: owner/repo or github:owner/repo:branch

    Examples:
        capseal greptile index myorg/myrepo
        capseal greptile index myorg/myrepo --branch develop
        capseal greptile index github:myorg/myrepo:main --reload
    """
    remote, repository, br = _parse_repo_string(repo)
    if branch != "main":
        br = branch

    click.echo(f"Indexing {remote}:{repository}:{br}...")

    result = _api_request(
        "/repositories",
        method="POST",
        data={
            "remote": remote,
            "repository": repository,
            "branch": br,
            "reload": reload,
        },
    )

    status = result.get("status", "submitted")
    click.echo(f"Status: {status}")

    if result.get("sha"):
        click.echo(f"SHA: {result['sha']}")
    if result.get("filesProcessed"):
        click.echo(f"Files processed: {result['filesProcessed']}")
    if result.get("numFiles"):
        click.echo(f"Total files: {result['numFiles']}")


@greptile_group.command("status")
@click.argument("repo")
@click.option("--branch", "-b", default="main", help="Branch")
def repo_status(repo: str, branch: str):
    """Check indexing status of a repository.

    Example:
        capseal greptile status myorg/myrepo
    """
    remote, repository, br = _parse_repo_string(repo)
    if branch != "main":
        br = branch

    # Encode the repo identifier
    repo_id = f"{remote}:{br}:{repository}".replace("/", "%2F")

    result = _api_request(f"/repositories/{repo_id}", method="GET")

    click.echo(f"Repository: {repository}")
    click.echo(f"Branch: {br}")
    click.echo(f"Status: {result.get('status', 'unknown')}")

    if result.get("filesProcessed"):
        click.echo(f"Files: {result['filesProcessed']}/{result.get('numFiles', '?')}")
    if result.get("sha"):
        click.echo(f"SHA: {result['sha'][:12]}")


@greptile_group.command("query")
@click.argument("question")
@click.option("--repo", "-r", multiple=True, required=True, help="Repository to query (can specify multiple)")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
def query_repo(question: str, repo: tuple[str, ...], json_output: bool):
    """Ask a question about indexed repositories.

    Examples:
        capseal greptile query "what trace schemas are defined?" -r myorg/myrepo
        capseal greptile query "how does authentication work?" -r org/repo1 -r org/repo2
    """
    repositories = []
    for r in repo:
        remote, repository, branch = _parse_repo_string(r)
        repositories.append({
            "remote": remote,
            "repository": repository,
            "branch": branch,
        })

    result = _api_request(
        "/query",
        method="POST",
        data={
            "messages": [{"role": "user", "content": question}],
            "repositories": repositories,
        },
    )

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        message = result.get("message", "No response")
        click.echo(message)

        sources = result.get("sources", [])
        if sources:
            click.echo("\n--- Sources ---")
            for src in sources[:5]:
                filepath = src.get("filepath", "unknown")
                lines = src.get("lineNumbers", [])
                if lines:
                    click.echo(f"  {filepath}:{lines[0]}-{lines[-1]}")
                else:
                    click.echo(f"  {filepath}")


@greptile_group.command("search")
@click.argument("query")
@click.option("--repo", "-r", multiple=True, required=True, help="Repository to search")
@click.option("--limit", "-n", default=10, help="Max results")
def search_repo(query: str, repo: tuple[str, ...], limit: int):
    """Search for code patterns in indexed repositories.

    Examples:
        capseal greptile search "TraceSchema class definition" -r myorg/myrepo
        capseal greptile search "def verify" -r myorg/myrepo -n 5
    """
    repositories = []
    for r in repo:
        remote, repository, branch = _parse_repo_string(r)
        repositories.append({
            "remote": remote,
            "repository": repository,
            "branch": branch,
        })

    # Use query endpoint with a search-focused prompt
    result = _api_request(
        "/query",
        method="POST",
        data={
            "messages": [{"role": "user", "content": f"Find code matching: {query}. Show file paths and line numbers."}],
            "repositories": repositories,
        },
    )

    message = result.get("message", "No results")
    click.echo(message)

    sources = result.get("sources", [])
    if sources:
        click.echo(f"\n--- Found in {len(sources)} files ---")
        for src in sources[:limit]:
            filepath = src.get("filepath", "unknown")
            repo_name = src.get("repository", "")
            summary = src.get("summary", "")[:80]
            click.echo(f"  {repo_name}/{filepath}")
            if summary:
                click.echo(f"    {summary}")


@greptile_group.command("schema-map")
@click.option("--repo", "-r", required=True, help="Repository to analyze")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file")
def schema_map(repo: str, output: Path | None):
    """Generate a map of trace schemas from the codebase.

    Queries the indexed repo to find all TraceSchema definitions
    and their field mappings.

    Example:
        capseal greptile schema-map -r myorg/bef-main -o schemas.json
    """
    remote, repository, branch = _parse_repo_string(repo)

    click.echo(f"Analyzing {repository} for trace schemas...")

    result = _api_request(
        "/query",
        method="POST",
        data={
            "messages": [{
                "role": "user",
                "content": """Find all TraceSchema definitions in the codebase.
For each schema, list:
1. Schema ID (the string identifier)
2. All fields with their names, types, and descriptions
3. The file path where it's defined

Format as JSON: {"schemas": [{"id": "...", "fields": [...], "file": "..."}]}"""
            }],
            "repositories": [{
                "remote": remote,
                "repository": repository,
                "branch": branch,
            }],
        },
    )

    message = result.get("message", "")

    if output:
        # Try to extract JSON from the response
        output.write_text(message)
        click.echo(f"Saved to: {output}")
    else:
        click.echo(message)


@greptile_group.command("capsule-context")
@click.argument("capsule_path", type=click.Path(exists=True, path_type=Path))
@click.option("--repo", "-r", required=True, help="Repository to query")
def capsule_context(capsule_path: Path, repo: str):
    """Get codebase context for a capsule's trace schema.

    Reads the capsule's trace_schema_id and queries Greptile
    to explain what code produced this trace.

    Example:
        capseal greptile capsule-context ./capsule.json -r myorg/bef-main
    """
    # Load capsule
    try:
        capsule = json.loads(capsule_path.read_text())
    except Exception as e:
        raise click.ClickException(f"Failed to load capsule: {e}")

    schema_id = capsule.get("trace_schema_id")
    if not schema_id:
        raise click.ClickException("Capsule has no trace_schema_id")

    remote, repository, branch = _parse_repo_string(repo)

    click.echo(f"Looking up schema '{schema_id}' in {repository}...")

    result = _api_request(
        "/query",
        method="POST",
        data={
            "messages": [{
                "role": "user",
                "content": f"""Find the trace schema with ID '{schema_id}'.
Explain:
1. What this schema represents
2. What code generates traces using this schema
3. The meaning of each field in the schema
4. Any related schemas or dependencies"""
            }],
            "repositories": [{
                "remote": remote,
                "repository": repository,
                "branch": branch,
            }],
        },
    )

    click.echo(result.get("message", "No context found"))

    sources = result.get("sources", [])
    if sources:
        click.echo("\n--- Relevant files ---")
        for src in sources[:5]:
            click.echo(f"  {src.get('filepath', 'unknown')}")


@greptile_group.command("review")
@click.option("--repo", "-r", required=True, help="Repository to review against")
@click.option("--context", "-c", default="latest", help="Context checkpoint name")
@click.option("--focus", "-f", help="Focus area (e.g., 'security', 'performance', 'style')")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
def review_diff(repo: str, context: str, focus: str | None, json_output: bool):
    """Review a diff context using Greptile's codebase understanding.

    This loads a context checkpoint (created with 'context save') and
    asks Greptile to review the changes against the indexed codebase.

    Examples:
        capseal greptile review -r myorg/myrepo
        capseal greptile review -r myorg/myrepo --focus security
        capseal greptile review -r myorg/myrepo -c merge_task
    """
    # Load context checkpoint
    try:
        from capseal.cli.context import load_context
        ctx = load_context(context)
        if not ctx:
            raise click.ClickException(f"No context checkpoint '{context}'. Run: capseal context save")
    except ImportError:
        raise click.ClickException("Context module not available")

    # Build diff summary for review
    summary = ctx.get("summary", {})
    files = ctx.get("files", [])
    diffs = ctx.get("diffs", [])  # Array of {path, patch, ...} objects

    # Build a concise diff representation
    diff_text = f"## Changes: {summary.get('comparison', 'unknown')}\n\n"
    diff_text += f"Files changed: {len(files)}\n\n"

    # Handle diffs as array of objects
    for diff_obj in diffs[:20]:  # Limit to 20 files
        if isinstance(diff_obj, dict):
            filepath = diff_obj.get("path", "unknown")
            patch = diff_obj.get("patch", "")
            if patch:
                diff_text += f"### {filepath}\n```diff\n{patch[:2000]}\n```\n\n"

    # Add uncommitted changes if present
    working_tree = ctx.get("working_tree") or {}
    if working_tree and working_tree.get("files"):
        diff_text += "\n## Uncommitted Changes\n\n"
        for f in working_tree["files"][:10]:
            diff_text += f"### {f['path']}\n```diff\n{f.get('diff', '')[:1500]}\n```\n\n"

    # Build review prompt
    focus_text = f" Focus especially on {focus} issues." if focus else ""
    review_prompt = f"""Review these code changes.{focus_text}

{diff_text[:15000]}

Provide:
1. **Summary**: What do these changes accomplish?
2. **Concerns**: Any bugs, security issues, or problems?
3. **Suggestions**: How could these changes be improved?
4. **Verdict**: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION

Be specific - reference file names and line patterns."""

    remote, repository, branch = _parse_repo_string(repo)

    click.echo(f"Reviewing {len(files)} changed files against {repository}...")

    result = _api_request(
        "/query",
        method="POST",
        data={
            "messages": [{"role": "user", "content": review_prompt}],
            "repositories": [{
                "remote": remote,
                "repository": repository,
                "branch": branch,
            }],
        },
    )

    if json_output:
        click.echo(json.dumps({
            "review": result.get("message", ""),
            "sources": result.get("sources", []),
            "context_id": ctx.get("checkpoint_id"),
            "files_reviewed": len(files),
        }, indent=2))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("CODE REVIEW")
        click.echo("=" * 60 + "\n")
        click.echo(result.get("message", "No review generated"))

        sources = result.get("sources", [])
        if sources:
            click.echo("\n--- Referenced in codebase ---")
            for src in sources[:5]:
                click.echo(f"  {src.get('filepath', 'unknown')}")


def greptile_review_api(
    repo: str,
    context_name: str = "latest",
    focus: str | None = None,
) -> dict:
    """API function for agents to call Greptile review.

    Returns structured review data that agents can process.
    """
    try:
        from capseal.cli.context import load_context
        ctx = load_context(context_name)
        if not ctx:
            return {"ok": False, "error": f"No context checkpoint: {context_name}"}
    except ImportError:
        return {"ok": False, "error": "Context module not available"}

    # Build diff summary
    summary = ctx.get("summary", {})
    files = ctx.get("files", [])
    diffs = ctx.get("diffs", [])  # Array of {path, patch, ...} objects

    diff_text = f"Changes: {summary.get('comparison', 'unknown')}\n"
    # Handle diffs as array of objects
    for diff_obj in diffs[:15]:
        if isinstance(diff_obj, dict):
            filepath = diff_obj.get("path", "unknown")
            patch = diff_obj.get("patch", "")
            if patch:
                diff_text += f"\n### {filepath}\n{patch[:1500]}\n"

    working_tree = ctx.get("working_tree") or {}
    if working_tree and working_tree.get("files"):
        for f in working_tree["files"][:5]:
            diff_text += f"\n### {f['path']} (uncommitted)\n{f.get('diff', '')[:1000]}\n"

    focus_text = f" Focus on {focus}." if focus else ""
    review_prompt = f"""Review these changes.{focus_text}

{diff_text[:12000]}

Return JSON:
{{"summary": "...", "concerns": ["..."], "suggestions": ["..."], "verdict": "APPROVE|REQUEST_CHANGES|NEEDS_DISCUSSION"}}"""

    try:
        remote, repository, branch = _parse_repo_string(repo)
        result = _api_request(
            "/query",
            method="POST",
            data={
                "messages": [{"role": "user", "content": review_prompt}],
                "repositories": [{
                    "remote": remote,
                    "repository": repository,
                    "branch": branch,
                }],
            },
        )
        return {
            "ok": True,
            "review": result.get("message", ""),
            "sources": result.get("sources", []),
            "files_reviewed": len(files),
            "context_id": ctx.get("checkpoint_id"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
