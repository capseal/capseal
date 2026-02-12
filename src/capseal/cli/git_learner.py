"""Learn risk model from git history — zero cost, instant.

Walks the project's commit history, extracts diffs, and classifies
each commit based on whether it introduced or fixed semgrep findings.
No LLM calls needed.

Usage:
    capseal learn . --from-git
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path


def learn_from_git(
    target: str,
    max_commits: int = 50,
    quiet: bool = False,
    max_duration_seconds: float | None = None,
    semgrep_timeout_seconds: int = 30,
) -> dict[str, dict]:
    """Walk git history and classify commits as pass/fail.

    Returns: {profile_name: {passes: int, fails: int, grid_idx: int}}
    """
    target_path = Path(target).resolve()

    # Check if this is a git repo
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=target_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if not quiet:
            import click
            click.echo("  Not a git repository. --from-git requires git history.")
        return {}

    # Get recent commit hashes
    result = subprocess.run(
        ["git", "log", "--oneline", f"-{max_commits}", "--format=%H"],
        cwd=target_path,
        capture_output=True,
        text=True,
    )
    commits = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]

    if len(commits) < 2:
        if not quiet:
            import click
            click.echo("  Not enough git history (need at least 2 commits).")
        return {}

    if not quiet:
        import click
        click.echo(f"  Analyzing {len(commits) - 1} commit pairs...")

    results: dict[str, dict] = {}

    start = time.monotonic()
    for i in range(len(commits) - 1):
        if max_duration_seconds is not None and (time.monotonic() - start) > max_duration_seconds:
            if not quiet:
                import click
                click.echo(f"  Time budget reached ({max_duration_seconds:.0f}s), stopping early.")
            break

        new_commit = commits[i]
        old_commit = commits[i + 1]

        # Get changed files
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", old_commit, new_commit],
            cwd=target_path,
            capture_output=True,
            text=True,
        )
        changed_files = [f for f in diff_result.stdout.strip().split("\n") if f.strip()]

        # Filter to code files only
        code_exts = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".cpp", ".h"}
        code_files = [f for f in changed_files if Path(f).suffix in code_exts]
        if not code_files:
            continue

        # Get diff stats for profile classification
        stat_result = subprocess.run(
            ["git", "diff", "--shortstat", old_commit, new_commit],
            cwd=target_path,
            capture_output=True,
            text=True,
        )
        lines_changed = _parse_lines_changed(stat_result.stdout)
        n_files = len(code_files)

        # Classify commit by comparing semgrep findings before/after
        old_findings = _count_findings_at_commit(
            target_path, old_commit, code_files, semgrep_timeout_seconds=semgrep_timeout_seconds
        )
        new_findings = _count_findings_at_commit(
            target_path, new_commit, code_files, semgrep_timeout_seconds=semgrep_timeout_seconds
        )

        if old_findings is None or new_findings is None:
            continue  # Skip if we can't analyze

        # Build profile
        profile = _classify_profile(lines_changed, n_files)
        grid_idx = _profile_to_grid_idx(lines_changed, n_files)

        if profile not in results:
            results[profile] = {"passes": 0, "fails": 0, "grid_idx": grid_idx}

        if new_findings > old_findings:
            results[profile]["fails"] += 1  # Introduced issues
        elif new_findings <= old_findings:
            results[profile]["passes"] += 1  # Fixed or maintained

        if not quiet and (i + 1) % 10 == 0:
            import click
            click.echo(f"  ... {i + 1}/{len(commits) - 1} commits analyzed")

    return results


def _count_findings_at_commit(
    target: Path,
    commit: str,
    files: list[str],
    semgrep_timeout_seconds: int = 30,
) -> int | None:
    """Count semgrep findings at a commit without checking out."""
    with tempfile.TemporaryDirectory(prefix="capseal_git_") as tmpdir:
        tmp_path = Path(tmpdir)

        for f in files:
            result = subprocess.run(
                ["git", "show", f"{commit}:{f}"],
                cwd=target,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue  # File didn't exist at this commit

            tmp_file = tmp_path / f
            tmp_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file.write_text(result.stdout)

        # Check if we have any files to scan
        if not any(tmp_path.rglob("*")):
            return 0

        # Run semgrep
        try:
            sem_result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", "--quiet", "."],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=semgrep_timeout_seconds,
            )
            data = json.loads(sem_result.stdout)
            return len(data.get("results", []))
        except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError):
            return None


def _parse_lines_changed(stat_output: str) -> int:
    """Parse 'N insertions, M deletions' from git diff --shortstat."""
    import re
    total = 0
    for match in re.findall(r"(\d+) insertion", stat_output):
        total += int(match)
    for match in re.findall(r"(\d+) deletion", stat_output):
        total += int(match)
    return total


def _classify_profile(lines_changed: int, n_files: int) -> str:
    """Generate a human-readable profile name."""
    if lines_changed < 10:
        size = "small"
    elif lines_changed < 50:
        size = "medium"
    else:
        size = "large"

    if n_files == 1:
        scope = "single-file"
    elif n_files <= 3:
        scope = "focused"
    else:
        scope = "broad"

    return f"{size} + {scope}"


def _profile_to_grid_idx(lines_changed: int, n_files: int) -> int:
    """Map profile features to a grid index compatible with Beta posteriors."""
    # Map to the same feature space as the semgrep-based learner
    # Feature: complexity (0-3), files (0-3), severity (0-3), size (0-3), coverage (0-3)
    # We only have lines_changed and n_files, so we estimate

    # Size/complexity from lines changed
    if lines_changed < 10:
        complexity = 0
        size = 0
    elif lines_changed < 30:
        complexity = 1
        size = 1
    elif lines_changed < 100:
        complexity = 2
        size = 2
    else:
        complexity = 3
        size = 3

    # Files
    if n_files <= 1:
        files = 0
    elif n_files <= 3:
        files = 1
    elif n_files <= 8:
        files = 2
    else:
        files = 3

    severity = 1  # Default to "warning" — we don't know actual severity
    coverage = 0  # Default to "untested" — we don't know coverage

    # Convert to grid index (same formula as features_to_grid_idx)
    try:
        from capseal.shared.features import features_to_grid_idx, discretize_features
        levels = [complexity, files, severity, size, coverage]
        return features_to_grid_idx(levels)
    except ImportError:
        # Fallback: simple hash
        return (complexity * 256 + files * 64 + severity * 16 + size * 4 + coverage) % 1024


__all__ = ["learn_from_git"]
