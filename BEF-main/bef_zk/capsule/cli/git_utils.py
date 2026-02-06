"""Git utilities for fast file operations - replaces rglob/os.walk with git ls-files."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterator


def repo_fingerprint(repo_path: str | Path) -> str:
    """Get fast repo fingerprint from git HEAD.

    Returns SHA of HEAD commit - changes only when repo changes.
    Much faster than computing tree hash via file walks.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: hash the path itself
    return hashlib.sha256(str(repo_path).encode()).hexdigest()[:16]


def tracked_files(repo_path: str | Path, patterns: list[str] | None = None) -> list[str]:
    """Get list of git-tracked files (fast, no filesystem walking).

    Args:
        repo_path: Path to git repository
        patterns: Optional glob patterns to filter (e.g., ["*.py", "src/**/*.ts"])

    Returns:
        List of relative file paths
    """
    cmd = ["git", "ls-files"]
    if patterns:
        for p in patterns:
            cmd.extend(["--", p])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return []


def tracked_files_with_status(repo_path: str | Path) -> list[dict]:
    """Get tracked files with modification status.

    Returns list of dicts: {path: str, status: str}
    Status: M=modified, A=added, D=deleted, ?=untracked
    """
    files = []

    # Get all tracked files
    for f in tracked_files(repo_path):
        files.append({"path": f, "status": ""})

    # Get modified files
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            status_map = {}
            for line in result.stdout.strip().split("\n"):
                if len(line) >= 3:
                    status = line[:2].strip()
                    path = line[3:]
                    status_map[path] = status

            for f in files:
                if f["path"] in status_map:
                    f["status"] = status_map[f["path"]]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return files


def diff_files(
    repo_path: str | Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
) -> list[str]:
    """Get list of files changed between two refs.

    Args:
        repo_path: Path to git repository
        base_ref: Base reference (e.g., "main", "HEAD~5")
        head_ref: Head reference (e.g., "HEAD", "feature-branch")

    Returns:
        List of changed file paths
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}..{head_ref}"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return []


def safe_glob(
    repo_path: str | Path,
    pattern: str = "*",
    max_results: int = 5000,
    tracked_only: bool = True,
) -> Iterator[Path]:
    """Safe glob that uses git ls-files instead of filesystem walking.

    Args:
        repo_path: Path to git repository
        pattern: Glob pattern (e.g., "*.py", "**/*.ts")
        max_results: Maximum number of results to return
        tracked_only: If True, only return git-tracked files

    Yields:
        Path objects for matching files
    """
    import fnmatch

    repo_path = Path(repo_path)

    if tracked_only:
        files = tracked_files(repo_path)
    else:
        # Fallback to limited filesystem walk
        files = []
        try:
            result = subprocess.run(
                ["find", ".", "-type", "f", "-maxdepth", "10"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                files = [f.lstrip("./") for f in result.stdout.strip().split("\n") if f]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    count = 0
    for f in files:
        if count >= max_results:
            break
        if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern):
            yield repo_path / f
            count += 1


def extract_symbols_from_diff(diff_text: str) -> list[str]:
    """Extract potential symbol names from a diff.

    Looks for:
    - Function definitions (def foo, function foo)
    - Class definitions (class Foo)
    - Imported modules
    - Exported names

    Returns:
        List of symbol names found
    """
    import re

    symbols = set()

    # Python function/class definitions
    for match in re.finditer(r'^[+-]\s*(?:def|class)\s+(\w+)', diff_text, re.MULTILINE):
        symbols.add(match.group(1))

    # JavaScript/TypeScript function definitions
    for match in re.finditer(r'^[+-]\s*(?:function|const|let|var|export)\s+(\w+)', diff_text, re.MULTILINE):
        symbols.add(match.group(1))

    # Import statements
    for match in re.finditer(r'^[+-]\s*(?:from|import)\s+[\w.]+\s+import\s+(\w+)', diff_text, re.MULTILINE):
        symbols.add(match.group(1))

    for match in re.finditer(r'^[+-]\s*import\s+\{([^}]+)\}', diff_text, re.MULTILINE):
        for name in match.group(1).split(","):
            name = name.strip().split(" as ")[0].strip()
            if name:
                symbols.add(name)

    return list(symbols)


def find_references_local(
    repo_path: str | Path,
    symbols: list[str],
    max_files: int = 25,
) -> dict[str, list[dict]]:
    """Find references to symbols using local ripgrep.

    Args:
        repo_path: Path to git repository
        symbols: List of symbol names to search for
        max_files: Maximum number of files to return

    Returns:
        Dict mapping symbol -> list of {file, line, context}
    """
    import shutil

    repo_path = Path(repo_path)
    results = {}

    rg_path = shutil.which("rg")
    if not rg_path:
        return results

    for symbol in symbols[:20]:  # Limit symbols to search
        try:
            result = subprocess.run(
                [
                    "rg", "-n", "--json",
                    "-g", "!*.min.js",
                    "-g", "!*.map",
                    "-g", "!node_modules/*",
                    "-g", "!.git/*",
                    "-g", "!dist/*",
                    "-g", "!build/*",
                    f"\\b{symbol}\\b",
                ],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import json
                matches = []
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("type") == "match":
                            match_data = data.get("data", {})
                            matches.append({
                                "file": match_data.get("path", {}).get("text", ""),
                                "line": match_data.get("line_number", 0),
                                "context": match_data.get("lines", {}).get("text", "")[:200],
                            })
                            if len(matches) >= max_files:
                                break
                    except json.JSONDecodeError:
                        continue

                if matches:
                    results[symbol] = matches

        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return results
