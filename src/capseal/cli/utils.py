"""Common CLI utilities - path resolution, root detection."""
from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start_path: Path | None = None) -> Path:
    """Find the capseal/BEF repository root.

    Searches upward from start_path (or __file__ if None) for marker directories.
    Falls back to cwd() if no markers found.

    This function should be used instead of hardcoded paths like:
        - Path(__file__).parents[N]
        - "/home/ryan/BEF-main"
        - os.environ.get("CAPSEAL_WORKSPACE_ROOT", "/some/hardcoded/path")

    Markers checked (in order):
        1. pyproject.toml + bef_zk/ (strongest - project root)
        2. bef_zk/ + scripts/ (legacy)
        3. .capseal/ directory (workspace root)
        4. .git/ directory (repo root fallback)

    Args:
        start_path: Where to start searching (default: caller's __file__)

    Returns:
        Path to repo root, or cwd() if no markers found
    """
    if start_path is None:
        start_path = Path(__file__).resolve()
    elif isinstance(start_path, str):
        start_path = Path(start_path).resolve()
    else:
        start_path = start_path.resolve()

    # Walk up directory tree
    for parent in [start_path] + list(start_path.parents):
        # Primary: pyproject.toml + bef_zk exists = definite project root
        if (parent / "pyproject.toml").exists() and (parent / "bef_zk").is_dir():
            return parent

        # Secondary: bef_zk + scripts = legacy marker combo
        if (parent / "bef_zk").is_dir() and (parent / "scripts").is_dir():
            return parent

        # Tertiary: .capseal directory = workspace initialized
        if (parent / ".capseal").is_dir():
            return parent

        # Quaternary: .git = repo root (weakest)
        if (parent / ".git").is_dir():
            return parent

    # Fallback to cwd
    return Path.cwd()


def get_workspace_root() -> Path:
    """Get workspace root, respecting CAPSEAL_WORKSPACE_ROOT if set.

    Order of precedence:
        1. CAPSEAL_WORKSPACE_ROOT environment variable
        2. find_repo_root() detection

    This is the preferred way to get the workspace root in CLI commands.
    """
    env_root = os.environ.get("CAPSEAL_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root)
    return find_repo_root()


def get_capseal_dir() -> Path:
    """Get the .capseal configuration directory.

    Creates it if it doesn't exist.
    """
    root = get_workspace_root()
    capseal_dir = root / ".capseal"
    capseal_dir.mkdir(parents=True, exist_ok=True)
    return capseal_dir


def run_subprocess(
    cmd: list[str],
    timeout: float = 60.0,
    capture_output: bool = True,
    text: bool = True,
    check: bool = False,
    **kwargs,
) -> "subprocess.CompletedProcess[str]":
    """Run subprocess with default timeout and error handling.

    This is a safer wrapper around subprocess.run that:
    - Enforces argv discipline (list only, no shell strings)
    - Applies a default timeout (60s)
    - Captures output by default
    - Returns structured result

    Args:
        cmd: Command as list of strings (NO shell strings)
        timeout: Timeout in seconds (default 60)
        capture_output: Capture stdout/stderr (default True)
        text: Return strings not bytes (default True)
        check: Raise on non-zero exit (default False)
        **kwargs: Additional subprocess.run arguments

    Returns:
        subprocess.CompletedProcess

    Raises:
        TypeError: If cmd is not a list
        subprocess.TimeoutExpired: If timeout exceeded
        subprocess.CalledProcessError: If check=True and non-zero exit
    """
    import subprocess

    if isinstance(cmd, (str, bytes)):
        raise TypeError(
            "cmd must be a list of args, not a shell string. "
            "Pass ['git', 'status'] not 'git status'"
        )

    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=capture_output,
        text=text,
        check=check,
        **kwargs,
    )


__all__ = ["find_repo_root", "get_workspace_root", "get_capseal_dir", "run_subprocess"]
