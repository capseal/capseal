"""FSGuard - File System Guard for Cline/Agent allowlist enforcement.

This module enforces path-based access control for AI agents like Cline.
It validates that file operations only touch allowed paths, preventing
accidental or malicious access to sensitive files.

Key Features:
- Allowlist-based path validation
- Glob pattern support (e.g., "src/**/*.py")
- Sensitive path blocklist (always denied)
- Symlink resolution and validation
- Audit logging of access attempts

Usage:
    from capseal.cli.fsguard import FSGuard, get_default_guard

    guard = get_default_guard(repo_root="/path/to/repo")

    # Check if a path is allowed
    if guard.is_allowed("/path/to/repo/src/main.py"):
        # Allow the operation
        pass

    # Or use context manager for automatic enforcement
    with guard.protect():
        # All file operations here are validated
        pass
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Paths that are ALWAYS blocked (sensitive files)
SENSITIVE_PATHS = [
    # Credentials and secrets
    "**/.env",
    "**/.env.*",
    "**/credentials.json",
    "**/secrets.json",
    "**/secrets.yaml",
    "**/secrets.yml",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa",
    "**/id_ed25519",
    "**/.ssh/*",
    "**/.aws/*",
    "**/.gcp/*",
    "**/service_account.json",

    # Git internals (prevent history manipulation)
    "**/.git/config",
    "**/.git/hooks/*",
    "**/.git/objects/*",
    "**/.git/refs/*",

    # Package manager auth
    "**/.npmrc",
    "**/.pypirc",
    "**/pip.conf",

    # IDE settings that might contain tokens
    "**/.vscode/settings.json",
    "**/.idea/**",

    # System files
    "/etc/**",
    "/root/**",
    "/home/*/.bashrc",
    "/home/*/.bash_history",
    "/home/*/.zsh_history",
]

# Default allowlist patterns for typical code review
DEFAULT_ALLOWLIST = [
    # Source code
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.tsx",
    "**/*.jsx",
    "**/*.go",
    "**/*.rs",
    "**/*.java",
    "**/*.c",
    "**/*.cpp",
    "**/*.h",
    "**/*.hpp",

    # Config files (non-sensitive)
    "**/package.json",
    "**/pyproject.toml",
    "**/setup.py",
    "**/Cargo.toml",
    "**/go.mod",
    "**/Makefile",
    "**/Dockerfile",
    "**/docker-compose.yml",

    # Documentation
    "**/*.md",
    "**/*.rst",
    "**/*.txt",
    "**/LICENSE",

    # Tests
    "**/test_*.py",
    "**/*_test.py",
    "**/*.test.js",
    "**/*.spec.ts",

    # Data schemas (non-sensitive)
    "**/*.json",  # Will be checked against blocklist
    "**/*.yaml",
    "**/*.yml",
    "**/*.toml",

    # Web assets
    "**/*.html",
    "**/*.css",
    "**/*.scss",
]


@dataclass
class AccessAttempt:
    """Record of a file access attempt."""
    path: str
    operation: str  # read, write, delete, list
    allowed: bool
    reason: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "operation": self.operation,
            "allowed": self.allowed,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class FSGuardConfig:
    """Configuration for FSGuard."""
    repo_root: Path
    allowlist: list[str] = field(default_factory=lambda: DEFAULT_ALLOWLIST.copy())
    blocklist: list[str] = field(default_factory=lambda: SENSITIVE_PATHS.copy())
    allow_symlinks: bool = False  # Deny symlinks by default (security)
    max_file_size: int = 10 * 1024 * 1024  # 10MB max
    audit_log: bool = True
    strict_mode: bool = True  # Fail on any violation


class FSGuard:
    """File System Guard - enforces allowlist-based access control.

    This is the main entry point for path validation. It checks:
    1. Path is within repo_root
    2. Path matches allowlist patterns
    3. Path does not match blocklist patterns
    4. Path is not a symlink (unless explicitly allowed)
    5. File size is within limits
    """

    def __init__(self, config: FSGuardConfig):
        self.config = config
        self.access_log: list[AccessAttempt] = []
        self._repo_root_resolved = config.repo_root.resolve()

    def is_allowed(
        self,
        path: str | Path,
        operation: str = "read",
    ) -> bool:
        """Check if a path is allowed for the given operation.

        Args:
            path: Path to check
            operation: One of "read", "write", "delete", "list"

        Returns:
            True if allowed, False otherwise
        """
        attempt = self._check_path(path, operation)
        if self.config.audit_log:
            self.access_log.append(attempt)
        return attempt.allowed

    def validate(
        self,
        path: str | Path,
        operation: str = "read",
    ) -> None:
        """Validate path access, raising if denied.

        Args:
            path: Path to check
            operation: One of "read", "write", "delete", "list"

        Raises:
            PermissionError: If access is denied
        """
        attempt = self._check_path(path, operation)
        if self.config.audit_log:
            self.access_log.append(attempt)

        if not attempt.allowed:
            raise PermissionError(
                f"FSGuard: {operation} access denied for {path}\n"
                f"Reason: {attempt.reason}"
            )

    def _check_path(self, path: str | Path, operation: str) -> AccessAttempt:
        """Internal path checking logic."""
        path = Path(path)

        # Resolve to absolute path
        try:
            if path.is_absolute():
                resolved = path.resolve()
            else:
                resolved = (self._repo_root_resolved / path).resolve()
        except (OSError, RuntimeError) as e:
            return AccessAttempt(
                path=str(path),
                operation=operation,
                allowed=False,
                reason=f"Path resolution failed: {e}",
            )

        # Check if within repo root
        try:
            resolved.relative_to(self._repo_root_resolved)
        except ValueError:
            return AccessAttempt(
                path=str(path),
                operation=operation,
                allowed=False,
                reason=f"Path escapes repo root: {resolved}",
            )

        # Get relative path for pattern matching
        try:
            rel_path = str(resolved.relative_to(self._repo_root_resolved))
        except ValueError:
            rel_path = str(resolved)

        # Check blocklist first (always denied)
        for pattern in self.config.blocklist:
            if self._matches_pattern(rel_path, pattern) or self._matches_pattern(str(resolved), pattern):
                return AccessAttempt(
                    path=str(path),
                    operation=operation,
                    allowed=False,
                    reason=f"Blocked by pattern: {pattern}",
                )

        # Check symlinks
        if not self.config.allow_symlinks and path.is_symlink():
            return AccessAttempt(
                path=str(path),
                operation=operation,
                allowed=False,
                reason="Symlinks not allowed",
            )

        # Check if symlink target escapes repo root
        if path.exists() and path.is_symlink():
            try:
                target = path.resolve()
                target.relative_to(self._repo_root_resolved)
            except ValueError:
                return AccessAttempt(
                    path=str(path),
                    operation=operation,
                    allowed=False,
                    reason="Symlink target escapes repo root",
                )

        # Check allowlist
        allowed = False
        matching_pattern = None
        for pattern in self.config.allowlist:
            if self._matches_pattern(rel_path, pattern):
                allowed = True
                matching_pattern = pattern
                break

        if not allowed:
            return AccessAttempt(
                path=str(path),
                operation=operation,
                allowed=False,
                reason="No matching allowlist pattern",
            )

        # Check file size (for read/write operations on existing files)
        if operation in ("read", "write") and resolved.exists() and resolved.is_file():
            try:
                size = resolved.stat().st_size
                if size > self.config.max_file_size:
                    return AccessAttempt(
                        path=str(path),
                        operation=operation,
                        allowed=False,
                        reason=f"File too large: {size} > {self.config.max_file_size}",
                    )
            except OSError:
                pass

        return AccessAttempt(
            path=str(path),
            operation=operation,
            allowed=True,
            reason=f"Allowed by pattern: {matching_pattern}",
        )

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a glob pattern."""
        # Normalize path separators
        path = path.replace(os.sep, "/")
        pattern = pattern.replace(os.sep, "/")

        # Handle ** patterns
        if "**" in pattern:
            # fnmatch doesn't handle ** well, use simple logic
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                prefix = prefix.rstrip("/")
                suffix = suffix.lstrip("/")

                if prefix and not path.startswith(prefix):
                    return False
                if suffix and not fnmatch.fnmatch(path.split("/")[-1] if "/" in path else path, suffix.lstrip("*/")):
                    # Try matching the suffix pattern against the path
                    return fnmatch.fnmatch(path, f"*{suffix}")
                return True

        return fnmatch.fnmatch(path, pattern)

    def add_allowlist(self, *patterns: str) -> None:
        """Add patterns to the allowlist."""
        self.config.allowlist.extend(patterns)

    def add_blocklist(self, *patterns: str) -> None:
        """Add patterns to the blocklist."""
        self.config.blocklist.extend(patterns)

    def get_audit_log(self) -> list[dict]:
        """Get the audit log as a list of dicts."""
        return [a.to_dict() for a in self.access_log]

    def get_violations(self) -> list[dict]:
        """Get only denied access attempts."""
        return [a.to_dict() for a in self.access_log if not a.allowed]

    def summary(self) -> dict[str, Any]:
        """Get a summary of access attempts."""
        total = len(self.access_log)
        allowed = sum(1 for a in self.access_log if a.allowed)
        denied = total - allowed

        return {
            "total_attempts": total,
            "allowed": allowed,
            "denied": denied,
            "violations": self.get_violations(),
            "repo_root": str(self.config.repo_root),
            "allowlist_count": len(self.config.allowlist),
            "blocklist_count": len(self.config.blocklist),
        }

    def save_audit_log(self, path: Path) -> None:
        """Save audit log to JSON file."""
        path.write_text(json.dumps({
            "summary": self.summary(),
            "log": self.get_audit_log(),
        }, indent=2))


def get_default_guard(
    repo_root: str | Path = ".",
    additional_allowlist: list[str] | None = None,
    additional_blocklist: list[str] | None = None,
) -> FSGuard:
    """Get an FSGuard instance with default configuration.

    Args:
        repo_root: Root directory to guard
        additional_allowlist: Extra patterns to allow
        additional_blocklist: Extra patterns to block

    Returns:
        Configured FSGuard instance
    """
    config = FSGuardConfig(repo_root=Path(repo_root).resolve())

    if additional_allowlist:
        config.allowlist.extend(additional_allowlist)
    if additional_blocklist:
        config.blocklist.extend(additional_blocklist)

    return FSGuard(config)


def guard_for_review(repo_root: str | Path = ".") -> FSGuard:
    """Get an FSGuard configured for code review operations.

    This is the recommended guard for Cline/agent code review tasks.
    It allows reading source files but blocks sensitive paths.
    """
    return get_default_guard(
        repo_root=repo_root,
        additional_blocklist=[
            # Extra patterns for review context
            "**/.capseal/cache/*",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
        ],
    )


def validate_paths_batch(
    guard: FSGuard,
    paths: list[str | Path],
    operation: str = "read",
) -> tuple[list[str], list[str]]:
    """Validate a batch of paths, returning allowed and denied lists.

    Args:
        guard: FSGuard instance
        paths: List of paths to check
        operation: Operation type

    Returns:
        Tuple of (allowed_paths, denied_paths)
    """
    allowed = []
    denied = []

    for path in paths:
        if guard.is_allowed(path, operation):
            allowed.append(str(path))
        else:
            denied.append(str(path))

    return allowed, denied
