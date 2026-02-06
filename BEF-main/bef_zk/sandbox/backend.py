"""Runtime backend interface - pluggable sandbox execution.

This module defines the RuntimeBackend protocol that all sandbox backends
must implement. This abstraction allows capseal to run on different platforms
with different isolation technologies while maintaining a consistent interface.

Supported Backends:
    - BubblewrapBackend: Linux, namespace-based (fast, lightweight)
    - FirejailBackend: Linux, seccomp + namespaces
    - NsjailBackend: Linux, Google's sandbox (most secure)
    - SandboxExecBackend: macOS, seatbelt profiles
    - DockerBackend: Cross-platform, container-based
    - NoneBackend: No isolation (development only)

Usage:
    from bef_zk.sandbox.backend import get_backend, RuntimeResult

    backend = get_backend()  # Auto-detect best available
    result = backend.run(
        command=["python", "script.py"],
        datasets=[Path("./data")],
        output_dir=Path("./output"),
    )
    print(f"Exit code: {result.returncode}")
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .detect import SandboxBackend, detect_sandbox_backend, get_sandbox_info

# Import redactor for sanitizing subprocess output
try:
    from bef_zk.capsule.cli.redact import redact_secrets
except ImportError:
    # Fallback if redact module not available (e.g., minimal install)
    def redact_secrets(text: str) -> str:
        return text


@dataclass
class RuntimeConfig:
    """Configuration for sandboxed execution."""

    # Filesystem access
    datasets: list[Path] = field(default_factory=list)  # Read-only dataset paths
    output_dir: Path | None = None                       # Write access for output
    workdir: Path | None = None                          # Working directory inside sandbox

    # Resource limits
    memory_mb: int = 4096          # Memory limit in MB
    cpu_time_sec: int = 300        # CPU time limit in seconds
    wall_time_sec: int = 600       # Wall clock time limit
    max_pids: int = 100            # Max processes
    max_files: int = 1024          # Max open files

    # Network
    network: bool = False          # Allow network access (default: no)

    # Environment
    env: dict[str, str] = field(default_factory=dict)  # Environment variables
    inherit_env: bool = False      # Inherit parent environment


@dataclass
class RuntimeResult:
    """Result of sandboxed execution."""
    returncode: int
    stdout: str
    stderr: str
    backend: str                   # Which backend was used
    duration_ms: float = 0         # Execution time in milliseconds
    resource_usage: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)  # Generated artifacts


@runtime_checkable
class RuntimeBackend(Protocol):
    """Protocol for sandbox runtime backends.

    All backends must implement these methods to be usable with capseal.
    """

    @property
    def name(self) -> str:
        """Return the backend identifier (e.g., 'bwrap', 'docker')."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    def get_info(self) -> dict[str, Any]:
        """Get backend version and configuration info."""
        ...

    def run(
        self,
        command: list[str],
        config: RuntimeConfig | None = None,
    ) -> RuntimeResult:
        """Execute a command in the sandbox."""
        ...


class BaseBackend(ABC):
    """Base class for runtime backends with common functionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend identifier."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Get backend info."""
        pass

    @abstractmethod
    def _build_command(
        self,
        command: list[str],
        config: RuntimeConfig,
    ) -> list[str]:
        """Build the full command with sandbox wrapper."""
        pass

    def run(
        self,
        command: list[str],
        config: RuntimeConfig | None = None,
    ) -> RuntimeResult:
        """Execute a command in the sandbox.

        Security model:
        - command MUST be a list[str], never a shell string (argv discipline)
        - full_cmd is the sandbox wrapper + user command as args
        - env is scrubbed baseline + explicit allowlist overlay
        - Exceptions are classified for receipt observability
        """
        import time

        # 1) Enforce argv discipline - reject shell strings
        if isinstance(command, (str, bytes)):
            raise TypeError(
                "command must be a list of args, not a shell string. "
                "Pass ['python', 'script.py', '--arg', value] not 'python script.py'"
            )

        config = config or RuntimeConfig()
        start = time.time()

        # Build sandboxed command (wrapper + user args)
        full_cmd = self._build_command(list(command), config)

        # 2) Explicit env policy: scrubbed baseline + allowlist overlay
        # Don't pass empty dict (breaks PATH) or None blindly
        base_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm"),
        }
        if config.inherit_env:
            # Inherit full env (less secure, but explicit choice)
            base_env = dict(os.environ)
        # Overlay user-specified env
        base_env.update(config.env)

        # 3) Execute with proper exception taxonomy for receipt observability
        try:
            result = subprocess.run(
                list(full_cmd),  # Ensure list
                capture_output=True,
                text=True,
                timeout=config.wall_time_sec,
                env=base_env,
                shell=False,  # Explicit guardrail - never shell=True
                check=False,  # We handle return codes ourselves
            )
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr

        except subprocess.TimeoutExpired as e:
            # Timeout - capture partial output if available
            returncode = 124  # Standard timeout exit code
            stdout = e.stdout or "" if hasattr(e, 'stdout') else ""
            stderr = (e.stderr or "") if hasattr(e, 'stderr') else ""
            stderr = f"{stderr}\nCommand timed out after {config.wall_time_sec}s".strip()

        except OSError as e:
            # ENOENT (command not found), EACCES (permission denied), etc.
            # These are classifiable system errors
            returncode = 127 if e.errno == 2 else 126  # 127=not found, 126=not executable
            stdout = ""
            stderr = f"OS error spawning process: [{e.errno}] {e.strerror}: {e.filename or full_cmd[0]}"

        except subprocess.SubprocessError as e:
            # Other subprocess module errors (rare)
            returncode = 1
            stdout = ""
            stderr = f"Subprocess error: {type(e).__name__}: {e}"

        duration = (time.time() - start) * 1000

        # Find generated artifacts
        artifacts = []
        if config.output_dir and config.output_dir.exists():
            artifacts = list(config.output_dir.rglob("*"))

        # Redact secrets from output before returning (P0 security control)
        # Tokens in subprocess output can leak via terminal scrollback, CI logs
        return RuntimeResult(
            returncode=returncode,
            stdout=redact_secrets(stdout),
            stderr=redact_secrets(stderr),
            backend=self.name,
            duration_ms=round(duration, 2),
            artifacts=artifacts,
            resource_usage={
                "env_keys": list(base_env.keys()),  # Record what env was passed
                "argv_len": len(full_cmd),
            },
        )


class BubblewrapBackend(BaseBackend):
    """Bubblewrap (bwrap) backend - fast Linux namespace isolation."""

    @property
    def name(self) -> str:
        return "bwrap"

    def is_available(self) -> bool:
        from .detect import _check_bwrap
        return _check_bwrap()

    def get_info(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["bwrap", "--version"],
                capture_output=True,
                text=True,
            )
            version = result.stdout.strip()
        except Exception:
            version = "unknown"

        return {
            "backend": self.name,
            "version": version,
            "features": ["user_namespaces", "mount_namespaces", "network_isolation"],
        }

    def _build_command(
        self,
        command: list[str],
        config: RuntimeConfig,
    ) -> list[str]:
        cmd = ["bwrap"]

        # Basic isolation
        cmd.extend(["--unshare-all"])

        # Mount root filesystem read-only
        cmd.extend(["--ro-bind", "/", "/"])

        # Mount /tmp
        cmd.extend(["--tmpfs", "/tmp"])

        # Mount datasets read-only
        for dataset in config.datasets:
            if dataset.exists():
                cmd.extend(["--ro-bind", str(dataset), str(dataset)])

        # Mount output directory read-write
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--bind", str(config.output_dir), str(config.output_dir)])

        # Set working directory
        if config.workdir:
            cmd.extend(["--chdir", str(config.workdir)])

        # Network isolation (default: no network)
        if not config.network:
            cmd.extend(["--unshare-net"])

        # Die with parent
        cmd.extend(["--die-with-parent"])

        # Add the actual command
        cmd.extend(["--"])
        cmd.extend(command)

        return cmd


class FirejailBackend(BaseBackend):
    """Firejail backend - seccomp + namespace isolation."""

    @property
    def name(self) -> str:
        return "firejail"

    def is_available(self) -> bool:
        from .detect import _check_firejail
        return _check_firejail()

    def get_info(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["firejail", "--version"],
                capture_output=True,
                text=True,
            )
            version = result.stdout.split("\n")[0].strip()
        except Exception:
            version = "unknown"

        return {
            "backend": self.name,
            "version": version,
            "features": ["seccomp", "namespaces", "caps"],
        }

    def _build_command(
        self,
        command: list[str],
        config: RuntimeConfig,
    ) -> list[str]:
        cmd = ["firejail"]

        # Quiet mode
        cmd.append("--quiet")

        # Private mode (isolated home)
        cmd.append("--private")

        # Network isolation
        if not config.network:
            cmd.append("--net=none")

        # Memory limit
        if config.memory_mb:
            cmd.extend(["--rlimit-as", str(config.memory_mb * 1024 * 1024)])

        # CPU time limit
        if config.cpu_time_sec:
            cmd.extend(["--rlimit-cpu", str(config.cpu_time_sec)])

        # Whitelist datasets
        for dataset in config.datasets:
            if dataset.exists():
                cmd.extend(["--whitelist", str(dataset)])

        # Whitelist output directory
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--whitelist", str(config.output_dir)])
            cmd.extend(["--read-write", str(config.output_dir)])

        # Add the actual command
        cmd.extend(["--"])
        cmd.extend(command)

        return cmd


class NoneBackend(BaseBackend):
    """No-sandbox backend - runs commands directly (development only)."""

    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True  # Always available

    def get_info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "version": "n/a",
            "features": [],
            "warning": "No isolation - use only for development",
        }

    def _build_command(
        self,
        command: list[str],
        config: RuntimeConfig,
    ) -> list[str]:
        # No sandbox wrapper - just return the command
        return command


class DockerBackend(BaseBackend):
    """Docker backend - container-based isolation (cross-platform)."""

    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image

    @property
    def name(self) -> str:
        return "docker"

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
            )
            version = result.stdout.strip()
        except Exception:
            version = "unknown"

        return {
            "backend": self.name,
            "version": version,
            "image": self.image,
            "features": ["container_isolation", "cross_platform"],
        }

    def _build_command(
        self,
        command: list[str],
        config: RuntimeConfig,
    ) -> list[str]:
        cmd = ["docker", "run", "--rm"]

        # Resource limits
        if config.memory_mb:
            cmd.extend(["-m", f"{config.memory_mb}m"])

        # Network
        if not config.network:
            cmd.extend(["--network", "none"])

        # Mount datasets read-only
        for dataset in config.datasets:
            if dataset.exists():
                cmd.extend(["-v", f"{dataset}:{dataset}:ro"])

        # Mount output directory read-write
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["-v", f"{config.output_dir}:{config.output_dir}"])

        # Working directory
        if config.workdir:
            cmd.extend(["-w", str(config.workdir)])

        # Environment variables
        for key, value in config.env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image and command
        cmd.append(self.image)
        cmd.extend(command)

        return cmd


# Backend registry
_BACKENDS: dict[str, type[BaseBackend]] = {
    "bwrap": BubblewrapBackend,
    "firejail": FirejailBackend,
    "docker": DockerBackend,
    "none": NoneBackend,
}


def get_backend(name: str | None = None) -> BaseBackend:
    """Get a runtime backend by name, or auto-detect the best available.

    Args:
        name: Backend name ('bwrap', 'firejail', 'docker', 'none').
              If None, auto-detect the best available.

    Returns:
        A RuntimeBackend instance.

    Raises:
        ValueError: If the requested backend is not available.
    """
    if name is None:
        # Auto-detect
        detected = detect_sandbox_backend()
        name = detected.value

    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")

    backend_class = _BACKENDS[name]
    backend = backend_class()

    if not backend.is_available():
        raise ValueError(f"Backend '{name}' is not available on this system")

    return backend


def list_backends() -> dict[str, dict[str, Any]]:
    """List all backends and their availability status."""
    result = {}
    for name, backend_class in _BACKENDS.items():
        try:
            backend = backend_class()
            result[name] = {
                "available": backend.is_available(),
                "info": backend.get_info() if backend.is_available() else None,
            }
        except Exception as e:
            result[name] = {
                "available": False,
                "error": str(e),
            }
    return result


__all__ = [
    "RuntimeConfig",
    "RuntimeResult",
    "RuntimeBackend",
    "BaseBackend",
    "BubblewrapBackend",
    "FirejailBackend",
    "DockerBackend",
    "NoneBackend",
    "get_backend",
    "list_backends",
]
