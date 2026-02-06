"""Detect available sandbox backends on the system."""
from __future__ import annotations

import shutil
import platform
import subprocess
from enum import Enum


class SandboxBackend(Enum):
    """Available sandbox backends."""
    BUBBLEWRAP = "bwrap"      # Linux - lightweight, namespace-based
    FIREJAIL = "firejail"    # Linux - seccomp + namespaces
    NSJAIL = "nsjail"        # Linux - Google's sandbox
    SANDBOX_EXEC = "sandbox-exec"  # macOS - seatbelt
    NONE = "none"            # No sandbox (fallback)


def _check_command(cmd: str) -> bool:
    """Check if a command exists and is executable."""
    return shutil.which(cmd) is not None


def _check_bwrap() -> bool:
    """Check if bubblewrap is available and functional."""
    if not _check_command("bwrap"):
        return False
    try:
        result = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _check_firejail() -> bool:
    """Check if firejail is available."""
    if not _check_command("firejail"):
        return False
    try:
        result = subprocess.run(
            ["firejail", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _check_nsjail() -> bool:
    """Check if nsjail is available."""
    if not _check_command("nsjail"):
        return False
    try:
        result = subprocess.run(
            ["nsjail", "--version"],
            capture_output=True,
            timeout=5,
        )
        # nsjail returns non-zero for --version but outputs version info
        return b"nsjail" in result.stdout or b"nsjail" in result.stderr
    except (subprocess.TimeoutExpired, OSError):
        return False


def _check_sandbox_exec() -> bool:
    """Check if macOS sandbox-exec is available."""
    if platform.system() != "Darwin":
        return False
    return _check_command("sandbox-exec")


def detect_sandbox_backend() -> SandboxBackend:
    """
    Detect the best available sandbox backend for this system.

    Priority order:
    1. bubblewrap (Linux) - fastest, most lightweight
    2. firejail (Linux) - good fallback with seccomp
    3. nsjail (Linux) - most secure but complex
    4. sandbox-exec (macOS) - native macOS sandbox
    5. none - no sandbox available
    """
    system = platform.system()

    if system == "Linux":
        if _check_bwrap():
            return SandboxBackend.BUBBLEWRAP
        if _check_firejail():
            return SandboxBackend.FIREJAIL
        if _check_nsjail():
            return SandboxBackend.NSJAIL
    elif system == "Darwin":
        if _check_sandbox_exec():
            return SandboxBackend.SANDBOX_EXEC

    return SandboxBackend.NONE


def is_sandbox_available() -> bool:
    """Check if any sandbox backend is available."""
    return detect_sandbox_backend() != SandboxBackend.NONE


def get_sandbox_info() -> dict:
    """Get information about available sandbox backends."""
    system = platform.system()
    return {
        "system": system,
        "detected_backend": detect_sandbox_backend().value,
        "available": {
            "bubblewrap": _check_bwrap() if system == "Linux" else False,
            "firejail": _check_firejail() if system == "Linux" else False,
            "nsjail": _check_nsjail() if system == "Linux" else False,
            "sandbox_exec": _check_sandbox_exec() if system == "Darwin" else False,
        },
        "sandbox_available": is_sandbox_available(),
    }
