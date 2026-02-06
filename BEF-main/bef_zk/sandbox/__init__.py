"""
Capseal Sandbox - isolated execution environment for proof generation.

When you run `capseal run`, the computation executes inside a sandboxed
environment with:
- No network access
- Read-only access to datasets
- Write access only to output directory
- Resource limits (memory, CPU time)
- Syscall filtering (seccomp)

Supported backends:
- bubblewrap (Linux) - lightweight namespace sandbox
- firejail (Linux) - seccomp + namespace sandbox
- sandbox-exec (macOS) - seatbelt sandbox
- wasm (cross-platform) - WebAssembly isolation [future]
"""
from .runner import SandboxRunner, SandboxConfig, SandboxBackend
from .detect import detect_sandbox_backend, is_sandbox_available

__all__ = [
    "SandboxRunner",
    "SandboxConfig",
    "SandboxBackend",
    "detect_sandbox_backend",
    "is_sandbox_available",
]
