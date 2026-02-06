"""Sandbox runner - executes capseal commands in isolated environments."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detect import SandboxBackend, detect_sandbox_backend


class SandboxBackendError(RuntimeError):
    """Signals that the selected sandbox backend cannot run this command."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""

    # Filesystem access
    datasets: list[Path] = field(default_factory=list)  # Read-only dataset paths
    output_dir: Path | None = None                       # Write access for output
    policy_path: Path | None = None                      # Read-only policy file

    # Resource limits
    memory_mb: int = 4096          # Memory limit in MB
    cpu_time_sec: int = 300        # CPU time limit in seconds
    wall_time_sec: int = 600       # Wall clock time limit
    max_files: int = 1024          # Max open files

    # Network
    network: bool = False          # Allow network access (default: no)

    # Sandbox backend
    backend: SandboxBackend | None = None  # Auto-detect if None

    # Python environment
    python_path: Path | None = None  # Python interpreter to use
    capseal_root: Path | None = None  # Capseal installation root


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    returncode: int
    stdout: str
    stderr: str
    sandbox_backend: str
    resource_usage: dict[str, Any] = field(default_factory=dict)


class SandboxRunner:
    """
    Run capseal commands inside an isolated sandbox.

    Example:
        runner = SandboxRunner(SandboxConfig(
            datasets=[Path("./data")],
            output_dir=Path("./output"),
        ))
        result = runner.run(["capseal", "run", "-p", "policy.json", ...])
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.backend = config.backend or detect_sandbox_backend()

        # Find Python and capseal paths
        self.python = config.python_path or Path(sys.executable)
        self.capseal_root = config.capseal_root or self._find_capseal_root()

    def _find_capseal_root(self) -> Path:
        """Find the capseal installation root."""
        # Try to find it relative to this file
        this_file = Path(__file__).resolve()
        # bef_zk/sandbox/runner.py -> bef_zk -> root
        root = this_file.parent.parent.parent
        if (root / "bef_zk").exists():
            return root
        # Fall back to current directory
        return Path.cwd()

    def run(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run a command inside the sandbox."""
        try:
            if self.backend == SandboxBackend.BUBBLEWRAP:
                return self._run_bwrap(cmd, **kwargs)
            elif self.backend == SandboxBackend.FIREJAIL:
                return self._run_firejail(cmd, **kwargs)
            elif self.backend == SandboxBackend.NSJAIL:
                return self._run_nsjail(cmd, **kwargs)
            elif self.backend == SandboxBackend.SANDBOX_EXEC:
                return self._run_sandbox_exec(cmd, **kwargs)
            else:
                return self._run_unsandboxed(cmd, **kwargs)
        except SandboxBackendError as exc:
            if self.backend == SandboxBackend.BUBBLEWRAP and exc.reason == "userns_unavailable":
                return self._fallback_from_bwrap(cmd, **kwargs)
            raise

    def _run_bwrap(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run command using bubblewrap (Linux)."""
        return self._run_bwrap_impl(cmd, try_net_ns=not self.config.network, **kwargs)

    def _fallback_from_bwrap(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Attempt alternative backends when bubblewrap cannot create user namespaces."""
        import shutil
        import warnings

        warnings.warn(
            "User namespaces unavailable; falling back to a less isolated sandbox backend.",
            RuntimeWarning,
        )

        fallback_order: list[SandboxBackend] = []
        if shutil.which("firejail"):
            fallback_order.append(SandboxBackend.FIREJAIL)
        if shutil.which("nsjail"):
            fallback_order.append(SandboxBackend.NSJAIL)
        fallback_order.append(SandboxBackend.NONE)

        handlers = {
            SandboxBackend.FIREJAIL: self._run_firejail,
            SandboxBackend.NSJAIL: self._run_nsjail,
            SandboxBackend.NONE: self._run_unsandboxed,
        }

        last_error: Exception | None = None
        for backend in fallback_order:
            handler = handlers[backend]
            previous_backend = self.backend
            self.backend = backend
            try:
                result = handler(cmd, **kwargs)
                isolation = result.resource_usage.setdefault("isolation", {})
                isolation.setdefault("fallback_from", "bwrap_userns")
                if backend == SandboxBackend.NONE:
                    isolation["network_degraded"] = True
                return result
            except FileNotFoundError as err:
                last_error = err
                continue
            finally:
                self.backend = previous_backend

        raise SandboxBackendError("fallback_unavailable", str(last_error) if last_error else "no backend")

    def _run_bwrap_impl(
        self, cmd: list[str], try_net_ns: bool = True, **kwargs
    ) -> SandboxResult:
        """Run command using bubblewrap with fallback for net namespace failures."""
        import shutil
        import tempfile

        bwrap_cmd = [
            "bwrap",
            # Die with parent process - prevents orphaned sandboxes
            "--die-with-parent",
            # New namespaces (except network, handled separately)
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ]

        # Network isolation - try to unshare, but may fail without CAP_NET_ADMIN
        if try_net_ns and not self.config.network:
            bwrap_cmd.append("--unshare-net")
        # If network requested or fallback, share host network
        # (no --unshare-net means shared)

        # Clear environment and set minimal required vars
        bwrap_cmd.append("--clearenv")
        capseal_root = str(self.capseal_root)
        python_root = str(self.python.parent.parent)
        bwrap_cmd.extend([
            "--setenv", "PATH", "/usr/bin:/bin:/usr/local/bin",
            "--setenv", "HOME", "/tmp",
            "--setenv", "PYTHONPATH", capseal_root,
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "LC_ALL", "C.UTF-8",
        ])

        tz_value = os.environ.get("TZ")
        if tz_value:
            bwrap_cmd.extend(["--setenv", "TZ", tz_value])

        # Restricted filesystem via pivot root pattern
        scratch_root = tempfile.mkdtemp(prefix="capseal_root_")
        bwrap_cmd.extend(["--bind", scratch_root, "/"])
        bwrap_cmd.extend([
            # Minimal system directories (read-only)
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            # Essential virtual filesystems
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            # Home as tmpfs (nothing from host)
            "--tmpfs", "/home",
        ])

        # Add lib64 if it exists
        if Path("/lib64").exists():
            bwrap_cmd.extend(["--ro-bind", "/lib64", "/lib64"])

        # Add /etc for SSL certs, resolv.conf, etc (read-only)
        # Only bind specific files to minimize exposure
        etc_files = [
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/ssl",
            "/etc/ca-certificates",
            "/etc/pki",  # RHEL/CentOS
            "/etc/passwd",  # For user lookup
            "/etc/group",
            "/etc/nsswitch.conf",
            "/etc/localtime",
        ]
        for etc_path in etc_files:
            if Path(etc_path).exists():
                bwrap_cmd.extend(["--ro-bind", etc_path, etc_path])

        # Python environment (read-only)
        bwrap_cmd.extend(["--ro-bind", python_root, python_root])

        # Capseal code (read-only)
        bwrap_cmd.extend(["--ro-bind", capseal_root, capseal_root])

        # Set working directory
        bwrap_cmd.extend(["--chdir", capseal_root])

        # Add dataset mounts (read-only) - keep original paths for simplicity
        for dataset in self.config.datasets:
            dataset = dataset.resolve()
            bwrap_cmd.extend(["--ro-bind", str(dataset), str(dataset)])

        # Add policy file (read-only)
        if self.config.policy_path:
            policy = self.config.policy_path.resolve()
            # Bind parent dir to allow policy loading
            policy_dir = str(policy.parent.resolve())
            bwrap_cmd.extend(["--ro-bind", policy_dir, policy_dir])

        # Add output directory (read-write)
        if self.config.output_dir:
            output = self.config.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            bwrap_cmd.extend(["--bind", str(output), str(output)])

        # Build resource limit prefix
        limit_cmd = []

        # Wall clock timeout
        limit_cmd.extend([
            "timeout",
            "--signal=KILL",
            str(self.config.wall_time_sec),
        ])

        # Add prlimit for memory/CPU if available
        if shutil.which("prlimit"):
            # Memory limit in bytes
            mem_bytes = self.config.memory_mb * 1024 * 1024
            limit_cmd = [
                "prlimit",
                f"--as={mem_bytes}",           # Virtual memory limit
                f"--cpu={self.config.cpu_time_sec}",  # CPU time limit
                f"--nofile={self.config.max_files}",  # Open files limit
                "--",
            ] + limit_cmd

        # Build final command
        full_cmd = limit_cmd + bwrap_cmd + ["--", *cmd]

        try:
            result = self._execute(full_cmd, **kwargs)
        finally:
            shutil.rmtree(scratch_root, ignore_errors=True)

        combined_output = (result.stderr or "").lower() + (result.stdout or "").lower()

        # Detect user-namespace failures (common on hardened hosts)
        user_ns_errors = (
            "uid map",
            "user namespace",
            "userns",
            "need newuidmap",
            "need newgidmap",
            "unprivileged user namespaces",
        )
        if result.returncode != 0 and any(err in combined_output for err in user_ns_errors):
            raise SandboxBackendError("userns_unavailable", result.stderr)

        # Check for network namespace failure and retry without it
        # Common error patterns when user lacks CAP_NET_ADMIN:
        # - "EPERM" / "eperm" (generic kernel code)
        # - "Operation not permitted" / "Permission denied" (verbose)
        # - "NETLINK_ROUTE" / "CLONE_NEWNET" (specific errors)
        # - "unshare" (bwrap's own message)
        net_ns_errors = (
            "eperm", "operation not permitted", "permission denied",
            "netlink_route", "clone_newnet", "unshare",
        )
        if result.returncode != 0 and try_net_ns and any(e in combined_output for e in net_ns_errors):
            import warnings
            warnings.warn(
                "Network namespace unavailable (unprivileged). "
                "Retrying with shared network. Network isolation NOT active.",
                RuntimeWarning,
            )
            return self._run_bwrap_impl(cmd, try_net_ns=False, **kwargs)

        # Annotate result with actual isolation guarantees
        network_isolated = try_net_ns and not self.config.network
        result.resource_usage["isolation"] = {
            "network": network_isolated,
            "network_degraded": not try_net_ns and not self.config.network,  # Wanted but couldn't
            "filesystem": True,
            "pid_namespace": True,
            "ipc_namespace": True,
            "uts_namespace": True,
            "cgroup_namespace": True,
            "memory_limit": shutil.which("prlimit") is not None,
            "cpu_limit": shutil.which("prlimit") is not None,
            "file_limit": shutil.which("prlimit") is not None,
            "wall_timeout": True,
            "pivot_root": True,
        }

        return result

    def _run_firejail(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run command using firejail (Linux)."""
        firejail_cmd = [
            "firejail",
            "--quiet",
            "--noprofile",

            # Namespaces
            "--noroot",
            "--nonewprivs",
            "--seccomp",

            # Network
            "--net=none" if not self.config.network else "",

            # Resource limits
            f"--rlimit-as={self.config.memory_mb * 1024 * 1024}",
            f"--timeout={self.config.wall_time_sec}",  # Seconds directly

            # Filesystem
            "--private-tmp",
            "--private-dev",
        ]

        # Remove empty strings
        firejail_cmd = [c for c in firejail_cmd if c]

        # Add read-only binds for datasets
        for i, dataset in enumerate(self.config.datasets):
            dataset = dataset.resolve()
            firejail_cmd.append(f"--read-only={dataset}")

        # Add capseal root as read-only
        firejail_cmd.append(f"--read-only={self.capseal_root}")

        # Add output as writable
        if self.config.output_dir:
            output = self.config.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            firejail_cmd.append(f"--read-write={output}")

        full_cmd = firejail_cmd + ["--", *cmd]
        result = self._execute(full_cmd, **kwargs)

        # Annotate isolation guarantees
        result.resource_usage["isolation"] = {
            "network": not self.config.network,
            "network_degraded": False,
            "filesystem": True,
            "pid_namespace": True,
            "ipc_namespace": True,
            "uts_namespace": True,
            "cgroup_namespace": False,  # firejail doesn't unshare cgroup by default
            "memory_limit": True,
            "cpu_limit": False,
            "file_limit": False,
            "wall_timeout": True,
            "seccomp": True,
            "pivot_root": False,
        }
        return result

    def _run_nsjail(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run command using nsjail (Linux)."""
        nsjail_cmd = [
            "nsjail",
            "--mode", "o",  # Once mode
            "--quiet",

            # Resource limits
            "--rlimit_as", str(self.config.memory_mb),
            "--time_limit", str(self.config.wall_time_sec),
            "--max_cpus", "1",

            # Basic mounts - keep original paths for pipeline compatibility
            "--bindmount_ro", "/usr:/usr",
            "--bindmount_ro", "/bin:/bin",
            "--bindmount_ro", "/lib:/lib",
            "--bindmount_ro", "/etc:/etc",
            "--mount", "none:/tmp:tmpfs:size=100M",
            "--mount", "none:/dev:tmpfs",
        ]

        # Network isolation - nsjail isolates by default, --disable_clone_newnet ENABLES network
        if self.config.network:
            nsjail_cmd.append("--disable_clone_newnet")
        # (no flag = network isolated by default)

        # Add lib64 if it exists
        if Path("/lib64").exists():
            nsjail_cmd.extend(["--bindmount_ro", "/lib64:/lib64"])

        # Python environment - keep original paths
        python_root = str(self.python.parent.parent)
        nsjail_cmd.extend(["--bindmount_ro", f"{python_root}:{python_root}"])

        # Capseal root - keep original path
        capseal_root = str(self.capseal_root)
        nsjail_cmd.extend(["--bindmount_ro", f"{capseal_root}:{capseal_root}"])

        # Add dataset mounts (read-only) - keep original paths!
        for dataset in self.config.datasets:
            dataset = dataset.resolve()
            nsjail_cmd.extend(["--bindmount_ro", f"{dataset}:{dataset}"])

        # Add policy file parent directory
        if self.config.policy_path:
            policy_dir = str(self.config.policy_path.resolve().parent)
            nsjail_cmd.extend(["--bindmount_ro", f"{policy_dir}:{policy_dir}"])

        # Add output mount (read-write)
        if self.config.output_dir:
            output = self.config.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            nsjail_cmd.extend(["--bindmount", f"{output}:{output}"])

        # Set working directory to capseal root
        nsjail_cmd.extend(["--cwd", capseal_root])

        full_cmd = nsjail_cmd + ["--", *cmd]
        result = self._execute(full_cmd, **kwargs)

        # Annotate isolation guarantees
        result.resource_usage["isolation"] = {
            "network": not self.config.network,
            "network_degraded": False,
            "filesystem": True,
            "pid_namespace": True,
            "ipc_namespace": True,
            "uts_namespace": True,
            "cgroup_namespace": True,
            "memory_limit": True,
            "cpu_limit": True,
            "file_limit": False,
            "wall_timeout": True,
            "pivot_root": False,
        }
        return result

    def _run_sandbox_exec(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run command using macOS sandbox-exec."""
        # Build seatbelt profile
        profile = self._build_seatbelt_profile()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as f:
            f.write(profile)
            profile_path = f.name

        try:
            sandbox_cmd = [
                "sandbox-exec",
                "-f", profile_path,
                *cmd,
            ]
            result = self._execute(sandbox_cmd, **kwargs)

            # Annotate isolation guarantees
            result.resource_usage["isolation"] = {
                "network": not self.config.network,
                "network_degraded": False,
                "filesystem": True,
                "pid_namespace": False,  # macOS sandbox doesn't isolate PIDs
                "ipc_namespace": False,
                "uts_namespace": False,
                "cgroup_namespace": False,
                "memory_limit": False,  # seatbelt doesn't limit memory
                "cpu_limit": False,
                "file_limit": False,
                "wall_timeout": False,
                "seatbelt": True,
                "pivot_root": False,
            }
            return result
        finally:
            os.unlink(profile_path)

    def _build_seatbelt_profile(self) -> str:
        """Build macOS seatbelt sandbox profile."""
        profile = """
(version 1)
(deny default)

; Allow basic process execution
(allow process-exec*)
(allow process-fork)
(allow signal)

; Allow reading system libraries
(allow file-read*
    (subpath "/usr/lib")
    (subpath "/System/Library")
    (subpath "/Library/Frameworks")
    (subpath "/usr/local")
    (literal "/dev/null")
    (literal "/dev/urandom")
    (literal "/dev/random")
)

; Allow Python
(allow file-read* (subpath "{python_root}"))

; Allow capseal code (read-only)
(allow file-read* (subpath "{capseal_root}"))

; Allow temp
(allow file-read* file-write* (subpath "/tmp"))
(allow file-read* file-write* (subpath "/private/tmp"))
(allow file-read* file-write* (subpath (param "TMPDIR")))

; Allow sysctl for process info
(allow sysctl-read)
(allow mach-lookup)
""".format(
            python_root=str(self.python.parent.parent),
            capseal_root=str(self.capseal_root),
        )

        # Add dataset read permissions
        for dataset in self.config.datasets:
            profile += f'\n(allow file-read* (subpath "{dataset.resolve()}"))'

        # Add output write permission
        if self.config.output_dir:
            output = self.config.output_dir.resolve()
            profile += f'\n(allow file-read* file-write* (subpath "{output}"))'

        # Add policy file permission
        if self.config.policy_path:
            policy = self.config.policy_path.resolve()
            profile += f'\n(allow file-read* (literal "{policy}"))'

        # Network
        if not self.config.network:
            profile += "\n(deny network*)"
        else:
            profile += "\n(allow network*)"

        return profile

    def _run_unsandboxed(self, cmd: list[str], **kwargs) -> SandboxResult:
        """Run without sandbox (fallback)."""
        import warnings
        warnings.warn(
            "No sandbox backend available. Running without isolation. "
            "Install bubblewrap (Linux) or use macOS for sandboxed execution.",
            RuntimeWarning,
        )
        result = self._execute(cmd, **kwargs)

        # No isolation guarantees
        result.resource_usage["isolation"] = {
            "network": False,
            "network_degraded": False,
            "filesystem": False,
            "pid_namespace": False,
            "ipc_namespace": False,
            "uts_namespace": False,
            "cgroup_namespace": False,
            "memory_limit": False,
            "cpu_limit": False,
            "file_limit": False,
            "wall_timeout": False,
            "pivot_root": False,
        }
        return result

    def _execute(
        self,
        cmd: list[str],
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute a command and capture output."""
        timeout = timeout or self.config.wall_time_sec

        # Build environment
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(self.capseal_root)
        if env:
            run_env.update(env)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                env=run_env,
                text=True,
            )
            return SandboxResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                sandbox_backend=self.backend.value,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                returncode=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=f"Timeout after {timeout}s",
                sandbox_backend=self.backend.value,
            )
        except Exception as e:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                sandbox_backend=self.backend.value,
            )


def run_sandboxed(
    cmd: list[str],
    *,
    datasets: list[Path] | None = None,
    output_dir: Path | None = None,
    policy_path: Path | None = None,
    **kwargs,
) -> SandboxResult:
    """
    Convenience function to run a command in a sandbox.

    Example:
        result = run_sandboxed(
            ["python", "-m", "bef_zk.capsule.cli", "run", ...],
            datasets=[Path("./data")],
            output_dir=Path("./output"),
        )
    """
    config = SandboxConfig(
        datasets=datasets or [],
        output_dir=output_dir,
        policy_path=policy_path,
        **kwargs,
    )
    runner = SandboxRunner(config)
    return runner.run(cmd)
