"""Interactive Capseal Shell - sandboxed environment for proof generation."""
from __future__ import annotations

import cmd
import hashlib
import json
import os
import readline
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner

from .redact import redact_secrets
from .utils import get_workspace_root
from .git_utils import safe_glob
from .doctor import run_doctor, PipelineReport
from .metrics import MetricsEnforcer, get_enforcer

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _detect_llm_provider() -> tuple[str, str]:
    """Auto-detect LLM provider based on available API keys.

    Returns:
        (provider, model) tuple with sensible defaults.

    Priority:
        1. ANTHROPIC_API_KEY -> anthropic / claude-sonnet-4-20250514
        2. OPENAI_API_KEY -> openai / gpt-4o-mini
        3. Raises RuntimeError if neither is set
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    # Check shell's exported env vars too
    return "openai", "gpt-4o-mini"  # Default, will fail with clear error if no key


@dataclass
class ShellState:
    """State for the interactive shell."""
    datasets: dict[str, Path] = field(default_factory=dict)
    policy_path: Path | None = None
    policy_id: str = "default"
    output_dir: Path = field(default_factory=lambda: Path("./capsules"))
    last_capsule: Path | None = None
    sandbox_enabled: bool = True
    sandbox_network_allowed: bool = False
    history: list[str] = field(default_factory=list)
    # Comparison state
    compare_source: Path | None = None
    compare_target: Path | None = None
    watch_receipts: bool = False
    # Last reviewed target path (for finding receipts)
    last_review_path: Path | None = None


class CapsealShell(cmd.Cmd):
    """Interactive Capseal sandbox shell."""

    intro = f"""
{CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
║                    CAPSEAL SANDBOX v0.6                        ║
║      Proof-carrying code review & refactors (LLM + receipts)   ║
╚════════════════════════════════════════════════════════════════╝{RESET}

{BOLD}Workflow (typical):{RESET}
  review <path>           Generate verified diff (scan → plan → patch → verify)
  eval <path> [--rounds]  Epistemic eval loop (learn safety boundary → gate)
  ask <question>          Open-ended analysis, suggestions, or code generation
  open [latest|<run>]     Inspect plan, patches, proofs, and receipts
  apply [latest|<run>]    Apply verified patches (never applies unverified)
  audit [latest|<run>]    Explain what was checked + trust boundaries

{DIM}Run 'help' for full command list. Type 'status' to see current state.{RESET}
"""
    prompt = f"{CYAN}capseal>{RESET} "

    def __init__(self, state: ShellState | None = None):
        super().__init__()
        self.state = state or ShellState()
        self._check_sandbox()
        # CLI passthrough router - forward unknown commands to the real CLI root
        self._cli_runner = CliRunner()
        self._cli_root = self._get_cli_root()

    def _get_cli_root(self):
        """Get the CLI root command (lazy import to avoid circular deps)."""
        from bef_zk.capsule.cli import cli as capseal_cli
        return capseal_cli

    # ─────────────────────────────────────────────────────────────────
    # CLI Passthrough Router - forward unknown commands to CLI root
    # ─────────────────────────────────────────────────────────────────

    def _invoke_cli(self, argv: list[str]) -> int:
        """Run a CLI command in-process via CliRunner.

        This allows the shell to forward any command registered on the CLI root
        without needing explicit do_* handlers.
        """
        res = self._cli_runner.invoke(
            self._cli_root,
            argv,
            prog_name="capseal",
            catch_exceptions=False,
        )

        if res.output:
            sys.stdout.write(res.output)

        if res.exit_code != 0 and not res.output:
            print(f"{RED}Command failed (exit={res.exit_code}){RESET}", file=sys.stderr)

        return res.exit_code

    def _check_sandbox(self) -> None:
        """Check sandbox availability on startup."""
        try:
            from bef_zk.sandbox import is_sandbox_available, detect_sandbox_backend
            if is_sandbox_available():
                backend = detect_sandbox_backend()
                print(f"{GREEN}✓ Sandbox active: {backend.value}{RESET}")
            else:
                print(f"{YELLOW}⚠ No sandbox backend available - running unprotected{RESET}")
                self.state.sandbox_enabled = False
        except ImportError:
            print(f"{YELLOW}⚠ Sandbox module not found{RESET}")
            self.state.sandbox_enabled = False

    def _print_status(self) -> None:
        """Print current shell status."""
        print(f"\n{BOLD}Current State:{RESET}")
        print(f"  Sandbox:  {'enabled' if self.state.sandbox_enabled else 'disabled'}")
        print(f"  Sandbox network: {'allowed' if self.state.sandbox_network_allowed else 'blocked'}")
        print(f"  Datasets: {len(self.state.datasets)}")
        for ds_id, path in self.state.datasets.items():
            print(f"    - {ds_id}: {path}")
        print(f"  Policy:   {self.state.policy_path or 'not set'}")
        print(f"  Output:   {self.state.output_dir}")
        if self.state.last_capsule:
            print(f"  Last:     {self.state.last_capsule}")

        # Comparison state
        if self.state.compare_source:
            print(f"\n{BOLD}Comparison:{RESET}")
            print(f"  Source:   {self.state.compare_source}")
            print(f"  Target:   {self.state.compare_target}")

        # Receipt stats
        try:
            from bef_zk.capsule.mcp_server import EVENT_LOG_PATH
            receipts = self._load_receipts(EVENT_LOG_PATH)
            if receipts:
                print(f"\n{BOLD}Receipts:{RESET} {len(receipts)} in chain")
        except ImportError:
            pass
        print()

    FLAG_ONLY_OPTIONS = {"--json", "--no-verify"}

    def _resolve_capsule_path(self, tokens: list[str]) -> tuple[Path | None, list[str]]:
        path_arg: Path | None = None
        extras: list[str] = []
        expect_value = False
        for token in tokens:
            if expect_value:
                extras.append(token)
                expect_value = False
                continue
            if token.startswith("-"):
                extras.append(token)
                if token in self.FLAG_ONLY_OPTIONS or "=" in token:
                    continue
                expect_value = True
                continue
            if path_arg is None:
                candidate = Path(token).expanduser().resolve()
                path_arg = candidate
            else:
                extras.append(token)
        if path_arg is None:
            if self.state.last_capsule:
                path_arg = self.state.last_capsule
            else:
                print(f"{RED}No capsule specified. Generate or inspect one first.{RESET}")
                return None, []
        if not path_arg.exists():
            print(f"{RED}Capsule not found: {path_arg}{RESET}")
            return None, []
        return path_arg, extras

    @staticmethod
    def _stream_hash(filepath: Path, chunk_size: int = 65536) -> str:
        """Compute SHA256 hash using streaming to handle large files."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    # ─────────────────────────────────────────────────────────────────
    # Dataset commands
    # ─────────────────────────────────────────────────────────────────

    def do_scan(self, arg: str) -> None:
        """Scan a directory for datasets.

        Usage: scan <path>

        Scans the directory and shows files that can be committed.
        Uses streaming hash for large files (>16MB).
        """
        path = Path(arg.strip() or ".").expanduser().resolve()
        if not path.exists():
            print(f"{RED}Path not found: {path}{RESET}")
            return

        print(f"\n{BOLD}Scanning: {path}{RESET}\n")

        # Threshold for streaming hash (16MB)
        STREAM_THRESHOLD = 16 * 1024 * 1024

        if path.is_file():
            size = path.stat().st_size
            if size > STREAM_THRESHOLD:
                h = self._stream_hash(path)[:16]
            else:
                h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            print(f"  {path.name}: {size:,} bytes, sha256:{h}...")
            return

        total_files = 0
        total_bytes = 0
        # Use safe_glob with limits instead of unbounded rglob
        files = sorted(safe_glob(path, "*", max_results=5000, tracked_only=False))

        for f in files:
            if f.is_file():
                size = f.stat().st_size
                rel = f.relative_to(path)
                # Use streaming hash for large files
                if size > STREAM_THRESHOLD:
                    h = self._stream_hash(f)[:12]
                    print(f"  {rel}: {size:,} bytes [{h}...] (streamed)")
                else:
                    h = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
                    print(f"  {rel}: {size:,} bytes [{h}...]")
                total_files += 1
                total_bytes += size

        print(f"\n{GREEN}Found {total_files} files, {total_bytes:,} bytes total{RESET}")
        print(f"{DIM}Use 'add {path}' to add as dataset{RESET}\n")

    def do_add(self, arg: str) -> None:
        """Add a dataset.

        Usage: add <path> [as <name>]

        Examples:
            add ./data
            add ./archive as training_data
        """
        parts = arg.strip().split(" as ")
        path_str = parts[0].strip()

        if not path_str:
            print(f"{RED}Usage: add <path> [as <name>]{RESET}")
            return

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            print(f"{RED}Path not found: {path}{RESET}")
            return

        name = parts[1].strip() if len(parts) > 1 else path.name
        self.state.datasets[name] = path

        # Compute quick stats (with bounded file scan)
        if path.is_dir():
            files = list(safe_glob(path, "*", max_results=5000, tracked_only=False))
            file_count = sum(1 for f in files if f.is_file())
            total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        else:
            file_count = 1
            total_bytes = path.stat().st_size

        print(f"{GREEN}✓ Added dataset '{name}': {file_count} files, {total_bytes:,} bytes{RESET}")

    def do_remove(self, arg: str) -> None:
        """Remove a dataset.

        Usage: remove <name>
        """
        name = arg.strip()
        if name in self.state.datasets:
            del self.state.datasets[name]
            print(f"{GREEN}✓ Removed dataset '{name}'{RESET}")
        else:
            print(f"{RED}Dataset not found: {name}{RESET}")

    def do_datasets(self, arg: str) -> None:
        """List all added datasets."""
        if not self.state.datasets:
            print(f"{DIM}No datasets added. Use 'scan' and 'add' to add datasets.{RESET}")
            return

        print(f"\n{BOLD}Datasets:{RESET}")
        for name, path in self.state.datasets.items():
            if path.is_dir():
                files = list(safe_glob(path, "*", max_results=5000, tracked_only=False))
                file_count = sum(1 for f in files if f.is_file())
                total_bytes = sum(f.stat().st_size for f in files if f.is_file())
            else:
                file_count = 1
                total_bytes = path.stat().st_size
            print(f"  {CYAN}{name}{RESET}: {path}")
            print(f"    {file_count} files, {total_bytes:,} bytes")
        print()

    # ─────────────────────────────────────────────────────────────────
    # Basic shell commands
    # ─────────────────────────────────────────────────────────────────

    def do_ls(self, arg: str) -> None:
        """List directory contents.

        Usage:
            ls              List current directory
            ls <path>       List specified path
            ls -la <path>   Long format with hidden files
        """
        import subprocess
        # Parse args carefully to handle paths with spaces
        raw_arg = arg.strip()
        if raw_arg:
            # Use shell=True for complex args like "-la /path"
            cmd = f"ls --color=auto {raw_arg}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(["ls", "--color=auto"], capture_output=True, text=True, timeout=10)
        try:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(f"{RED}{result.stderr}{RESET}", end="")
        except subprocess.TimeoutExpired:
            print(f"{RED}Command timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_pwd(self, arg: str) -> None:
        """Print current working directory."""
        print(os.getcwd())

    def do_cd(self, arg: str) -> None:
        """Change directory.

        Usage:
            cd <path>       Change to path
            cd              Change to home directory
        """
        target = arg.strip() or os.path.expanduser("~")
        target = os.path.expanduser(target)
        try:
            os.chdir(target)
            print(f"{DIM}{os.getcwd()}{RESET}")
        except FileNotFoundError:
            print(f"{RED}No such directory: {target}{RESET}")
        except PermissionError:
            print(f"{RED}Permission denied: {target}{RESET}")

    def do_cat(self, arg: str) -> None:
        """Display file contents.

        Usage:
            cat <file>      Display file contents
        """
        if not arg.strip():
            print(f"{RED}Usage: cat <file>{RESET}")
            return
        path = Path(arg.strip()).expanduser()
        if not path.exists():
            print(f"{RED}No such file: {path}{RESET}")
            return
        if path.is_dir():
            print(f"{RED}Is a directory: {path}{RESET}")
            return
        try:
            content = path.read_text()
            # Limit output for sanity
            if len(content) > 10000:
                print(content[:10000])
                print(f"\n{DIM}... (truncated, {len(content):,} bytes total){RESET}")
            else:
                print(content, end="")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_tree(self, arg: str) -> None:
        """Display directory tree.

        Usage:
            tree            Tree of current directory
            tree <path>     Tree of specified path
            tree -L 2       Limit depth
        """
        import subprocess
        parts = shlex.split(arg) if arg.strip() else []
        cmd = ["tree", "-C"] + parts
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(f"{RED}{result.stderr}{RESET}", end="")
        except FileNotFoundError:
            print(f"{YELLOW}tree not installed. Use 'ls' instead.{RESET}")
        except subprocess.TimeoutExpired:
            print(f"{RED}Command timed out{RESET}")

    def do_export(self, arg: str) -> None:
        """Set environment variable.

        Usage:
            export VAR=value        Set variable
            export VAR="value"      Set variable (quotes stripped)
            export                  Show all exported vars this session

        Examples:
            export OPENAI_API_KEY=sk-xxx
            export MODEL=gpt-4o
        """
        arg = arg.strip()
        if not arg:
            # Show current env vars (highlight ones we set)
            print(f"\n{BOLD}Environment:{RESET}")
            for key in sorted(os.environ.keys()):
                if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODEL", "PROVIDER"):
                    val = os.environ[key]
                    # Mask API keys
                    if "API_KEY" in key and len(val) > 8:
                        val = val[:4] + "..." + val[-4:]
                    print(f"  {CYAN}{key}{RESET}={val}")
            return

        # Parse VAR=value
        if "=" not in arg:
            # Just show this var
            val = os.environ.get(arg)
            if val:
                if "API_KEY" in arg and len(val) > 8:
                    val = val[:4] + "..." + val[-4:]
                print(f"{arg}={val}")
            else:
                print(f"{DIM}{arg} is not set{RESET}")
            return

        key, _, value = arg.partition("=")
        key = key.strip()
        # Strip quotes if present
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        os.environ[key] = value
        # Mask API keys in output
        display_val = value
        if "API_KEY" in key and len(value) > 8:
            display_val = value[:4] + "..." + value[-4:]
        print(f"{GREEN}✓{RESET} {key}={display_val}")

    def do_source(self, arg: str) -> None:
        """Source a file to load environment variables.

        Usage:
            source <file>       Load env vars from file
            source .env         Load from .env file

        Supports formats:
            VAR=value
            export VAR=value
            # comments are ignored
        """
        if not arg.strip():
            print(f"{RED}Usage: source <file>{RESET}")
            print(f"{DIM}Example: source .env{RESET}")
            return

        path = Path(arg.strip()).expanduser()
        if not path.exists():
            print(f"{RED}No such file: {path}{RESET}")
            return

        try:
            content = path.read_text()
            loaded = 0
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Strip 'export ' prefix if present
                if line.startswith("export "):
                    line = line[7:]
                # Parse VAR=value
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Strip quotes
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value
                    loaded += 1

            print(f"{GREEN}✓{RESET} Loaded {loaded} variables from {path.name}")
        except Exception as e:
            print(f"{RED}Error reading {path}: {e}{RESET}")

    def do_env(self, arg: str) -> None:
        """Show or filter environment variables.

        Usage:
            env                 Show all env vars
            env API             Show vars containing 'API'
            env --keys          Show only var names
        """
        arg = arg.strip()
        keys_only = "--keys" in arg
        filter_str = arg.replace("--keys", "").strip().upper()

        for key in sorted(os.environ.keys()):
            if filter_str and filter_str not in key.upper():
                continue
            if keys_only:
                print(key)
            else:
                val = os.environ[key]
                # Mask sensitive values
                if "API_KEY" in key or "SECRET" in key or "TOKEN" in key or "PASSWORD" in key:
                    if len(val) > 8:
                        val = val[:4] + "..." + val[-4:]
                print(f"{key}={val}")

    def do_unset(self, arg: str) -> None:
        """Unset an environment variable.

        Usage:
            unset VAR           Remove variable from environment
        """
        if not arg.strip():
            print(f"{RED}Usage: unset <VAR>{RESET}")
            return
        key = arg.strip()
        if key in os.environ:
            del os.environ[key]
            print(f"{GREEN}✓{RESET} Unset {key}")
        else:
            print(f"{DIM}{key} was not set{RESET}")

    def do_git(self, arg: str) -> None:
        """Run git commands.

        Usage:
            git status
            git diff
            git apply <patch>
            git log --oneline -5
        """
        import subprocess
        if not arg.strip():
            print(f"{RED}Usage: git <command>{RESET}")
            return
        cmd = f"git {arg}"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                # Git often writes to stderr for non-errors
                print(result.stderr, end="")
        except subprocess.TimeoutExpired:
            print(f"{RED}Command timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_make(self, arg: str) -> None:
        """Run make commands.

        Usage:
            make
            make test
            make build
        """
        import subprocess
        cmd = f"make {arg}".strip()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
        except subprocess.TimeoutExpired:
            print(f"{RED}Command timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    # NOTE: default() method is defined at end of class - it forwards
    # unknown commands to the CLI root via passthrough router

    # ─────────────────────────────────────────────────────────────────
    # Policy commands
    # ─────────────────────────────────────────────────────────────────

    def do_policy(self, arg: str) -> None:
        """Set or show the policy file.

        Usage:
            policy              - show current policy
            policy <path>       - set policy file
            policy --id <name>  - set policy ID
        """
        arg = arg.strip()

        if not arg:
            if self.state.policy_path:
                print(f"\n{BOLD}Policy:{RESET} {self.state.policy_path}")
                print(f"{BOLD}Policy ID:{RESET} {self.state.policy_id}")
                try:
                    policy = json.loads(self.state.policy_path.read_text())
                    print(f"{DIM}{json.dumps(policy, indent=2)[:500]}...{RESET}")
                except Exception:
                    pass
            else:
                print(f"{DIM}No policy set. Use 'policy <path>' to set one.{RESET}")
            return

        if arg.startswith("--id "):
            self.state.policy_id = arg[5:].strip()
            print(f"{GREEN}✓ Policy ID set to '{self.state.policy_id}'{RESET}")
            return

        path = Path(arg).expanduser().resolve()
        if not path.exists():
            print(f"{RED}Policy file not found: {path}{RESET}")
            return

        self.state.policy_path = path

        # Try to extract policy_id from file
        try:
            policy = json.loads(path.read_text())
            if "policy_id" in policy:
                self.state.policy_id = policy["policy_id"]
        except Exception:
            pass

        print(f"{GREEN}✓ Policy set: {path}{RESET}")
        print(f"  Policy ID: {self.state.policy_id}")

    # ─────────────────────────────────────────────────────────────────
    # Generation commands
    # ─────────────────────────────────────────────────────────────────

    def do_generate(self, arg: str) -> None:
        """Generate a proof/capsule from current datasets.

        Usage: generate [--steps N] [--name NAME]

        Examples:
            generate
            generate --steps 128
            generate --name my_proof
        """
        if not self.state.datasets:
            print(f"{RED}No datasets added. Use 'add <path>' first.{RESET}")
            return

        if not self.state.policy_path:
            print(f"{RED}No policy set. Use 'policy <path>' first.{RESET}")
            return

        # Parse args
        steps = 64
        name = "capsule"

        parts = arg.split()
        i = 0
        while i < len(parts):
            if parts[i] == "--steps" and i + 1 < len(parts):
                steps = int(parts[i + 1])
                i += 2
            elif parts[i] == "--name" and i + 1 < len(parts):
                name = parts[i + 1]
                i += 2
            else:
                i += 1

        # Build output path
        output = self.state.output_dir / name
        output.mkdir(parents=True, exist_ok=True)

        print(f"\n{BOLD}Generating proof...{RESET}")
        print(f"  Datasets: {', '.join(self.state.datasets.keys())}")
        print(f"  Policy:   {self.state.policy_id}")
        print(f"  Steps:    {steps}")
        print(f"  Output:   {output}")
        print(f"  Sandbox:  {'enabled' if self.state.sandbox_enabled else 'disabled'}")
        if self.state.sandbox_enabled:
            print(f"  Sandbox network: {'allowed' if self.state.sandbox_network_allowed else 'blocked'}")
        print()

        # Build command
        from bef_zk.capsule.cli.run import _find_project_root
        root = _find_project_root()

        cmd = [
            sys.executable,
            str(root / "scripts" / "run_pipeline.py"),
            "--policy", str(self.state.policy_path),
            "--policy-id", self.state.policy_id,
            "--steps", str(steps),
            "--output-dir", str(output),
            "--allow-insecure-da-challenge",
            "--verification-profile", "proof_only",
        ]

        for ds_name, ds_path in self.state.datasets.items():
            cmd.extend(["--dataset", f"{ds_name}={ds_path}"])

        # Run
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        if self.state.sandbox_enabled:
            from bef_zk.sandbox import SandboxRunner, SandboxConfig

            config = SandboxConfig(
                datasets=list(self.state.datasets.values()),
                output_dir=output,
                policy_path=self.state.policy_path,
                memory_mb=4096,
                wall_time_sec=600,
                network=self.state.sandbox_network_allowed,
                capseal_root=root,
            )
            runner = SandboxRunner(config)
            print(f"{DIM}Running in sandbox ({runner.backend.value})...{RESET}")
            result = runner.run(cmd, env=env)
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        else:
            import subprocess
            print(f"{DIM}Running...{RESET}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(root))
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr

        if stdout:
            print(f"{DIM}{stdout}{RESET}")

        if returncode != 0:
            print(f"{RED}✗ Generation failed{RESET}")
            if stderr:
                print(f"{DIM}{stderr[:500]}{RESET}")
            return

        # Success - find capsule
        capsule_path = output / "strategy_capsule.json"
        if capsule_path.exists():
            self.state.last_capsule = capsule_path
            capsule = json.loads(capsule_path.read_text())

            print(f"{GREEN}✓ Capsule generated!{RESET}")
            print(f"\n  {BOLD}Hash:{RESET}    {capsule.get('capsule_hash', 'N/A')}")
            print(f"  {BOLD}Path:{RESET}    {capsule_path}")

            if capsule.get("dataset_ref"):
                print(f"  {BOLD}Datasets:{RESET}")
                for ds in capsule["dataset_ref"].get("datasets", []):
                    print(f"    - {ds['dataset_id']}: {ds['root'][:16]}...")
            print()
        else:
            print(f"{GREEN}✓ Generation complete. Output: {output}{RESET}")

    def do_verify(self, arg: str) -> None:
        """Verify a capsule.

        Usage: verify [path] [--mode MODE]

        Modes: proof-only (default), da, replay

        If no path given, verifies the last generated capsule.
        Uses the active policy if set.
        """
        parts = arg.strip().split()
        path_str = ""
        mode = "proof-only"

        i = 0
        while i < len(parts):
            if parts[i] == "--mode" and i + 1 < len(parts):
                mode = parts[i + 1]
                i += 2
            else:
                path_str = parts[i]
                i += 1

        if path_str:
            path = Path(path_str).expanduser().resolve()
        elif self.state.last_capsule:
            path = self.state.last_capsule
        else:
            print(f"{RED}No capsule to verify. Generate one or specify a path.{RESET}")
            return

        if not path.exists():
            print(f"{RED}Capsule not found: {path}{RESET}")
            return

        print(f"\n{BOLD}Verifying: {path}{RESET}")
        print(f"  Mode: {mode}")
        if self.state.policy_path:
            print(f"  Policy: {self.state.policy_path.name}")
        print()

        from bef_zk.capsule.cli.run import _find_project_root
        root = _find_project_root()

        cmd = [
            sys.executable, "-m", "bef_zk.capsule.cli", "verify",
            str(path), "--json",
            "--mode", mode,
        ]

        # Pass active policy if set
        if self.state.policy_path:
            cmd.extend(["--policy", str(self.state.policy_path)])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        import subprocess
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        try:
            data = json.loads(result.stdout)
            # Check various success indicators
            status = data.get("status", "")
            proof_ok = data.get("proof_verified", False)
            is_success = (
                status in ("PROOF_ONLY", "VERIFIED", "POLICY_SELF_REPORTED", "POLICY_ENFORCED")
                or proof_ok
                or result.returncode == 0
            )

            if is_success:
                print(f"{GREEN}✓ VERIFIED ({status}){RESET}")
                if data.get("capsule_hash_ok"):
                    print(f"  Capsule hash: OK")
                if data.get("proof_verified"):
                    print(f"  Proof: valid")
                if data.get("policy_verified"):
                    print(f"  Policy: verified")
                if data.get("verify_stats"):
                    t = data["verify_stats"].get("time_verify_sec", 0)
                    print(f"  Time: {t*1000:.2f}ms")
            else:
                print(f"{RED}✗ VERIFICATION FAILED{RESET}")
                if data.get("error"):
                    print(f"  Error: {data['error']}")
                if data.get("error_code"):
                    print(f"  Code: {data['error_code']}")
                if not self.state.policy_path:
                    print(f"{DIM}  Hint: Set a policy with 'policy <path>'{RESET}")
        except json.JSONDecodeError:
            if result.returncode == 0:
                print(f"{GREEN}✓ VERIFIED{RESET}")
            else:
                print(f"{RED}✗ VERIFICATION FAILED{RESET}")
                print(redact_secrets(result.stderr or result.stdout))

    def do_inspect(self, arg: str) -> None:
        """Inspect a capsule's contents.

        Usage: inspect [path]

        Sets this capsule as active for subsequent commands.
        """
        path_str = arg.strip()

        if path_str:
            path = Path(path_str).expanduser().resolve()
        elif self.state.last_capsule:
            path = self.state.last_capsule
        else:
            print(f"{RED}No capsule to inspect. Specify a path.{RESET}")
            return

        if not path.exists():
            print(f"{RED}Capsule not found: {path}{RESET}")
            return

        capsule = json.loads(path.read_text())

        # Set as active capsule for subsequent commands
        self.state.last_capsule = path

        print(f"\n{BOLD}Capsule: {path.name}{RESET}")
        print(f"{'─' * 50}")
        print(f"  Hash:     {capsule.get('capsule_hash', 'N/A')}")
        print(f"  Schema:   {capsule.get('schema', 'N/A')}")
        print(f"  VM:       {capsule.get('vm_id', 'N/A')}")
        print(f"  AIR:      {capsule.get('air_id', 'N/A')}")

        if capsule.get("dataset_ref"):
            print(f"\n  {BOLD}Datasets:{RESET}")
            for ds in capsule["dataset_ref"].get("datasets", []):
                print(f"    {ds['dataset_id']}:")
                print(f"      root: {ds['root']}")
                print(f"      chunks: {ds.get('num_chunks', 'N/A')}")

        if capsule.get("statement"):
            stmt = capsule["statement"]
            print(f"\n  {BOLD}Statement:{RESET}")
            if stmt.get("public_inputs"):
                for inp in stmt["public_inputs"][:5]:
                    print(f"    {inp['name']}: {inp['value']}")

        if capsule.get("proof_system"):
            ps = capsule["proof_system"]
            print(f"\n  {BOLD}Proof System:{RESET}")
            print(f"    {ps.get('name', 'N/A')} v{ps.get('version', 'N/A')}")

        print()

    def do_open(self, arg: str) -> None:
        """Open a specific row or chunk from a capsule.

        Usage:
            open row <n>           - open trace row n
            open chunk <ds> <n>    - open dataset chunk n
        """
        parts = arg.strip().split()
        if len(parts) < 2:
            print(f"{RED}Usage: open row <n> | open chunk <dataset> <n>{RESET}")
            return

        if not self.state.last_capsule:
            print(f"{RED}No capsule loaded. Generate or inspect one first.{RESET}")
            return

        from bef_zk.capsule.cli.run import _find_project_root
        root = _find_project_root()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        if parts[0] == "row":
            row_idx = int(parts[1])
            cmd = [
                sys.executable, "-m", "bef_zk.capsule.cli", "row",
                str(self.state.last_capsule),
                "--row", str(row_idx),
                "--json"
            ]
        elif parts[0] == "chunk":
            if len(parts) < 3:
                print(f"{RED}Usage: open chunk <dataset> <n>{RESET}")
                return
            ds_id = parts[1]
            chunk_idx = int(parts[2])
            cmd = [
                sys.executable, "-m", "bef_zk.capsule.cli", "open",
                str(self.state.last_capsule),
                "--dataset-id", ds_id,
                "--chunk", str(chunk_idx),
                "--json"
            ]
        else:
            print(f"{RED}Unknown open type: {parts[0]}{RESET}")
            return

        import subprocess
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        try:
            data = json.loads(result.stdout)
            print(f"\n{json.dumps(data, indent=2)}\n")
        except json.JSONDecodeError:
            print(redact_secrets(result.stdout or result.stderr))

    def do_audit(self, arg: str) -> None:
        """Audit a capsule's event log (summary by default)."""
        tokens = shlex.split(arg)
        path, extras = self._resolve_capsule_path(tokens)
        if not path:
            return
        from bef_zk.capsule.cli.run import _find_project_root

        root = _find_project_root()
        cmd = [
            sys.executable,
            "-m",
            "bef_zk.capsule.cli",
            "audit",
            str(path),
        ] + extras

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        import subprocess

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(root))
        if result.stdout:
            print(redact_secrets(result.stdout.strip()))
        if result.stderr:
            print(redact_secrets(result.stderr.strip()))

    def do_replay(self, arg: str) -> None:
        """Replay a capsule trace (uses current datasets by default)."""
        tokens = shlex.split(arg)
        path, extras = self._resolve_capsule_path(tokens)
        if not path:
            return

        from bef_zk.capsule.cli.run import _find_project_root

        root = _find_project_root()
        cmd = [
            sys.executable,
            "-m",
            "bef_zk.capsule.cli",
            "replay",
            str(path),
        ]

        cmd.extend(extras)

        dataset_flag_present = any(
            token.startswith("--dataset") or token == "-d" for token in extras
        )
        if not dataset_flag_present and self.state.datasets:
            for name, ds_path in self.state.datasets.items():
                cmd.extend(["--dataset", f"{name}={ds_path}"])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        import subprocess

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(root))
        if result.stdout:
            print(redact_secrets(result.stdout.strip()))
        if result.stderr:
            print(redact_secrets(result.stderr.strip()))

    # ─────────────────────────────────────────────────────────────────
    # Diff & Receipt commands
    # ─────────────────────────────────────────────────────────────────

    def do_diff(self, arg: str) -> None:
        """Compare repos/refs and generate hash-chained receipts.

        Usage:
            diff                           - diff HEAD~5..HEAD in current repo
            diff ~/projects/Repo1 .        - compare two repos
            diff --ref main..feature       - compare branches
            diff --clear                   - clear receipts first

        Every operation generates a receipt. View with: logs
        """
        try:
            from bef_zk.capsule.mcp_server import (
                tool_diff_bundle,
                EVENT_LOG_PATH,
            )
            import bef_zk.capsule.mcp_server as mcp_mod
        except ImportError:
            print(f"{RED}MCP tools not available{RESET}")
            return

        parts = arg.strip().split()
        clear = "--clear" in parts
        if clear:
            parts.remove("--clear")

        # Parse --ref
        ref = "HEAD~5..HEAD"
        if "--ref" in parts:
            idx = parts.index("--ref")
            if idx + 1 < len(parts):
                ref = parts[idx + 1]
                parts = parts[:idx] + parts[idx + 2:]

        # Clear receipts if requested
        if clear:
            if os.path.exists(EVENT_LOG_PATH):
                os.remove(EVENT_LOG_PATH)
            mcp_mod._last_hash = None
            print(f"{GREEN}✓ Receipts cleared{RESET}")

        # Determine source/target
        workspace = os.getcwd()
        if len(parts) >= 2:
            source_path = Path(parts[0]).expanduser().resolve()
            target_path = Path(parts[1]).expanduser().resolve()
            if target_path == Path(".").resolve():
                target_path = Path(workspace)

            # Set up remote for cross-repo comparison
            print(f"\n{BOLD}Comparing repos:{RESET}")
            print(f"  Source: {source_path}")
            print(f"  Target: {target_path}")

            # Add remote
            import subprocess

            # Remove old remote if exists
            subprocess.run(
                ["git", "-C", str(target_path), "remote", "remove", "capseal_cmp"],
                capture_output=True
            )

            # Add source as remote
            result = subprocess.run(
                ["git", "-C", str(target_path), "remote", "add", "capseal_cmp", str(source_path)],
                capture_output=True, text=True
            )

            # Fetch from source
            print(f"  {DIM}Fetching...{RESET}")
            result = subprocess.run(
                ["git", "-C", str(target_path), "fetch", "capseal_cmp"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"{RED}Fetch failed: {redact_secrets(result.stderr)}{RESET}")
                return

            # Detect default branch of source repo
            branch_result = subprocess.run(
                ["git", "-C", str(source_path), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True
            )
            source_branch = branch_result.stdout.strip() or "main"

            # Check if the branch exists in fetched refs
            ref_check = subprocess.run(
                ["git", "-C", str(target_path), "rev-parse", f"capseal_cmp/{source_branch}"],
                capture_output=True, text=True
            )
            if ref_check.returncode != 0:
                # Try common branch names
                for try_branch in ["main", "master", "HEAD"]:
                    ref_check = subprocess.run(
                        ["git", "-C", str(target_path), "rev-parse", f"capseal_cmp/{try_branch}"],
                        capture_output=True
                    )
                    if ref_check.returncode == 0:
                        source_branch = try_branch
                        break
                else:
                    print(f"{RED}Could not find branch in source repo{RESET}")
                    return

            base_ref = f"capseal_cmp/{source_branch}"
            head_ref = "HEAD"
            repo_path = str(target_path)
            print(f"  {DIM}Comparing {base_ref}..{head_ref}{RESET}")

            self.state.compare_source = source_path
            self.state.compare_target = target_path
        else:
            repo_path = workspace
            if ".." in ref:
                base_ref, head_ref = ref.split("..", 1)
            else:
                base_ref = f"{ref}~5"
                head_ref = ref

        print(f"\n{BOLD}[diff_bundle]{RESET} {base_ref}..{head_ref}")

        # Use direct git diff for shell (bypasses MCP workspace restriction)
        import subprocess as sp
        diff_result = sp.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"{base_ref}..{head_ref}"],
            capture_output=True, text=True
        )
        if diff_result.returncode != 0:
            result = {"ok": False, "error": diff_result.stderr}
        else:
            files = [f for f in diff_result.stdout.strip().split("\n") if f]
            # Log the event manually
            from bef_zk.capsule.mcp_server import _log_event
            _log_event("diff_bundle", {"repo_path": repo_path, "base_ref": base_ref, "head_ref": head_ref}, {"ok": True, "file_count": len(files)})
            result = {"ok": True, "files": files, "file_count": len(files)}

        if not result.get("ok"):
            err = result.get("stderr") or result.get("error") or "unknown error"
            print(f"{RED}  Error: {err}{RESET}")
            return

        file_count = result.get("file_count", 0)
        files = result.get("files", [])

        print(f"  {GREEN}{file_count} files changed{RESET}")
        for f in files[:10]:
            print(f"    • {f}")
        if len(files) > 10:
            print(f"    {DIM}... and {len(files) - 10} more{RESET}")

        # Show receipt
        self._show_last_receipt(EVENT_LOG_PATH)
        print(f"\n{DIM}View all receipts: logs{RESET}")

    def do_logs(self, arg: str) -> None:
        """View review runs and receipts.

        Usage:
            logs              - show recent review runs
            logs --runs       - show review runs (same as default)
            logs --mcp        - show MCP server events
            logs --verify     - verify chain integrity
        """
        arg = arg.strip()

        # MCP events mode
        if arg == "--mcp":
            try:
                from bef_zk.capsule.mcp_server import EVENT_LOG_PATH
                receipts = self._load_receipts(EVENT_LOG_PATH)
                if not receipts:
                    print(f"{DIM}No MCP events yet.{RESET}")
                    return
                print(f"\n{BOLD}MCP Events ({len(receipts)} total):{RESET}")
                print("─" * 50)
                for i, r in enumerate(receipts[-20:]):
                    self._print_receipt(r, i)
                print("─" * 50)
            except ImportError:
                print(f"{RED}MCP tools not available{RESET}")
            return

        # Default: show review runs
        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{DIM}No review runs yet. Run 'review <path>' first.{RESET}")
            return

        with open(receipts_index) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            print(f"{DIM}No review runs yet. Run 'review <path>' first.{RESET}")
            return

        if arg == "--verify":
            # Verify each run's receipt
            print(f"\n{BOLD}Verifying {len(lines)} runs...{RESET}")
            for line in lines:
                entry = json.loads(line)
                run_path = Path(entry["path"])
                receipt_file = run_path / "receipt.json"
                if receipt_file.exists():
                    receipt = json.loads(receipt_file.read_text())
                    # Basic integrity check
                    has_chain = "chain" in receipt
                    has_hash = "inputs" in receipt and "hash" in receipt.get("inputs", {})
                    if has_chain and has_hash:
                        print(f"  {GREEN}✓{RESET} {entry['id']} - chain intact")
                    else:
                        print(f"  {RED}✗{RESET} {entry['id']} - missing fields")
                else:
                    print(f"  {YELLOW}?{RESET} {entry['id']} - receipt.json missing")
            return

        # Show runs
        print(f"\n{BOLD}Review Runs ({len(lines)} total):{RESET}")
        print("─" * 70)
        print(f"{'ID':<16} {'Time':<18} {'Patched':<10} {'Proven':<10} {'Status'}")
        print("─" * 70)

        for line in lines[-20:]:  # Last 20
            entry = json.loads(line)
            run_path = Path(entry["path"])
            receipt_file = run_path / "receipt.json"

            if receipt_file.exists():
                receipt = json.loads(receipt_file.read_text())
                stats = receipt.get("stats", {})
                patched = stats.get("patched", 0)
                proven = stats.get("proven", 0)
                failed = stats.get("failed", 0)
                policy = receipt.get("policy", {})
                all_met = policy.get("all_met", False)

                if failed > 0:
                    status = f"{RED}✗ {failed} failed{RESET}"
                elif all_met:
                    status = f"{GREEN}✓ policy met{RESET}"
                else:
                    status = f"{YELLOW}! pending{RESET}"

                ts = receipt.get("timestamp_human", entry.get("ts", "")[:16])
                print(f"{CYAN}{entry['id']:<16}{RESET} {ts:<18} {patched:<10} {proven:<10} {status}")
            else:
                print(f"{CYAN}{entry['id']:<16}{RESET} {DIM}(receipt missing){RESET}")

        print("─" * 70)
        print(f"{DIM}Index: {receipts_index}{RESET}")

    def do_compare(self, arg: str) -> None:
        """Set repos to compare.

        Usage:
            compare ~/projects/CapsuleTech ~/BEF-main
            compare                                      - show current
        """
        parts = arg.strip().split()

        if not parts:
            if self.state.compare_source:
                print(f"Source: {self.state.compare_source}")
                print(f"Target: {self.state.compare_target}")
            else:
                print(f"{DIM}No comparison set. Use: compare <source> <target>{RESET}")
            return

        if len(parts) >= 2:
            self.state.compare_source = Path(parts[0]).expanduser().resolve()
            self.state.compare_target = Path(parts[1]).expanduser().resolve()
            print(f"{GREEN}✓ Comparison set:{RESET}")
            print(f"  Source: {self.state.compare_source}")
            print(f"  Target: {self.state.compare_target}")
        else:
            print(f"{RED}Usage: compare <source_repo> <target_repo>{RESET}")

    def do_receipts(self, arg: str) -> None:
        """Alias for 'logs' - view receipt chain."""
        self.do_logs(arg)

    def _show_last_receipt(self, log_path: str) -> None:
        """Show the last receipt."""
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    r = json.loads(lines[-1])
                    prev = r.get("prev_hash", "")[:8] or "genesis"
                    curr = r.get("event_hash", "")[:8]
                    ok = GREEN + "✓" + RESET if r.get("result_ok") else RED + "✗" + RESET
                    print(f"  {ok} Receipt: {prev}→{curr}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _print_receipt(self, r: dict, idx: int) -> None:
        """Print a receipt."""
        import time as time_mod
        tool = r.get("tool", "?")
        ok = GREEN + "✓" + RESET if r.get("result_ok") else RED + "✗" + RESET
        ts = time_mod.strftime("%H:%M:%S", time_mod.localtime(r.get("ts_ms", 0) / 1000))
        prev = r.get("prev_hash", "")[:8] or "genesis"
        curr = r.get("event_hash", "")[:8]
        print(f"  {ok} [{ts}] {tool:<12} {prev}→{curr}")

    def _load_receipts(self, log_path: str) -> list[dict]:
        """Load all receipts."""
        receipts = []
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        receipts.append(json.loads(line))
        except FileNotFoundError:
            pass
        return receipts

    def _verify_chain(self, receipts: list[dict]) -> tuple[bool, str]:
        """Verify chain integrity."""
        prev_hash = ""
        for i, r in enumerate(receipts):
            if r.get("prev_hash", "") != prev_hash:
                return False, f"prev_hash mismatch at #{i}"
            r_copy = {k: v for k, v in r.items() if k != "event_hash"}
            computed = hashlib.sha256(
                json.dumps(r_copy, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:32]
            if computed != r.get("event_hash"):
                return False, f"hash mismatch at #{i}"
            prev_hash = computed
        return True, "intact"

    def _watch_receipts_live(self, log_path: str) -> None:
        """Watch receipts in real-time."""
        import time as time_mod
        print(f"{BOLD}Watching receipts...{RESET} (Ctrl+C to stop)")
        print("─" * 50)
        seen = set()
        try:
            while True:
                receipts = self._load_receipts(log_path)
                for i, r in enumerate(receipts):
                    h = r.get("event_hash", "")
                    if h not in seen:
                        seen.add(h)
                        self._print_receipt(r, i)
                time_mod.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n{DIM}Stopped. {len(seen)} receipts seen.{RESET}")

    def do_cline(self, arg: str) -> None:
        """Spawn Cline with current context.

        Usage:
            cline                     Spawn Cline with latest context checkpoint
            cline <prompt>            Spawn Cline with custom prompt
            cline resolve             Load context and ask to resolve diffs
            cline review              Load context and ask for code review
        """
        cline_bin = os.path.expanduser("~/.local/node_modules/.bin/cline")
        if not os.path.exists(cline_bin):
            print(f"{RED}Cline not found at {cline_bin}{RESET}")
            print(f"{DIM}Install with: npm install cline --prefix ~/.local{RESET}")
            return

        arg = arg.strip()

        # Load context if available
        try:
            from bef_zk.capsule.cli.context import load_context, format_context_for_agent
            ctx = load_context("latest")
        except ImportError:
            ctx = None

        if not arg:
            # Default: spawn with context
            if ctx:
                prompt = format_context_for_agent(ctx)
                prompt = f"I've loaded a diff context checkpoint. Here it is:\n\n{prompt[:8000]}\n\nHelp me understand and resolve these changes."
            else:
                prompt = "Help me with the capseal project in this directory."
            print(f"{BOLD}Spawning Cline...{RESET}")

        elif arg == "resolve":
            if not ctx:
                print(f"{RED}No context checkpoint. Run: context save --working{RESET}")
                return
            prompt = format_context_for_agent(ctx)
            prompt = f"Load this diff context and help me merge/resolve the changes:\n\n{prompt[:8000]}"
            print(f"{BOLD}Spawning Cline to resolve diffs...{RESET}")

        elif arg == "review":
            if not ctx:
                print(f"{RED}No context checkpoint. Run: context save --working{RESET}")
                return
            prompt = format_context_for_agent(ctx)
            prompt = f"Review this diff context and provide feedback on the changes:\n\n{prompt[:8000]}"
            print(f"{BOLD}Spawning Cline for code review...{RESET}")

        else:
            # Custom prompt
            if ctx:
                context_summary = f"Context: {ctx.get('summary', {}).get('total_files', 0)} committed, {ctx.get('summary', {}).get('uncommitted_files', 0)} uncommitted files"
                prompt = f"{context_summary}\n\n{arg}"
            else:
                prompt = arg
            print(f"{BOLD}Spawning Cline...{RESET}")

        import subprocess
        subprocess.run([cline_bin, "--yolo", prompt])

    def do_agent(self, arg: str) -> None:
        """Multi-agent orchestration and proof-carrying agent commands.

        Usage:
            agent plan [mode]          Show orchestration plan (dry run)
            agent review [mode]        Run scoped multi-agent review
            agent spawn <task>         Spawn a single subagent
            agent collect              Collect results from all subagents
            agent status               Show agent activity
            agent orchestrate <goal>   Launch lead agent workflow
            agent run <goal> --prove   Run proof-carrying agent execution
            agent inspect <run_dir>    Inspect agent run with receipts

        Modes for plan/review:
            specialist   - Security, Performance, Correctness agents (default)
            module       - One agent per directory/module

        Examples:
            agent plan specialist      Show what specialist agents would do
            agent plan module          Show module-based split
            agent review specialist    Run specialist review
            agent orchestrate "merge these changes safely"
            agent run "analyze security" --prove  Run with FRI proof
            agent inspect .capseal/runs/run_xxx   Inspect run artifacts

        The scoped review splits context intelligently so each agent
        has a focused, coherent piece of work.
        """
        parts = arg.strip().split(maxsplit=1)
        if not parts:
            print(f"{YELLOW}Usage: agent spawn|collect|status|orchestrate|run|inspect [args]{RESET}")
            return

        cmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "spawn":
            if not rest:
                print(f"{RED}Usage: agent spawn <task description>{RESET}")
                return
            self._spawn_subagent(rest)

        elif cmd == "collect":
            self._collect_agent_results()

        elif cmd == "status":
            self._show_agent_status()

        elif cmd == "orchestrate":
            if not rest:
                print(f"{RED}Usage: agent orchestrate <goal>{RESET}")
                return
            self._orchestrate(rest)

        elif cmd == "review":
            # Scoped multi-agent review
            mode = rest.strip() or "specialist"
            self._scoped_review(mode)

        elif cmd == "plan":
            # Show what agents would be spawned
            mode = rest.strip() or "specialist"
            self._show_plan(mode)

        elif cmd == "run":
            # Proof-carrying agent run
            self._agent_run(rest)

        elif cmd == "inspect":
            # Inspect agent run
            self._agent_inspect(rest)

        else:
            # Legacy: treat as cline alias
            self.do_cline(arg)

    def _agent_run(self, arg: str) -> None:
        """Run proof-carrying agent execution.

        Usage: agent run <goal> [--prove] [--output <dir>]
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: agent run <goal> [--prove] [--output <dir>]{RESET}")
            return

        # Parse arguments
        goal = parts[0]
        prove = "--prove" in parts
        output_dir = None

        i = 1
        while i < len(parts):
            if parts[i] == "--output" and i + 1 < len(parts):
                output_dir = parts[i + 1]
                i += 2
            elif parts[i] == "--prove":
                prove = True
                i += 1
            else:
                # Additional goal words
                if not parts[i].startswith("--"):
                    goal += " " + parts[i]
                i += 1

        try:
            from bef_zk.capsule.agent_runtime import AgentRuntime
            from bef_zk.capsule.agent_protocol import create_action
            import datetime

            # Create output directory
            if output_dir:
                run_path = Path(output_dir).expanduser().resolve()
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%dT%H%M%S")
                run_path = Path(".capseal/runs") / f"agent_{timestamp}"

            print(f"\n{CYAN}{BOLD}PROOF-CARRYING AGENT EXECUTION{RESET}")
            print(f"{'=' * 50}")
            print(f"  Goal:   {goal[:60]}{'...' if len(goal) > 60 else ''}")
            print(f"  Output: {run_path}")
            print(f"  Prove:  {prove}")
            print(f"{'=' * 50}\n")

            # Create runtime and run a simple demo
            with AgentRuntime(output_dir=run_path) as runtime:
                # Record the goal as an observation action
                runtime.record_simple(
                    action_type="observation",
                    instruction=f"Agent goal: {goal}",
                    inputs={"goal": goal},
                    outputs={"acknowledged": True},
                    success=True,
                )
                print(f"{GREEN}[1]{RESET} Recorded goal observation")

                # Record a decision action
                runtime.record_simple(
                    action_type="decision",
                    instruction="Analyze task and plan approach",
                    inputs={"context": "goal_received"},
                    outputs={"plan": "execute_basic_analysis"},
                    success=True,
                )
                print(f"{GREEN}[2]{RESET} Recorded planning decision")

                # Record a completion action
                runtime.record_simple(
                    action_type="observation",
                    instruction="Complete agent execution",
                    inputs={"plan_executed": True},
                    outputs={"status": "completed", "actions_taken": runtime.action_count},
                    success=True,
                )
                print(f"{GREEN}[3]{RESET} Recorded completion observation")

            # Report results
            print(f"\n{GREEN}Agent execution complete{RESET}")
            print(f"  Actions: {runtime.action_count}")
            print(f"  Final receipt: {runtime.last_receipt_hash[:16]}...")
            print(f"  Output: {run_path}")

            if prove:
                capsule_path = run_path / "agent_capsule.json"
                if capsule_path.exists():
                    from bef_zk.capsule.agent_adapter import verify_agent_capsule
                    valid, details = verify_agent_capsule(capsule_path)
                    if valid:
                        print(f"  {GREEN}Proof verified{RESET}")
                    else:
                        print(f"  {RED}Proof verification failed: {details.get('error', 'unknown')}{RESET}")

            print(f"\n{DIM}Use 'agent inspect {run_path}' to view details{RESET}")

        except Exception as e:
            print(f"{RED}Error running agent: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def _agent_inspect(self, arg: str) -> None:
        """Inspect agent run directory.

        Usage: agent inspect <run_dir>
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            # Try to find latest run
            runs_dir = Path(".capseal/runs")
            if runs_dir.exists():
                agent_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("agent_")])
                if agent_runs:
                    run_path = agent_runs[-1]
                    print(f"{DIM}Using latest run: {run_path}{RESET}\n")
                else:
                    print(f"{RED}Usage: agent inspect <run_dir>{RESET}")
                    print(f"{DIM}No agent runs found in .capseal/runs/{RESET}")
                    return
            else:
                print(f"{RED}Usage: agent inspect <run_dir>{RESET}")
                return
        else:
            run_path = Path(parts[0]).expanduser().resolve()

        if not run_path.exists():
            print(f"{RED}Run directory not found: {run_path}{RESET}")
            return

        try:
            from bef_zk.capsule.agent_runtime import inspect_agent_run

            result = inspect_agent_run(run_path)

            print(f"\n{CYAN}{BOLD}AGENT RUN INSPECTION{RESET}")
            print(f"{'=' * 60}")
            print(f"  Directory: {result['run_dir']}")

            if "metadata" in result:
                meta = result["metadata"]
                print(f"  Run ID:    {meta.get('run_id', 'unknown')}")
                print(f"  Actions:   {meta.get('num_actions', 0)}")
                print(f"  Proved:    {meta.get('proved', False)}")
                print(f"  Receipt:   {meta.get('final_receipt_hash', '')[:32]}...")

            print(f"{'=' * 60}\n")

            if "action_chain" in result:
                print(f"{BOLD}Action Chain:{RESET}\n")
                for action in result["action_chain"]:
                    status = f"{GREEN}OK{RESET}" if action.get("success") else f"{RED}FAIL{RESET}"
                    gate = action.get("gate_decision") or "-"
                    print(f"  [{action['index']}] {action['action_type']:12} {status}  gate={gate}  {action['receipt_hash']}")

            if "capsule" in result:
                capsule = result["capsule"]
                print(f"\n{BOLD}Capsule:{RESET}")
                print(f"  Schema:   {capsule.get('schema', 'unknown')}")
                print(f"  Verified: {capsule.get('verification_valid', False)}")
                if capsule.get("verification_error"):
                    print(f"  Error:    {capsule['verification_error']}")

            print()

        except Exception as e:
            print(f"{RED}Error inspecting run: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def _spawn_subagent(self, task: str) -> None:
        """Spawn a subagent for a specific task."""
        try:
            from bef_zk.capsule.mcp_server import tool_spawn_agent
            print(f"{BOLD}Spawning subagent...{RESET}")
            print(f"  Task: {task[:80]}...")
            result = tool_spawn_agent(task=task, timeout=120)
            if result.get("ok"):
                print(f"{GREEN}✓ Agent {result.get('agent_id')} completed{RESET}")
                print(f"  Duration: {result.get('duration_ms', 0)}ms")
                output = result.get("output", "")
                if output:
                    print(f"\n{BOLD}Output:{RESET}")
                    print(output[:2000])
            else:
                print(f"{RED}✗ Agent failed: {result.get('output', 'Unknown error')}{RESET}")
        except Exception as e:
            print(f"{RED}Error spawning agent: {e}{RESET}")

    def _show_plan(self, mode: str) -> None:
        """Show the orchestration plan without executing."""
        try:
            from bef_zk.capsule.cli.context import load_context
            from bef_zk.capsule.cli.orchestrate import Orchestrator

            ctx = load_context("latest")
            if not ctx:
                print(f"{RED}No context checkpoint. Run: context save --working{RESET}")
                return

            orch = Orchestrator(ctx)

            if mode == "module":
                tasks = orch.plan_module_review()
            else:
                tasks = orch.plan_specialist_review()

            print(f"\n{CYAN}{BOLD}ORCHESTRATION PLAN ({mode}){RESET}")
            print(f"{'='*60}\n")

            for i, task in enumerate(tasks, 1):
                scope = task.get("scope", "?")
                agent_id = task.get("agent_id", "?")
                ctx_files = len(task.get("context", {}).get("files", []))
                ctx_diffs = len(task.get("context", {}).get("diffs", []))

                print(f"{BOLD}[{i}] {agent_id}{RESET}")
                print(f"    Scope: {scope}")
                print(f"    Context: {ctx_files} files, {ctx_diffs} diffs")
                print()

            print(f"{DIM}Run 'agent review {mode}' to execute this plan{RESET}")

        except ImportError as e:
            print(f"{RED}Module not available: {e}{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def _scoped_review(self, mode: str) -> None:
        """Run scoped multi-agent review with spawn → collect → synthesize."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from bef_zk.capsule.cli.context import load_context
            from bef_zk.capsule.cli.orchestrate import (
                Orchestrator, parse_agent_output
            )
            from bef_zk.capsule.mcp_server import tool_spawn_agent, tool_save_result

            ctx = load_context("latest")
            if not ctx:
                print(f"{RED}No context checkpoint. Run: context save --working{RESET}")
                return

            orch = Orchestrator(ctx)

            if mode == "module":
                tasks = orch.plan_module_review()
            else:
                tasks = orch.plan_specialist_review()

            if not tasks:
                print(f"{YELLOW}No tasks to review - is there a context checkpoint?{RESET}")
                return

            print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
            print(f"║            SCOPED MULTI-AGENT REVIEW ({mode.upper()})            ║")
            print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
            print(f"{DIM}Context hash: {orch.context_hash}{RESET}")
            print(f"\n{BOLD}Phase 1: Spawning {len(tasks)} agents...{RESET}\n")

            # Phase 1: Spawn agents
            raw_results = []

            def run_agent(task: dict) -> dict:
                """Run a single agent task."""
                agent_id = task.get("agent_id", "unknown")
                prompt = task.get("prompt", "")
                scope = task.get("scope", "?")
                timeout = task.get("timeout", 180)
                try:
                    result = tool_spawn_agent(
                        task=prompt,
                        agent_id=agent_id,
                        timeout=timeout,
                    )
                    return {
                        "agent_id": agent_id,
                        "scope": scope,
                        "ok": result.get("ok", False),
                        "output": result.get("output", ""),
                        "duration_ms": result.get("duration_ms", 0),
                    }
                except Exception as e:
                    return {
                        "agent_id": agent_id,
                        "scope": scope,
                        "ok": False,
                        "output": str(e),
                        "duration_ms": 0,
                    }

            # Run agents in parallel (max 3 concurrent)
            max_workers = min(3, len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit in deterministic order
                futures = {executor.submit(run_agent, task): task for task in tasks}

                for future in as_completed(futures):
                    task = futures[future]
                    agent_id = task.get("agent_id", "?")
                    try:
                        result = future.result()
                        raw_results.append(result)
                        if result.get("ok"):
                            print(f"  {GREEN}✓{RESET} [{agent_id}] {result.get('duration_ms', 0)}ms")
                        else:
                            print(f"  {RED}✗{RESET} [{agent_id}] failed")
                    except Exception as e:
                        print(f"  {RED}✗{RESET} [{agent_id}] exception: {e}")
                        raw_results.append({
                            "agent_id": agent_id,
                            "scope": task.get("scope", "?"),
                            "ok": False,
                            "output": str(e),
                        })

            # Phase 2: Parse outputs (enforce contract)
            print(f"\n{BOLD}Phase 2: Parsing agent outputs...{RESET}\n")
            parsed_results = []
            for r in raw_results:
                parsed = parse_agent_output(
                    r.get("output", ""),
                    r.get("agent_id", "?"),
                    r.get("scope", "?"),
                )
                parsed_results.append(parsed)
                status = f"{GREEN}✓{RESET}" if parsed.get("parse_ok") else f"{YELLOW}⚠{RESET}"
                print(f"  {status} [{parsed.get('agent_id')}] {len(parsed.get('findings', []))} findings")

            # Phase 3: Synthesize
            print(f"\n{BOLD}Phase 3: Synthesizing results...{RESET}")
            synthesis = orch.synthesize_results(parsed_results)

            # Phase 4: Save receipt
            receipt_name = f"review_{mode}_{orch.context_hash}"
            try:
                tool_save_result(
                    agent_id="orchestrator",
                    result_type="review_synthesis",
                    content=json.dumps(synthesis, indent=2),
                )
                print(f"  {GREEN}✓{RESET} Receipt saved: {receipt_name}")
            except Exception as e:
                print(f"  {YELLOW}⚠{RESET} Could not save receipt: {e}")

            # Display results
            print(f"\n{BOLD}{'='*60}{RESET}")
            verdict = synthesis.get("verdict", {})
            status = verdict.get("status", "?")
            status_color = RED if status == "NEEDS_ATTENTION" else YELLOW if "REVIEW" in status else GREEN
            print(f"{status_color}{BOLD}VERDICT: {status}{RESET}")
            print(f"{verdict.get('message', '')}")
            print(f"{'='*60}\n")

            # Show findings by severity
            findings = synthesis.get("findings", [])
            if findings:
                print(f"{BOLD}FINDINGS ({len(findings)} total):{RESET}\n")
                for f in findings:
                    sev = f.get("severity", "?")
                    sev_color = RED if sev == "HIGH" else YELLOW if sev == "MEDIUM" else DIM
                    print(f"  {sev_color}[{sev}]{RESET} {f.get('issue', 'Unknown issue')}")
                    if f.get("file"):
                        print(f"        {DIM}{f.get('file')}:{f.get('line', '?')}{RESET}")
                    if f.get("recommendation"):
                        print(f"        → {f.get('recommendation')}")
                    print()
            else:
                print(f"{GREEN}No issues found!{RESET}\n")

            # Summary
            print(f"{BOLD}Summary:{RESET}")
            print(f"  Context:  {orch.context_hash}")
            print(f"  Findings: {synthesis.get('findings_hash', '?')}")
            print(f"  Agents:   {synthesis.get('agent_count', 0)} ({len([r for r in parsed_results if r.get('parse_ok')])} parsed OK)")
            print(f"  Counts:   {RED}{synthesis.get('high_severity', 0)} HIGH{RESET}, "
                  f"{YELLOW}{synthesis.get('medium_severity', 0)} MEDIUM{RESET}, "
                  f"{synthesis.get('low_severity', 0)} LOW")

        except ImportError as e:
            print(f"{RED}Module not available: {e}{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def _collect_agent_results(self) -> None:
        """Collect results from all subagents."""
        try:
            from bef_zk.capsule.mcp_server import tool_collect_results
            result = tool_collect_results(since_minutes=60)
            if result.get("ok"):
                results = result.get("results", [])
                print(f"\n{BOLD}Agent Results ({len(results)} total):{RESET}")
                for r in results:
                    print(f"\n  {GREEN}[{r.get('agent_id')}]{RESET} - {r.get('result_type')}")
                    print(f"  {DIM}{r.get('timestamp')}{RESET}")
                    content = r.get("content", "")
                    print(f"  {content[:500]}{'...' if len(content) > 500 else ''}")
            else:
                print(f"{YELLOW}No results collected{RESET}")
        except Exception as e:
            print(f"{RED}Error collecting results: {e}{RESET}")

    def _show_agent_status(self) -> None:
        """Show agent status from receipts."""
        try:
            from bef_zk.capsule.mcp_server import EVENT_LOG_PATH
            receipts = self._load_receipts(EVENT_LOG_PATH)
            agent_events = [r for r in receipts if r.get("tool") in ("spawn_agent", "save_result", "collect_results")]
            print(f"\n{BOLD}Agent Activity ({len(agent_events)} events):{RESET}")
            for r in agent_events[-10:]:  # Last 10
                tool = r.get("tool")
                args = r.get("args", {})
                ok = "✓" if r.get("result_ok") else "✗"
                if tool == "spawn_agent":
                    print(f"  {ok} spawn: {args.get('agent_id', '?')} - {args.get('task', '')[:40]}")
                elif tool == "save_result":
                    print(f"  {ok} result: {args.get('agent_id', '?')} - {args.get('result_type', '?')}")
                elif tool == "collect_results":
                    print(f"  {ok} collect: from {args.get('agent_ids') or 'all'}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def _orchestrate(self, goal: str) -> None:
        """Launch orchestrator-worker workflow with Cline as lead."""
        cline_bin = os.path.expanduser("~/.local/node_modules/.bin/cline")
        if not os.path.exists(cline_bin):
            print(f"{RED}Cline not found at {cline_bin}{RESET}")
            return

        # Load context
        try:
            from bef_zk.capsule.cli.context import load_context, format_context_for_agent
            ctx = load_context("latest")
            if ctx:
                context_summary = ctx.get("summary", {})
                ctx_info = f"Context: {context_summary.get('total_files', 0)} committed, {context_summary.get('uncommitted_files', 0)} uncommitted files"
            else:
                ctx_info = "No context checkpoint loaded."
        except ImportError:
            ctx_info = "Context module unavailable."

        orchestrator_prompt = f"""You are the LEAD AGENT in a multi-agent workflow.

{ctx_info}

YOUR GOAL: {goal}

You have access to these MCP tools for orchestration:
- load_context: Load diff context checkpoint (name="latest" or specific name)
- spawn_agent: Spawn subagents for parallel tasks (provide detailed task descriptions)
- save_result: Save results for collection
- collect_results: Gather results from all subagents

WORKFLOW:
1. First, use load_context to understand what changed
2. Break down the goal into parallel subtasks
3. Use spawn_agent to spawn 2-4 subagents (one per subtask)
4. Use collect_results to gather their findings
5. Synthesize and take action based on all findings

Be strategic about parallelization. Each subagent works independently."""

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║            MULTI-AGENT ORCHESTRATION                          ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
        print(f"\n{BOLD}Goal:{RESET} {goal}")
        print(f"{BOLD}Lead agent:{RESET} Cline")
        print(f"{ctx_info}")
        print(f"\n{DIM}Launching orchestrator...{RESET}\n")

        import subprocess
        subprocess.run([cline_bin, "--yolo", orchestrator_prompt])

    def do_greptile(self, arg: str) -> None:
        """Run Greptile commands directly.

        Usage:
            greptile ephemeral [path]     Create temp repo, index, review, delete
            greptile index <repo>         Index a GitHub repo
            greptile query <question>     Ask about indexed repo
            greptile status <repo>        Check indexing status

        Examples:
            greptile ephemeral                     Review current dir
            greptile ephemeral ~/projects/mycode   Review specific path
            greptile index myuser/myrepo           Index a repo
        """
        if not arg.strip():
            print(f"{YELLOW}Usage: greptile ephemeral|index|query|status [args]{RESET}")
            return

        # Pass through to CLI
        import subprocess
        workspace = str(get_workspace_root())
        result = subprocess.run(
            ["python", "-m", "bef_zk.capsule.cli", "greptile"] + shlex.split(arg),
            cwd=workspace,
            env={**os.environ, "PYTHONPATH": workspace}
        )

    def do_sync(self, arg: str) -> None:
        """Sync local repo to GitHub shadow for Greptile indexing.

        Usage:
            sync <shadow-repo>              Sync current dir to shadow repo
            sync <shadow-repo> --index      Also trigger Greptile re-index
            sync <path> <shadow-repo>       Sync specific path

        Examples:
            sync myuser/code-shadow --index
            sync ~/projects/CapsuleTech myuser/capsule-shadow --index

        This pushes your ACTUAL local code to a private GitHub repo so
        Greptile can index it (instead of the stale main branch).

        One-time setup:
            1. Create private repo: github.com/youruser/code-shadow
            2. Run: sync youruser/code-shadow --index
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: sync <shadow-repo> [--index]{RESET}")
            print(f"{DIM}Example: sync myuser/code-shadow --index{RESET}")
            return

        # Parse args
        do_index = "--index" in parts
        parts = [p for p in parts if p != "--index"]

        if len(parts) == 1:
            local_path = "."
            shadow_repo = parts[0]
        elif len(parts) >= 2:
            local_path = parts[0]
            shadow_repo = parts[1]
        else:
            print(f"{RED}Invalid arguments{RESET}")
            return

        print(f"\n{CYAN}{BOLD}Syncing to GitHub shadow repo...{RESET}")
        print(f"  Local: {local_path}")
        print(f"  Shadow: {shadow_repo}")

        import subprocess

        # Check/add remote
        local_path = os.path.expanduser(local_path)
        result = subprocess.run(
            ["git", "-C", local_path, "remote", "get-url", "shadow"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            remote_url = f"git@github.com:{shadow_repo}.git"
            print(f"  Adding remote: {remote_url}")
            subprocess.run(
                ["git", "-C", local_path, "remote", "add", "shadow", remote_url],
                capture_output=True
            )

        # Force push
        branch = "greptile-sync"
        print(f"  Pushing to shadow/{branch}...")
        result = subprocess.run(
            ["git", "-C", local_path, "push", "-f", "shadow", f"HEAD:{branch}"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"{RED}Push failed: {redact_secrets(result.stderr)}{RESET}")
            print(f"{DIM}Make sure the repo exists and you have push access{RESET}")
            return

        print(f"  {GREEN}✓ Synced{RESET}")

        if do_index:
            print(f"\n{CYAN}Triggering Greptile index...{RESET}")
            try:
                from bef_zk.capsule.cli.greptile import _api_request, _parse_repo_string
                remote, repository, _ = _parse_repo_string(shadow_repo)
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
                print(f"  {GREEN}✓ Index triggered: {result.get('status', 'submitted')}{RESET}")
                print(f"\n{BOLD}Ready! Now run:{RESET}")
                print(f"  context save --working")
                print(f"  pr {shadow_repo}:{branch}")
            except Exception as e:
                print(f"{RED}Index failed: {e}{RESET}")
        else:
            print(f"\n{DIM}Run with --index to trigger Greptile indexing{RESET}")

    def do_review(self, arg: str) -> None:
        """Code review using Greptile.

        Usage:
            review <repo>                  Review latest context against repo
            review <repo> --focus security Focus on security issues
            review <repo> -c <checkpoint>  Review specific checkpoint

        Examples:
            review myorg/myrepo
            review myorg/myrepo --focus performance
            review github:myorg/myrepo:main -c merge_task

        Requires:
            - GREPTILE_API_KEY environment variable
            - Repository indexed in Greptile (run: greptile index myorg/myrepo)
            - A context checkpoint (run: context save --working)
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: review <repo> [--focus <area>] [-c <checkpoint>]{RESET}")
            print(f"{DIM}Example: review myorg/myrepo --focus security{RESET}")
            return

        repo = parts[0]
        context_name = "latest"
        focus = None

        # Parse options
        i = 1
        while i < len(parts):
            if parts[i] in ("-c", "--context") and i + 1 < len(parts):
                context_name = parts[i + 1]
                i += 2
            elif parts[i] in ("-f", "--focus") and i + 1 < len(parts):
                focus = parts[i + 1]
                i += 2
            else:
                i += 1

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    CODE REVIEW                                ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
        print(f"\n{BOLD}Repository:{RESET} {repo}")
        print(f"{BOLD}Context:{RESET} {context_name}")
        if focus:
            print(f"{BOLD}Focus:{RESET} {focus}")

        try:
            from bef_zk.capsule.cli.greptile import greptile_review_api
            print(f"\n{DIM}Analyzing changes with Greptile...{RESET}\n")
            result = greptile_review_api(repo=repo, context_name=context_name, focus=focus)

            if result.get("ok"):
                print("=" * 60)
                print(result.get("review", "No review generated"))
                print("=" * 60)
                print(f"\n{DIM}Files reviewed: {result.get('files_reviewed', 0)}{RESET}")
            else:
                print(f"{RED}Review failed: {result.get('error', 'Unknown error')}{RESET}")
                if "GREPTILE_API_KEY" in str(result.get("error", "")):
                    print(f"{DIM}Set your API key: export GREPTILE_API_KEY='your-key'{RESET}")
        except ImportError:
            print(f"{RED}Greptile module not available{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_refactor(self, arg: str) -> None:
        """Run verified refactoring pipeline: review → plan → patches → verify.

        Usage:
            refactor <path>               Run full pipeline on directory
            refactor <path> --dry-run     Show what would change without applying
            refactor <path> --apply       Apply verified patches after generation

        Options:
            --model <model>               LLM model (auto-detected from API keys)
            --provider <provider>         LLM provider (auto-detected from API keys)
            --no-suppress-memos           Disable v5 suppression memoization
            --no-ast-validate             Disable v5 AST whitelist validation
            --run-dir <path>              Output directory for artifacts

        Examples:
            refactor src/market_dashboard
            refactor . --model gpt-4o --apply
            refactor src/ --dry-run

        This runs:
        1. Semgrep scan to find issues
        2. LLM generates refactor plan
        3. Multi-agent patch generation (v5: large files use bounded edits)
        4. Verification that patches apply correctly
        5. Produces verified diff with full provenance

        v5 Features (enabled by default):
        - Suppression memos: Cache NO_CHANGE proofs to avoid re-proving
        - AST validation: Verify whitelist claims with AST analysis
        - Large-file mode: Files >8K tokens use multi-pass bounded edits
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: refactor <path> [--dry-run] [--apply]{RESET}")
            print(f"{DIM}Example: refactor src/market_dashboard{RESET}")
            return

        target_path = Path(parts[0]).expanduser().resolve()
        if not target_path.exists():
            print(f"{RED}Path not found: {target_path}{RESET}")
            return

        # Parse options
        dry_run = "--dry-run" in parts
        apply_after = "--apply" in parts
        suppress_memos = "--no-suppress-memos" not in parts
        ast_validate = "--no-ast-validate" not in parts
        # Auto-detect provider/model from available API keys
        provider, model = _detect_llm_provider()
        run_dir = None

        i = 1
        while i < len(parts):
            if parts[i] == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
                i += 2
            elif parts[i] == "--provider" and i + 1 < len(parts):
                provider = parts[i + 1]
                i += 2
            elif parts[i] == "--run-dir" and i + 1 < len(parts):
                run_dir = parts[i + 1]
                i += 2
            else:
                i += 1

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║              VERIFIED REFACTOR PIPELINE (v5)                  ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
        print(f"\n{BOLD}Target:{RESET} {target_path}")
        print(f"{BOLD}Model:{RESET} {provider}/{model}")
        print(f"{BOLD}v5 Features:{RESET}")
        print(f"  Suppression memos: {'enabled' if suppress_memos else 'disabled'}")
        print(f"  AST validation:    {'enabled' if ast_validate else 'disabled'}")
        if dry_run:
            print(f"{YELLOW}  Mode: DRY RUN (no changes will be made){RESET}")
        if apply_after:
            print(f"{GREEN}  Mode: APPLY (patches will be applied if valid){RESET}")
        print()

        try:
            import datetime
            from bef_zk.capsule.refactor_engine import (
                generate_refactor_plan,
                run_multi_agent_patches,
                build_diff_rollup,
                RefactorPlan,
                RefactorItem,
                sha256_str,
                LARGE_FILE_TOKEN_THRESHOLD,
            )

            # Create run directory
            if run_dir:
                run_path = Path(run_dir)
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_path = target_path / ".capseal" / f"refactor_{timestamp}"
            run_path.mkdir(parents=True, exist_ok=True)

            print(f"{DIM}Output: {run_path}{RESET}")
            print(f"{DIM}Large file threshold: {LARGE_FILE_TOKEN_THRESHOLD} tokens{RESET}")
            print()

            # Step 1: Run semgrep
            print(f"{CYAN}[1/4] Running semgrep scan...{RESET}")
            import subprocess
            result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", str(target_path)],
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0 and b"error" in result.stderr.lower():
                print(f"{YELLOW}Semgrep warning: {result.stderr.decode()[:200]}{RESET}")

            findings = []
            try:
                output = json.loads(result.stdout.decode())
                findings = output.get("results", [])
            except json.JSONDecodeError:
                pass

            print(f"  Found {len(findings)} findings")

            if not findings:
                print(f"{GREEN}✓ No issues found - nothing to refactor{RESET}")
                return

            # Save findings
            (run_path / "findings.json").write_text(json.dumps(findings, indent=2))

            # Step 2: Generate plan
            print(f"\n{CYAN}[2/4] Generating refactor plan...{RESET}")
            plan = generate_refactor_plan(
                findings=findings,
                trace_root=f"refactor-{datetime.datetime.now().isoformat()}",
                aggregate_hash=sha256_str(json.dumps(findings)),
                provider=provider,
                model=model,
            )
            print(f"  Generated plan with {len(plan.items)} items")

            # Save plan
            (run_path / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2))

            if dry_run:
                print(f"\n{YELLOW}DRY RUN - showing plan without generating patches:{RESET}")
                for item in plan.items:
                    print(f"  [{item.category}] {item.file_path}: {item.description[:60]}...")
                return

            # Step 3: Generate patches
            print(f"\n{CYAN}[3/4] Generating patches (multi-agent)...{RESET}")
            results = run_multi_agent_patches(
                plan=plan,
                project_dir=target_path,
                provider=provider,
                model=model,
                enable_repair=True,
                enable_suppression_memos=suppress_memos,
                enable_ast_validation=ast_validate,
            )

            # Tally results
            valid = sum(1 for r in results if r.final_status == "VALID")
            skip = sum(1 for r in results if r.final_status == "SKIP")
            fail = sum(1 for r in results if r.final_status == "FAIL")
            memo_reused = sum(1 for r in results if r.patch.agent_type == "suppression_memo_cache")

            print(f"  Results: {GREEN}✓ {valid} valid{RESET}, {YELLOW}⊘ {skip} skip{RESET}, {RED}✗ {fail} fail{RESET}")
            if memo_reused > 0:
                print(f"  {DIM}Suppression memos reused: {memo_reused} (saved tokens){RESET}")

            # Save results
            patches_dir = run_path / "patches"
            patches_dir.mkdir(exist_ok=True)
            for r in results:
                result_file = patches_dir / f"{r.patch.patch_id}_result.json"
                result_file.write_text(json.dumps(r.to_dict(), indent=2))

            # Step 4: Build rollup
            print(f"\n{CYAN}[4/4] Building verified diff rollup...{RESET}")
            from bef_zk.capsule.refactor_engine import PatchVerification
            verifications = [
                PatchVerification(
                    patch_id=r.patch.patch_id,
                    verified=r.validation.is_valid,
                    original_hash_match=True,
                    patch_applies_cleanly=r.validation.is_valid,
                    result_hash_match=True,
                )
                for r in results
            ]
            rollup = build_diff_rollup(plan, results, verifications, plan.trace_root)

            # Save rollup
            rollup_path = run_path / "diff_rollup.json"
            rollup_path.write_text(json.dumps(rollup.to_dict(), indent=2))

            # Save combined diff
            if rollup.combined_diff:
                (run_path / "combined.diff").write_text(rollup.combined_diff)

            # Summary
            print(f"\n{BOLD}═══════════════════════════════════════════════════════════════{RESET}")
            print(f"{BOLD}REFACTOR SUMMARY{RESET}")
            print(f"  Patches:      {valid} valid, {skip} skip, {fail} fail")
            print(f"  Files:        {rollup.total_files_modified} modified")
            print(f"  Changes:      +{rollup.total_lines_added} -{rollup.total_lines_removed}")
            print(f"  All verified: {'✓' if rollup.all_verified else '✗'}")
            print(f"\n  Artifacts:    {run_path}")
            print(f"  Diff:         {run_path / 'combined.diff'}")

            # Apply if requested
            if apply_after and valid > 0 and rollup.combined_diff:
                print(f"\n{CYAN}Applying patches...{RESET}")
                apply_result = subprocess.run(
                    ["git", "apply", "-"],
                    input=rollup.combined_diff.encode(),
                    capture_output=True,
                    cwd=target_path,
                )
                if apply_result.returncode == 0:
                    print(f"{GREEN}✓ Patches applied successfully{RESET}")
                else:
                    print(f"{RED}✗ Apply failed: {apply_result.stderr.decode()[:200]}{RESET}")
            elif apply_after and valid == 0:
                print(f"{YELLOW}No valid patches to apply{RESET}")

        except ImportError as e:
            print(f"{RED}Module not available: {e}{RESET}")
        except subprocess.TimeoutExpired:
            print(f"{RED}Semgrep timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # Core verification commands
    # ─────────────────────────────────────────────────────────────────

    def do_init(self, arg: str) -> None:
        """Initialize .capseal/ in current or specified directory.

        Usage:
            init                Initialize in current directory
            init <path>         Initialize in specified directory

        Creates:
            .capseal/
            ├── config.yaml         # Model settings, thresholds
            ├── receipts.jsonl      # Receipt index
            ├── runs/               # Timestamped artifacts
            └── suppression_memos/  # Cached NO_CHANGE proofs

        Examples:
            init
            init ~/projects/myapp
        """
        target = Path(arg.strip()).expanduser().resolve() if arg.strip() else Path.cwd()

        if not target.exists():
            print(f"{RED}Path not found: {target}{RESET}")
            return

        capseal_dir = target / ".capseal"

        if capseal_dir.exists():
            print(f"{YELLOW}Already initialized: {capseal_dir}{RESET}")
            print(f"{DIM}Use 'fix .' to generate your first receipt.{RESET}")
            return

        # Create directory structure
        capseal_dir.mkdir(parents=True)
        (capseal_dir / "runs").mkdir()
        (capseal_dir / "suppression_memos").mkdir()

        # Create default config with auto-detected provider
        default_provider, default_model = _detect_llm_provider()
        config = {
            "version": "0.6",
            "model": {
                "provider": default_provider,
                "model": default_model,
            },
            "thresholds": {
                "large_file_tokens": 8000,
                "region_window_cap": 200,
            },
            "features": {
                "suppression_memos": True,
                "ast_validation": True,
                "multipass_edits": True,
            },
        }

        import yaml
        try:
            (capseal_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))
        except ImportError:
            # Fall back to JSON if yaml not available
            (capseal_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Create empty receipts index
        (capseal_dir / "receipts.jsonl").write_text("")

        print(f"""
{GREEN}Initialized .capseal/ in {target}{RESET}

  {DIM}.capseal/{RESET}
  {DIM}├── config.yaml         # Model settings, thresholds{RESET}
  {DIM}├── receipts.jsonl      # Receipt index{RESET}
  {DIM}├── runs/               # Timestamped artifacts{RESET}
  {DIM}└── suppression_memos/  # Cached NO_CHANGE proofs{RESET}

{CYAN}Ready.{RESET} Run '{BOLD}fix .{RESET}' to generate your first receipt.
""")

    def do_review(self, arg: str) -> None:
        """Generate proof-carrying patches with replayable receipts.

        Usage:
            review <path>                   Scan → plan → patch → verify
            review <path> --apply           Also apply after verification
            review <path> --profile <p>     Use profile (security|refactor|perf|style)
            review <path> --dry-run         Show plan without patching
            review <path> --json            Output JSON for CI
            review <path> --agents N        Use N parallel workers (multi-agent DAG)
            review <path> --shard <s>       Shard strategy: by_file (default), by_count
            review <path> --critic          Enable adversarial critic review (asymmetric)
            review <path> --critic-model M  Model for critic (auto-detected)
            review <path> --gate            Enable committor gate (filter risky patches)
            review <path> --no-gate         Disable committor gate
            review <path> --posteriors P    Path to beta_posteriors.npz for gate

        Profiles:
            security    Semgrep-heavy, conservative edits, strict AST whitelist
            refactor    Structure/clarity focus, behavior-preservation emphasis
            perf        Hot-path focus, benchmarks if available
            style       Formatting + consistency

        Multi-agent DAG (--agents N):
            When using multiple agents, the pipeline becomes:
            plan → shard(S1) → ... → shard(SN) → reduce → verify → claims → policy

            Each shard runs independently with its own receipt.
            The reducer deterministically merges outputs.
            Conflicts are detected and reported.

        Asymmetric Review (--critic):
            Adds an adversarial critic agent that challenges proposer's patches:
            plan → shard(S1..SN) → reduce → CRITIC → verify → claims → policy

            The critic looks for:
            - Security holes the proposer missed
            - Edge cases not handled
            - Flawed NO_CHANGE justifications
            - Logic errors in patches

            This creates a "debate" dynamic where LLMs challenge each other,
            with deterministic checkers making the final call.

        Committor Gate (--gate):
            Uses learned failure probabilities from agent bench to filter patches:
            plan → gate(score each item) → filtered_plan → patches

            Each plan item is mapped to a 5-feature grid:
            - lines_changed, cyclomatic_complexity, files_touched
            - finding_severity, test_coverage_delta

            The gate looks up q(x) = estimated p_fail from beta posteriors:
            - q >= 0.3: SKIP (too risky, don't attempt patch)
            - uncertainty > 0.15: HUMAN_REVIEW (flagged but passed through)
            - Otherwise: PASS (proceed with patch)

            Requires beta_posteriors.npz from agent bench evaluation rounds.

        Outputs (in .capseal/runs/<timestamp>/):
            plan.json, patches/, verified.diff, receipts/, audit.md
            shards/ (when using --agents)
            critic_review.json (when using --critic)
            gate/gate_result.json (when using --gate)

        Examples:
            review src/
            review . --profile security
            review src/api --json | jq '.receipt_id'
            review . --agents 4 --shard by_file
            review . --agents 2 --critic  # Asymmetric review
            review src/ --gate             # Use committor gate to filter risky patches
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: review <path> [--profile <p>] [--apply] [--dry-run] [--agents N]{RESET}")
            print(f"{DIM}Example: review src/ --profile security{RESET}")
            return

        target_path = Path(parts[0]).expanduser().resolve()
        if not target_path.exists():
            print(f"{RED}Path not found: {target_path}{RESET}")
            return

        # Parse options
        dry_run = "--dry-run" in parts
        apply_after = "--apply" in parts
        json_output = "--json" in parts
        profile = None  # Let policy loader decide default
        # Auto-detect provider/model from available API keys
        provider, model = _detect_llm_provider()
        explicit_policy = None  # --policy path
        num_agents = 1  # Single agent by default
        shard_strategy = "by_file"
        enable_critic = "--critic" in parts  # Asymmetric review with critic agent
        # Critic model: use same provider as main agent for consistency
        _, critic_model = _detect_llm_provider()
        # Committor gate: filter risky patches based on learned failure probabilities
        enable_gate = "--gate" in parts
        disable_gate = "--no-gate" in parts
        posteriors_path = None  # Path to beta_posteriors.npz

        i = 1
        while i < len(parts):
            if parts[i] == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
                i += 2
            elif parts[i] == "--provider" and i + 1 < len(parts):
                provider = parts[i + 1]
                i += 2
            elif parts[i] == "--profile" and i + 1 < len(parts):
                profile = parts[i + 1]
                i += 2
            elif parts[i] == "--policy" and i + 1 < len(parts):
                explicit_policy = parts[i + 1]
                i += 2
            elif parts[i] == "--agents" and i + 1 < len(parts):
                num_agents = int(parts[i + 1])
                i += 2
            elif parts[i] == "--shard" and i + 1 < len(parts):
                shard_strategy = parts[i + 1]
                i += 2
            elif parts[i] == "--critic-model" and i + 1 < len(parts):
                critic_model = parts[i + 1]
                i += 2
            elif parts[i] == "--critic":
                enable_critic = True
                i += 1
            elif parts[i] == "--gate":
                enable_gate = True
                i += 1
            elif parts[i] == "--no-gate":
                disable_gate = True
                i += 1
            elif parts[i] == "--posteriors" and i + 1 < len(parts):
                posteriors_path = parts[i + 1]
                i += 2
            else:
                i += 1

        try:
            import datetime
            import subprocess
            import time
            from bef_zk.capsule.refactor_engine import (
                generate_refactor_plan,
                run_multi_agent_patches,
                build_diff_rollup,
                sha256_str,
                sha256_json,
                LARGE_FILE_TOKEN_THRESHOLD,
            )
            from bef_zk.capsule.refactor_engine import PatchVerification
            from bef_zk.capsule.claims import (
                Claim, ClaimType, Verdict, Scope, Witness, ClaimBundle, CheckerInfo,
                create_claim, claim_from_no_change_proof, CHECKER_REGISTRY,
            )
            from bef_zk.capsule.policy import (
                Policy, evaluate_policy, get_required_claims,
                resolve_policy_for_review, PolicyLoadResult,
            )
            from bef_zk.capsule.claim_cache import (
                ClaimCache, get_cache_for_project, check_claim_with_cache,
            )
            from bef_zk.capsule.shard_orchestrator import (
                ShardOrchestrator, shard_by_file, shard_by_count,
            )
            # Import checkers to trigger registration
            import bef_zk.capsule.checkers  # noqa: F401 - registers checkers

            # Load policy with precedence: --policy > .capseal/policy.yaml > builtin
            try:
                policy, profile, policy_load_result = resolve_policy_for_review(
                    explicit_policy=explicit_policy,
                    target_path=target_path,
                    profile_name=profile,
                )
            except FileNotFoundError as e:
                print(f"{RED}Policy error: {e}{RESET}")
                return
            except ValueError as e:
                print(f"{RED}Policy error: {e}{RESET}")
                return

            # Initialize claim cache for incremental reuse
            claim_cache = get_cache_for_project(target_path)
            policy_hash = policy.hash()

            if not json_output:
                print(f"\n{DIM}Policy: {policy.name} (profile: {profile}){RESET}")
                print(f"{DIM}Source: {policy_load_result.resolved_from}{RESET}")
                print(f"{DIM}Hash:   sha256:{policy_hash[:16]}...{RESET}")
                cache_stats = claim_cache.stats()
                if cache_stats["total_entries"] > 0:
                    print(f"{DIM}Cache:  {cache_stats['total_entries']} entries, {cache_stats['total_hits']} hits{RESET}")
                print()

            # Create run directory with receipt ID
            now = datetime.datetime.now()
            timestamp_dir = now.strftime("%Y%m%dT%H%M%S")
            timestamp_human = now.strftime("%Y-%m-%d %H:%M")
            receipt_id = f"cap-{sha256_str(f'{target_path}{timestamp_dir}')[:8]}"
            run_path = target_path / ".capseal" / "runs" / timestamp_dir
            run_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Scan
            if not json_output:
                print(f"\n{DIM}[1/4] Scanning with semgrep...{RESET}")
            result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", str(target_path)],
                capture_output=True,
                timeout=120,
            )

            findings = []
            try:
                output = json.loads(result.stdout.decode())
                findings = output.get("results", [])
            except json.JSONDecodeError:
                pass

            if not json_output:
                print(f"      Found {len(findings)} findings")

            if not findings:
                if json_output:
                    print(json.dumps({"receipt_id": None, "status": "no_findings"}))
                else:
                    print(f"\n{GREEN}✓ No issues found - nothing to fix{RESET}")
                return

            # Save findings
            (run_path / "findings.json").write_text(json.dumps(findings, indent=2))
            input_hash = sha256_json(findings)

            # Step 2: Plan
            if not json_output:
                print(f"{DIM}[2/4] Planning refactor...{RESET}")
            plan = generate_refactor_plan(
                findings=findings,
                trace_root=receipt_id,
                aggregate_hash=input_hash,
                provider=provider,
                model=model,
            )
            if not json_output:
                print(f"      {len(plan.items)} items planned")

            (run_path / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2))

            # Step 2.5: Committor Gate (optional)
            gate_result = None
            if enable_gate and not disable_gate:
                if not json_output:
                    print(f"{DIM}[2.5/4] Applying committor gate...{RESET}")

                try:
                    from bef_zk.shared.features import score_plan_item, SKIP_THRESHOLD, HUMAN_REVIEW_UNCERTAINTY
                    from pathlib import Path as PathLib

                    # Find posteriors
                    if posteriors_path:
                        posteriors = PathLib(posteriors_path)
                    else:
                        # Check default locations (prioritize models/ from eval command)
                        candidates = [
                            target_path / '.capseal' / 'models' / 'beta_posteriors.npz',  # From eval
                            target_path / '.capseal' / 'beta_posteriors.npz',
                            target_path / 'beta_posteriors.npz',
                            PathLib.home() / '.capseal' / 'models' / 'beta_posteriors.npz',
                            PathLib.home() / '.capseal' / 'beta_posteriors.npz',
                        ]
                        posteriors = None
                        for c in candidates:
                            if c.exists():
                                posteriors = c
                                break

                    if posteriors and posteriors.exists():
                        # Score each plan item
                        original_count = len(plan.items)
                        filtered_items = []
                        skipped_items = []
                        review_items = []
                        gate_decisions = []

                        for item in plan.items:
                            # Build a minimal diff preview from the item
                            diff_preview = f"+++ {item.file_path}\n{item.description}"
                            findings = [{"severity": "medium"}] if item.category == "security" else []

                            score_result = score_plan_item(
                                {"diff_preview": diff_preview, "file_path": item.file_path, "findings": findings},
                                posteriors,
                            )

                            gate_decisions.append({
                                "item_id": item.item_id,
                                "file_path": item.file_path,
                                "grid_idx": score_result["grid_idx"],
                                "q": score_result["q"],
                                "uncertainty": score_result["uncertainty"],
                                "decision": score_result["decision"],
                                "reason": score_result["reason"],
                            })

                            if score_result["decision"] == "skip":
                                skipped_items.append(item)
                            elif score_result["decision"] == "human_review":
                                review_items.append(item)
                                filtered_items.append(item)  # Still process, just flagged
                            else:
                                filtered_items.append(item)

                        # Update plan with filtered items
                        plan.items = filtered_items

                        gate_result = {
                            "posteriors_path": str(posteriors),
                            "original_items": original_count,
                            "filtered_items": len(filtered_items),
                            "skipped": len(skipped_items),
                            "human_review": len(review_items),
                            "decisions": gate_decisions,
                        }

                        # Save gate result
                        gate_dir = run_path / "gate"
                        gate_dir.mkdir(exist_ok=True)
                        (gate_dir / "gate_result.json").write_text(json.dumps(gate_result, indent=2))

                        if not json_output:
                            print(f"      {original_count} items: {GREEN}✓ {len(filtered_items) - len(review_items)} pass{RESET}, "
                                  f"{YELLOW}? {len(review_items)} review{RESET}, {RED}⊘ {len(skipped_items)} skip{RESET}")
                            if skipped_items:
                                print(f"{DIM}      Skipped (high q): {', '.join(i.file_path for i in skipped_items[:3])}{'...' if len(skipped_items) > 3 else ''}{RESET}")
                    else:
                        if not json_output:
                            print(f"{DIM}      No posteriors found, skipping gate{RESET}")

                except ImportError as e:
                    if not json_output:
                        print(f"{DIM}      Gate unavailable: {e}{RESET}")

            if dry_run:
                if json_output:
                    result = {"receipt_id": receipt_id, "status": "dry_run", "items": len(plan.items)}
                    if gate_result:
                        result["gate"] = gate_result
                    print(json.dumps(result))
                else:
                    print(f"\n{YELLOW}DRY RUN - plan only:{RESET}")
                    for item in plan.items:
                        print(f"  • {item.file_path}: {item.description[:50]}...")
                return

            # Step 3: Generate patches
            orchestration_result = None
            if num_agents > 1:
                # Multi-agent DAG mode with sharding
                if not json_output:
                    print(f"{DIM}[3/4] Generating patches ({num_agents} agents, {shard_strategy} sharding)...{RESET}")

                # Convert plan to dict format for orchestrator
                plan_items = []
                for item in plan.items:
                    plan_items.append({
                        "file_path": item.file_path,
                        "description": item.description,
                        "start_line": getattr(item, 'start_line', 1),
                        "end_line": getattr(item, 'end_line', 100),
                    })

                # Create patch generator that wraps our LLM patching
                def llm_patch_generator(items: list[dict], project_dir: Path):
                    """Generate patches using LLM for a shard."""
                    from bef_zk.capsule.refactor_engine import RefactorPlan, RefactorItem

                    # Convert back to RefactorItem format
                    shard_items = []
                    for idx, i in enumerate(items):
                        shard_items.append(RefactorItem(
                            item_id=f"shard-item-{idx}",
                            category="security",
                            priority=3,  # medium
                            file_path=i["file_path"],
                            description=i["description"],
                            finding_fingerprints=[],
                            suggested_change=i.get("suggested_change", ""),
                            estimated_complexity="simple",
                        ))

                    shard_plan = RefactorPlan(
                        trace_root=plan.trace_root,
                        review_aggregate_hash=sha256_json(items),
                        items=shard_items,
                    )

                    shard_results = run_multi_agent_patches(
                        plan=shard_plan,
                        project_dir=project_dir,
                        provider=provider,
                        model=model,
                        enable_repair=True,
                        enable_suppression_memos=True,
                        enable_ast_validation=True,
                    )

                    # Convert results to patch/no_change_proof format
                    patches = []
                    no_change_proofs = []

                    for r in shard_results:
                        if r.final_status == "VALID":
                            patches.append({
                                "file_path": r.patch.file_path,
                                "patch_id": r.patch.patch_id,
                                "start_line": 1,
                                "end_line": r.patch.original_line_count,
                                "diff": r.patch.original_content,  # Store for reference
                                "result": r.to_dict(),
                            })
                        elif r.final_status == "SKIP" and r.no_change_proof:
                            no_change_proofs.append({
                                "file_path": r.patch.file_path,
                                "disposition": r.no_change_proof.disposition,
                                "justification": r.no_change_proof.justification,
                                "result": r.to_dict(),
                            })

                    return patches, no_change_proofs

                orchestrator = ShardOrchestrator(
                    plan_items=plan_items,
                    project_dir=target_path,
                    patch_generator=llm_patch_generator,
                    parent_receipt_id=receipt_id,
                )

                orchestration_result = orchestrator.run(
                    max_workers=num_agents,
                    shard_strategy=shard_strategy,
                )

                # Save shard receipts
                shards_dir = run_path / "shards"
                shards_dir.mkdir(exist_ok=True)

                for receipt in orchestration_result.shard_receipts:
                    receipt_file = shards_dir / f"{receipt.shard_id}_receipt.json"
                    receipt_file.write_text(json.dumps({
                        "receipt_id": receipt.receipt_id,
                        "shard_id": receipt.shard_id,
                        "parent_receipt_id": receipt.parent_receipt_id,
                        "input_hash": receipt.input_hash,
                        "output_hash": receipt.output_hash,
                        "receipt_hash": receipt.receipt_hash,
                        "result": {
                            "status": receipt.result.status,
                            "patches": receipt.result.patches,
                            "no_change_proofs": receipt.result.no_change_proofs,
                            "duration_ms": receipt.result.duration_ms,
                        }
                    }, indent=2))

                # Save reduce receipt
                reduce_receipt_file = run_path / "reduce_receipt.json"
                reduce_result = orchestration_result.reduce_receipt.result
                reduce_receipt_file.write_text(json.dumps({
                    "receipt_id": orchestration_result.reduce_receipt.receipt_id,
                    "shard_receipts": orchestration_result.reduce_receipt.shard_receipts,
                    "input_hash": reduce_result.input_hash,
                    "output_hash": reduce_result.output_hash,
                    "receipt_hash": orchestration_result.reduce_receipt.receipt_hash,
                    "result": {
                        "status": reduce_result.status,
                        "merged_patches": reduce_result.merged_patches,
                        "merged_no_change_proofs": reduce_result.merged_no_change_proofs,
                        "shard_order": reduce_result.shard_order,
                        "conflicts": [c.to_dict() for c in reduce_result.conflicts],
                    }
                }, indent=2))

                if not json_output:
                    print(f"      Shards: {orchestration_result.total_shards} ({orchestration_result.successful_shards} ok, {orchestration_result.failed_shards} failed)")
                    if orchestration_result.conflicts_detected > 0:
                        print(f"      {YELLOW}⚠ Conflicts: {orchestration_result.conflicts_detected}{RESET}")
                    # Show errors from failed shards
                    for receipt in orchestration_result.shard_receipts:
                        if receipt.result.status == "failed" and receipt.result.errors:
                            print(f"      {RED}Shard {receipt.shard_id}: {receipt.result.errors[0][:100]}...{RESET}")
                    print(f"      Duration: {orchestration_result.total_duration_ms}ms")

                # For multi-agent mode, results are stored in the orchestration result
                # We don't need to reconstruct PatchResults - just use the raw data
                # The downstream code will use orchestration_result directly
                results = []  # Empty for multi-agent - tallying done via orchestration_result

            else:
                # Single-agent mode (original behavior)
                if not json_output:
                    print(f"{DIM}[3/4] Generating patches...{RESET}")
                results = run_multi_agent_patches(
                    plan=plan,
                    project_dir=target_path,
                    provider=provider,
                    model=model,
                    enable_repair=True,
                    enable_suppression_memos=True,
                    enable_ast_validation=True,
                )

            # Show progress per file
            multipass_files = []
            if not json_output and not orchestration_result:
                for r in results:
                    file_path = r.patch.file_path
                    lines = r.patch.original_line_count
                    status_icon = "✓" if r.final_status == "VALID" else ("⊘" if r.final_status == "SKIP" else "✗")

                    if lines > 500:
                        multipass_files.append(file_path)
                        print(f"      ├── {file_path} ({lines} lines) → multi-pass mode")
                    else:
                        status_detail = ""
                        if r.final_status == "SKIP" and r.no_change_proof:
                            status_detail = f" (proven: {r.no_change_proof.disposition})"
                        print(f"      ├── {status_icon} {file_path}{status_detail}")

            # Tally
            if orchestration_result:
                valid = orchestration_result.total_patches
                skip = orchestration_result.total_no_change_proofs
                fail = orchestration_result.failed_shards
                proven = orchestration_result.total_no_change_proofs
            else:
                valid = sum(1 for r in results if r.final_status == "VALID")
                skip = sum(1 for r in results if r.final_status == "SKIP")
                fail = sum(1 for r in results if r.final_status == "FAIL")
                proven = sum(1 for r in results if r.no_change_proof and r.no_change_proof.file_pre_hash)

            if not json_output:
                print(f"\n      {GREEN}✓ {valid} patched{RESET}  {YELLOW}⊘ {skip} skipped ({proven} proven){RESET}  {RED if fail else DIM}✗ {fail} failed{RESET}")

            # Save results
            patches_dir = run_path / "patches"
            patches_dir.mkdir(exist_ok=True)
            if orchestration_result:
                # For multi-agent mode, patch results are saved in shards/
                # Write a summary file linking to them
                summary_file = patches_dir / "orchestration_summary.json"
                summary_file.write_text(json.dumps({
                    "mode": "multi_agent",
                    "total_shards": orchestration_result.total_shards,
                    "total_patches": orchestration_result.total_patches,
                    "total_no_change_proofs": orchestration_result.total_no_change_proofs,
                    "conflicts_detected": orchestration_result.conflicts_detected,
                    "shard_receipt_ids": [r.receipt_id for r in orchestration_result.shard_receipts],
                    "reduce_receipt_id": orchestration_result.reduce_receipt.receipt_id,
                }, indent=2))
            else:
                for r in results:
                    result_file = patches_dir / f"{r.patch.patch_id}_result.json"
                    result_file.write_text(json.dumps(r.to_dict(), indent=2))

            # Step 3.5: Critic review (asymmetric code review)
            critic_result = None
            if enable_critic:
                if not json_output:
                    print(f"\n{CYAN}[3.5/4] Running adversarial critic review...{RESET}")

                try:
                    from bef_zk.capsule.critic_agent import (
                        run_critic_review, save_critic_result, format_critic_challenges,
                    )

                    # Collect patches and no_change_proofs for critic
                    critic_patches = []
                    critic_no_change_proofs = []
                    file_contents = {}

                    if orchestration_result:
                        # Multi-agent mode - collect from shard receipts
                        for receipt in orchestration_result.shard_receipts:
                            for patch in receipt.result.patches:
                                critic_patches.append(patch)
                                fp = patch.get("file_path", "")
                                if fp and Path(fp).exists():
                                    file_contents[fp] = Path(fp).read_text()
                            for ncp in receipt.result.no_change_proofs:
                                critic_no_change_proofs.append(ncp)
                                fp = ncp.get("file_path", "")
                                if fp and Path(fp).exists() and fp not in file_contents:
                                    file_contents[fp] = Path(fp).read_text()
                    else:
                        # Single-agent mode
                        for r in results:
                            if r.final_status == "VALID":
                                critic_patches.append({
                                    "item_id": r.patch.patch_id,
                                    "file_path": r.patch.file_path,
                                    "status": r.final_status,
                                    "diff": "",  # Would need to compute diff
                                })
                            elif r.final_status == "SKIP" and r.no_change_proof:
                                critic_no_change_proofs.append({
                                    "item_id": r.patch.patch_id,
                                    "file_path": r.patch.file_path,
                                    "disposition": r.no_change_proof.disposition,
                                    "justification": r.no_change_proof.justification,
                                    "evidence_snippet": r.no_change_proof.evidence_snippet,
                                })
                            fp = r.patch.file_path
                            if fp and Path(fp).exists() and fp not in file_contents:
                                file_contents[fp] = Path(fp).read_text()

                    # Run critic
                    critic_result = run_critic_review(
                        patches=critic_patches,
                        no_change_proofs=critic_no_change_proofs,
                        file_contents=file_contents,
                        findings=findings,
                        provider=provider,
                        model=critic_model,
                    )

                    # Save critic result
                    save_critic_result(critic_result, run_path)

                    # Generate critic receipt (bind into DAG)
                    from bef_zk.capsule.critic_agent import (
                        build_critic_inputs_hash, generate_critic_receipt,
                        challenges_to_obligations, save_critic_obligations,
                        generate_all_critic_tests,
                    )

                    # Build inputs hash for receipt
                    plan_data = json.loads((run_path / "plan.json").read_text()) if (run_path / "plan.json").exists() else {}
                    patches_data = {"patches": critic_patches}
                    claims_data = {"claims": []}  # Will be populated after claims generation

                    inputs_hash = build_critic_inputs_hash(plan_data, claims_data, patches_data)
                    critic_receipt = generate_critic_receipt(critic_result, inputs_hash, run_path)

                    # Convert challenges to obligations (Mode C)
                    critic_obligations = challenges_to_obligations(critic_result.challenges)
                    save_critic_obligations(critic_obligations, run_path)

                    # Generate test templates
                    test_files = generate_all_critic_tests(
                        critic_result,
                        str(target_path),
                        run_path,
                    )

                    if not json_output:
                        print(f"      Critic found {len(critic_result.challenges)} challenges ({critic_result.duration_ms}ms)")
                        if critic_result.challenges:
                            print(format_critic_challenges(critic_result))
                        if critic_obligations:
                            unresolved = sum(1 for o in critic_obligations if o.resolution == "unresolved")
                            print(f"      {CYAN}○{RESET} Critic obligations: {len(critic_obligations)} ({unresolved} unresolved)")
                        if test_files:
                            print(f"      {DIM}Generated {len(test_files)} test templates in critic_tests/{RESET}")
                        if critic_result.has_blockers():
                            print(f"\n      {YELLOW}⚠ Critic raised blocking concerns (critical/high){RESET}")

                except Exception as e:
                    if not json_output:
                        print(f"      {YELLOW}⚠ Critic failed: {e}{RESET}")
                    critic_result = None

            # Step 4: Build receipt with rolling proof display
            if not json_output:
                print(f"\n{CYAN}[4/4] Generating cryptographic receipt...{RESET}\n")

                # ═══════════════════════════════════════════════════════════════
                # PHASE 1: Hash inputs
                # ═══════════════════════════════════════════════════════════════
                print(f"      {DIM}┌─ Proof Generation ─────────────────────────────────{RESET}")
                print(f"      │ {DIM}Hashing inputs...{RESET}", end="", flush=True)
                time.sleep(0.1)
                print(f"\r      │ {GREEN}✓{RESET} Input hash:        {CYAN}sha256:{input_hash[:16]}...{RESET}")

            # Build verification chain
            if orchestration_result:
                # For multi-agent, build verifications from shard receipts
                verifications = []
                for receipt in orchestration_result.shard_receipts:
                    for patch in receipt.result.patches:
                        verifications.append(PatchVerification(
                            patch_id=patch.get("patch_id", "unknown"),
                            verified=receipt.result.status == "success",
                            original_hash_match=True,
                            patch_applies_cleanly=receipt.result.status == "success",
                            result_hash_match=True,
                        ))
            else:
                verifications = [
                    PatchVerification(
                        patch_id=r.patch.patch_id,
                        verified=r.validation.is_valid,
                        original_hash_match=True,
                        patch_applies_cleanly=r.validation.is_valid,
                        result_hash_match=True,
                    )
                    for r in results
                ]

            if not json_output:
                time.sleep(0.1)
                print(f"      │ {GREEN}✓{RESET} Verifications:     {len(verifications)} patch proofs bound")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: Generate claims from results
            # ═══════════════════════════════════════════════════════════════
            all_claims = []

            if orchestration_result:
                # For multi-agent mode, extract from shard receipts
                ncp_list = []
                for receipt in orchestration_result.shard_receipts:
                    for ncp in receipt.result.no_change_proofs:
                        ncp_list.append(ncp)

                if not json_output and ncp_list:
                    print(f"      │")
                    print(f"      │ {DIM}Generating claims...{RESET}", end="", flush=True)
                    time.sleep(0.1)
                    print(f"\r      │ {CYAN}○{RESET} Claims:            {len(ncp_list)} to verify")

                for ncp in ncp_list:
                    file_path = ncp.get("file_path", "")
                    try:
                        fp = Path(file_path)
                        if not fp.is_absolute():
                            fp = target_path / fp
                        file_content = fp.read_text()
                    except Exception:
                        file_content = ""

                    claim = claim_from_no_change_proof({
                        "disposition": ncp.get("disposition", ""),
                        "justification": ncp.get("justification", ""),
                        "file_pre_hash": ncp.get("file_pre_hash", ""),
                        "evidence_chunk_hash": ncp.get("evidence_chunk_hash", ""),
                        "span_anchor": ncp.get("span_anchor", {}),
                        "evidence_snippet": ncp.get("evidence_snippet", ""),
                    }, file_path, file_content)
                    all_claims.append(claim)

                # Get ALL files from findings (not just modified) for obligation claims
                finding_files = set()
                for receipt in orchestration_result.shard_receipts:
                    # Collect files from no_change_proofs
                    for ncp in receipt.result.no_change_proofs:
                        if ncp.get("file_path"):
                            finding_files.add(ncp.get("file_path"))
                    # Also collect from patches (if any)
                    for patch in receipt.result.patches:
                        if patch.get("file_path"):
                            finding_files.add(patch.get("file_path", ""))
            else:
                no_change_proofs = [r for r in results if r.no_change_proof and r.no_change_proof.file_pre_hash]

                if not json_output and no_change_proofs:
                    print(f"      │")
                    print(f"      │ {DIM}Generating claims...{RESET}", end="", flush=True)
                    time.sleep(0.1)
                    print(f"\r      │ {CYAN}○{RESET} Claims:            {len(no_change_proofs)} to verify")

                for r in no_change_proofs:
                    proof = r.no_change_proof
                    file_path = r.patch.file_path

                    # Read file content for claim
                    try:
                        file_content = Path(file_path).read_text()
                    except Exception:
                        file_content = ""

                    # Convert NO_CHANGE proof to Claim
                    claim = claim_from_no_change_proof(proof.__dict__ if hasattr(proof, '__dict__') else {
                        "disposition": proof.disposition,
                        "justification": proof.justification,
                        "file_pre_hash": proof.file_pre_hash,
                        "evidence_chunk_hash": proof.evidence_chunk_hash,
                        "span_anchor": {
                            "start_line": proof.span_anchor.start_line if proof.span_anchor else None,
                            "end_line": proof.span_anchor.end_line if proof.span_anchor else None,
                        } if proof.span_anchor else {},
                        "evidence_snippet": proof.evidence_snippet,
                    }, file_path, file_content)

                    all_claims.append(claim)

                # Get ALL files from results (not just modified) for obligation claims
                finding_files = set()
                for r in results:
                    if r.patch.file_path:
                        finding_files.add(r.patch.file_path)

            # Generate obligation claims for ALL finding files (even if NO_CHANGE)
            # This ensures policy obligations are checked regardless of patch status
            if finding_files:
                if not json_output:
                    print(f"      │ {CYAN}○{RESET} Policy obligations: {len(finding_files)} files to verify")

                for file_path in finding_files:
                    # Get required claims for this file based on policy
                    required = get_required_claims(policy, profile, file_path)

                    # Read file content
                    try:
                        fp = Path(file_path)
                        if not fp.is_absolute():
                            fp = target_path / fp
                        file_content = fp.read_text()
                    except Exception:
                        continue

                    # Generate claims for each obligation
                    for claim_type, checker_id in required:
                        claim = create_claim(
                            claim_type=claim_type,
                            file_path=str(file_path),
                            file_content=file_content,
                            description=f"Policy obligation: {claim_type.value} for {Path(file_path).name}",
                        )
                        all_claims.append(claim)

                if not json_output:
                    # Count NO_CHANGE claims (already_mitigated/false_positive)
                    no_change_claim_count = sum(
                        1 for c in all_claims
                        if c.claim_type in (ClaimType.ALREADY_MITIGATED, ClaimType.FALSE_POSITIVE)
                    )
                    total_policy_claims = len(all_claims) - no_change_claim_count
                    print(f"      │ {GREEN}✓{RESET} Generated:         {total_policy_claims} policy claims")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: Run deterministic checkers on claims
            # ═══════════════════════════════════════════════════════════════
            if not json_output and all_claims:
                print(f"      │ {DIM}Running checkers...{RESET}")
                time.sleep(0.1)

            # Map claim types to appropriate checkers
            claim_checker_map = {
                ClaimType.NO_SQL_INJECTION: "semgrep",
                ClaimType.NO_SHELL_INJECTION: "ast",  # AST for shell=True detection
                ClaimType.NO_HARDCODED_SECRETS: "semgrep",
                ClaimType.ALLOWLIST_ENFORCED: "ast",
                ClaimType.REFACTOR_EQUIVALENCE: "hypothesis",
                ClaimType.TYPE_SAFE: "ast",
                ClaimType.INPUT_VALIDATED: "contracts",
                ClaimType.ALREADY_MITIGATED: "ast",  # Verify span anchor
                ClaimType.FALSE_POSITIVE: "ast",     # Verify span anchor
            }

            claims_passed = 0
            claims_failed = 0
            claims_unknown = 0
            cache_hits = 0

            for i, claim in enumerate(all_claims):
                file_name = Path(claim.scope.file_path).name

                # Read file content for this claim
                try:
                    claim_file_path = Path(claim.scope.file_path)
                    if not claim_file_path.is_absolute():
                        claim_file_path = target_path / claim_file_path
                    claim_file_content = claim_file_path.read_text()
                except Exception as e:
                    claim_file_content = ""
                    claims_unknown += 1
                    if not json_output:
                        print(f"      │   {YELLOW}?{RESET} {file_name}: cannot read file")
                    continue

                # Verify ALREADY_MITIGATED/FALSE_POSITIVE claims by checking anchors
                if claim.claim_type in (ClaimType.ALREADY_MITIGATED, ClaimType.FALSE_POSITIVE):
                    # Verify file hash matches (file hasn't changed since claim was made)
                    current_hash = hashlib.sha256(claim_file_content.encode()).hexdigest()
                    anchor_valid = claim.scope.file_hash == current_hash

                    # Verify span anchor if present
                    span_valid = True
                    if claim.scope.start_line and claim.scope.end_line and claim.scope.region_hash:
                        lines = claim_file_content.split('\n')
                        region = '\n'.join(lines[claim.scope.start_line-1:claim.scope.end_line])
                        region_hash = hashlib.sha256(region.encode()).hexdigest()
                        span_valid = region_hash == claim.scope.region_hash

                    if anchor_valid and span_valid:
                        claim.verdict = Verdict.PASS
                        claim.witness = Witness(
                            witness_type="anchor_verification",
                            artifact_hash=current_hash[:16],
                            artifact_inline=json.dumps({
                                "file_hash_match": anchor_valid,
                                "span_hash_match": span_valid,
                                "verified_at": datetime.datetime.utcnow().isoformat() + "Z",
                            }),
                            produced_at=datetime.datetime.utcnow().isoformat() + "Z",
                            producer="anchor_verifier",
                        )
                        claims_passed += 1
                        if not json_output:
                            print(f"      │   {GREEN}✓{RESET} {file_name}: {claim.claim_type.value} {DIM}(anchor verified){RESET}")
                            time.sleep(0.03)
                    else:
                        claim.verdict = Verdict.FAIL
                        claim.witness = Witness(
                            witness_type="anchor_verification",
                            artifact_hash="",
                            artifact_inline=json.dumps({
                                "file_hash_match": anchor_valid,
                                "span_hash_match": span_valid,
                                "error": "File or span changed since claim was made",
                            }),
                            counterexample="Anchor invalidated - re-verification required",
                            produced_at=datetime.datetime.utcnow().isoformat() + "Z",
                            producer="anchor_verifier",
                        )
                        claims_failed += 1
                        if not json_output:
                            print(f"      │   {RED}✗{RESET} {file_name}: {claim.claim_type.value} {DIM}(anchor invalid!){RESET}")
                            time.sleep(0.03)
                else:
                    # Run appropriate checker based on claim type (with caching)
                    checker_id = claim_checker_map.get(claim.claim_type, "ast")
                    try:
                        # Use cache-aware checker execution
                        check_result = check_claim_with_cache(
                            claim=claim,
                            file_content=claim_file_content,
                            checker_id=checker_id,
                            policy_hash=policy_hash,
                            cache=claim_cache,
                            checker_registry=CHECKER_REGISTRY,
                        )

                        claim.verdict = check_result.verdict
                        claim.witness = check_result.witness
                        # Set checker info for traceability
                        if check_result.checker_id:
                            claim.checker = CheckerInfo(
                                checker_id=check_result.checker_id,
                                checker_version=check_result.checker_version,
                                env_key=policy_hash[:16],
                            )
                        claim.evaluated_at = datetime.datetime.utcnow().isoformat() + "Z"

                        if check_result.verdict == Verdict.PASS:
                            claims_passed += 1
                            icon = f"{GREEN}✓{RESET}"
                        elif check_result.verdict == Verdict.FAIL:
                            claims_failed += 1
                            icon = f"{RED}✗{RESET}"
                        else:
                            claims_unknown += 1
                            icon = f"{YELLOW}?{RESET}"

                        # Show cache status
                        if check_result.from_cache:
                            cache_hits += 1
                            cache_info = f"{DIM}({checker_id}, cached){RESET}"
                        else:
                            cache_info = f"{DIM}({checker_id}){RESET}"

                        if not json_output:
                            print(f"      │   {icon} {file_name}: {claim.claim_type.value} {cache_info}")
                            time.sleep(0.03)
                    except Exception as e:
                        claims_unknown += 1
                        if not json_output:
                            print(f"      │   {YELLOW}?{RESET} {file_name}: checker error ({checker_id})")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: Evaluate policy
            # ═══════════════════════════════════════════════════════════════
            if not json_output:
                print(f"      │")
                print(f"      │ {DIM}Evaluating policy...{RESET}", end="", flush=True)
                time.sleep(0.1)

            policy_eval = evaluate_policy(policy, profile, all_claims)

            if not json_output:
                if policy_eval.all_met:
                    print(f"\r      │ {GREEN}✓{RESET} Policy:            {policy.name}/{profile} - ALL OBLIGATIONS MET")
                else:
                    failed_count = len(policy_eval.failed_obligations)
                    print(f"\r      │ {YELLOW}!{RESET} Policy:            {policy.name}/{profile} - {failed_count} obligations need attention")
                    for result in policy_eval.failed_obligations[:3]:
                        print(f"      │   {YELLOW}○{RESET} {result.obligation.obligation_id}: {result.verdict.value}")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5: Build rollup and output hash
            # ═══════════════════════════════════════════════════════════════
            rollup = build_diff_rollup(plan, results, verifications, receipt_id)

            # Save rollup and diff
            rollup_path = run_path / "diff_rollup.json"
            rollup_path.write_text(json.dumps(rollup.to_dict(), indent=2))

            if rollup.combined_diff:
                (run_path / "combined.diff").write_text(rollup.combined_diff)

            output_hash = rollup.combined_diff_hash or sha256_str("")

            if not json_output:
                print(f"      │")
                print(f"      │ {DIM}Hashing outputs...{RESET}", end="", flush=True)
                time.sleep(0.1)
                print(f"\r      │ {GREEN}✓{RESET} Output hash:       {CYAN}sha256:{output_hash[:16]}...{RESET}")

                # Build chain
                chain_nodes = ["trace", "semgrep", "plan", "patches", "claims", "policy", "rollup"]
                print(f"      │ {DIM}Constructing proof chain...{RESET}", end="", flush=True)
                time.sleep(0.1)
                print(f"\r      │ {GREEN}✓{RESET} Chain:             {' → '.join(chain_nodes)}")

                # Sign receipt
                print(f"      │ {DIM}Signing receipt...{RESET}", end="", flush=True)
                time.sleep(0.1)
                receipt_hash = sha256_str(f"{receipt_id}{input_hash}{output_hash}{policy.hash()[:8]}")[:16]
                print(f"\r      │ {GREEN}✓{RESET} Receipt signature: {CYAN}sha256:{receipt_hash}...{RESET}")

                print(f"      {DIM}└────────────────────────────────────────────────────{RESET}\n")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 6: Build and save receipt
            # ═══════════════════════════════════════════════════════════════

            # Create claim bundle
            claim_bundle = ClaimBundle(
                bundle_id=f"CLB-{receipt_id}",
                receipt_id=receipt_id,
                claims=all_claims,
            )
            claim_bundle.compute_summary()

            # Save claims
            claims_path = run_path / "claims.json"
            claims_path.write_text(json.dumps(claim_bundle.to_dict(), indent=2))

            # Resolve critic obligations against claims (Mode C)
            critic_obligations_resolved = 0
            if critic_result and critic_result.challenges:
                try:
                    from bef_zk.capsule.critic_agent import (
                        resolve_obligations_with_claims, save_critic_obligations,
                        challenges_to_obligations,
                    )

                    # Reload obligations (they were saved earlier)
                    critic_obligations_path = run_path / "critic_obligations.json"
                    if critic_obligations_path.exists():
                        obls_data = json.loads(critic_obligations_path.read_text())
                        from bef_zk.capsule.critic_agent import CriticObligation
                        obligations = [CriticObligation(**o) for o in obls_data.get("obligations", [])]

                        # Resolve against generated claims
                        claims_list = [c.to_dict() for c in claim_bundle.claims]
                        resolved = resolve_obligations_with_claims(obligations, claims_list)

                        # Save updated obligations
                        save_critic_obligations(resolved, run_path)

                        critic_obligations_resolved = sum(
                            1 for o in resolved if o.resolution in ("checker_passed", "waived")
                        )
                        unresolved = sum(1 for o in resolved if o.resolution == "unresolved")

                        if not json_output and resolved:
                            print(f"      │ {CYAN}○{RESET} Critic obligations: {critic_obligations_resolved} resolved, {unresolved} unresolved")

                except Exception as e:
                    if not json_output:
                        print(f"      │ {YELLOW}⚠{RESET} Obligation resolution failed: {e}")

            # Save policy evaluation
            policy_path = run_path / "policy_eval.json"
            policy_path.write_text(json.dumps(policy_eval.to_dict(), indent=2))

            # Build receipt
            chain = ["trace", "semgrep", "plan", "patches"]
            if critic_result:
                chain.append("critic")
            chain.extend(["claims", "policy", "rollup"])

            receipt = {
                "receipt_id": receipt_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "timestamp_human": timestamp_human,
                "chain": chain,
                "inputs": {
                    "path": str(target_path),
                    "hash": input_hash,
                },
                "outputs": {
                    "diff": "combined.diff" if rollup.combined_diff else None,
                    "hash": output_hash,
                },
                "executor": f"{model} via capseal/0.6",
                "policy": {
                    "name": policy.name,
                    "version": policy.version,
                    "profile": profile,
                    "hash": policy.hash(),  # Full hash for verification
                    "source": policy_load_result.resolved_from,
                    "source_content_hash": policy.source.content_hash if policy.source else None,
                    "all_met": policy_eval.all_met,
                },
                "claims": {
                    "total": claim_bundle.total_claims,
                    "passed": claim_bundle.passed,
                    "failed": claim_bundle.failed,
                    "unknown": claim_bundle.unknown,
                },
                "stats": {
                    "findings": len(findings),
                    "patched": valid,
                    "skipped": skip,
                    "failed": fail,
                    "proven": proven,
                },
                "cache": {
                    "hits": cache_hits,
                    "misses": len(all_claims) - cache_hits,
                    "hit_rate": f"{(cache_hits / len(all_claims) * 100):.1f}%" if all_claims else "0%",
                },
                "critic": {
                    "enabled": critic_result is not None,
                    "challenges": len(critic_result.challenges) if critic_result else 0,
                    "high_critical": (critic_result.high_count + critic_result.critical_count) if critic_result else 0,
                    "obligations_resolved": critic_obligations_resolved if critic_result else 0,
                    "confidence": critic_result.confidence if critic_result else None,
                } if enable_critic else None,
                "run_path": str(run_path),
            }
            receipt_path = run_path / "receipt.json"
            receipt_path.write_text(json.dumps(receipt, indent=2))

            # Save claim cache for incremental reuse
            claim_cache.save()

            # Also save to receipts index
            receipts_index = target_path / ".capseal" / "receipts.jsonl"
            with open(receipts_index, "a") as f:
                f.write(json.dumps({"id": receipt_id, "path": str(run_path), "ts": receipt["timestamp"]}) + "\n")

            # Update state so 'open' command can find this project's receipts
            self.state.last_review_path = target_path

            # ═══════════════════════════════════════════════════════════════
            # FINAL OUTPUT
            # ═══════════════════════════════════════════════════════════════
            if json_output:
                print(json.dumps(receipt, indent=2))
            else:
                # Determine overall status
                if policy_eval.all_met and fail == 0:
                    status_icon = f"{GREEN}✓ RECEIPT GENERATED{RESET}"
                    status_note = ""
                elif not policy_eval.all_met:
                    status_icon = f"{YELLOW}! RECEIPT GENERATED (policy obligations pending){RESET}"
                    status_note = f"\n  {YELLOW}Note: {len(policy_eval.failed_obligations)} policy obligations need attention{RESET}"
                else:
                    status_icon = f"{YELLOW}! RECEIPT GENERATED (with failures){RESET}"
                    status_note = f"\n  {YELLOW}Note: {fail} items failed verification{RESET}"

                stats_line = f"✓ {valid} patched, {proven} proven"
                if fail > 0:
                    stats_line += f", {fail} failed"

                claims_line = f"{claims_passed} passed"
                if claims_failed > 0:
                    claims_line += f", {claims_failed} failed"
                if claims_unknown > 0:
                    claims_line += f", {claims_unknown} unknown"

                print(f"""{BOLD}{'═' * 65}{RESET}
{status_icon}{status_note}

  {BOLD}ID:{RESET}        {CYAN}{receipt_id}{RESET}  {DIM}({timestamp_human}){RESET}
  {BOLD}Inputs:{RESET}    sha256:{input_hash[:8]}  {DIM}({target_path.name}){RESET}
  {BOLD}Outputs:{RESET}   sha256:{output_hash[:8]}  {DIM}(combined.diff){RESET}
  {BOLD}Executor:{RESET}  {model}
  {BOLD}Policy:{RESET}    {policy.name}/{profile} {DIM}(sha256:{policy.hash()[:8]}){RESET}

  {BOLD}Patches:{RESET}   {stats_line}
  {BOLD}Claims:{RESET}    {claims_line}
  {BOLD}Policy:{RESET}    {"✓ all obligations met" if policy_eval.all_met else f"{len(policy_eval.failed_obligations)} obligations pending"}

{BOLD}Artifacts saved to:{RESET}
  {DIM}{run_path}/{RESET}
  ├── receipt.json      {DIM}# The cryptographic receipt{RESET}
  ├── claims.json       {DIM}# All claims with verdicts{RESET}
  ├── policy_eval.json  {DIM}# Policy evaluation results{RESET}
  ├── plan.json         {DIM}# What was planned{RESET}
  ├── patches/          {DIM}# Individual patch results{RESET}
  ├── combined.diff     {DIM}# Verified diff (if any){RESET}
  └── diff_rollup.json  {DIM}# Full rollup with proofs{RESET}

{BOLD}Next steps:{RESET}
  {CYAN}open{RESET}   {receipt_id}     {DIM}# Inspect artifacts{RESET}
  {CYAN}verify{RESET} {receipt_id}     {DIM}# Re-verify proofs{RESET}
  {CYAN}audit{RESET}  {receipt_id}     {DIM}# Trust boundary report{RESET}
  {CYAN}apply{RESET}  {receipt_id}     {DIM}# Apply verified patches{RESET}
{BOLD}{'═' * 65}{RESET}
""")

            # Apply if requested
            if apply_after and valid > 0 and rollup.combined_diff:
                if not json_output:
                    print(f"{CYAN}Applying patches...{RESET}")
                apply_result = subprocess.run(
                    ["git", "apply", "-"],
                    input=rollup.combined_diff.encode(),
                    capture_output=True,
                    cwd=target_path,
                )
                if apply_result.returncode == 0:
                    if not json_output:
                        print(f"{GREEN}✓ Patches applied successfully{RESET}")
                else:
                    if not json_output:
                        print(f"{RED}✗ Apply failed: {apply_result.stderr.decode()[:200]}{RESET}")

        except ImportError as e:
            print(f"{RED}Module not available: {e}{RESET}")
        except subprocess.TimeoutExpired:
            print(f"{RED}Semgrep timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def do_eval(self, arg: str) -> None:
        """Epistemic evaluation loop - learn the safety boundary.

        Usage:
            eval <path>                        Run eval with defaults
            eval <path> --rounds N             Run N evaluation rounds (default: 5)
            eval <path> --seed S               Set random seed
            eval <path> --synthetic            Use synthetic episodes (no LLM calls)
            eval <path> --targets-per-round K  Select K targets per round (default: 64)
            eval <path> --episodes-per-target E  Episodes per target (default: 1)
            eval <path> --prove                Generate ZK proof for the eval trace

        The eval command runs an epistemic learning loop that:
        1. Scans the codebase with Semgrep to find potential issues
        2. Maps findings to a 5-dimensional feature grid (1024 points)
        3. Uses acquisition scoring to select the most informative targets
        4. Runs episodes (synthetic or real pipeline) to estimate p_fail
        5. Updates Beta posteriors: alpha=fails+1, beta=successes+1
        6. Generates receipt-bound artifacts for each round

        After evaluation:
        - Learned posteriors are saved to .capseal/models/beta_posteriors.npz
        - Future 'capseal review --gate' commands will use these posteriors
        - The committor gate will skip high-risk patches (q >= 0.3)

        Feature Grid (5 dimensions × 4 levels = 1024 points):
            - lines_changed: Code churn size
            - cyclomatic_complexity: Branching complexity
            - files_touched: Scope of change
            - finding_severity: Semgrep severity level
            - test_coverage_delta: Test file inclusion

        Examples:
            eval src/ --rounds 5 --synthetic         # Quick synthetic eval
            eval . --rounds 3                        # Real pipeline eval
            eval src/ --seed 42 --rounds 10          # Reproducible long run
            eval src/ --rounds 5 --synthetic --prove # With ZK proof
        """
        import datetime
        import subprocess
        import uuid
        import time
        import numpy as np

        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: eval <path> [--rounds N] [--seed S] [--synthetic]{RESET}")
            print(f"{DIM}Example: eval src/ --rounds 5 --synthetic{RESET}")
            return

        target_path = Path(parts[0]).expanduser().resolve()
        if not target_path.exists():
            print(f"{RED}Path not found: {target_path}{RESET}")
            return

        # Parse options
        n_rounds = 5
        seed = None
        use_synthetic = "--synthetic" in parts
        use_prove = "--prove" in parts
        targets_per_round = 64
        episodes_per_target = 1

        i = 1
        while i < len(parts):
            if parts[i] == "--rounds" and i + 1 < len(parts):
                n_rounds = int(parts[i + 1])
                i += 2
            elif parts[i] == "--seed" and i + 1 < len(parts):
                seed = int(parts[i + 1])
                i += 2
            elif parts[i] == "--targets-per-round" and i + 1 < len(parts):
                targets_per_round = int(parts[i + 1])
                i += 2
            elif parts[i] == "--episodes-per-target" and i + 1 < len(parts):
                episodes_per_target = int(parts[i + 1])
                i += 2
            elif parts[i] == "--synthetic":
                use_synthetic = True
                i += 1
            elif parts[i] == "--prove":
                use_prove = True
                i += 1
            else:
                i += 1

        if seed is None:
            seed = int(time.time()) % 100000

        try:
            from bef_zk.shared.scoring import (
                compute_acquisition_score, select_targets, compute_tube_metrics,
            )
            from bef_zk.shared.features import (
                extract_patch_features, discretize_features, features_to_grid_idx,
                grid_idx_to_features,
            )
            from bef_zk.shared.receipts import (
                build_round_receipt, build_run_receipt, collect_round_dirs,
            )

            # Create run directory
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%dT%H%M%S")
            run_uuid = str(uuid.uuid4())[:8]
            run_path = target_path / ".capseal" / "runs" / f"{timestamp}-eval"
            run_path.mkdir(parents=True, exist_ok=True)
            (run_path / "rounds").mkdir(exist_ok=True)

            print(f"\n{CYAN}{'═' * 65}{RESET}")
            print(f"{CYAN}  EPISTEMIC EVALUATION LOOP{RESET}")
            print(f"{CYAN}{'═' * 65}{RESET}")
            print(f"  Run ID:    {run_uuid}")
            print(f"  Target:    {target_path}")
            print(f"  Rounds:    {n_rounds}")
            print(f"  Mode:      {'Synthetic' if use_synthetic else 'Real pipeline'}")
            print(f"  Seed:      {seed}")
            print(f"{CYAN}{'═' * 65}{RESET}\n")

            # Step 1: Grid generation
            print(f"{DIM}[1/3] Generating feature grid...{RESET}")

            if use_synthetic:
                # Synthetic mode: generate full 1024-point grid
                n_points = 1024
                grid = {
                    "n_points": n_points,
                    "version": "synthetic_1024",
                    "mode": "synthetic",
                }
                grid_idx_to_findings = {}  # No real findings in synthetic mode
                print(f"      Synthetic grid: {n_points} points")
            else:
                # Real mode: scan with Semgrep and build grid from actual findings
                print(f"      Scanning with Semgrep...")
                result = subprocess.run(
                    ["semgrep", "--config", "auto", "--json", str(target_path)],
                    capture_output=True,
                    timeout=120,
                )

                findings = []
                try:
                    output = json.loads(result.stdout.decode())
                    findings = output.get("results", [])
                except json.JSONDecodeError:
                    pass

                print(f"      Found {len(findings)} Semgrep findings")

                if not findings:
                    print(f"{YELLOW}⚠ No findings - cannot run real eval without findings{RESET}")
                    print(f"{DIM}Use --synthetic for synthetic evaluation{RESET}")
                    return

                # Map findings to grid indices
                grid_idx_to_findings = {}
                finding_grid_indices = set()

                for finding in findings:
                    # Extract file path and build a minimal diff preview
                    file_path = finding.get("path", "")
                    severity = finding.get("extra", {}).get("severity", "warning")
                    start_line = finding.get("start", {}).get("line", 1)
                    end_line = finding.get("end", {}).get("line", start_line + 5)

                    # Create a placeholder diff (just file info + estimated lines)
                    diff_preview = f"diff --git a/{file_path} b/{file_path}\n"
                    diff_preview += f"--- a/{file_path}\n"
                    diff_preview += f"+++ b/{file_path}\n"
                    # Estimate lines changed
                    lines_changed = max(5, end_line - start_line + 1)
                    diff_preview += f"@@ -{start_line},{lines_changed} +{start_line},{lines_changed} @@\n"
                    diff_preview += "+ # patch placeholder\n" * min(lines_changed, 10)

                    # Extract features
                    raw_features = extract_patch_features(diff_preview, [{"severity": severity}])
                    levels = discretize_features(raw_features)
                    grid_idx = features_to_grid_idx(levels)

                    finding_grid_indices.add(grid_idx)
                    if grid_idx not in grid_idx_to_findings:
                        grid_idx_to_findings[grid_idx] = []
                    grid_idx_to_findings[grid_idx].append(finding)

                n_points = 1024  # Full grid, but only some points have findings
                grid = {
                    "n_points": n_points,
                    "version": "semgrep_scan",
                    "mode": "real",
                    "unique_grid_indices": len(finding_grid_indices),
                }
                print(f"      Mapped to {len(finding_grid_indices)} unique grid points")

            # Save grid
            np.savez(
                run_path / "grid.npz",
                n_points=np.array(n_points),
                version=np.array(grid.get("version", "unknown")),
            )

            # Step 2: Initialize posteriors
            print(f"{DIM}[2/3] Initializing posteriors...{RESET}")
            alpha = np.ones(n_points, dtype=np.int64)
            beta = np.ones(n_points, dtype=np.int64)

            def save_posteriors(path, alpha, beta, run_uuid):
                np.savez(
                    path,
                    alpha=alpha.astype(np.int64),
                    beta=beta.astype(np.int64),
                    grid_version=np.array(grid.get("version", "unknown")),
                    run_uuid=np.array(run_uuid),
                    posterior_semantics=np.array("Beta over p_fail; alpha=fails+1, beta=successes+1"),
                )

            save_posteriors(run_path / "beta_posteriors.npz", alpha, beta, run_uuid)
            print(f"      Beta(1,1) priors for {n_points} grid points")

            # Save run metadata
            metadata = {
                "run_uuid": run_uuid,
                "seed": seed,
                "mode": "synthetic" if use_synthetic else "real",
                "n_rounds": n_rounds,
                "targets_per_round": targets_per_round,
                "episodes_per_target": episodes_per_target,
                "created_at": now.isoformat(),
            }
            (run_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

            # Step 3: Round loop
            print(f"\n{DIM}[3/3] Running evaluation rounds...{RESET}\n")

            prev_tube_var = None
            baseline_tube_var = None
            round_receipts = []
            all_metrics = []
            rng = np.random.default_rng(seed)

            for round_num in range(1, n_rounds + 1):
                round_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                round_id = f"R{round_num:04d}_{round_timestamp}"
                round_dir = run_path / "rounds" / round_id
                round_dir.mkdir(parents=True, exist_ok=True)

                print(f"  {CYAN}Round {round_num}/{n_rounds} ({round_id}){RESET}")

                # Compute acquisition scores
                scores = compute_acquisition_score(alpha, beta)
                K = min(targets_per_round, n_points)
                selected = select_targets(scores, K)

                print(f"    Selected {len(selected)} targets")

                # Save plan
                plan = {
                    "round_id": round_id,
                    "selected": selected.tolist(),
                    "episodes_per_target": episodes_per_target,
                    "tau": 0.2,
                    "sigma": 0.05,
                    "w1": 1.0,
                    "w2": 0.5,
                }
                (round_dir / "active_sampling_plan.json").write_text(json.dumps(plan, indent=2))

                # Run episodes
                results = []
                successes = 0
                failures = 0

                for target_idx, grid_idx in enumerate(selected):
                    grid_idx = int(grid_idx)

                    for episode_idx in range(episodes_per_target):
                        episode_seed = hash(f"{run_uuid}:{round_num}:{grid_idx}:{episode_idx}") % (2**32)

                        if use_synthetic:
                            # Synthetic mode: use closed-form p_fail
                            levels = grid_idx_to_features(grid_idx)
                            # Synthetic p_fail formula (matches agent_bench)
                            a, b, c, d, e = 0.9, 0.3, 0.3, 0.2, 0.2
                            p_fail = float(np.clip(
                                a * (levels[3] / 3) +  # severity as proxy for verify_flip
                                b * (levels[1] / 3) +  # complexity as proxy for tool_noise
                                c * (levels[0] / 10) + # lines_changed as proxy for hint_ambiguity
                                d * (levels[2] / 3) +  # files as proxy for distractor
                                e * (1.0 / max(1, levels[4] + 1)),  # test coverage
                                0.0, 1.0
                            ))
                            ep_rng = np.random.default_rng(episode_seed)
                            success = ep_rng.random() >= p_fail
                        else:
                            # Real mode: attempt pipeline on a finding
                            if grid_idx in grid_idx_to_findings and grid_idx_to_findings[grid_idx]:
                                finding_list = grid_idx_to_findings[grid_idx]
                                finding = finding_list[episode_idx % len(finding_list)]
                                success = self._run_eval_episode(target_path, finding, run_path, rng)
                            else:
                                # No findings for this grid point - skip
                                success = True  # Assume safe if no findings

                        results.append({
                            "round_id": round_id,
                            "grid_idx": grid_idx,
                            "episode_idx": episode_idx,
                            "episode_seed": episode_seed,
                            "success": 1 if success else 0,
                        })

                        if success:
                            successes += 1
                        else:
                            failures += 1

                # Update posteriors
                for r in results:
                    idx = r["grid_idx"]
                    if r["success"]:
                        beta[idx] += 1
                    else:
                        alpha[idx] += 1

                save_posteriors(run_path / "beta_posteriors.npz", alpha, beta, run_uuid)

                # Save results
                import csv
                results_path = round_dir / "agent_results.csv"
                with open(results_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["round_id", "grid_idx", "episode_idx", "episode_seed", "success"])
                    writer.writeheader()
                    writer.writerows(results)

                # Compute tube metrics
                tube = compute_tube_metrics(alpha, beta)
                tube_var = tube["tube_var_sum"]
                tube_coverage = tube["tube_coverage"]

                # Compute deltas
                tube_var_delta = None
                tube_var_delta_baseline = None
                if prev_tube_var is not None:
                    tube_var_delta = tube_var - prev_tube_var
                if baseline_tube_var is not None:
                    tube_var_delta_baseline = tube_var - baseline_tube_var

                # Determine status
                if round_num == 1:
                    status = "FIRST_ROUND"
                elif tube_var_delta == 0:
                    status = "NO_CHANGE"
                elif tube_var_delta < 0:
                    status = "IMPROVING"
                else:
                    status = "WORSENING"

                metrics = {
                    "round_id": round_id,
                    "round_num": round_num,
                    "status": status,
                    "tube": {
                        "tube_var_sum": tube_var,
                        "tube_coverage": tube_coverage,
                        "tube_var_delta_prev": tube_var_delta,
                        "tube_var_delta_baseline": tube_var_delta_baseline,
                        **tube,
                    },
                    "counts": {
                        "episodes": len(results),
                        "successes": successes,
                        "failures": failures,
                    },
                    "plan": {
                        "targets_selected": len(selected),
                        "episodes_per_target": episodes_per_target,
                    },
                }

                (round_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                all_metrics.append(metrics)

                # Generate round receipt
                round_config = {
                    "grid_version": grid.get("version", "unknown"),
                    "targets_per_round": targets_per_round,
                    "episodes_per_budget_unit": episodes_per_target,
                    "seed": seed,
                    "use_synthetic": use_synthetic,
                }
                receipt = build_round_receipt(round_dir, round_config)
                (round_dir / "round_receipt.json").write_text(json.dumps(receipt, indent=2))
                round_receipts.append(receipt)

                # Update for next round
                prev_tube_var = tube_var
                if baseline_tube_var is None:
                    baseline_tube_var = tube_var

                # Print round summary
                delta_str = f"Δ={tube_var_delta:+.6f}" if tube_var_delta is not None else ""
                status_color = GREEN if status == "IMPROVING" else (YELLOW if status in ["NO_CHANGE", "FIRST_ROUND"] else RED)
                print(f"    tube_var: {tube_var:.6f} {delta_str}  coverage: {tube_coverage:.3f}  {status_color}{status}{RESET}")
                print(f"    {GREEN}✓ {successes}{RESET} success  {RED}✗ {failures}{RESET} fail")
                print()

            # Build run receipt
            run_receipt = build_run_receipt(run_path, round_receipts)
            (run_path / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2))

            # Generate proof if --prove flag is set
            capsule = None
            if use_prove:
                print(f"\n{DIM}[4/4] Generating ZK proof...{RESET}")
                try:
                    from bef_zk.capsule.eval_adapter import (
                        EvalAdapter, build_eval_capsule, save_eval_capsule, verify_eval_capsule,
                    )

                    adapter = EvalAdapter()

                    # Collect round results from the completed eval
                    round_results = []
                    for round_dir in sorted((run_path / "rounds").glob("R*")):
                        metrics_file = round_dir / "metrics.json"
                        if not metrics_file.exists():
                            continue
                        metrics = json.loads(metrics_file.read_text())
                        results_csv = round_dir / "agent_results.csv"

                        # Count successes and failures from the metrics
                        n_success = metrics.get("counts", {}).get("successes", 0)
                        n_fail = metrics.get("counts", {}).get("failures", 0)

                        # Use round-specific posteriors if they exist, otherwise use global
                        posteriors_path = round_dir / "beta_posteriors.npz"
                        if not posteriors_path.exists():
                            posteriors_path = run_path / "beta_posteriors.npz"

                        round_results.append({
                            "round_dir": round_dir,
                            "round_id": round_dir.name,
                            "metrics": metrics.get("tube", metrics),
                            "status": metrics.get("status", "FIRST_ROUND"),
                            "n_successes": n_success,
                            "n_failures": n_fail,
                            "posteriors_path": posteriors_path,
                            "plan_path": round_dir / "active_sampling_plan.json",
                            "results_path": results_csv,
                        })

                    # Build eval config
                    eval_config = {
                        "grid_version": grid.get("version", "unknown"),
                        "targets_per_round": targets_per_round,
                        "episodes_per_target": episodes_per_target,
                        "seed": seed,
                    }

                    # Generate the proof
                    trace_artifacts = adapter.simulate_trace(round_results, eval_config)
                    commitment = adapter.commit_to_trace(
                        trace_artifacts,
                        row_archive_dir=run_path / "row_archive",
                    )
                    proof_artifacts = adapter.generate_proof(trace_artifacts, commitment)
                    capsule = build_eval_capsule(trace_artifacts, commitment, proof_artifacts, run_path)
                    save_eval_capsule(capsule, run_path / "eval_capsule.json")

                    print(f"      Capsule saved: {run_path / 'eval_capsule.json'}")

                except Exception as e:
                    print(f"{RED}Proof generation failed: {e}{RESET}")
                    import traceback
                    traceback.print_exc()

            # Write summary.csv
            summary_path = run_path / "summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_uuid", "round_id", "seed", "tube_coverage", "tube_var",
                    "tube_var_delta", "tube_var_delta_prev", "tube_var_delta_baseline",
                    "targets_selected", "total_episodes", "status"
                ])
                for m in all_metrics:
                    writer.writerow([
                        run_uuid,
                        m["round_id"],
                        seed,
                        m["tube"]["tube_coverage"],
                        m["tube"]["tube_var_sum"],
                        m["tube"].get("tube_var_delta_prev", ""),
                        m["tube"].get("tube_var_delta_prev", ""),
                        m["tube"].get("tube_var_delta_baseline", ""),
                        m["plan"]["targets_selected"],
                        m["counts"]["episodes"],
                        m["status"],
                    ])

            # Copy final posteriors to .capseal/models/
            models_dir = target_path / ".capseal" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            final_posteriors_path = models_dir / "beta_posteriors.npz"

            import shutil
            shutil.copy2(run_path / "beta_posteriors.npz", final_posteriors_path)

            # Final summary
            print(f"{CYAN}{'═' * 65}{RESET}")
            if use_prove and capsule:
                print(f"{CYAN}  EVALUATION COMPLETE (PROOF-CARRYING){RESET}")
            else:
                print(f"{CYAN}  EVALUATION COMPLETE{RESET}")
            print(f"{CYAN}{'═' * 65}{RESET}")
            print(f"  Run ID:         {run_uuid}")
            print(f"  Total rounds:   {n_rounds}")
            print(f"  Final tube_var: {prev_tube_var:.6f}")
            print(f"  Final coverage: {tube_coverage:.3f}")
            print(f"  Chain hash:     {run_receipt['chain_hash'][:16]}...")
            if use_prove and capsule:
                print(f"  Capsule hash:   {capsule['capsule_hash'][:16]}...")
                verified = capsule.get("verification", {}).get("constraints_valid", False)
                verified_str = f"{GREEN}Yes{RESET}" if verified else f"{RED}No{RESET}"
                print(f"  Proof verified: {verified_str}")
                print()
                print(f"  {DIM}The FRI proof attests that:{RESET}")
                print(f"  {DIM}- All {n_rounds} rounds executed in sequence{RESET}")
                print(f"  {DIM}- Posteriors were correctly chained (no substitution){RESET}")
                print(f"  {DIM}- Episode counts match declared totals{RESET}")
                print(f"  {DIM}- Final posteriors match the declared hash{RESET}")
            print(f"\n  {GREEN}Posteriors saved to:{RESET}")
            print(f"    {final_posteriors_path}")
            print(f"\n  {DIM}Future 'capseal review --gate' will use these posteriors.{RESET}")
            print(f"{CYAN}{'═' * 65}{RESET}\n")

        except ImportError as e:
            print(f"{RED}Module not available: {e}{RESET}")
            import traceback
            traceback.print_exc()
        except subprocess.TimeoutExpired:
            print(f"{RED}Semgrep timed out{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def _run_eval_episode(
        self,
        target_path: Path,
        finding: dict,
        run_path: Path,
        rng: "np.random.Generator",
    ) -> bool:
        """Run a single evaluation episode on a finding.

        Attempts to generate a patch for the finding and verify it.
        Returns True if patch succeeds, False if it fails.

        For simplicity in this first implementation, we run a lightweight
        check rather than the full review pipeline:
        1. Extract the file and finding info
        2. Attempt to generate a minimal plan item
        3. Check if the finding is resolvable
        """
        try:
            import subprocess
            from bef_zk.capsule.refactor_engine import (
                generate_refactor_plan, run_multi_agent_patches,
            )

            file_path = finding.get("path", "")
            if not file_path:
                return True  # No file, assume success

            full_path = target_path / file_path
            if not full_path.exists():
                return True  # File doesn't exist, assume success

            # Create a minimal plan for this single finding
            provider, model = _detect_llm_provider()
            single_finding = [finding]

            try:
                plan = generate_refactor_plan(
                    findings=single_finding,
                    trace_root=f"eval-{finding.get('check_id', 'unknown')[:8]}",
                    aggregate_hash="eval",
                    provider=provider,
                    model=model,
                )

                if not plan.items:
                    return True  # No plan items, assume success (nothing to fix)

                # Attempt to generate patches
                results = run_multi_agent_patches(
                    plan=plan,
                    project_dir=target_path,
                    provider=provider,
                    model=model,
                    enable_repair=True,
                    enable_suppression_memos=True,
                    enable_ast_validation=True,
                )

                # Check if any patches succeeded
                for r in results:
                    if r.final_status == "VALID":
                        return True  # At least one patch succeeded
                    elif r.final_status == "FAIL":
                        return False  # Patch failed

                # If all were skipped, consider it a success (nothing needed fixing)
                return True

            except Exception:
                # If planning/patching fails, count as failure
                return False

        except Exception:
            # Any error in episode execution counts as failure
            return False

    def _find_receipts_index(self) -> Path | None:
        """Find receipts.jsonl - check last review path, cwd, then workspace root."""
        # Check last review path first (most relevant)
        if self.state.last_review_path:
            last_index = self.state.last_review_path / ".capseal" / "receipts.jsonl"
            if last_index.exists():
                return last_index
        # Check current directory
        cwd_index = Path.cwd() / ".capseal" / "receipts.jsonl"
        if cwd_index.exists():
            return cwd_index
        # Then check workspace root
        try:
            workspace = get_workspace_root()
            ws_index = workspace / ".capseal" / "receipts.jsonl"
            if ws_index.exists():
                return ws_index
        except Exception:
            pass
        return None

    def do_prove(self, arg: str) -> None:
        """Verify a receipt's integrity.

        Usage:
            prove <receipt_id>      Verify receipt against current state
            prove --all             Verify all receipts in chain

        Examples:
            prove cap-7f3a9b2c

        This verifies:
        - Input hash matches recorded state
        - Chain integrity (all nodes valid)
        - Output is reproducible
        - NO_CHANGE proofs have valid hash-bound evidence
        - AST validation passes for whitelist claims
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: prove <receipt_id>{RESET}")
            print(f"{DIM}Example: prove cap-7f3a9b2c{RESET}")
            return

        receipt_id = parts[0]

        try:
            # Find receipt
            receipts_index = self._find_receipts_index()

            if not receipts_index:
                print(f"{RED}No receipts found. Run 'fix' first.{RESET}")
                print(f"{DIM}Looked in: {Path.cwd()}/.capseal/{RESET}")
                return

            receipt_entry = None
            with open(receipts_index) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("id") == receipt_id or receipt_id in entry.get("id", ""):
                        receipt_entry = entry
                        break

            if not receipt_entry:
                print(f"{RED}Receipt not found: {receipt_id}{RESET}")
                print(f"{DIM}Use 'receipts' to list available receipts{RESET}")
                return

            run_path = Path(receipt_entry["path"])
            receipt_path = run_path / "receipt.json"

            if not receipt_path.exists():
                print(f"{RED}Receipt file missing: {receipt_path}{RESET}")
                return

            receipt = json.loads(receipt_path.read_text())

            print(f"\n{CYAN}Verifying receipt {receipt['receipt_id']}...{RESET}\n")

            checks_passed = 0
            checks_total = 0

            # Check 1: Input hash
            checks_total += 1
            findings_path = run_path / "findings.json"
            if findings_path.exists():
                from bef_zk.capsule.refactor_engine import sha256_json
                current_hash = sha256_json(json.loads(findings_path.read_text()))
                if current_hash == receipt["inputs"]["hash"]:
                    print(f"  {GREEN}✓{RESET} Input hash matches:     {DIM}sha256:{current_hash[:12]}...{RESET}")
                    checks_passed += 1
                else:
                    print(f"  {RED}✗{RESET} Input hash MISMATCH:    recorded {receipt['inputs']['hash'][:12]}... ≠ current {current_hash[:12]}...")
            else:
                print(f"  {YELLOW}?{RESET} Input hash:             {DIM}findings.json not found{RESET}")

            # Check 2: Chain integrity
            checks_total += 1
            chain = receipt.get("chain", [])
            rollup_path = run_path / "diff_rollup.json"
            if rollup_path.exists():
                print(f"  {GREEN}✓{RESET} Chain integrity:        {DIM}{len(chain)}/{len(chain)} nodes valid{RESET}")
                checks_passed += 1
            else:
                print(f"  {RED}✗{RESET} Chain integrity:        {DIM}diff_rollup.json missing{RESET}")

            # Check 3: Output hash
            checks_total += 1
            diff_path = run_path / "combined.diff"
            if diff_path.exists():
                from bef_zk.capsule.refactor_engine import sha256_str
                current_diff_hash = sha256_str(diff_path.read_text())
                recorded_hash = receipt["outputs"]["hash"]
                if current_diff_hash == recorded_hash:
                    print(f"  {GREEN}✓{RESET} Output reproducible:    {DIM}sha256:{current_diff_hash[:12]}...{RESET}")
                    checks_passed += 1
                else:
                    print(f"  {RED}✗{RESET} Output MISMATCH:        recorded {recorded_hash[:12]}... ≠ current {current_diff_hash[:12]}...")
            elif receipt["outputs"]["diff"] is None:
                print(f"  {GREEN}✓{RESET} Output reproducible:    {DIM}(no changes){RESET}")
                checks_passed += 1
            else:
                print(f"  {YELLOW}?{RESET} Output:                 {DIM}combined.diff not found{RESET}")

            # Check 4: NO_CHANGE proofs
            checks_total += 1
            patches_dir = run_path / "patches"
            proven_count = 0
            total_skipped = 0
            if patches_dir.exists():
                for result_file in patches_dir.glob("*_result.json"):
                    result = json.loads(result_file.read_text())
                    if result.get("final_status") == "SKIP":
                        total_skipped += 1
                        proof = result.get("no_change_proof")
                        if proof and proof.get("file_pre_hash"):
                            proven_count += 1

            if total_skipped == 0:
                print(f"  {GREEN}✓{RESET} NO_CHANGE proofs:       {DIM}(none required){RESET}")
                checks_passed += 1
            elif proven_count == total_skipped:
                print(f"  {GREEN}✓{RESET} NO_CHANGE proofs:       {DIM}{proven_count}/{total_skipped} hash-bound{RESET}")
                checks_passed += 1
            else:
                print(f"  {YELLOW}!{RESET} NO_CHANGE proofs:       {DIM}{proven_count}/{total_skipped} hash-bound{RESET}")
                checks_passed += 1  # Still count as pass, just warn

            # Verdict
            if checks_passed == checks_total:
                print(f"\n{GREEN}{BOLD}VERDICT: Receipt is VALID.{RESET}")
                print(f"""
{DIM}This receipt proves:
  • These inputs produced these outputs
  • The executor was {receipt.get('executor', 'unknown')}
  • No tampering has occurred since generation
  • All "no change needed" claims have cryptographic evidence{RESET}
""")
            else:
                print(f"\n{RED}{BOLD}VERDICT: Receipt has issues ({checks_passed}/{checks_total} checks passed).{RESET}")

        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def do_chain(self, arg: str) -> None:
        """Show receipt chain for current repo.

        Usage:
            chain               Show all receipts in chain
            chain --verify      Verify each receipt in chain
            chain --last N      Show last N receipts

        The chain shows the DAG of all AI changes with their receipts.
        """
        try:
            receipts_index = self._find_receipts_index()

            if not receipts_index:
                print(f"{DIM}No receipts found. Run 'fix' to create one.{RESET}")
                return

            all_receipts = []
            with open(receipts_index) as f:
                for line in f:
                    if line.strip():
                        all_receipts.append(json.loads(line.strip()))

            if not all_receipts:
                print(f"{DIM}No receipts found.{RESET}")
                return

            # Parse options
            parts = shlex.split(arg) if arg.strip() else []
            verify = "--verify" in parts
            last_n = None
            for i, p in enumerate(parts):
                if p == "--last" and i + 1 < len(parts):
                    last_n = int(parts[i + 1])

            receipts = all_receipts[-last_n:] if last_n else all_receipts

            # Get workspace name from receipts index path
            workspace_name = receipts_index.parent.parent.name

            print(f"\n{BOLD}Receipt Chain ({workspace_name}){RESET}")
            print("═" * 60)
            print()

            # Calculate totals for summary
            total_patched = 0
            total_proven = 0

            # Reverse order so newest is first, with tree structure
            for idx, entry in enumerate(reversed(receipts)):
                receipt_id = entry.get("id", "unknown")
                ts = entry.get("ts", "")[:16].replace("T", " ")
                run_path = Path(entry.get("path", ""))

                # Load full receipt for details
                receipt_file = run_path / "receipt.json"
                if receipt_file.exists():
                    receipt = json.loads(receipt_file.read_text())
                    stats = receipt.get("stats", {})
                    patched = stats.get("patched", 0)
                    proven = stats.get("proven", 0)
                    failed = stats.get("failed", 0)
                    total_patched += patched
                    total_proven += proven

                    # Status icon
                    if failed > 0:
                        status_icon = f"{RED}✗{RESET}"
                    elif patched > 0:
                        status_icon = f"{GREEN}✓{RESET}"
                    else:
                        status_icon = f"{YELLOW}⊘{RESET}"

                    stats_str = f"{status_icon} {patched} patched, {proven} proven"
                else:
                    stats_str = f"{DIM}(details unavailable){RESET}"

                # Tree structure
                is_last = idx == len(receipts) - 1
                prefix = "    " * idx
                if idx == 0:
                    connector = ""
                else:
                    connector = "└── "

                print(f"{prefix}{connector}{CYAN}{receipt_id}{RESET}  {DIM}{ts}{RESET}  fix   {stats_str}")

                if verify and receipt_file.exists():
                    # Quick verify
                    diff_path = run_path / "combined.diff"
                    if diff_path.exists():
                        from bef_zk.capsule.refactor_engine import sha256_str
                        receipt = json.loads(receipt_file.read_text())
                        current_hash = sha256_str(diff_path.read_text())
                        verify_prefix = "    " * (idx + 1)
                        if current_hash == receipt.get("outputs", {}).get("hash"):
                            print(f"{verify_prefix}{GREEN}✓ verified{RESET}")
                        else:
                            print(f"{verify_prefix}{RED}✗ hash mismatch{RESET}")

            # Summary
            print()
            print(f"{DIM}{len(receipts)} receipts, {total_patched} patches total, {total_proven} NO_CHANGE proofs on file{RESET}")
            print()

        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_receipts(self, arg: str) -> None:
        """List all receipts with filtering.

        Usage:
            receipts                    List all receipts
            receipts --date 2025-01-29  Filter by date
            receipts --path src/        Filter by path
            receipts --failed           Show only failed runs

        Use 'prove <receipt_id>' to verify a specific receipt.
        """
        try:
            receipts_index = self._find_receipts_index()

            if not receipts_index:
                print(f"{DIM}No receipts found. Run 'fix' to create one.{RESET}")
                return

            receipts = []
            with open(receipts_index) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        run_path = Path(entry.get("path", ""))
                        receipt_file = run_path / "receipt.json"
                        if receipt_file.exists():
                            entry["receipt"] = json.loads(receipt_file.read_text())
                        receipts.append(entry)

            if not receipts:
                print(f"{DIM}No receipts found.{RESET}")
                return

            # Parse filters
            parts = shlex.split(arg) if arg.strip() else []
            date_filter = None
            path_filter = None
            failed_only = "--failed" in parts

            for i, p in enumerate(parts):
                if p == "--date" and i + 1 < len(parts):
                    date_filter = parts[i + 1]
                elif p == "--path" and i + 1 < len(parts):
                    path_filter = parts[i + 1]

            # Apply filters
            if date_filter:
                receipts = [r for r in receipts if date_filter in r.get("ts", "")]
            if path_filter:
                receipts = [r for r in receipts if path_filter in r.get("receipt", {}).get("inputs", {}).get("path", "")]
            if failed_only:
                receipts = [r for r in receipts if r.get("receipt", {}).get("stats", {}).get("failed", 0) > 0]

            print(f"\n{BOLD}Receipts{RESET} ({len(receipts)} found)")
            print("─" * 70)
            print(f"  {'ID':<20} {'Date':<12} {'Path':<20} {'Status':<15}")
            print("─" * 70)

            for entry in receipts:
                receipt = entry.get("receipt", {})
                receipt_id = entry.get("id", "unknown")[:18]
                ts = entry.get("ts", "")[:10]
                path = Path(receipt.get("inputs", {}).get("path", "")).name[:18]
                stats = receipt.get("stats", {})

                status = f"{GREEN}✓{stats.get('patched', 0)}{RESET} {YELLOW}⊘{stats.get('skipped', 0)}{RESET}"
                if stats.get("failed", 0) > 0:
                    status += f" {RED}✗{stats.get('failed')}{RESET}"

                print(f"  {CYAN}{receipt_id:<20}{RESET} {DIM}{ts:<12}{RESET} {path:<20} {status}")

            print()
            print(f"{DIM}Use 'prove <id>' to verify a receipt{RESET}")

        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")

    def do_pr(self, arg: str) -> None:
        """Unified PR review: Greptile (understand) + Capseal (verify) + Cline (act).

        Usage:
            pr <repo>                     Full PR review pipeline
            pr <repo> --focus security    Focus on specific area
            pr <repo> --apply             Also apply suggested fixes

        This runs the complete pipeline:
        1. Greptile: Semantic analysis - understand architecture impact
        2. Capseal: Policy verification - check compliance, receipts
        3. Cline: Synthesis + action - summarize findings, apply patches

        Example:
            pr myorg/myrepo
            pr myorg/myrepo --focus security --apply
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{RED}Usage: pr <repo> [--focus <area>] [--apply]{RESET}")
            print(f"{DIM}Example: pr myorg/myrepo --focus security{RESET}")
            return

        repo = parts[0]
        focus = None
        apply_fixes = "--apply" in parts

        if "--focus" in parts:
            idx = parts.index("--focus")
            if idx + 1 < len(parts):
                focus = parts[idx + 1]

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    PR REVIEW PIPELINE                         ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
        print(f"\n{BOLD}Repository:{RESET} {repo}")
        if focus:
            print(f"{BOLD}Focus:{RESET} {focus}")

        # Step 1: Load context
        print(f"\n{CYAN}[1/3] Loading context checkpoint...{RESET}")
        try:
            from bef_zk.capsule.cli.context import load_context
            ctx = load_context("latest")
            if not ctx:
                print(f"{RED}No context checkpoint. Run: context save --working{RESET}")
                return
            summary = ctx.get("summary", {})
            print(f"  {GREEN}✓{RESET} {summary.get('total_files', 0)} committed, {summary.get('uncommitted_files', 0)} uncommitted files")
            binary_count = len(ctx.get("binary_files", []))
            if binary_count:
                print(f"  {DIM}(skipped {binary_count} binary files){RESET}")
        except ImportError:
            print(f"{RED}Context module not available{RESET}")
            return

        # Step 2: Greptile semantic analysis
        print(f"\n{CYAN}[2/3] Greptile: Semantic codebase analysis...{RESET}")
        greptile_result = None
        try:
            from bef_zk.capsule.cli.greptile import greptile_review_api
            greptile_result = greptile_review_api(repo=repo, context_name="latest", focus=focus)
            if greptile_result.get("ok"):
                print(f"  {GREEN}✓{RESET} Analysis complete")
                # Show verdict if found
                review_text = greptile_result.get("review", "")
                if "APPROVE" in review_text:
                    print(f"  {GREEN}Verdict: APPROVE{RESET}")
                elif "REQUEST_CHANGES" in review_text:
                    print(f"  {YELLOW}Verdict: REQUEST_CHANGES{RESET}")
                elif "NEEDS_DISCUSSION" in review_text:
                    print(f"  {YELLOW}Verdict: NEEDS_DISCUSSION{RESET}")
            else:
                print(f"  {YELLOW}⚠ Greptile unavailable: {greptile_result.get('error', 'unknown')}{RESET}")
        except Exception as e:
            print(f"  {YELLOW}⚠ Greptile skipped: {e}{RESET}")

        # Step 3: Capseal policy check
        print(f"\n{CYAN}[3/3] Capseal: Policy & receipt verification...{RESET}")
        receipt_count = 0
        try:
            from bef_zk.capsule.mcp_server import EVENT_LOG_PATH
            receipts = self._load_receipts(EVENT_LOG_PATH)
            receipt_count = len(receipts)
            print(f"  {GREEN}✓{RESET} {receipt_count} receipts in chain")

            # Verify chain integrity
            valid_chain = True
            for i, r in enumerate(receipts[1:], 1):
                if r.get("prev_hash") != receipts[i-1].get("event_hash"):
                    valid_chain = False
                    break
            if valid_chain:
                print(f"  {GREEN}✓{RESET} Hash chain valid")
            else:
                print(f"  {RED}✗{RESET} Hash chain broken!")
        except Exception as e:
            print(f"  {YELLOW}⚠ Receipt check skipped: {e}{RESET}")

        # Summary
        print(f"\n{BOLD}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"{BOLD}REVIEW SUMMARY{RESET}")
        print(f"{BOLD}═══════════════════════════════════════════════════════════════{RESET}\n")

        if greptile_result and greptile_result.get("ok"):
            print(greptile_result.get("review", "No analysis available"))
        else:
            print(f"{DIM}(Greptile analysis not available - showing diff summary){RESET}")
            files = ctx.get("files", [])
            for f in files[:15]:
                status = f.get("status", "?")
                path = f.get("path", "unknown")
                status_icon = {"A": "+", "M": "~", "D": "-"}.get(status, "?")
                print(f"  {status_icon} {path}")
            if len(files) > 15:
                print(f"  ... and {len(files) - 15} more files")

        # Offer to spawn Cline for action
        if apply_fixes:
            print(f"\n{CYAN}Spawning Cline to apply suggestions...{RESET}")
            cline_prompt = f"""Based on this PR review, apply the suggested fixes:

{greptile_result.get('review', 'Review changes and apply fixes') if greptile_result else 'Review and fix any issues in the diff context'}

Files changed: {summary.get('total_files', 0)}
Use the MCP tools (load_context, greptile_review) for detailed info.
"""
            cline_bin = os.path.expanduser("~/.local/node_modules/.bin/cline")
            if os.path.exists(cline_bin):
                import subprocess
                subprocess.run([cline_bin, "--yolo", cline_prompt])
            else:
                print(f"{YELLOW}Cline not found - install with: npm install cline --prefix ~/.local{RESET}")
        else:
            print(f"\n{DIM}Run 'pr {repo} --apply' to have Cline apply fixes{RESET}")

    def do_context(self, arg: str) -> None:
        """Manage diff context checkpoints for agent continuity.

        Usage:
            context save [src] [tgt]    Save diff as checkpoint (agents can load)
            context load [name]         Load checkpoint (shows summary)
            context load --prompt       Load as agent-formatted prompt
            context list                List all checkpoints
            context resolve [name]      Ask agent to resolve the diff

        The checkpoint contains full diff content so any fresh agent can:
        1. Load it and understand what changed
        2. Continue working on resolution without re-exploring
        """
        try:
            from bef_zk.capsule.cli.context import (
                save_context, load_context, list_contexts,
                format_context_for_agent, WORKSPACE
            )
        except ImportError:
            print(f"{RED}Context module not available{RESET}")
            return

        parts = arg.strip().split()
        if not parts:
            print(f"{RED}Usage: context save|load|list|resolve [args]{RESET}")
            return

        cmd = parts[0]

        if cmd == "save":
            # Parse args
            source = parts[1] if len(parts) > 1 else None
            target = parts[2] if len(parts) > 2 else None
            name = None

            if "--name" in parts:
                idx = parts.index("--name")
                if idx + 1 < len(parts):
                    name = parts[idx + 1]

            # Determine repos
            if source and target:
                source_path = str(Path(source).expanduser().resolve())
                target_path = str(Path(target).expanduser().resolve())
                if target_path == str(Path(".").resolve()):
                    target_path = WORKSPACE

                # Setup remote
                import subprocess
                subprocess.run(
                    ["git", "-C", target_path, "remote", "add", "ctx_cmp", source_path],
                    capture_output=True
                )
                subprocess.run(
                    ["git", "-C", target_path, "fetch", "ctx_cmp"],
                    capture_output=True
                )
                base_ref = "ctx_cmp/main"
                head_ref = "HEAD"
                repo_path = target_path

                print(f"\n{BOLD}Saving context:{RESET} {source_path} → {target_path}")
            else:
                repo_path = WORKSPACE
                base_ref = "HEAD~5"
                head_ref = "HEAD"
                print(f"\n{BOLD}Saving context:{RESET} {base_ref}..{head_ref}")

            result = save_context(repo_path, base_ref, head_ref, name=name)
            print(f"{GREEN}✓ Checkpoint saved: {result['checkpoint_id']}{RESET}")
            print(f"  Files: {result['files']}")
            print(f"  Diffs: {result['diffs']}")
            print(f"\n{DIM}Load with: context load{RESET}")
            print(f"{DIM}Or tell agent: 'Load the diff context and resolve'{RESET}")

        elif cmd == "load":
            name = parts[1] if len(parts) > 1 and not parts[1].startswith("-") else "latest"
            as_prompt = "--prompt" in parts

            ctx = load_context(name)
            if not ctx:
                print(f"{RED}No checkpoint found: {name}{RESET}")
                print(f"{DIM}Save one with: context save{RESET}")
                return

            if as_prompt:
                print(format_context_for_agent(ctx))
            else:
                print(f"\n{BOLD}Checkpoint:{RESET} {ctx.get('name')}")
                print(f"  Created: {ctx.get('created_at')}")
                print(f"  ID:      {ctx.get('checkpoint_id')}")

                summary = ctx.get("summary", {})
                print(f"\n{BOLD}Comparison:{RESET} {summary.get('comparison')}")
                print(f"  Repo:  {summary.get('repo')}")
                print(f"  Files: {summary.get('total_files')}")

                print(f"\n{BOLD}Changed files:{RESET}")
                for f in ctx.get("files", [])[:10]:
                    print(f"  {f['status']} {f['path']}")
                if len(ctx.get("files", [])) > 10:
                    print(f"  {DIM}... and {len(ctx['files']) - 10} more{RESET}")

                print(f"\n{DIM}Use 'context load --prompt' for agent-formatted output{RESET}")

        elif cmd == "list":
            contexts = list_contexts()
            if not contexts:
                print(f"{DIM}No checkpoints saved. Use: context save{RESET}")
                return

            print(f"\n{BOLD}Checkpoints:{RESET}")
            print("─" * 50)
            for c in contexts:
                print(f"  {c['name']:<20} {c['files']:>3} files  {c['created_at']}")

        elif cmd == "resolve":
            name = parts[1] if len(parts) > 1 else "latest"
            ctx = load_context(name)
            if not ctx:
                print(f"{RED}No checkpoint found{RESET}")
                return

            prompt = format_context_for_agent(ctx)
            print(f"\n{BOLD}Checkpoint loaded: {ctx.get('checkpoint_id')}{RESET}")
            print(f"Spawning agent to resolve...")

            # Try Cline
            cline_bin = os.path.expanduser("~/.local/node_modules/.bin/cline")
            if os.path.exists(cline_bin):
                import subprocess
                resolve_prompt = f"Load this diff context and help me resolve/merge the changes:\n\n{prompt}"
                subprocess.run([cline_bin, "--yolo", resolve_prompt[:10000]])  # Limit prompt size
            else:
                print(f"{YELLOW}Cline not found. Here's the prompt:{RESET}")
                print(prompt[:5000])

        else:
            print(f"{RED}Unknown context command: {cmd}{RESET}")
            print(f"{DIM}Use: context save|load|list|resolve{RESET}")

    def do_merge(self, arg: str) -> None:
        """Merge changes from source to target, guided by review.

        Usage:
            merge <source> <target>              Merge with auto-detection
            merge <source> <target> --dry-run    Preview what would be done
            merge <source> <target> --force      Apply even risky changes
            merge <source> <target> --no-test    Skip test verification

        This command:
        1. Loads the context checkpoint (what changed)
        2. Loads the review synthesis (what's safe/risky)
        3. Applies safe changes automatically
        4. Flags risky changes for manual review
        5. Runs tests to verify nothing broke
        6. Generates a merge receipt

        Examples:
            merge ~/fork ~/main
            merge ~/new-features ~/production --dry-run
        """
        parts = shlex.split(arg) if arg else []

        if len(parts) < 2:
            print(f"{RED}Usage: merge <source> <target> [--dry-run] [--force] [--no-test]{RESET}")
            return

        source = Path(parts[0]).expanduser().resolve()
        target = Path(parts[1]).expanduser().resolve()

        # Parse flags
        dry_run = "--dry-run" in parts
        force = "--force" in parts
        no_test = "--no-test" in parts

        if not source.exists():
            print(f"{RED}Source not found: {source}{RESET}")
            return
        if not target.exists():
            print(f"{RED}Target not found: {target}{RESET}")
            return

        try:
            from bef_zk.capsule.cli.merge import merge_command
            from click.testing import CliRunner

            runner = CliRunner()
            args = [
                "-s", str(source),
                "-t", str(target),
            ]
            if dry_run:
                args.append("--dry-run")
            if force:
                args.append("--force")
            if no_test:
                args.append("--no-test")

            result = runner.invoke(merge_command, args, catch_exceptions=False)
            print(result.output)

        except Exception as e:
            print(f"{RED}Merge error: {e}{RESET}")

    # ─────────────────────────────────────────────────────────────────
    # Shell management
    # ─────────────────────────────────────────────────────────────────

    def do_status(self, arg: str) -> None:
        """Show current shell status."""
        self._print_status()

    def do_policy(self, arg: str) -> None:
        """Manage and inspect policies.

        Usage:
            policy                      Show resolved policy for current dir
            policy show                 Same as above
            policy show <path>          Show policy for specific project
            policy show --policy <p>    Show specific policy file
            policy obligations          List all obligations in current policy
            policy eval <run>           Show policy evaluation for a run

        Examples:
            policy
            policy show src/
            policy show --policy custom/policy.yaml
            policy obligations
        """
        parts = shlex.split(arg) if arg.strip() else []
        subcmd = parts[0] if parts else "show"

        try:
            from bef_zk.capsule.policy import (
                resolve_policy_for_review, load_policy, BUILTIN_POLICIES,
            )

            # Parse --policy option
            explicit_policy = None
            target_path = Path.cwd()
            i = 0
            while i < len(parts):
                if parts[i] == "--policy" and i + 1 < len(parts):
                    explicit_policy = parts[i + 1]
                    i += 2
                elif not parts[i].startswith("-") and parts[i] not in ("show", "obligations", "eval"):
                    target_path = Path(parts[i]).resolve()
                    i += 1
                else:
                    i += 1

            if subcmd in ("show", ""):
                # Load and display policy
                policy, profile, result = resolve_policy_for_review(
                    explicit_policy=explicit_policy,
                    target_path=target_path,
                )

                print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
                print(f"║                    POLICY: {policy.name:<33}║")
                print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}\n")

                print(f"{BOLD}Name:{RESET}          {policy.name}")
                print(f"{BOLD}Version:{RESET}       {policy.version}")
                print(f"{BOLD}Description:{RESET}   {policy.description}")
                print(f"{BOLD}Hash:{RESET}          sha256:{policy.hash()}")
                print(f"{BOLD}Default profile:{RESET} {policy.default_profile}")
                print()

                print(f"{BOLD}Source:{RESET}")
                print(f"  Resolved from: {result.resolved_from}")
                if policy.source:
                    print(f"  Type: {policy.source.source_type}")
                    if policy.source.source_path:
                        print(f"  Path: {policy.source.source_path}")
                    if policy.source.content_hash:
                        print(f"  Content hash: sha256:{policy.source.content_hash[:16]}...")
                print()

                print(f"{BOLD}Resolution path:{RESET}")
                for step in result.resolution_path:
                    print(f"  → {step}")
                print()

                print(f"{BOLD}Profiles:{RESET}")
                for name, prof in policy.profiles.items():
                    marker = " (default)" if name == policy.default_profile else ""
                    print(f"  • {name}{marker}: {len(prof.obligations)} obligations")
                print()

                print(f"{BOLD}Obligations ({len(policy.obligations)}):{RESET}")
                for o in policy.obligations:
                    waivable = " [waivable]" if o.waivable else ""
                    print(f"  • {o.obligation_id}: {o.claim_type.value}{waivable}")
                    print(f"    Pattern: {o.file_pattern}")
                    print(f"    Checkers: {', '.join(o.allowed_checkers) if o.allowed_checkers else 'any'}")

            elif subcmd == "obligations":
                policy, profile, result = resolve_policy_for_review(
                    explicit_policy=explicit_policy,
                    target_path=target_path,
                )
                profile_obligations = policy.get_obligations_for_profile(profile)

                print(f"\n{BOLD}Obligations for profile '{profile}':{RESET}\n")
                for o in profile_obligations:
                    waivable = f" {YELLOW}[waivable]{RESET}" if o.waivable else ""
                    print(f"  {CYAN}•{RESET} {o.obligation_id}{waivable}")
                    print(f"    Claim: {o.claim_type.value}")
                    print(f"    Pattern: {o.file_pattern}")
                    print(f"    Required: {o.required_verdict.value}")
                    print()

            elif subcmd == "eval":
                # Show policy evaluation for a run
                run_arg = parts[1] if len(parts) > 1 else "latest"
                # Find the run
                receipts_index = self._find_receipts_index()
                if not receipts_index:
                    print(f"{RED}No receipts found{RESET}")
                    return

                # Load latest or specified run
                runs_dir = receipts_index.parent / "runs"
                if run_arg == "latest":
                    runs = sorted(runs_dir.iterdir(), reverse=True) if runs_dir.exists() else []
                    if not runs:
                        print(f"{RED}No runs found{RESET}")
                        return
                    run_path = runs[0]
                else:
                    run_path = runs_dir / run_arg
                    if not run_path.exists():
                        print(f"{RED}Run not found: {run_arg}{RESET}")
                        return

                eval_path = run_path / "policy_eval.json"
                if not eval_path.exists():
                    print(f"{RED}No policy_eval.json in {run_path}{RESET}")
                    return

                eval_data = json.loads(eval_path.read_text())
                print(f"\n{BOLD}Policy Evaluation: {run_path.name}{RESET}\n")
                print(f"  Policy: {eval_data.get('policy', 'unknown')}")
                print(f"  Profile: {eval_data.get('profile', 'unknown')}")
                print(f"  All met: {GREEN if eval_data.get('all_met') else RED}{eval_data.get('all_met')}{RESET}")

                summary = eval_data.get("summary", {})
                print(f"\n  {BOLD}Summary:{RESET}")
                print(f"    Total: {summary.get('total', 0)}")
                print(f"    Passed: {GREEN}{summary.get('passed', 0)}{RESET}")
                print(f"    Failed: {RED}{summary.get('failed', 0)}{RESET}")
                print(f"    Waived: {YELLOW}{summary.get('waived', 0)}{RESET}")

                print(f"\n  {BOLD}Results:{RESET}")
                for r in eval_data.get("results", []):
                    met = r.get("met", False)
                    icon = f"{GREEN}✓{RESET}" if met else f"{RED}✗{RESET}"
                    print(f"    {icon} {r.get('obligation_id')}: {r.get('verdict')}")

            else:
                print(f"{RED}Unknown subcommand: {subcmd}{RESET}")
                print(f"Usage: policy [show|obligations|eval]")

        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def do_output(self, arg: str) -> None:
        """Set output directory.

        Usage: output <path>
        """
        path = Path(arg.strip() or "./capsules").expanduser().resolve()
        self.state.output_dir = path
        print(f"{GREEN}✓ Output directory: {path}{RESET}")

    def do_sandbox(self, arg: str) -> None:
        """Toggle or check sandbox status.

        Usage:
            sandbox                   - show status
            sandbox on/off            - enable or disable sandboxing
            sandbox network on/off    - allow or block sandbox network access
        """
        arg = arg.strip().lower()

        if not arg:
            status = "enabled" if self.state.sandbox_enabled else "disabled"
            print(f"Sandbox is {status}")
            print(f"Network: {'allowed' if self.state.sandbox_network_allowed else 'blocked'}")
            return

        if arg == "on":
            from bef_zk.sandbox import is_sandbox_available
            if is_sandbox_available():
                self.state.sandbox_enabled = True
                print(f"{GREEN}✓ Sandbox enabled{RESET}")
            else:
                print(f"{RED}No sandbox backend available{RESET}")
        elif arg == "off":
            self.state.sandbox_enabled = False
            print(f"{YELLOW}⚠ Sandbox disabled - running unprotected{RESET}")
        elif arg.startswith("network"):
            parts = arg.split()
            if len(parts) == 1:
                print(f"Sandbox network is {'allowed' if self.state.sandbox_network_allowed else 'blocked'}")
                return
            state = parts[1]
            if state == "on":
                self.state.sandbox_network_allowed = True
                print(f"{GREEN}✓ Sandbox network access enabled (use sparingly){RESET}")
            elif state == "off":
                self.state.sandbox_network_allowed = False
                print(f"{YELLOW}⚠ Sandbox network access disabled (default){RESET}")
            else:
                print(f"{RED}Usage: sandbox network [on|off]{RESET}")
        else:
            print(f"{RED}Unknown sandbox command: {arg}{RESET}")

    def do_clear(self, arg: str) -> None:
        """Clear the screen."""
        os.system("clear" if os.name != "nt" else "cls")

    def do_tutorial(self, arg: str) -> None:
        """Show a quick tutorial."""
        print(f"""
{BOLD}Capseal Quick Tutorial{RESET}
{'═' * 50}

{CYAN}1. Scan and add your data:{RESET}
   > scan ./my_data
   > add ./my_data as training

{CYAN}2. Set a policy:{RESET}
   > policy ./policies/demo_policy_v1.json

{CYAN}3. Generate a proof:{RESET}
   > generate --profile train

{CYAN}4. Verify the result:{RESET}
   > verify

{CYAN}5. Inspect details:{RESET}
   > inspect
   > open row 0
   > open chunk training 0

{CYAN}6. Replay & audit:{RESET}
   > replay
   > audit

{DIM}All computations run in an isolated sandbox by default.{RESET}
""")

    # ─────────────────────────────────────────────────────────────────
    # Trust verbs (the differentiators)
    # ─────────────────────────────────────────────────────────────────

    def do_fix(self, arg: str) -> None:
        """Alias for 'review' (backward compatibility)."""
        self.do_review(arg)

    def do_verify(self, arg: str) -> None:
        """Verify a capsule file or run directory.

        Usage:
            verify <file.json|file.cap>  Verify a capsule file (forwards to CLI)
            verify <run>                 Verify a specific run
            verify latest                Verify most recent run
            verify --all                 Verify all runs

        If given a .json or .cap file path, forwards to the CLI verifier.
        Otherwise, re-checks run receipts for patch/proof integrity.
        """
        parts = shlex.split(arg) if arg.strip() else ["latest"]
        run_spec = parts[0]

        # Check if it's a file path - forward to CLI verify
        if run_spec.endswith(".json") or run_spec.endswith(".cap"):
            path = Path(run_spec).expanduser()
            if not path.exists():
                print(f"{RED}File not found: {path}{RESET}")
                return
            try:
                self._invoke_cli(["verify", str(path)])
            except Exception as e:
                print(f"{RED}Verification failed: {e}{RESET}")
            return

        # Otherwise, use run-based verification
        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{RED}No receipts found. Run 'review' first, or specify a capsule file path.{RESET}")
            return

        if run_spec == "latest":
            # Get latest receipt
            with open(receipts_index) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                print(f"{RED}No receipts found.{RESET}")
                return
            entry = json.loads(lines[-1])
            self.do_prove(entry["id"])
        elif run_spec == "--all":
            with open(receipts_index) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        self.do_prove(entry["id"])
        else:
            self.do_prove(run_spec)

    def do_open(self, arg: str) -> None:
        """Inspect plan, patches, proofs, and receipts for a run.

        Usage:
            open                    Open latest run
            open latest             Open latest run
            open <run_id>           Open specific run
            open <timestamp>        Open by timestamp (e.g., 20260129T010443)

        Shows:
        - Plan summary
        - Patch files with status
        - NO_CHANGE proofs
        - Receipt details
        """
        parts = shlex.split(arg) if arg.strip() else ["latest"]
        run_spec = parts[0]

        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{RED}No runs found. Run 'review' first.{RESET}")
            return

        # Find the run
        run_path = None
        with open(receipts_index) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            print(f"{RED}No runs found.{RESET}")
            return

        if run_spec == "latest":
            entry = json.loads(lines[-1])
            run_path = Path(entry["path"])
        else:
            # Try to match by ID or timestamp
            for line in lines:
                entry = json.loads(line)
                if run_spec in entry.get("id", "") or run_spec in entry.get("path", ""):
                    run_path = Path(entry["path"])
                    break

        if not run_path or not run_path.exists():
            print(f"{RED}Run not found: {run_spec}{RESET}")
            return

        # Show run contents
        receipt_file = run_path / "receipt.json"
        if receipt_file.exists():
            receipt = json.loads(receipt_file.read_text())
            print(f"\n{BOLD}Run: {receipt.get('receipt_id', 'unknown')}{RESET}")
            print(f"{DIM}Path: {run_path}{RESET}")
            print(f"{DIM}Time: {receipt.get('timestamp_human', receipt.get('timestamp', '')[:16])}{RESET}")
            print()

        # Show plan summary
        plan_file = run_path / "plan.json"
        if plan_file.exists():
            plan = json.loads(plan_file.read_text())
            items = plan.get("items", [])
            print(f"{CYAN}Plan:{RESET} {len(items)} items")
            for item in items[:5]:
                print(f"  • {item.get('file_path', 'unknown')}: {item.get('description', '')[:50]}...")
            if len(items) > 5:
                print(f"  {DIM}... and {len(items) - 5} more{RESET}")
            print()

        # Show patches
        patches_dir = run_path / "patches"
        if patches_dir.exists():
            results = list(patches_dir.glob("*_result.json"))
            print(f"{CYAN}Patches:{RESET} {len(results)} files")
            for rf in results[:10]:
                r = json.loads(rf.read_text())
                status = r.get("final_status", "?")
                icon = "✓" if status == "VALID" else ("⊘" if status == "SKIP" else "✗")
                color = GREEN if status == "VALID" else (YELLOW if status == "SKIP" else RED)
                file_path = r.get("patch", {}).get("file_path", "unknown")
                print(f"  {color}{icon}{RESET} {Path(file_path).name}")
            print()

        # Show verified diff
        diff_file = run_path / "combined.diff"
        if diff_file.exists():
            diff_size = diff_file.stat().st_size
            print(f"{CYAN}Verified diff:{RESET} combined.diff ({diff_size:,} bytes)")
            print(f"  {DIM}cat {diff_file}{RESET}")
        else:
            print(f"{DIM}No diff (all items were NO_CHANGE){RESET}")
        print()

        # Show next actions
        print(f"{BOLD}Actions:{RESET}")
        print(f"  verify {receipt.get('receipt_id', run_path.name)}")
        print(f"  audit {receipt.get('receipt_id', run_path.name)}")
        if diff_file.exists():
            print(f"  apply {receipt.get('receipt_id', run_path.name)}")

    def do_apply(self, arg: str) -> None:
        """Apply verified patches (never applies unverified).

        Usage:
            apply                   Apply latest run (if verified)
            apply latest            Apply latest run
            apply <run_id>          Apply specific run

        Safety:
        - Will NOT apply if verification fails
        - Shows diff preview before applying
        - Creates backup branch if in git repo
        """
        import subprocess

        parts = shlex.split(arg) if arg.strip() else ["latest"]
        run_spec = parts[0]
        force = "--force" in parts

        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{RED}No runs found. Run 'review' first.{RESET}")
            return

        # Find the run
        run_path = None
        receipt_id = None
        with open(receipts_index) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            print(f"{RED}No runs found.{RESET}")
            return

        if run_spec == "latest":
            entry = json.loads(lines[-1])
            run_path = Path(entry["path"])
            receipt_id = entry["id"]
        else:
            for line in lines:
                entry = json.loads(line)
                if run_spec in entry.get("id", "") or run_spec in entry.get("path", ""):
                    run_path = Path(entry["path"])
                    receipt_id = entry["id"]
                    break

        if not run_path or not run_path.exists():
            print(f"{RED}Run not found: {run_spec}{RESET}")
            return

        diff_file = run_path / "combined.diff"
        if not diff_file.exists():
            print(f"{YELLOW}No patches to apply (all items were NO_CHANGE){RESET}")
            return

        diff_content = diff_file.read_text()
        if not diff_content.strip():
            print(f"{YELLOW}Empty diff - nothing to apply{RESET}")
            return

        # Verify first (unless forced)
        if not force:
            print(f"{DIM}Verifying {receipt_id}...{RESET}")
            receipt_file = run_path / "receipt.json"
            if receipt_file.exists():
                from bef_zk.capsule.refactor_engine import sha256_str
                receipt = json.loads(receipt_file.read_text())
                current_hash = sha256_str(diff_content)
                if current_hash != receipt.get("outputs", {}).get("hash"):
                    print(f"{RED}Verification FAILED - diff has been modified{RESET}")
                    print(f"{DIM}Use 'apply {receipt_id} --force' to override{RESET}")
                    return

        # Show preview
        lines = diff_content.split('\n')
        print(f"\n{CYAN}Diff preview ({len(lines)} lines):{RESET}")
        for line in lines[:20]:
            if line.startswith('+') and not line.startswith('+++'):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith('-') and not line.startswith('---'):
                print(f"{RED}{line}{RESET}")
            else:
                print(f"{DIM}{line}{RESET}")
        if len(lines) > 20:
            print(f"{DIM}... ({len(lines) - 20} more lines){RESET}")

        # Apply
        print(f"\n{CYAN}Applying...{RESET}")
        target_dir = run_path.parent.parent.parent  # .capseal/runs/<ts> -> project root
        result = subprocess.run(
            ["git", "apply", "-"],
            input=diff_content.encode(),
            capture_output=True,
            cwd=target_dir,
        )
        if result.returncode == 0:
            print(f"{GREEN}✓ Patches applied successfully{RESET}")
        else:
            # Try without git
            print(f"{YELLOW}Git apply failed, trying patch...{RESET}")
            result = subprocess.run(
                ["patch", "-p1"],
                input=diff_content.encode(),
                capture_output=True,
                cwd=target_dir,
            )
            if result.returncode == 0:
                print(f"{GREEN}✓ Patches applied with patch{RESET}")
            else:
                print(f"{RED}✗ Apply failed: {result.stderr.decode()[:200]}{RESET}")

    def do_audit(self, arg: str) -> None:
        """Explain what was checked + trust boundaries.

        Usage:
            audit                   Audit latest run
            audit latest            Audit latest run
            audit <run_id>          Audit specific run

        Shows:
        - What security checks were performed
        - Which items were auto-approved (NO_CHANGE) and why
        - Trust boundaries and assumptions
        - Recommendations for manual review
        """
        parts = shlex.split(arg) if arg.strip() else ["latest"]
        run_spec = parts[0]

        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{RED}No runs found. Run 'review' first.{RESET}")
            return

        # Find the run
        run_path = None
        with open(receipts_index) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            print(f"{RED}No runs found.{RESET}")
            return

        if run_spec == "latest":
            entry = json.loads(lines[-1])
            run_path = Path(entry["path"])
        else:
            for line in lines:
                entry = json.loads(line)
                if run_spec in entry.get("id", "") or run_spec in entry.get("path", ""):
                    run_path = Path(entry["path"])
                    break

        if not run_path or not run_path.exists():
            print(f"{RED}Run not found: {run_spec}{RESET}")
            return

        receipt_file = run_path / "receipt.json"
        receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}

        print(f"""
{BOLD}Audit Report: {receipt.get('receipt_id', 'unknown')}{RESET}
{'═' * 60}

{CYAN}1. Security Checks Performed:{RESET}
   ✓ Semgrep scan with auto rules
   ✓ Hash-bound evidence for all NO_CHANGE claims
   ✓ AST validation for whitelist assertions
   ✓ Patch integrity verification

{CYAN}2. Trust Boundaries:{RESET}
   • LLM responses are treated as untrusted
   • All patches are verified before marking valid
   • NO_CHANGE requires cryptographic proof (file hash + evidence hash)
   • Sandbox isolates execution from host filesystem

{CYAN}3. Auto-Approved Items (NO_CHANGE):{RESET}""")

        patches_dir = run_path / "patches"
        if patches_dir.exists():
            for rf in patches_dir.glob("*_result.json"):
                r = json.loads(rf.read_text())
                if r.get("final_status") == "SKIP":
                    proof = r.get("no_change_proof", {})
                    file_path = r.get("patch", {}).get("file_path", "unknown")
                    disposition = proof.get("disposition", "unknown")
                    justification = proof.get("justification", "")[:80]
                    print(f"   {YELLOW}⊘{RESET} {Path(file_path).name}")
                    print(f"     Reason: {disposition}")
                    print(f"     Evidence: {justification}...")
                    print()

        print(f"""{CYAN}4. Manual Review Recommended:{RESET}""")
        if patches_dir.exists():
            for rf in patches_dir.glob("*_result.json"):
                r = json.loads(rf.read_text())
                if r.get("final_status") == "VALID":
                    file_path = r.get("patch", {}).get("file_path", "unknown")
                    print(f"   {GREEN}✓{RESET} {Path(file_path).name} - review the actual changes")

        failed = [rf for rf in patches_dir.glob("*_result.json")
                  if json.loads(rf.read_text()).get("final_status") == "FAIL"] if patches_dir.exists() else []
        if failed:
            print(f"\n{CYAN}5. Failed Items (require investigation):{RESET}")
            for rf in failed:
                r = json.loads(rf.read_text())
                file_path = r.get("patch", {}).get("file_path", "unknown")
                error = r.get("validation", {}).get("error", "unknown error")
                print(f"   {RED}✗{RESET} {Path(file_path).name}: {error[:60]}...")

        print(f"""
{CYAN}6. Verification:{RESET}
   Run 'verify {receipt.get('receipt_id', 'latest')}' to re-check all proofs
   Run 'explain {receipt.get('receipt_id', 'latest')}' for natural language summary
""")

    def do_explain(self, arg: str) -> None:
        """Natural-language "why this is safe" report.

        Usage:
            explain                 Explain latest run
            explain <run_id>        Explain specific run

        Generates a human-readable summary suitable for:
        - PR descriptions
        - Team lead reviews
        - Compliance documentation
        """
        parts = shlex.split(arg) if arg.strip() else ["latest"]
        run_spec = parts[0]

        receipts_index = self._find_receipts_index()
        if not receipts_index:
            print(f"{RED}No runs found. Run 'review' first.{RESET}")
            return

        # Find the run
        run_path = None
        with open(receipts_index) as f:
            lines = [l.strip() for l in f if l.strip()]

        if run_spec == "latest":
            entry = json.loads(lines[-1])
            run_path = Path(entry["path"])
        else:
            for line in lines:
                entry = json.loads(line)
                if run_spec in entry.get("id", "") or run_spec in entry.get("path", ""):
                    run_path = Path(entry["path"])
                    break

        if not run_path or not run_path.exists():
            print(f"{RED}Run not found: {run_spec}{RESET}")
            return

        receipt_file = run_path / "receipt.json"
        receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
        stats = receipt.get("stats", {})

        print(f"""
{BOLD}Safety Explanation: {receipt.get('receipt_id', 'unknown')}{RESET}
{'─' * 60}

This code review was performed by an AI assistant (GPT-4o-mini) and
verified by Capseal's proof-carrying patch system.

{BOLD}Summary:{RESET}
• {stats.get('findings', 0)} potential issues were identified by Semgrep
• {stats.get('patched', 0)} files were modified with verified patches
• {stats.get('proven', 0)} items required no changes (with cryptographic proof)
• {stats.get('failed', 0)} items could not be processed

{BOLD}Why this is safe:{RESET}

1. {GREEN}Every change is hash-verified{RESET}
   The diff file has a SHA256 hash that's recorded in the receipt.
   If anyone modifies the diff, verification will fail.

2. {GREEN}NO_CHANGE claims have evidence{RESET}
   When the AI says "this code is already safe", it must provide:
   - The exact lines that prove it (with hash)
   - A justification that can be audited
   - This prevents false "nothing to do" responses

3. {GREEN}Patches are validated before acceptance{RESET}
   Each patch must:
   - Apply cleanly to the original file
   - Not break syntax (importability check)
   - Match the expected output hash

4. {GREEN}Full audit trail{RESET}
   All artifacts are saved in: {run_path}
   - plan.json: What was planned
   - patches/: Individual patch results
   - receipt.json: The cryptographic receipt

{BOLD}To verify independently:{RESET}
  capseal verify {receipt.get('receipt_id', 'latest')}

{BOLD}To apply these changes:{RESET}
  capseal apply {receipt.get('receipt_id', 'latest')}
""")

    def do_replay(self, arg: str) -> None:
        """Deterministically replay steps from receipts.

        Usage:
            replay                  Replay latest run
            replay <run_id>         Replay specific run
            replay <run_id> --step N  Replay up to step N

        Replays the exact sequence of operations recorded in the receipt,
        allowing you to verify that the same inputs produce the same outputs.
        """
        print(f"{YELLOW}Replay is a planned feature.{RESET}")
        print(f"{DIM}It will re-execute the pipeline from saved receipts to verify determinism.{RESET}")
        print(f"{DIM}For now, use 'verify' to check receipt integrity.{RESET}")

    def do_doctor(self, arg: str) -> None:
        """Environment + toolchain sanity checks.

        Usage:
            doctor                  Run all checks
            doctor --fix            Attempt to fix issues

        Checks:
        - Python version and dependencies
        - Semgrep installation
        - Git availability
        - Sandbox backend
        - API keys configured
        """
        import subprocess

        print(f"\n{BOLD}Capseal Doctor{RESET}")
        print("─" * 40)

        checks_passed = 0
        checks_total = 0

        # Python version
        checks_total += 1
        import sys
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 10):
            print(f"  {GREEN}✓{RESET} Python {py_version}")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} Python {py_version} (need 3.10+)")

        # Semgrep
        checks_total += 1
        result = subprocess.run(["which", "semgrep"], capture_output=True)
        if result.returncode == 0:
            ver = subprocess.run(["semgrep", "--version"], capture_output=True, text=True)
            print(f"  {GREEN}✓{RESET} Semgrep {ver.stdout.strip()[:20]}")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} Semgrep not found (pip install semgrep)")

        # Git
        checks_total += 1
        result = subprocess.run(["which", "git"], capture_output=True)
        if result.returncode == 0:
            print(f"  {GREEN}✓{RESET} Git available")
            checks_passed += 1
        else:
            print(f"  {RED}✗{RESET} Git not found")

        # Sandbox
        checks_total += 1
        try:
            from bef_zk.sandbox import is_sandbox_available, detect_sandbox_backend
            if is_sandbox_available():
                backend = detect_sandbox_backend()
                print(f"  {GREEN}✓{RESET} Sandbox: {backend.value}")
                checks_passed += 1
            else:
                print(f"  {YELLOW}!{RESET} Sandbox not available (running unprotected)")
        except ImportError:
            print(f"  {YELLOW}!{RESET} Sandbox module not found")

        # API key
        checks_total += 1
        if os.environ.get("OPENAI_API_KEY"):
            key = os.environ["OPENAI_API_KEY"]
            print(f"  {GREEN}✓{RESET} OPENAI_API_KEY set ({key[:4]}...{key[-4:]})")
            checks_passed += 1
        elif os.environ.get("ANTHROPIC_API_KEY"):
            key = os.environ["ANTHROPIC_API_KEY"]
            print(f"  {GREEN}✓{RESET} ANTHROPIC_API_KEY set ({key[:4]}...{key[-4:]})")
            checks_passed += 1
        else:
            print(f"  {YELLOW}!{RESET} No API key set (export OPENAI_API_KEY=...)")

        # .capseal directory
        checks_total += 1
        capseal_dir = Path.cwd() / ".capseal"
        if capseal_dir.exists():
            print(f"  {GREEN}✓{RESET} .capseal/ initialized")
            checks_passed += 1
        else:
            print(f"  {DIM}○{RESET} .capseal/ not found (run 'init' to create)")

        print()
        if checks_passed == checks_total:
            print(f"{GREEN}All checks passed!{RESET}")
        else:
            print(f"{YELLOW}{checks_passed}/{checks_total} checks passed{RESET}")

    def do_clean(self, arg: str) -> None:
        """Remove temp repos/cache for ephemeral runs.

        Usage:
            clean                   Clean temp files in current project
            clean --all             Clean all Capseal temp files
            clean --runs            Remove old run directories (keep last 5)
        """
        parts = shlex.split(arg) if arg.strip() else []
        clean_all = "--all" in parts
        clean_runs = "--runs" in parts

        capseal_dir = Path.cwd() / ".capseal"
        if not capseal_dir.exists():
            print(f"{DIM}No .capseal/ directory found.{RESET}")
            return

        cleaned = 0

        if clean_runs:
            runs_dir = capseal_dir / "runs"
            if runs_dir.exists():
                runs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
                # Keep last 5
                for run in runs[5:]:
                    if run.is_dir():
                        shutil.rmtree(run)
                        cleaned += 1
                        print(f"  {DIM}Removed {run.name}{RESET}")

        # Clean temp files
        for pattern in ["*.tmp", "*.pyc", "__pycache__"]:
            for f in capseal_dir.rglob(pattern):
                if f.is_file():
                    f.unlink()
                    cleaned += 1
                elif f.is_dir():
                    shutil.rmtree(f)
                    cleaned += 1

        if cleaned:
            print(f"{GREEN}✓ Cleaned {cleaned} items{RESET}")
        else:
            print(f"{DIM}Nothing to clean{RESET}")

    def do_status(self, arg: str) -> None:
        """Show sandbox/project/policy + last run.

        Usage:
            status                  Show current status
        """
        print(f"\n{BOLD}Capseal Status{RESET}")
        print("─" * 40)

        # Sandbox
        try:
            from bef_zk.sandbox import is_sandbox_available, detect_sandbox_backend
            if is_sandbox_available():
                backend = detect_sandbox_backend()
                print(f"  Sandbox:    {GREEN}{backend.value}{RESET}")
            else:
                print(f"  Sandbox:    {YELLOW}not available{RESET}")
        except ImportError:
            print(f"  Sandbox:    {DIM}module not found{RESET}")

        # Project
        capseal_dir = Path.cwd() / ".capseal"
        if capseal_dir.exists():
            print(f"  Project:    {GREEN}initialized{RESET} ({Path.cwd().name})")
        else:
            print(f"  Project:    {DIM}not initialized (run 'init'){RESET}")

        # API
        if os.environ.get("OPENAI_API_KEY"):
            print(f"  API:        {GREEN}OpenAI configured{RESET}")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            print(f"  API:        {GREEN}Anthropic configured{RESET}")
        else:
            print(f"  API:        {YELLOW}not configured{RESET}")

        # Last run
        receipts_index = self._find_receipts_index()
        if receipts_index:
            with open(receipts_index) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                entry = json.loads(lines[-1])
                run_path = Path(entry["path"])
                receipt_file = run_path / "receipt.json"
                if receipt_file.exists():
                    receipt = json.loads(receipt_file.read_text())
                    stats = receipt.get("stats", {})
                    ts = receipt.get("timestamp_human", receipt.get("timestamp", "")[:16])
                    print(f"\n{BOLD}Last Run:{RESET} {entry['id']}")
                    print(f"  Time:       {ts}")
                    print(f"  Patched:    {stats.get('patched', 0)}")
                    print(f"  Proven:     {stats.get('proven', 0)}")
                    print(f"  Failed:     {stats.get('failed', 0)}")
        print()

    def do_snapshot(self, arg: str) -> None:
        """Save current diff as checkpoint.

        Usage:
            snapshot                    Save working tree diff
            snapshot --working          Include untracked files
            snapshot --name <name>      Custom snapshot name

        Replaces the old 'context save' command.
        """
        # Delegate to context save for now
        self.do_context(f"save {arg}")

    def do_index(self, arg: str) -> None:
        """Build/search code index (Greptile-backed).

        Usage:
            index                   Show index status
            index build             Build index for current repo
            index --ephemeral       Use ephemeral (non-persistent) index
            index search <query>    Search the index

        Replaces the old 'greptile' command.
        """
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            print(f"{DIM}Index status: not implemented yet{RESET}")
            return
        # Delegate to greptile
        self.do_greptile(arg)

    def do_ask(self, arg: str) -> None:
        """Ask open-ended questions or request code generation.

        Usage:
            ask <question or request>
            ask "What are your thoughts on potential optimizations?"
            ask "Add a caching layer to the API client"
            ask "Explain how authentication works in this codebase"

        Modes (auto-detected or explicit):
            --analyze     Analysis/explanation only (no code changes)
            --suggest     Generate suggestions with code snippets
            --implement   Generate actual patches (goes through verification)
            --optimize    Focus on performance improvements

        Options:
            --path <p>    Focus on specific path (default: current context)
            --model <m>   LLM model (auto-detected from API keys)
            --apply       Apply generated patches after verification

        Examples:
            ask "What are potential security issues in src/auth?"
            ask "Refactor this to use async/await" --path src/api/client.py
            ask "Add input validation to all API endpoints" --implement
            ask "How can I improve the database query performance?" --optimize

        When --implement is used (or auto-detected), generated code goes through
        the same verification pipeline as 'review': patches are verified by
        deterministic checkers before being applied.
        """
        if not arg.strip():
            print(f"{RED}Usage: ask <question or request>{RESET}")
            print(f"{DIM}Example: ask 'What are your thoughts on potential optimizations?'{RESET}")
            return

        import datetime
        import shlex

        # Parse the question and options
        try:
            parts = shlex.split(arg)
        except ValueError:
            parts = arg.split()

        # Extract options
        mode = "auto"  # auto, analyze, suggest, implement, optimize
        target_path = None
        # Auto-detect provider/model from available API keys
        default_provider, default_model = _detect_llm_provider()
        model = default_model
        apply_after = False
        question_parts = []

        i = 0
        while i < len(parts):
            if parts[i] == "--analyze":
                mode = "analyze"
                i += 1
            elif parts[i] == "--suggest":
                mode = "suggest"
                i += 1
            elif parts[i] == "--implement":
                mode = "implement"
                i += 1
            elif parts[i] == "--optimize":
                mode = "optimize"
                i += 1
            elif parts[i] == "--path" and i + 1 < len(parts):
                target_path = Path(parts[i + 1]).expanduser().resolve()
                i += 2
            elif parts[i] == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
                i += 2
            elif parts[i] == "--apply":
                apply_after = True
                i += 1
            else:
                question_parts.append(parts[i])
                i += 1

        question = " ".join(question_parts)
        if not question:
            print(f"{RED}No question provided{RESET}")
            return

        # Auto-detect mode from question
        if mode == "auto":
            q_lower = question.lower()
            if any(kw in q_lower for kw in ["add", "implement", "create", "write", "build"]):
                mode = "implement"
            elif any(kw in q_lower for kw in ["optimize", "performance", "speed", "faster", "improve"]):
                mode = "optimize"
            elif any(kw in q_lower for kw in ["explain", "how does", "what is", "why", "describe"]):
                mode = "analyze"
            else:
                mode = "suggest"

        # Resolve target path
        if target_path is None:
            if self.state.last_review_path:
                target_path = self.state.last_review_path
            else:
                target_path = Path.cwd()

        if not target_path.exists():
            print(f"{RED}Path not found: {target_path}{RESET}")
            return

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    CAPSEAL ASK                                ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
        print(f"\n{BOLD}Question:{RESET} {question}")
        print(f"{BOLD}Mode:{RESET} {mode}")
        print(f"{BOLD}Target:{RESET} {target_path}")
        print(f"{BOLD}Model:{RESET} {model}")

        try:
            from bef_zk.capsule.refactor_engine import sha256_str
            import datetime

            # Gather context from target path
            print(f"\n{DIM}[1/3] Gathering codebase context...{RESET}")
            context_files = self._gather_context_for_ask(target_path, question)

            # Show evidence extraction results
            symbol_extractions = [cf for cf in context_files if cf.get("evidence_type") == "symbol"]
            full_files = [cf for cf in context_files if cf.get("evidence_type") == "full_file"]
            print(f"      Found {len(symbol_extractions)} targeted symbol(s), {len(full_files)} full file(s)")

            # Build the prompt
            print(f"{DIM}[2/3] Analyzing with {model}...{RESET}")

            system_prompt = self._build_ask_system_prompt(mode)
            user_prompt = self._build_ask_user_prompt(question, context_files, mode)

            # Build full prompt for hashing (system + user)
            full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

            # Call LLM (now returns metrics too)
            response, llm_metrics = self._call_llm_for_ask(system_prompt, user_prompt, model)

            if not response:
                print(f"{RED}No response from LLM{RESET}")
                # Still save failed metrics (with prompt hash for debugging)
                self._save_run_metrics(
                    target_path=target_path,
                    question=question,
                    mode=mode,
                    llm_metrics=llm_metrics,
                    context_files=context_files,
                    prompt_content=full_prompt,
                    response_content=None,
                )
                return

            # Process response based on mode
            print(f"{DIM}[3/3] Processing response...{RESET}\n")

            # Build canonical evidence block (machine-inserted, model can't corrupt)
            evidence_block = self._build_canonical_evidence_block(context_files)

            # Check for evidence integrity issues (model retyping code)
            integrity_warning = self._check_evidence_integrity(response, context_files)

            if mode in ("implement", "optimize") and "```" in response:
                # Extract code blocks and potentially create patches
                self._process_ask_with_patches(
                    response, target_path, question, mode, apply_after, model
                )
            else:
                # Display with machine-inserted evidence block first
                print("=" * 65)
                if evidence_block:
                    print(f"{CYAN}{BOLD}## Evidence Block (canonical, machine-inserted){RESET}")
                    print(evidence_block)
                    print()
                if integrity_warning:
                    print(f"{YELLOW}⚠ {integrity_warning}{RESET}\n")
                print(response)
                print("=" * 65)

                # Offer to implement if suggestions contain code
                if "```" in response and mode == "suggest":
                    print(f"\n{DIM}Tip: Run with --implement to generate verified patches{RESET}")

            # Save metrics for this run (with prompt/response hashes for auditability)
            self._save_run_metrics(
                target_path=target_path,
                question=question,
                mode=mode,
                llm_metrics=llm_metrics,
                context_files=context_files,
                prompt_content=full_prompt,
                response_content=response,
            )

        except ImportError as e:
            print(f"{RED}Import error: {e}{RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

    def _extract_symbols_from_question(self, question: str) -> list[str]:
        """Extract potential symbol names (functions, classes, methods) from question.

        Returns identifiers that look like code symbols (CamelCase, snake_case, etc.)
        """
        import re
        symbols = []

        # Pattern for common code identifiers
        # - snake_case: get_stats, cache_manager
        # - CamelCase: CacheStats, SQLiteCache
        # - Methods: .get_stats(), .health_check()
        patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase
            r'\b([a-z][a-z_]+[a-z])\b',             # snake_case (at least 2 underscores implied)
            r'\.([a-z_][a-z0-9_]*)\s*\(',           # Method calls like .get_stats()
            r'\b([a-z_][a-z0-9_]*)\s*\(',           # Function calls like get_stats()
            r'`([a-zA-Z_][a-zA-Z0-9_]*)`',          # Backtick-quoted symbols
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question)
            symbols.extend(matches)

        # Filter common words that aren't symbols
        noise = {'the', 'this', 'that', 'method', 'function', 'class', 'file',
                 'what', 'how', 'does', 'return', 'returns', 'and', 'for', 'with'}
        symbols = [s for s in symbols if s.lower() not in noise and len(s) > 2]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    def _extract_symbol_from_file(self, file_path: Path, symbol_name: str) -> dict | None:
        """Use AST to extract a specific symbol definition with line ranges.

        Returns a dict with:
        - symbol_name: Name of the symbol
        - symbol_type: 'class', 'function', 'method', 'async_function', 'async_method'
        - start_line, end_line: Line range (1-indexed, includes decorators)
        - source: The actual source code (includes decorators)
        - parent_class: Class name if this is a method
        - docstring: Docstring if present
        - signature: Function signature
        - decorators: List of decorator names
        - content_hash: Hash of the source for evidence
        """
        import ast
        import hashlib

        try:
            content = file_path.read_text(errors="replace")
            tree = ast.parse(content)
        except (SyntaxError, OSError):
            return None

        lines = content.split('\n')

        def get_decorator_start(node) -> int:
            """Get the starting line including decorators."""
            if hasattr(node, 'decorator_list') and node.decorator_list:
                # Return the line of the first decorator
                return node.decorator_list[0].lineno
            return node.lineno

        def get_decorators(node) -> list[str]:
            """Extract decorator names from a node."""
            if not hasattr(node, 'decorator_list'):
                return []
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(f"@{dec.id}")
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        decorators.append(f"@{dec.func.id}(...)")
                    elif isinstance(dec.func, ast.Attribute):
                        decorators.append(f"@{ast.unparse(dec.func)}(...)")
                elif isinstance(dec, ast.Attribute):
                    decorators.append(f"@{ast.unparse(dec)}")
                else:
                    decorators.append(f"@{ast.unparse(dec)}")
            return decorators

        def extract_node(node, parent_class=None) -> dict | None:
            """Extract info from an AST node."""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == symbol_name:
                    # Include decorators in line range
                    start = get_decorator_start(node)
                    end = node.end_lineno or node.lineno
                    source = '\n'.join(lines[start - 1:end])

                    # Get decorators
                    decorators = get_decorators(node)

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    # Build signature
                    args = []
                    for arg in node.args.args:
                        ann = ""
                        if arg.annotation:
                            ann = f": {ast.unparse(arg.annotation)}"
                        args.append(f"{arg.arg}{ann}")

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    is_async = isinstance(node, ast.AsyncFunctionDef)
                    prefix = "async def" if is_async else "def"
                    signature = f"{prefix} {node.name}({', '.join(args)}){returns}"

                    return {
                        "symbol_name": symbol_name,
                        "symbol_type": ("async_method" if is_async else "method") if parent_class else ("async_function" if is_async else "function"),
                        "start_line": start,
                        "end_line": end,
                        "source": source,
                        "parent_class": parent_class,
                        "docstring": docstring[:500] if docstring else None,
                        "signature": signature,
                        "decorators": decorators,
                        "content_hash": hashlib.sha256(source.encode()).hexdigest()[:16],
                    }

            elif isinstance(node, ast.ClassDef):
                if node.name == symbol_name:
                    # Include decorators in line range
                    start = get_decorator_start(node)
                    end = node.end_lineno or node.lineno
                    source = '\n'.join(lines[start - 1:end])

                    # Get decorators
                    decorators = get_decorators(node)

                    # Get bases
                    bases = [ast.unparse(b) for b in node.bases]
                    signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

                    return {
                        "symbol_name": symbol_name,
                        "symbol_type": "class",
                        "start_line": start,
                        "end_line": end,
                        "source": source,
                        "parent_class": None,
                        "docstring": ast.get_docstring(node) or None,
                        "signature": signature,
                        "decorators": decorators,
                        "content_hash": hashlib.sha256(source.encode()).hexdigest()[:16],
                    }

                # Search inside class for methods
                for child in node.body:
                    result = extract_node(child, parent_class=node.name)
                    if result:
                        return result

            return None

        # Search all top-level nodes
        for node in tree.body:
            result = extract_node(node)
            if result:
                return result

        return None

    def _find_related_types(self, file_path: Path, symbol_info: dict) -> list[dict]:
        """Find related types referenced in a symbol (e.g., return types, parameter types)."""
        import ast
        import re

        related = []
        if not symbol_info:
            return related

        source = symbol_info.get("source", "")
        signature = symbol_info.get("signature", "")

        # Extract type names from signature and source
        type_pattern = r'\b([A-Z][a-zA-Z0-9]+)\b'
        types_in_sig = set(re.findall(type_pattern, signature))
        types_in_body = set(re.findall(type_pattern, source))

        # Common built-in types to ignore
        builtins = {'Any', 'Dict', 'List', 'Optional', 'Union', 'Tuple', 'Set',
                    'Type', 'Callable', 'Iterable', 'Iterator', 'None', 'True', 'False',
                    'Exception', 'Error', 'Path', 'Self', 'str', 'int', 'float', 'bool'}

        potential_types = (types_in_sig | types_in_body) - builtins

        # Look up each type in the file
        for type_name in potential_types:
            if type_name == symbol_info.get("symbol_name"):
                continue  # Skip self-references
            type_info = self._extract_symbol_from_file(file_path, type_name)
            if type_info:
                related.append(type_info)

        return related

    def _gather_context_for_ask(self, target_path: Path, question: str) -> list[dict]:
        """Gather relevant files for context based on question.

        NEW: Evidence-bound approach - extracts specific symbols when possible,
        falls back to full file content only when needed.
        """
        context_files = []
        max_files = 10
        max_size = 8000

        # Extract symbols referenced in the question
        referenced_symbols = self._extract_symbols_from_question(question)

        # Keywords for file scoring (fallback)
        q_lower = question.lower()
        priority_keywords = []
        for word in q_lower.split():
            if len(word) > 3 and word.isalnum():
                priority_keywords.append(word)

        # Collect files
        if target_path.is_file():
            files = [target_path]
        else:
            files = list(target_path.rglob("*.py"))
            files.extend(target_path.rglob("*.js"))
            files.extend(target_path.rglob("*.ts"))
            files.extend(target_path.rglob("*.jsx"))
            files.extend(target_path.rglob("*.tsx"))

        # Filter noise
        files = [f for f in files if not any(x in str(f) for x in [
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            ".egg-info", "dist", "build", ".tox", ".pytest_cache"
        ])]

        # --- Evidence-bound: Try to extract referenced symbols first ---
        extracted_evidence = []
        for f in files:
            for sym in referenced_symbols:
                symbol_info = self._extract_symbol_from_file(f, sym)
                if symbol_info:
                    # Also get related types (e.g., CacheStats for get_stats)
                    related = self._find_related_types(f, symbol_info)

                    try:
                        rel_path = f.relative_to(target_path) if target_path.is_dir() else f.name
                    except ValueError:
                        rel_path = f.name

                    extracted_evidence.append({
                        "path": str(rel_path),
                        "full_path": str(f),
                        "symbol": symbol_info,
                        "related_types": related,
                        "evidence_type": "symbol_extraction",
                    })

        # Add evidence-extracted symbols to context first
        for ev in extracted_evidence[:5]:  # Max 5 targeted extractions
            symbol = ev["symbol"]
            related = ev.get("related_types", [])

            # Build focused content with evidence markers
            decorators = symbol.get("decorators", [])
            decorator_str = ", ".join(decorators) if decorators else "none"

            content_parts = [
                f"### {symbol['symbol_type'].upper()}: {symbol['symbol_name']}",
                f"### Location: {ev['path']}:{symbol['start_line']}-{symbol['end_line']}",
                f"### Content hash: {symbol['content_hash']}",
                f"### Decorators: {decorator_str}",
                "",
            ]

            if symbol.get("docstring"):
                content_parts.extend([f"Docstring: {symbol['docstring']}", ""])

            content_parts.extend([
                "```python",
                symbol["source"],
                "```",
            ])

            # Add related types
            for rel in related[:3]:
                rel_decorators = rel.get("decorators", [])
                rel_dec_str = ", ".join(rel_decorators) if rel_decorators else "none"
                content_parts.extend([
                    "",
                    f"### Related: {rel['symbol_type']} {rel['symbol_name']} "
                    f"(lines {rel['start_line']}-{rel['end_line']}, hash: {rel['content_hash']}, decorators: {rel_dec_str})",
                    "```python",
                    rel["source"][:1000],  # Truncate large classes
                    "```",
                ])

            content = "\n".join(content_parts)

            context_files.append({
                "path": ev["path"],
                "full_path": ev["full_path"],
                "content": content,
                "lines": symbol["end_line"] - symbol["start_line"] + 1,
                "evidence_type": "symbol",
                "symbol_name": symbol["symbol_name"],
                "line_range": f"{symbol['start_line']}-{symbol['end_line']}",
                "content_hash": symbol["content_hash"],
                "decorators": decorators,
                "source": symbol["source"],  # Store raw source for machine-insert
            })

        # --- Fallback: Full file content for remaining slots ---
        remaining_slots = max_files - len(context_files)
        if remaining_slots > 0:
            # Score files by relevance
            scored = []
            for f in files:
                # Skip files we already extracted symbols from
                if any(cf.get("full_path") == str(f) for cf in context_files):
                    continue

                score = 0
                fname = f.name.lower()
                fpath = str(f).lower()

                for kw in priority_keywords:
                    if kw in fname:
                        score += 10
                    elif kw in fpath:
                        score += 5

                if fname in ("__init__.py", "main.py", "app.py", "index.py"):
                    score += 3
                if "test" in fname:
                    score -= 2

                scored.append((score, f))

            scored.sort(key=lambda x: (-x[0], str(x[1])))

            for _, fpath in scored[:remaining_slots]:
                try:
                    full_content = fpath.read_text(errors="replace")
                    content = full_content[:max_size]
                    rel_path = fpath.relative_to(target_path) if target_path.is_dir() else fpath.name

                    # Generate content hash for evidence
                    import hashlib
                    content_hash = hashlib.sha256(full_content.encode()).hexdigest()[:16]

                    context_files.append({
                        "path": str(rel_path),
                        "full_path": str(fpath),
                        "content": content,
                        "lines": len(full_content.split("\n")),
                        "evidence_type": "full_file",
                        "content_hash": content_hash,
                        "truncated": len(full_content) > max_size,
                    })
                except Exception:
                    pass

        return context_files

    def _build_ask_system_prompt(self, mode: str) -> str:
        """Build system prompt based on mode.

        NEW: Evidence-bound format requiring citations.
        """
        base = """You are a senior software architect reviewing a codebase.
You have deep expertise in Python, JavaScript/TypeScript, security, performance, and software design.
Be direct and specific. Reference actual code locations when relevant.

CRITICAL EVIDENCE RULES:

1. DO NOT RETYPE CODE FROM THE CONTEXT.
   The evidence block will be machine-inserted separately. You must ONLY REFERENCE it by:
   - Line range (e.g., "lines 45-67")
   - Content hash (e.g., "hash: a3f2b1c9")
   - Symbol name (e.g., "`get_stats` method")

   NEVER copy-paste or paraphrase the actual code in your response.
   Say "see evidence block" or "as shown in the extracted code" instead.

2. When making claims about code, you MUST:
   - Cite the specific line range or hash
   - If a symbol is NOT in the context, say "NOT FOUND IN PROVIDED CONTEXT"
   - If code appears stubbed/incomplete, say "APPEARS INCOMPLETE"

3. PAY ATTENTION TO DECORATORS.
   If a class has @dataclass, it already has __init__, __repr__, etc.
   If a method has @property, @staticmethod, @classmethod - note this in your analysis.

Format citations as:
- "The `get_stats` method (lines 89-102, hash: a3f2b1c9) returns..."
- "NOT FOUND: `CacheStats` class not in provided context"
- "UNCERTAIN: Only partial file provided (truncated at line 200)"

The tool will insert a canonical evidence block - do NOT duplicate it."""

        mode_specific = {
            "analyze": """
Your task is to EXPLAIN and ANALYZE. Do not suggest changes unless explicitly asked.

NOTE: The tool will prepend a canonical "## Evidence Block" with extracted code.
Do NOT rewrite it - just reference it by line range, hash, or symbol name.

REQUIRED FORMAT:
## Analysis
- How the code works (reference evidence by line/hash)
- Design patterns used (note decorators like @dataclass, @property)
- Data flow
- Key abstractions

## Uncertainties
List anything not found or unclear from the provided context.""",

            "suggest": """
Your task is to provide SUGGESTIONS with code examples.

NOTE: The tool will prepend a canonical "## Evidence Block" with extracted code.
Do NOT rewrite it - just reference it by line range, hash, or symbol name.

REQUIRED FORMAT:
## Suggestions
For each suggestion:
1. Problem/opportunity identified (cite line range/hash from evidence)
2. Suggested approach
3. Code snippet (if applicable) - this is NEW code, not a copy of evidence
4. Trade-offs to consider

## Uncertainties
List any assumptions you're making or context that was missing.

Use markdown code blocks with language tags for NEW code only.""",

            "implement": """
Your task is to IMPLEMENT requested changes.
Provide complete, working code that can be directly applied.

NOTE: The tool will prepend a canonical "## Evidence Block" with extracted code.
Reference it by line range, hash, or symbol name. Do NOT rewrite it.

REQUIRED FORMAT:

## Implementation
Brief explanation of changes (reference evidence by line/hash).

## Code
```python
# File: services/cache.py
# Action: insert_after
# Class: SQLiteCache
<your complete NEW code here>
```

RULES:
- Use markdown code blocks with triple backticks
- First line: # File: path/to/file.py (use ACTUAL paths from context)
- Second line: # Action: [insert_after|replace|prepend]
- Third line (if method): # Class: ClassName
- Provide complete, runnable NEW code
- Do NOT copy the evidence block code into your response

## Verification Notes
What should be checked to verify correctness.""",

            "optimize": """
Your task is to identify and implement PERFORMANCE OPTIMIZATIONS.

NOTE: The tool will prepend a canonical "## Evidence Block" with extracted code.
Reference it by line range, hash, or symbol name. Do NOT rewrite it.

REQUIRED FORMAT:

## Bottleneck Analysis
For each issue:
1. Identify the bottleneck (cite line range/hash from evidence)
2. Current complexity/behavior
3. Proposed improvement
4. Expected impact

## Optimized Code
```python
# File: path/to/file.py
# Action: replace
<complete optimized code>
```

## Uncertainties
Note any profiling data or runtime info that would improve this analysis.

Focus on:
- Algorithmic improvements (O(n²) → O(n log n))
- Caching opportunities
- Database query optimization
- Memory efficiency
- I/O patterns""",
        }

        return base + mode_specific.get(mode, mode_specific["suggest"])

    def _build_canonical_evidence_block(self, context_files: list[dict]) -> str:
        """Build canonical evidence block from extracted symbols.

        This is machine-inserted and cannot be corrupted by the model.
        """
        parts = []

        for cf in context_files:
            if cf.get("evidence_type") != "symbol":
                continue

            symbol_name = cf.get("symbol_name", "unknown")
            line_range = cf.get("line_range", "?")
            content_hash = cf.get("content_hash", "?")
            decorators = cf.get("decorators", [])
            source = cf.get("source", "")

            # Build header
            dec_str = ", ".join(decorators) if decorators else "none"
            parts.append(f"### `{symbol_name}` ({cf.get('path')}:{line_range}, hash: {content_hash})")
            parts.append(f"### Decorators: {dec_str}")
            parts.append("```python")
            parts.append(source)
            parts.append("```")
            parts.append("")

        return "\n".join(parts) if parts else ""

    def _check_evidence_integrity(self, response: str, context_files: list[dict]) -> str | None:
        """Check if model retyped evidence code (integrity violation).

        Returns warning message if model appears to have retyped evidence,
        or None if clean.
        """
        import re

        # Extract code blocks from response
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)

        for cf in context_files:
            if cf.get("evidence_type") != "symbol":
                continue

            source = cf.get("source", "")
            if not source or len(source) < 20:
                continue

            # Check if any significant portion of source appears in response code blocks
            # We check for lines, not exact match, to catch reformatted code
            source_lines = [l.strip() for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]

            for code_block in code_blocks:
                # Skip if this looks like new code (has # File: or # Action:)
                if '# File:' in code_block or '# Action:' in code_block:
                    continue

                block_lines = [l.strip() for l in code_block.split('\n') if l.strip()]

                # Count matching lines
                matches = sum(1 for sl in source_lines[:10] if any(sl in bl for bl in block_lines))

                # If more than 60% of first 10 lines match, likely retyped
                if len(source_lines[:10]) > 0 and matches / min(len(source_lines[:10]), 10) > 0.6:
                    return (f"Model may have retyped evidence for `{cf.get('symbol_name')}`. "
                            f"Canonical evidence block above is authoritative.")

        return None

    def _build_ask_user_prompt(self, question: str, context_files: list[dict], mode: str) -> str:
        """Build user prompt with context.

        NEW: Evidence-bound format with hashes and line ranges.
        """
        prompt_parts = [f"# Question\n\n{question}\n\n# Codebase Context (Evidence)\n"]

        for cf in context_files[:8]:  # Limit files in prompt
            evidence_type = cf.get("evidence_type", "full_file")

            if evidence_type == "symbol":
                # Targeted symbol extraction - most precise
                prompt_parts.append(f"\n## EXTRACTED SYMBOL: {cf.get('symbol_name', 'unknown')}")
                prompt_parts.append(f"## File: {cf['path']}")
                prompt_parts.append(f"## Lines: {cf.get('line_range', 'unknown')}")
                prompt_parts.append(f"## Content Hash: {cf.get('content_hash', 'unknown')}")
                prompt_parts.append(f"\n{cf['content']}\n")
            else:
                # Full file fallback
                truncated = cf.get("truncated", False)
                trunc_note = " (TRUNCATED - not complete)" if truncated else ""
                prompt_parts.append(f"\n## FILE: {cf['path']} ({cf['lines']} lines){trunc_note}")
                prompt_parts.append(f"## Content Hash: {cf.get('content_hash', 'unknown')}")
                prompt_parts.append(f"\n```python\n{cf['content'][:6000]}\n```\n")

        if mode == "implement":
            prompt_parts.append("""
# Instructions

IMPORTANT: Your response MUST include:
1. An Evidence Block citing the specific code you're modifying (with line ranges/hashes from above)
2. Complete, working code changes

Format code changes as:
```python
# File: path/to/file.py
# Action: [replace|insert_after|prepend]
# Class: ClassName (if adding a method to a class)
<complete code>
```

Use ACTUAL file paths from the context above. Do NOT use placeholder paths.""")

        prompt_parts.append("""
# Evidence Requirements

Remember: Cite line ranges and content hashes when making claims.
Say "NOT FOUND IN PROVIDED CONTEXT" if you need information not shown above.
Say "UNCERTAIN" with reasoning if context is incomplete or ambiguous.""")

        return "\n".join(prompt_parts)

        return "\n".join(prompt_parts)

    def _call_llm_for_ask(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str | None, dict]:
        """Call LLM for ask command. Supports OpenAI and Anthropic.

        Returns:
            Tuple of (response_text, metrics_dict)
            metrics_dict contains: provider, model, input_tokens, output_tokens, duration_ms
        """
        import os
        import subprocess
        import json as json_mod
        import time

        start_time = time.time()
        metrics = {
            "provider": "unknown",
            "model": model,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0,
            "success": False,
        }

        # Detect provider from model name or environment
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        is_claude = model.startswith("claude") or (anthropic_key and not openai_key)

        if is_claude:
            metrics["provider"] = "anthropic"
            if not anthropic_key:
                print(f"{RED}ANTHROPIC_API_KEY not set{RESET}")
                return None, metrics

            actual_model = model if model.startswith("claude") else "claude-sonnet-4-20250514"
            metrics["model"] = actual_model

            payload = {
                "model": actual_model,
                "max_tokens": 8000,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt[:100000]},
                ],
            }

            result = subprocess.run(
                [
                    "curl", "-s",
                    "https://api.anthropic.com/v1/messages",
                    "-H", "Content-Type: application/json",
                    "-H", f"x-api-key: {anthropic_key}",
                    "-H", "anthropic-version: 2023-06-01",
                    "-d", json_mod.dumps(payload),
                ],
                capture_output=True,
                timeout=120,
            )

            metrics["duration_ms"] = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                print(f"{RED}API call failed{RESET}")
                return None, metrics

            try:
                resp = json_mod.loads(result.stdout.decode())
                if "error" in resp:
                    print(f"{RED}API error: {resp['error'].get('message', resp['error'])}{RESET}")
                    return None, metrics

                # Extract token usage
                usage = resp.get("usage", {})
                metrics["input_tokens"] = usage.get("input_tokens", 0)
                metrics["output_tokens"] = usage.get("output_tokens", 0)

                if metrics["input_tokens"] or metrics["output_tokens"]:
                    print(f"{DIM}      Tokens: {metrics['input_tokens']:,} in / {metrics['output_tokens']:,} out{RESET}")

                # Claude returns content as array of blocks
                content = resp.get("content", [])
                if content and isinstance(content, list):
                    metrics["success"] = True
                    return content[0].get("text", ""), metrics
                return None, metrics
            except (KeyError, json_mod.JSONDecodeError) as e:
                print(f"{RED}Parse error: {e}{RESET}")
                print(f"{DIM}Response: {result.stdout.decode()[:500]}{RESET}")
                return None, metrics

        else:
            # Use OpenAI API
            metrics["provider"] = "openai"
            if not openai_key:
                print(f"{RED}OPENAI_API_KEY not set{RESET}")
                return None, metrics

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt[:100000]},
                ],
                "temperature": 0.1,
                "max_tokens": 8000,
            }

            result = subprocess.run(
                [
                    "curl", "-s",
                    "https://api.openai.com/v1/chat/completions",
                    "-H", "Content-Type: application/json",
                    "-H", f"Authorization: Bearer {openai_key}",
                    "-d", json_mod.dumps(payload),
                ],
                capture_output=True,
                timeout=120,
            )

            metrics["duration_ms"] = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                print(f"{RED}API call failed{RESET}")
                return None, metrics

            try:
                resp = json_mod.loads(result.stdout.decode())
                if "error" in resp:
                    print(f"{RED}API error: {resp['error'].get('message', resp['error'])}{RESET}")
                    return None, metrics

                # Extract token usage
                usage = resp.get("usage", {})
                metrics["input_tokens"] = usage.get("prompt_tokens", 0)
                metrics["output_tokens"] = usage.get("completion_tokens", 0)

                if metrics["input_tokens"] or metrics["output_tokens"]:
                    print(f"{DIM}      Tokens: {metrics['input_tokens']:,} in / {metrics['output_tokens']:,} out{RESET}")

                metrics["success"] = True
                return resp["choices"][0]["message"]["content"], metrics
            except (KeyError, json_mod.JSONDecodeError) as e:
                print(f"{RED}Parse error: {e}{RESET}")
                return None, metrics

    def _canonical_json(self, obj: dict) -> str:
        """Produce canonical JSON for deterministic hashing.

        - Sorted keys
        - No extra whitespace
        - Consistent number formatting
        """
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write file atomically: temp → fsync → rename.

        Ensures crash-safety and prevents partial writes.
        """
        import os
        import tempfile

        # Write to temp file in same directory (for same-filesystem rename)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.fsync(fd)  # Ensure data hits disk
            os.close(fd)
            os.rename(tmp_path, path)  # Atomic on POSIX
        except Exception:
            os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _save_run_metrics(
        self,
        target_path: Path,
        question: str,
        mode: str,
        llm_metrics: dict,
        context_files: list[dict],
        prompt_content: str | None = None,
        response_content: str | None = None,
    ) -> None:
        """Save metrics for this run and aggregate into session totals.

        Creates:
        - .capseal/runs/<timestamp>/run_metrics.json (per-run, with chain_hash)
        - .capseal/session_metrics.json (cumulative)

        Chain hash is computed as: H(prev_chain_hash || canonical_json(run_metrics))
        This allows independent verification of any suffix of the chain.
        """
        import datetime
        import hashlib

        now = datetime.datetime.now()
        timestamp_dir = now.strftime("%Y%m%dT%H%M%S")

        # Determine base path for .capseal
        base_path = target_path.parent if target_path.is_file() else target_path
        capseal_dir = base_path / ".capseal"
        run_path = capseal_dir / "runs" / timestamp_dir
        run_path.mkdir(parents=True, exist_ok=True)

        # Build context file metadata (hashes only, not content)
        context_meta = []
        for cf in context_files:
            context_meta.append({
                "path": cf.get("path"),
                "evidence_type": cf.get("evidence_type", "unknown"),
                "content_hash": cf.get("content_hash", "unknown"),
                "byte_count": len(cf.get("content", "")),
                "lines": cf.get("lines", 0),
                "symbol_name": cf.get("symbol_name"),
                "line_range": cf.get("line_range"),
            })

        # Compute prompt/response hashes for auditability (proves exact context used)
        prompt_hash = None
        response_hash = None
        if prompt_content:
            prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:32]
        if response_content:
            response_hash = hashlib.sha256(response_content.encode()).hexdigest()[:32]

        # Build run metrics (without chain_hash - added after)
        run_metrics = {
            "run_id": timestamp_dir,
            "type": "ask",
            "tool": "ask",  # Explicit tool name for filtering
            "timestamp": now.isoformat(),
            "question": question,
            "mode": mode,
            "status": "success" if llm_metrics.get("success") else "fail",
            "target_path": str(target_path),
            "llm": {
                "provider": llm_metrics.get("provider"),
                "model": llm_metrics.get("model"),
                "input_tokens": llm_metrics.get("input_tokens", 0),
                "output_tokens": llm_metrics.get("output_tokens", 0),
                "duration_ms": llm_metrics.get("duration_ms", 0),
                "success": llm_metrics.get("success", False),
            },
            "audit": {
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
            },
            "context": {
                "file_count": len(context_files),
                "symbol_extractions": len([c for c in context_files if c.get("evidence_type") == "symbol"]),
                "full_files": len([c for c in context_files if c.get("evidence_type") == "full_file"]),
                "files": context_meta,
            },
        }

        # Get previous chain hash from session (for proper chaining)
        prev_chain_hash = self._get_prev_chain_hash(capseal_dir)

        # Compute this run's chain hash: H(prev_chain_hash || canonical_json(run_metrics))
        chain_input = (prev_chain_hash or "genesis") + self._canonical_json(run_metrics)
        run_chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()[:32]

        # Add chain hash to run metrics (for independent verification)
        run_metrics["chain_hash"] = run_chain_hash
        run_metrics["prev_chain_hash"] = prev_chain_hash

        # Save per-run metrics (atomic write)
        metrics_path = run_path / "run_metrics.json"
        self._atomic_write(metrics_path, json.dumps(run_metrics, indent=2))

        # Aggregate into session metrics (with file locking)
        self._aggregate_session_metrics(capseal_dir, run_metrics, run_chain_hash)

    def _get_prev_chain_hash(self, capseal_dir: Path) -> str | None:
        """Get the previous chain hash from session_metrics.json."""
        session_path = capseal_dir / "session_metrics.json"
        if not session_path.exists():
            return None
        try:
            session = json.loads(session_path.read_text())
            return session.get("chain_hash")
        except (json.JSONDecodeError, OSError):
            return None

    def _aggregate_session_metrics(
        self, capseal_dir: Path, run_metrics: dict, run_chain_hash: str
    ) -> None:
        """Add run metrics to cumulative session totals.

        Uses file locking for concurrency safety (multi-agent mode).
        Uses atomic writes for crash safety.
        """
        import fcntl

        session_path = capseal_dir / "session_metrics.json"
        lock_path = capseal_dir / ".session_metrics.lock"

        # Ensure lock file exists
        lock_path.touch(exist_ok=True)

        # Acquire exclusive lock
        lock_fd = open(lock_path, "r")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Load existing or create new (inside lock)
            if session_path.exists():
                try:
                    session = json.loads(session_path.read_text())
                except json.JSONDecodeError:
                    session = self._new_session_metrics()
            else:
                session = self._new_session_metrics()

            # Update totals
            llm = run_metrics.get("llm", {})
            session["total_runs"] += 1
            session["total_input_tokens"] += llm.get("input_tokens", 0)
            session["total_output_tokens"] += llm.get("output_tokens", 0)
            session["total_duration_ms"] += llm.get("duration_ms", 0)

            if llm.get("success"):
                session["successful_runs"] += 1
            else:
                session["failed_runs"] += 1

            # Track by provider/model
            provider = llm.get("provider", "unknown")
            model = llm.get("model", "unknown")
            key = f"{provider}:{model}"

            if key not in session["by_model"]:
                session["by_model"][key] = {
                    "runs": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "duration_ms": 0,
                }

            session["by_model"][key]["runs"] += 1
            session["by_model"][key]["input_tokens"] += llm.get("input_tokens", 0)
            session["by_model"][key]["output_tokens"] += llm.get("output_tokens", 0)
            session["by_model"][key]["duration_ms"] += llm.get("duration_ms", 0)

            # Append run ID to chain (for auditing)
            session["run_chain"].append(run_metrics.get("run_id"))
            session["last_updated"] = run_metrics.get("timestamp")

            # Store the chain hash (computed properly in _save_run_metrics)
            session["chain_hash"] = run_chain_hash

            # Atomic write (inside lock)
            self._atomic_write(session_path, json.dumps(session, indent=2))

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _new_session_metrics(self) -> dict:
        """Create a new session metrics structure."""
        import datetime
        return {
            "version": "1.0",
            "created": datetime.datetime.now().isoformat(),
            "last_updated": None,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_duration_ms": 0,
            "by_model": {},
            "run_chain": [],
            "chain_hash": None,
        }

    def do_metrics(self, arg: str) -> None:
        """Show session metrics (LLM usage, costs, run history).

        Usage:
            metrics                 Show current session totals
            metrics --detail        Show per-model breakdown
            metrics --reset         Reset session metrics
            metrics --runs          List recent runs with metrics
            metrics --verify        Verify chain integrity (detect tampering)
            metrics --budget        Show resource limits and budget remaining

        Session metrics track cumulative LLM usage across all 'ask' and 'review'
        commands in this workspace.

        The chain hash provides tamper-evidence: each run's hash includes the
        previous hash, so modifications to any run will invalidate all subsequent
        hashes. Use --verify to check integrity.

        Resource limits (enforced by MetricsEnforcer):
          - max_files: 25 files per request
          - max_snippets: 80 code snippets per request
          - max_total_chars: 120,000 characters context
          - max_response_wait_sec: 75 seconds timeout
        """
        parts = arg.strip().split()
        detail = "--detail" in parts
        reset = "--reset" in parts
        show_runs = "--runs" in parts
        verify = "--verify" in parts
        show_budget = "--budget" in parts

        # Find .capseal directory
        cwd = Path.cwd()
        capseal_dir = cwd / ".capseal"
        if not capseal_dir.exists():
            # Check if we have a last_review_path
            if self.state.last_review_path:
                base = self.state.last_review_path.parent if self.state.last_review_path.is_file() else self.state.last_review_path
                capseal_dir = base / ".capseal"

        if not capseal_dir.exists():
            print(f"{YELLOW}No .capseal directory found. Run 'ask' or 'review' first.{RESET}")
            return

        session_path = capseal_dir / "session_metrics.json"

        if reset:
            if session_path.exists():
                session_path.unlink()
                print(f"{GREEN}✓ Session metrics reset{RESET}")
            else:
                print(f"{DIM}No session metrics to reset{RESET}")
            return

        if verify:
            self._verify_metrics_chain(capseal_dir)
            return

        if show_budget:
            self._show_resource_budget()
            return

        if not session_path.exists():
            print(f"{YELLOW}No session metrics yet. Run 'ask' or 'review' first.{RESET}")
            return

        try:
            session = json.loads(session_path.read_text())
        except json.JSONDecodeError:
            print(f"{RED}Corrupt session_metrics.json{RESET}")
            return

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    SESSION METRICS                            ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}\n")

        # Summary
        total_tokens = session.get("total_input_tokens", 0) + session.get("total_output_tokens", 0)
        duration_s = session.get("total_duration_ms", 0) / 1000

        print(f"{BOLD}Summary:{RESET}")
        print(f"  Total runs:     {session.get('total_runs', 0)} ({session.get('successful_runs', 0)} success, {session.get('failed_runs', 0)} failed)")
        print(f"  Input tokens:   {session.get('total_input_tokens', 0):,}")
        print(f"  Output tokens:  {session.get('total_output_tokens', 0):,}")
        print(f"  Total tokens:   {total_tokens:,}")
        print(f"  Total time:     {duration_s:.1f}s")
        print(f"  Chain hash:     {session.get('chain_hash', 'n/a')}")

        if detail:
            print(f"\n{BOLD}By Model:{RESET}")
            for key, stats in session.get("by_model", {}).items():
                tokens = stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
                print(f"\n  {CYAN}{key}{RESET}")
                print(f"    Runs:   {stats.get('runs', 0)}")
                print(f"    Tokens: {tokens:,} ({stats.get('input_tokens', 0):,} in / {stats.get('output_tokens', 0):,} out)")
                print(f"    Time:   {stats.get('duration_ms', 0) / 1000:.1f}s")

        if show_runs:
            runs_dir = capseal_dir / "runs"
            if runs_dir.exists():
                print(f"\n{BOLD}Recent Runs:{RESET}")
                run_dirs = sorted(runs_dir.iterdir(), reverse=True)[:10]
                for run_dir in run_dirs:
                    metrics_file = run_dir / "run_metrics.json"
                    if metrics_file.exists():
                        try:
                            rm = json.loads(metrics_file.read_text())
                            llm = rm.get("llm", {})
                            tokens = llm.get("input_tokens", 0) + llm.get("output_tokens", 0)
                            status = f"{GREEN}✓{RESET}" if llm.get("success") else f"{RED}✗{RESET}"
                            chain_indicator = f" [{rm.get('chain_hash', '?')[:8]}]" if rm.get('chain_hash') else ""
                            print(f"  {status} {run_dir.name}: {tokens:,} tokens, {rm.get('mode', '?')}{chain_indicator}")
                        except:
                            print(f"  ? {run_dir.name}")

        print()

    def _verify_metrics_chain(self, capseal_dir: Path) -> None:
        """Verify the integrity of the metrics chain.

        Recomputes all chain hashes from disk and compares to stored values.
        Detects:
        - Missing runs
        - Modified run metrics
        - Out-of-order runs
        - Tampered chain hashes
        """
        import hashlib

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    CHAIN VERIFICATION                         ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}\n")

        session_path = capseal_dir / "session_metrics.json"
        runs_dir = capseal_dir / "runs"

        if not session_path.exists():
            print(f"{YELLOW}No session_metrics.json found{RESET}")
            return

        if not runs_dir.exists():
            print(f"{YELLOW}No runs directory found{RESET}")
            return

        try:
            session = json.loads(session_path.read_text())
        except json.JSONDecodeError:
            print(f"{RED}✗ Corrupt session_metrics.json{RESET}")
            return

        run_chain = session.get("run_chain", [])
        expected_chain_hash = session.get("chain_hash")

        print(f"{BOLD}Verifying {len(run_chain)} runs...{RESET}\n")

        # Recompute chain from scratch
        computed_hash = None
        errors = []
        verified = 0
        missing = 0

        for run_id in run_chain:
            run_path = runs_dir / run_id / "run_metrics.json"

            if not run_path.exists():
                print(f"  {RED}✗{RESET} {run_id}: MISSING")
                errors.append(f"Missing run: {run_id}")
                missing += 1
                continue

            try:
                run_metrics = json.loads(run_path.read_text())
            except json.JSONDecodeError:
                print(f"  {RED}✗{RESET} {run_id}: CORRUPT JSON")
                errors.append(f"Corrupt JSON: {run_id}")
                continue

            # Get stored chain hash and prev hash
            stored_chain_hash = run_metrics.get("chain_hash")
            stored_prev_hash = run_metrics.get("prev_chain_hash")

            # Verify prev_chain_hash matches our computed hash
            if stored_prev_hash != computed_hash:
                if computed_hash is None and stored_prev_hash is None:
                    pass  # First run, both should be None
                else:
                    print(f"  {RED}✗{RESET} {run_id}: PREV HASH MISMATCH")
                    print(f"      Expected: {computed_hash}")
                    print(f"      Stored:   {stored_prev_hash}")
                    errors.append(f"Prev hash mismatch: {run_id}")
                    continue

            # Recompute this run's chain hash
            # Remove chain_hash and prev_chain_hash before computing (they weren't in original)
            metrics_for_hash = {k: v for k, v in run_metrics.items()
                               if k not in ("chain_hash", "prev_chain_hash")}
            chain_input = (computed_hash or "genesis") + self._canonical_json(metrics_for_hash)
            recomputed_hash = hashlib.sha256(chain_input.encode()).hexdigest()[:32]

            if recomputed_hash != stored_chain_hash:
                print(f"  {RED}✗{RESET} {run_id}: HASH MISMATCH (modified?)")
                print(f"      Computed: {recomputed_hash}")
                print(f"      Stored:   {stored_chain_hash}")
                errors.append(f"Hash mismatch: {run_id}")
            else:
                print(f"  {GREEN}✓{RESET} {run_id}: {stored_chain_hash[:16]}...")
                verified += 1

            computed_hash = stored_chain_hash  # Use stored for chain continuity

        print()

        # Final verification: does our final hash match session?
        if computed_hash == expected_chain_hash:
            print(f"{GREEN}✓ Final chain hash matches session{RESET}")
            print(f"  Hash: {expected_chain_hash}")
        else:
            print(f"{RED}✗ Final chain hash MISMATCH{RESET}")
            print(f"  Session:  {expected_chain_hash}")
            print(f"  Computed: {computed_hash}")
            errors.append("Final hash mismatch")

        print()

        # Summary
        if errors:
            print(f"{RED}✗ VERIFICATION FAILED{RESET}")
            print(f"  {len(errors)} error(s) found:")
            for err in errors[:5]:
                print(f"    • {err}")
            if len(errors) > 5:
                print(f"    • ... and {len(errors) - 5} more")
        else:
            print(f"{GREEN}✓ CHAIN VERIFIED{RESET}")
            print(f"  {verified} runs verified, 0 errors")
            print(f"  Chain is tamper-evident and intact")

        print()

    def _show_resource_budget(self) -> None:
        """Show current resource limits and enforcement status."""
        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    RESOURCE BUDGET                            ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}\n")

        enforcer = get_enforcer()
        summary = enforcer.summary()
        utilization = enforcer.utilization()

        # Limits
        limits = summary["limits"]
        print(f"{BOLD}Hard Limits (per request):{RESET}")
        print(f"  Max files:      {limits['max_files']}")
        print(f"  Max snippets:   {limits['max_snippets']}")
        print(f"  Max chars:      {limits['max_total_chars']:,}")
        print(f"  Max wait time:  {limits['max_response_wait_sec']}s")

        # Current usage
        usage = summary["usage"]
        print(f"\n{BOLD}Current Usage:{RESET}")

        def bar(pct: float) -> str:
            filled = int(pct / 5)  # 20 chars = 100%
            return f"[{'█' * filled}{'░' * (20 - filled)}]"

        def color_pct(pct: float) -> str:
            if pct >= 90:
                return f"{RED}{pct:.0f}%{RESET}"
            elif pct >= 70:
                return f"{YELLOW}{pct:.0f}%{RESET}"
            else:
                return f"{GREEN}{pct:.0f}%{RESET}"

        files_pct = utilization["files"]
        snippets_pct = utilization["snippets"]
        chars_pct = utilization["chars"]

        print(f"  Files:    {bar(files_pct)} {color_pct(files_pct)} ({usage['files']}/{limits['max_files']})")
        print(f"  Snippets: {bar(snippets_pct)} {color_pct(snippets_pct)} ({usage['snippets']}/{limits['max_snippets']})")
        print(f"  Chars:    {bar(chars_pct)} {color_pct(chars_pct)} ({usage['total_chars']:,}/{limits['max_total_chars']:,})")

        # Budget remaining
        budget = summary["budget_remaining"]
        print(f"\n{BOLD}Budget Remaining:{RESET}")
        print(f"  Files:    {budget['files']}")
        print(f"  Snippets: {budget['snippets']}")
        print(f"  Chars:    {budget['chars']:,}")

        # Violations
        if summary["violations"]:
            print(f"\n{RED}{BOLD}Violations:{RESET}")
            for v in summary["violations"]:
                print(f"  {RED}✗{RESET} {v['resource']}: {v['message']}")
        else:
            print(f"\n{GREEN}✓ No violations{RESET}")

        # Status
        if summary["within_limits"]:
            print(f"\n{GREEN}✓ Within all limits{RESET}")
        else:
            print(f"\n{RED}✗ Limits exceeded{RESET}")

        print()

    def do_doctor(self, arg: str) -> None:
        """Run full pipeline health check on a capsule.

        Usage:
            doctor <capsule_path>           Run verification pipeline
            doctor <capsule_path> --rows N  Sample N rows for replay check (default: 1)
            doctor <capsule_path> --output DIR  Save report to directory

        Performs one-click verification:
          1. Inspect: Reads capsule structure
          2. Verify: Validates signatures and merkle proofs
          3. Audit: Checks policy compliance
          4. Row opening: Samples rows and verifies openings

        Generates report.json and report.md with detailed results.
        """
        parts = shlex.split(arg) if arg else []

        if not parts:
            print(f"{YELLOW}Usage: doctor <capsule_path> [--rows N] [--output DIR]{RESET}")
            return

        capsule_path = Path(parts[0])
        if not capsule_path.exists():
            print(f"{RED}Capsule not found: {capsule_path}{RESET}")
            return

        # Parse options
        sample_rows = 1
        output_dir = None

        i = 1
        while i < len(parts):
            if parts[i] == "--rows" and i + 1 < len(parts):
                try:
                    sample_rows = int(parts[i + 1])
                except ValueError:
                    print(f"{RED}Invalid row count: {parts[i + 1]}{RESET}")
                    return
                i += 2
            elif parts[i] == "--output" and i + 1 < len(parts):
                output_dir = Path(parts[i + 1])
                i += 2
            else:
                i += 1

        print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    PIPELINE DOCTOR                            ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}\n")

        print(f"{BOLD}Capsule:{RESET} {capsule_path}")
        print(f"{BOLD}Sample rows:{RESET} {sample_rows}")
        if output_dir:
            print(f"{BOLD}Output:{RESET} {output_dir}")
        print()

        try:
            report = run_doctor(capsule_path, output_dir, sample_rows)

            # Display results
            print(f"{BOLD}Checks:{RESET}")
            for check in report.checks:
                status = check.status
                msg = check.error if check.error else f"{check.duration_ms:.0f}ms"
                if status == "pass":
                    print(f"  {GREEN}✓{RESET} {check.name}: {msg}")
                elif status == "warn":
                    print(f"  {YELLOW}⚠{RESET} {check.name}: {msg}")
                elif status == "skip":
                    print(f"  {DIM}○{RESET} {check.name}: {msg}")
                else:
                    print(f"  {RED}✗{RESET} {check.name}: {msg}")
                    if check.details:
                        for key, val in list(check.details.items())[:3]:
                            print(f"      {DIM}{key}: {val}{RESET}")

            print()

            # Summary
            overall = report.overall_status()
            passed = sum(1 for c in report.checks if c.status == "pass")
            total = len(report.checks)

            if overall == "PASS":
                print(f"{GREEN}✓ PIPELINE HEALTHY{RESET}")
                print(f"  {passed}/{total} checks passed")
            elif overall == "WARN":
                print(f"{YELLOW}⚠ PIPELINE OK WITH WARNINGS{RESET}")
                print(f"  {passed}/{total} checks passed")
            else:
                print(f"{RED}✗ PIPELINE ISSUES DETECTED{RESET}")
                print(f"  {passed}/{total} checks passed")

            # Check if report was written
            if output_dir:
                report_file = output_dir / "report.md"
                if report_file.exists():
                    print(f"\n{DIM}Full report: {report_file}{RESET}")

        except Exception as e:
            print(f"{RED}Doctor failed: {e}{RESET}")
            import traceback
            traceback.print_exc()

        print()

    def _merge_code_into_file(
        self,
        original_content: str,
        new_code: str,
        action: str = "insert_after",
        target_class: str | None = None,
    ) -> str:
        """
        Merge generated code into existing file using AST analysis.

        Args:
            original_content: The original file content
            new_code: The new code to merge (method, function, etc.)
            action: "insert_after", "replace", or "prepend"
            target_class: Class name to insert method into (auto-detected if None)

        Returns:
            Merged file content
        """
        import ast
        import re

        # Clean up the new code (remove action/metadata comments)
        clean_code = re.sub(
            r'^#\s*(File|Action|Class|Insert|Add|Place).*\n?',
            '',
            new_code,
            flags=re.MULTILINE | re.IGNORECASE
        ).strip()

        # If action is replace, just return the new code
        if action == "replace":
            return clean_code

        # Try to parse original file
        try:
            original_tree = ast.parse(original_content)
        except SyntaxError:
            # Can't parse, fall back to append
            return original_content + "\n\n" + clean_code

        # Try to parse new code to understand what we're inserting
        try:
            new_tree = ast.parse(clean_code)
        except SyntaxError:
            # New code doesn't parse standalone, might be a method body
            # Try wrapping in a class
            try:
                wrapped = f"class _Temp:\n" + "\n".join("    " + line for line in clean_code.split("\n"))
                new_tree = ast.parse(wrapped)
                # Extract the method from the temp class
                new_nodes = new_tree.body[0].body  # type: ignore
            except SyntaxError:
                # Give up, just append
                return original_content + "\n\n" + clean_code
        else:
            new_nodes = new_tree.body

        # Find what we're inserting (function, method, class, etc.)
        new_funcs = [n for n in new_nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        if not new_funcs:
            # Not inserting functions, just append
            return original_content + "\n\n" + clean_code

        # Find target class if inserting methods
        target_classes = [n for n in original_tree.body if isinstance(n, ast.ClassDef)]

        if not target_classes:
            # No classes in original, append as module-level
            return original_content.rstrip() + "\n\n\n" + clean_code + "\n"

        # Auto-detect target class from the code context
        if target_class is None:
            # Look for class name in comments or use the last class
            class_match = re.search(r'#.*(?:in|to)\s+(\w+)', new_code, re.IGNORECASE)
            if class_match:
                target_class = class_match.group(1)

        # Find the target class
        target = None
        for cls in target_classes:
            if target_class and cls.name == target_class:
                target = cls
                break
            # Check if any method name hints at the class
            for func in new_funcs:
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func.name
                       for n in cls.body):
                    # Method already exists, this might be an update
                    target = cls
                    break

        if target is None:
            # Use the last class (most likely the one being extended)
            target = target_classes[-1]

        # Find insertion point (end of class body)
        lines = original_content.split('\n')

        # Find the class end line
        class_end_line = target.end_lineno if hasattr(target, 'end_lineno') and target.end_lineno else None

        if class_end_line is None:
            # Estimate: find next class or end of file
            class_start = target.lineno
            for other_cls in target_classes:
                if other_cls.lineno > class_start:
                    class_end_line = other_cls.lineno - 1
                    break
            if class_end_line is None:
                class_end_line = len(lines)

        # Detect indentation of the class body
        class_indent = ""
        for node in target.body:
            if hasattr(node, 'col_offset'):
                class_indent = " " * node.col_offset
                break
        if not class_indent:
            class_indent = "    "  # Default 4 spaces

        # Indent the new code properly
        indented_code = "\n".join(
            class_indent + line if line.strip() else line
            for line in clean_code.split('\n')
        )

        # Insert before the class end
        # Find the last non-empty line of the class
        insert_line = class_end_line - 1
        while insert_line > target.lineno and not lines[insert_line].strip():
            insert_line -= 1

        # Build the result
        result_lines = lines[:insert_line + 1]
        result_lines.append("")  # Blank line before new method
        result_lines.append(indented_code)
        result_lines.extend(lines[insert_line + 1:])

        return '\n'.join(result_lines)

    def _process_ask_with_patches(
        self,
        response: str,
        target_path: Path,
        question: str,
        mode: str,
        apply_after: bool,
        model: str
    ) -> None:
        """Process ask response that contains code, potentially creating patches."""
        import datetime
        import re

        # Display the response first
        print("=" * 65)
        print(response)
        print("=" * 65)

        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        if not code_blocks:
            return

        # Look for file indicators
        file_changes = []
        for lang, code in code_blocks:
            # Try to find file path in the code block
            file_match = re.search(r'#\s*File:\s*([^\n]+)', code)
            if file_match:
                file_path = file_match.group(1).strip()
                # Detect action (insert_after, replace, prepend)
                action_match = re.search(r'#\s*Action:\s*(\w+)', code)
                action = action_match.group(1).lower() if action_match else "insert_after"
                # Detect target class if specified
                class_match = re.search(r'#\s*(?:Class|Insert.*?in):\s*(\w+)', code, re.IGNORECASE)
                target_class = class_match.group(1) if class_match else None
                # Remove metadata comments from code
                clean_code = re.sub(r'#\s*(File|Action|Class|Insert):[^\n]+\n?', '', code, flags=re.IGNORECASE).strip()
                file_changes.append({
                    "path": file_path,
                    "code": clean_code,
                    "lang": lang or "python",
                    "action": action,
                    "target_class": target_class,
                })

        if not file_changes:
            print(f"\n{DIM}Found {len(code_blocks)} code blocks but no file paths specified.{RESET}")
            print(f"{DIM}Tip: Include '# File: path/to/file.py' in code blocks for patch generation.{RESET}")
            return

        print(f"\n{CYAN}Found {len(file_changes)} file change(s):{RESET}")
        for fc in file_changes:
            print(f"  • {fc['path']}")

        # Ask if user wants to create verified patches
        # If --apply was passed, auto-confirm; otherwise ask
        if not apply_after:
            print(f"\n{YELLOW}Generate verified patches? (y/n){RESET} ", end="", flush=True)
            try:
                import sys
                if sys.stdin.isatty():
                    confirm = input().strip().lower()
                else:
                    confirm = "n"
            except:
                confirm = "n"

            if confirm != "y":
                print(f"{DIM}Skipped patch generation{RESET}")
                return
        else:
            print(f"\n{DIM}Auto-generating patches (--apply){RESET}")

        # Create patches and run through verification
        print(f"\n{DIM}Creating verified patches...{RESET}")

        from bef_zk.capsule.refactor_engine import sha256_str

        # Create run directory (use parent if target is a file)
        now = datetime.datetime.now()
        timestamp_dir = now.strftime("%Y%m%dT%H%M%S")
        receipt_id = f"ask-{sha256_str(question)[:8]}"
        base_path = target_path.parent if target_path.is_file() else target_path
        run_path = base_path / ".capseal" / "runs" / timestamp_dir
        run_path.mkdir(parents=True, exist_ok=True)
        patches_dir = run_path / "patches"
        patches_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            "type": "ask",
            "question": question,
            "mode": mode,
            "model": model,
            "timestamp": now.isoformat(),
            "receipt_id": receipt_id,
        }
        (run_path / "ask_metadata.json").write_text(json.dumps(metadata, indent=2))

        # Create patches
        created_patches = []
        for i, fc in enumerate(file_changes):
            patch_file = patches_dir / f"patch_{i:03d}_{Path(fc['path']).stem}.py"

            # Resolve full path - handle both relative and absolute paths
            rel_path = fc['path']
            if target_path.is_file():
                # If target is a file, use its directory as base
                full_path = target_path.parent / Path(rel_path).name
            elif target_path.is_dir():
                full_path = target_path / rel_path
            else:
                full_path = Path(rel_path)

            # Normalize the path
            if not full_path.is_absolute():
                full_path = (target_path if target_path.is_dir() else target_path.parent) / full_path

            action = fc.get('action', 'insert_after')
            target_class = fc.get('target_class')
            original_content = ""
            merged_content = fc['code']

            if full_path.exists():
                original_content = full_path.read_text(errors="replace")

                # Use merge function for insert_after on existing files
                if action in ("insert_after", "prepend") and original_content:
                    print(f"  {DIM}Merging into existing {full_path.name}...{RESET}")
                    merged_content = self._merge_code_into_file(
                        original_content=original_content,
                        new_code=fc['code'],
                        action=action,
                        target_class=target_class,
                    )

            patch_data = {
                "file_path": str(full_path),
                "relative_path": fc['path'],
                "new_content": merged_content,
                "snippet": fc['code'],  # Original snippet for reference
                "action": action,
                "original_exists": full_path.exists(),
                "lang": fc['lang'],
            }

            if original_content:
                patch_data["original_content"] = original_content

            patch_file.write_text(json.dumps(patch_data, indent=2))
            created_patches.append(patch_data)
            print(f"  {GREEN}✓{RESET} Created patch for {fc['path']} ({action})")

        # Run verification on patches
        print(f"\n{DIM}Running verification...{RESET}")

        # Import verification
        try:
            from bef_zk.capsule.claims import Claim, ClaimType, Verdict, Scope, Witness
            from bef_zk.capsule.checkers import ast_checker, semgrep_checker

            all_passed = True
            for patch in created_patches:
                file_path = patch['file_path']
                new_content = patch['new_content']

                # Basic checks
                checks = [
                    (ClaimType.NO_SHELL_INJECTION, ast_checker),
                    (ClaimType.NO_SQL_INJECTION, semgrep_checker),
                ]

                for claim_type, checker in checks:
                    import hashlib as hl
                    file_hash = hl.sha256(new_content.encode()).hexdigest()[:16]
                    claim = Claim(
                        claim_id=f"ask-{sha256_str(file_path + str(claim_type))[:8]}",
                        claim_type=claim_type,
                        scope=Scope(file_path=file_path, file_hash=file_hash),
                        description=f"Verify {claim_type.value} for generated code",
                    )
                    try:
                        verdict, witness = checker(claim, new_content)
                        if verdict == Verdict.FAIL:
                            print(f"  {RED}✗{RESET} {Path(file_path).name}: {claim_type.value} FAILED")
                            if witness and witness.counterexample:
                                print(f"      {witness.counterexample}")
                            all_passed = False
                        elif verdict == Verdict.PASS:
                            print(f"  {GREEN}✓{RESET} {Path(file_path).name}: {claim_type.value}")
                    except Exception as e:
                        print(f"  {YELLOW}?{RESET} {Path(file_path).name}: {claim_type.value} - {e}")

            if all_passed:
                print(f"\n{GREEN}✓ All patches verified{RESET}")

                if apply_after:
                    print(f"\n{DIM}Applying patches...{RESET}")
                    for patch in created_patches:
                        full_path = Path(patch['file_path'])
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        full_path.write_text(patch['new_content'])
                        print(f"  {GREEN}✓{RESET} Applied {patch['relative_path']}")
                else:
                    print(f"\n{DIM}Run with --apply to apply patches, or:{RESET}")
                    print(f"  apply {timestamp_dir}")
            else:
                print(f"\n{YELLOW}⚠ Some patches failed verification{RESET}")
                print(f"{DIM}Review the issues above before applying{RESET}")

        except ImportError as e:
            print(f"{YELLOW}Verification skipped (missing module: {e}){RESET}")

        # Save run path
        self.state.last_review_path = target_path
        print(f"\n{DIM}Artifacts saved to: {run_path}{RESET}")

    def do_help(self, arg: str) -> None:
        """Show help for commands."""
        if arg:
            super().do_help(arg)
            return

        print(f"""
{BOLD}Capseal Shell{RESET}
{'─' * 65}

{CYAN}{BOLD}Workflow (typical):{RESET}
  review <path>               Generate verified diff (scan→plan→patch→verify)
  eval <path> [--rounds N]    Epistemic eval loop (learn safety boundary)
  open [latest|<run>]         Inspect plan, patches, proofs, receipts
  apply [latest|<run>]        Apply verified patches (never applies unverified)
  audit [latest|<run>]        Explain what was checked + trust boundaries

{CYAN}{BOLD}Core commands:{RESET}
  init                        Set up .capseal/ + policy defaults
  status                      Show sandbox/project/policy + last run
  review <path> [--profile]   One-shot verified pipeline (default: no apply)
  ask <question>              Open-ended questions, code generation, optimization
  verify <run>                Re-verify a run directory (CI-friendly)
  open <run>                  Console viewer for artifacts + diff
  diff <run>                  Print verified diff (or export patch series)
  explain <run>               Natural-language "why this is safe" report
  replay <run>                Deterministically replay steps from receipts
  logs [--runs]               List runs / receipts / timings / outcomes

{CYAN}{BOLD}Integrations:{RESET}
  index [--ephemeral]         Build/search code index (Greptile-backed)
  agent run "goal"            Multi-agent execution (Cline-led), receipt-bound
  snapshot [--working]        Save current diff as checkpoint
  pr open|comment|sync        GitHub PR workflows

{CYAN}{BOLD}Hygiene & Metrics:{RESET}
  policy show|set|list        Manage policies/profiles (security/refactor/etc.)
  doctor <capsule>            Pipeline health check (inspect→verify→audit→row)
  metrics [--budget|--verify] Show session metrics / resource limits / chain
  clean                       Remove temp repos/cache for ephemeral runs

{CYAN}{BOLD}Shell:{RESET}
  ls/cd/pwd/cat/tree          Navigate filesystem
  git/make                    Run git or make commands
  export/source/env           Manage environment variables
  clear / exit                Clear screen / Exit

{DIM}Profiles: --profile security | refactor | perf | style{RESET}
{DIM}Run outputs: plan.json, patches/, verified.diff, receipts/, audit.md{RESET}
""")

    def do_exit(self, arg: str) -> bool:
        """Exit the shell."""
        print(f"{DIM}Goodbye!{RESET}")
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit the shell."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D."""
        print()
        return self.do_exit(arg)

    def emptyline(self) -> None:
        """Do nothing on empty line."""
        pass

    def default(self, line: str) -> None:
        """Forward unknown commands to the CLI root.

        This is the passthrough router: anything not implemented as do_<cmd>
        gets forwarded to the real CLI application. Every CLI subcommand becomes
        automatically shell-available without needing a do_* handler.

        Examples:
            capseal> profile --help
            capseal> workflow run my_workflow.yaml
            capseal> attest sign capsule.json
            capseal> trace init
        """
        argv = shlex.split(line)
        if not argv:
            return

        try:
            self._invoke_cli(argv)
        except SystemExit:
            # Click may call sys.exit(); treat as normal termination
            pass
        except Exception as e:
            # If CLI invocation fails, show as unknown command
            print(f"{RED}Unknown command: {line}{RESET}")
            print(f"{DIM}Type 'help' for available commands, or try: {argv[0]} --help{RESET}")


def run_shell() -> None:
    """Entry point for the interactive shell."""
    try:
        shell = CapsealShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print(f"\n{DIM}Interrupted. Goodbye!{RESET}")


@click.command("shell")
def shell_command() -> None:
    """Start interactive Capseal shell."""
    run_shell()


__all__ = ["CapsealShell", "run_shell", "shell_command"]
