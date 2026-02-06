#!/usr/bin/env python3
"""AgentEvalBench v1 - Refactor Harness.

Minimal, robust episode runner for evaluating Claude Code on refactor tasks.

CLI entrypoints:
    python -m agent_bench.refactor_harness_v1 init --repo ~/projects/market-dashboard --run-dir artifacts/refactor_eval
    python -m agent_bench.refactor_harness_v1 episode --run-dir ... --task-id <id> --spec-level <0-3> --knobs <json>
    python -m agent_bench.refactor_harness_v1 verify --run-dir ... --episode <id>
    python -m agent_bench.refactor_harness_v1 batch --run-dir ... --tasks QUICK_START_TASKS --spec-level <0-3>
    python -m agent_bench.refactor_harness_v1 smoke --repo ... --run-dir ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import tasks
from agent_bench.refactor_tasks_v1 import (
    ALL_TASKS,
    QUICK_START_TASKS,
    BOUNDARY_TASKS,
    RefactorTask,
    get_task,
    get_prompt,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class KnobSettings:
    """Knob settings for an episode."""
    spec_level: int = 0  # 0-3 (task prompt precision)
    tool_noise_level: int = 0  # 0-3 (0=normal, 1+=degraded tools)
    verify_flip_rate: float = 0.0  # 0..1 (probability of flipping verifier result)
    max_reads: int = 0  # 0 = unlimited
    max_greps: int = 0
    max_lsp_calls: int = 0
    max_bash_calls: int = 0
    max_bytes_read: int = 0
    max_diff_lines: int = 0  # 0 = use task default
    max_files_touched: int = 0  # 0 = use task default
    sandbox_mode: bool = True  # Just record; don't fight platform restrictions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnobSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "KnobSettings":
        return cls.from_dict(json.loads(s))

    def hash(self) -> str:
        """Short hash for identifying knob configuration."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:8]


@dataclass
class RunMeta:
    """Metadata for a run directory."""
    run_id: str
    repo_path: str
    base_commit: str
    created_at: str
    worktree_base: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMeta":
        return cls(**d)


@dataclass
class VerifyResult:
    """Result of running the verifier."""
    cmd: str
    exit_code: int
    stdout_path: str
    stderr_path: str
    duration_s: float
    real_pass: bool  # Actual verifier result
    injected_flip: bool  # Whether we injected a flip
    final_pass: bool  # Result after flip injection


@dataclass
class EpisodeResult:
    """Final result of an episode."""
    episode_id: str
    task_id: str
    spec_level: int
    knobs_hash: str
    verifier_exit: int
    injected_flip: bool
    passed: bool
    reason: str  # "pass", "verifier_failed", "budget_exceeded", "constraint_violated", etc.
    time_s: float
    diff_lines: int
    files_touched: int


# =============================================================================
# Git worktree management
# =============================================================================

def get_repo_base_commit(repo_path: Path) -> str:
    """Get the current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def create_worktree(repo_path: Path, worktree_path: Path, commit: str) -> None:
    """Create a git worktree for isolated episode execution."""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    # Create worktree at specific commit (detached HEAD)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), commit],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


def remove_worktree(repo_path: Path, worktree_path: Path) -> None:
    """Remove a git worktree."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_path,
        capture_output=True,
        check=False,  # Don't fail if already removed
    )


def get_diff_stats(worktree_path: Path) -> Tuple[int, int, List[str]]:
    """Get diff statistics: (lines_changed, files_touched, file_list)."""
    result = subprocess.run(
        ["git", "diff", "--numstat", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )

    lines_changed = 0
    files = []

    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split("\t")
            if len(parts) >= 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                removed = int(parts[1]) if parts[1] != "-" else 0
                lines_changed += added + removed
                files.append(parts[2])

    return lines_changed, len(files), files


def get_diff_patch(worktree_path: Path) -> str:
    """Get the full diff as a patch."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return result.stdout


# =============================================================================
# Run directory management
# =============================================================================

def init_run_dir(repo_path: Path, run_dir: Path) -> RunMeta:
    """Initialize a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes").mkdir(exist_ok=True)
    (run_dir / "worktrees").mkdir(exist_ok=True)

    repo_path = repo_path.resolve()
    base_commit = get_repo_base_commit(repo_path)

    meta = RunMeta(
        run_id=uuid.uuid4().hex[:8],
        repo_path=str(repo_path),
        base_commit=base_commit,
        created_at=datetime.now().isoformat(),
        worktree_base=str(run_dir / "worktrees"),
    )

    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    # Initialize episodes.csv with header
    csv_path = run_dir / "episodes.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id", "task_id", "spec_level", "knobs_hash",
                "verifier_exit", "injected_flip", "passed", "reason",
                "time_s", "diff_lines", "files_touched",
            ])

    return meta


def load_run_meta(run_dir: Path) -> RunMeta:
    """Load run metadata."""
    with open(run_dir / "run_meta.json") as f:
        return RunMeta.from_dict(json.load(f))


# =============================================================================
# Episode management
# =============================================================================

def create_episode_dir(run_dir: Path, task: RefactorTask, knobs: KnobSettings) -> Tuple[str, Path]:
    """Create an episode directory with initial artifacts."""
    episode_id = f"{task.task_id}_{knobs.hash()}_{uuid.uuid4().hex[:6]}"
    episode_dir = run_dir / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "logs").mkdir(exist_ok=True)

    # Write knobs.json
    with open(episode_dir / "knobs.json", "w") as f:
        json.dump(knobs.to_dict(), f, indent=2)

    # Write task.json
    task_meta = {
        "task_id": task.task_id,
        "category": task.category.value,
        "title": task.title,
        "target_files": task.target_files,
        "acceptance_criteria": task.acceptance_criteria,
        "verifier_cmd": task.verifier_cmd,
        "difficulty": task.difficulty,
        "max_diff_lines": task.max_diff_lines,
        "max_files_touched": task.max_files_touched,
    }
    with open(episode_dir / "task.json", "w") as f:
        json.dump(task_meta, f, indent=2)

    # Write prompt.txt
    prompt = get_prompt(task, knobs.spec_level)

    # Add tool noise instructions if needed
    if knobs.tool_noise_level >= 1:
        prompt = f"""CONSTRAINT: LSP tools are unavailable for this task.
Do NOT use: goToDefinition, findReferences, hover, documentSymbol, workspaceSymbol.
Use only: grep, glob, read, edit, bash.

---

{prompt}"""

    with open(episode_dir / "prompt.txt", "w") as f:
        f.write(prompt)

    # Write agent_notes.md template
    with open(episode_dir / "agent_notes.md", "w") as f:
        f.write(f"""# Agent Notes: {task.task_id}

Episode: {episode_id}
Task: {task.title}
Spec Level: {knobs.spec_level}

## Approach


## Changes Made


## Issues Encountered


## Time Spent

""")

    return episode_id, episode_dir


def run_verifier(
    worktree_path: Path,
    verifier_cmd: str,
    episode_dir: Path,
    flip_rate: float = 0.0,
    timeout: int = 600,
) -> VerifyResult:
    """Run the verifier command and capture results."""
    start_time = time.time()

    # Replace repo path in verifier command with worktree path
    # The verifier_cmd uses /home/ryan/projects/market-dashboard
    cmd = verifier_cmd.replace("/home/ryan/projects/market-dashboard", str(worktree_path))

    stdout_path = episode_dir / "logs" / "verify_stdout.txt"
    stderr_path = episode_dir / "logs" / "verify_stderr.txt"

    try:
        with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=worktree_path,
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=timeout,
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1
        with open(stderr_path, "a") as f:
            f.write(f"\n\nTIMEOUT after {timeout}s\n")
    except Exception as e:
        exit_code = -2
        with open(stderr_path, "a") as f:
            f.write(f"\n\nERROR: {e}\n")

    duration = time.time() - start_time
    real_pass = exit_code == 0

    # Inject flip if configured
    injected_flip = False
    final_pass = real_pass
    if flip_rate > 0 and random.random() < flip_rate:
        injected_flip = True
        final_pass = not real_pass

    verify_result = VerifyResult(
        cmd=cmd,
        exit_code=exit_code,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        duration_s=duration,
        real_pass=real_pass,
        injected_flip=injected_flip,
        final_pass=final_pass,
    )

    # Write verify.json
    with open(episode_dir / "verify.json", "w") as f:
        json.dump(asdict(verify_result), f, indent=2)

    return verify_result


def check_constraints(
    task: RefactorTask,
    knobs: KnobSettings,
    diff_lines: int,
    files_touched: int,
    changed_files: List[str],
) -> Tuple[bool, str]:
    """Check if the episode meets constraints. Returns (passed, reason)."""
    # Determine limits
    max_diff = knobs.max_diff_lines if knobs.max_diff_lines > 0 else task.max_diff_lines
    max_files = knobs.max_files_touched if knobs.max_files_touched > 0 else task.max_files_touched

    if max_diff > 0 and diff_lines > max_diff:
        return False, f"budget_exceeded:diff_lines ({diff_lines} > {max_diff})"

    if max_files > 0 and files_touched > max_files:
        return False, f"budget_exceeded:files_touched ({files_touched} > {max_files})"

    # Check forbidden paths (task-specific, parsed from task.notes if present)
    # For now, just check the task_10 forbidden path constraint
    if task.task_id == "forbidden_path_refactor":
        for f in changed_files:
            if "capsuletech" in f:
                return False, f"constraint_violated:forbidden_path ({f})"

    return True, "pass"


def record_episode_result(
    run_dir: Path,
    result: EpisodeResult,
) -> None:
    """Append episode result to episodes.csv."""
    csv_path = run_dir / "episodes.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.episode_id,
            result.task_id,
            result.spec_level,
            result.knobs_hash,
            result.verifier_exit,
            result.injected_flip,
            result.passed,
            result.reason,
            f"{result.time_s:.2f}",
            result.diff_lines,
            result.files_touched,
        ])


def finalize_episode(
    episode_dir: Path,
    worktree_path: Path,
    task: RefactorTask,
    knobs: KnobSettings,
    verify_result: VerifyResult,
    episode_id: str,
    start_time: float,
) -> EpisodeResult:
    """Finalize an episode: record diff, check constraints, write result."""
    # Get diff stats
    diff_lines, files_touched, changed_files = get_diff_stats(worktree_path)

    # Write diff.patch
    patch = get_diff_patch(worktree_path)
    with open(episode_dir / "diff.patch", "w") as f:
        f.write(patch)

    # Write changed_files.txt
    with open(episode_dir / "changed_files.txt", "w") as f:
        f.write("\n".join(changed_files))

    # Check constraints
    constraints_passed, constraint_reason = check_constraints(
        task, knobs, diff_lines, files_touched, changed_files
    )

    # Determine final result
    total_time = time.time() - start_time

    if not verify_result.final_pass:
        passed = False
        reason = "verifier_failed"
    elif not constraints_passed:
        passed = False
        reason = constraint_reason
    else:
        passed = True
        reason = "pass"

    result = EpisodeResult(
        episode_id=episode_id,
        task_id=task.task_id,
        spec_level=knobs.spec_level,
        knobs_hash=knobs.hash(),
        verifier_exit=verify_result.exit_code,
        injected_flip=verify_result.injected_flip,
        passed=passed,
        reason=reason,
        time_s=total_time,
        diff_lines=diff_lines,
        files_touched=files_touched,
    )

    # Write result.json
    with open(episode_dir / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a run directory."""
    repo_path = Path(args.repo).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()

    if not repo_path.exists():
        print(f"Error: repo does not exist: {repo_path}")
        return 1

    if not (repo_path / ".git").exists():
        print(f"Error: not a git repo: {repo_path}")
        return 1

    meta = init_run_dir(repo_path, run_dir)

    print(f"Initialized run directory: {run_dir}")
    print(f"  Run ID: {meta.run_id}")
    print(f"  Repo: {meta.repo_path}")
    print(f"  Base commit: {meta.base_commit}")
    print(f"  Worktrees: {meta.worktree_base}")

    return 0


def cmd_episode(args: argparse.Namespace) -> int:
    """Run a single episode (manual agent mode)."""
    run_dir = Path(args.run_dir).expanduser().resolve()

    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        print("Run 'init' first.")
        return 1

    meta = load_run_meta(run_dir)
    repo_path = Path(meta.repo_path)

    # Get task
    try:
        task = get_task(args.task_id)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Parse knobs
    knobs = KnobSettings()
    knobs.spec_level = args.spec_level
    if args.knobs:
        knobs = KnobSettings.from_json(args.knobs)
        knobs.spec_level = args.spec_level  # CLI overrides

    print(f"\n{'='*70}")
    print(f"EPISODE: {task.task_id}")
    print(f"{'='*70}")
    print(f"Task: {task.title}")
    print(f"Spec Level: {knobs.spec_level}")
    print(f"Knobs Hash: {knobs.hash()}")
    print()

    # Create episode directory
    episode_id, episode_dir = create_episode_dir(run_dir, task, knobs)
    print(f"Episode ID: {episode_id}")
    print(f"Episode Dir: {episode_dir}")

    # Create worktree
    worktree_path = Path(meta.worktree_base) / episode_id
    print(f"\nCreating worktree at: {worktree_path}")
    create_worktree(repo_path, worktree_path, meta.base_commit)

    # Write worktree path to episode
    with open(episode_dir / "worktree_path.txt", "w") as f:
        f.write(str(worktree_path))

    start_time = time.time()

    # Check for auto-agent
    claude_cmd = shutil.which("claude")
    if claude_cmd and args.agent_cmd:
        print(f"\nRunning agent: {args.agent_cmd}")
        # TODO: Implement auto-agent execution
        print("Auto-agent not yet implemented. Falling back to manual mode.")
        claude_cmd = None

    if not claude_cmd or not args.agent_cmd:
        # Manual mode
        print(f"""
{'='*70}
MANUAL AGENT MODE
{'='*70}

1. Open a new terminal and run:
   cd {worktree_path}

2. Read the task prompt:
   cat {episode_dir}/prompt.txt

3. Execute the refactor as Claude Code would.

4. When finished, record notes in:
   {episode_dir}/agent_notes.md

5. Then run verification:
   python -m agent_bench.refactor_harness_v1 verify \\
       --run-dir {run_dir} \\
       --episode {episode_id}

{'='*70}
""")
        print("Waiting for manual completion...")
        print("(Press Ctrl+C to cancel, or run 'verify' command when ready)")

        # In manual mode, we just set up and exit
        # The user will run 'verify' separately
        return 0

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify and finalize an episode."""
    run_dir = Path(args.run_dir).expanduser().resolve()
    episode_id = args.episode

    episode_dir = run_dir / "episodes" / episode_id
    if not episode_dir.exists():
        print(f"Error: episode not found: {episode_dir}")
        return 1

    meta = load_run_meta(run_dir)
    repo_path = Path(meta.repo_path)

    # Load task and knobs
    with open(episode_dir / "task.json") as f:
        task_meta = json.load(f)
    task = get_task(task_meta["task_id"])

    with open(episode_dir / "knobs.json") as f:
        knobs = KnobSettings.from_dict(json.load(f))

    # Get worktree path
    worktree_path_file = episode_dir / "worktree_path.txt"
    if worktree_path_file.exists():
        worktree_path = Path(worktree_path_file.read_text().strip())
    else:
        worktree_path = Path(meta.worktree_base) / episode_id

    if not worktree_path.exists():
        print(f"Error: worktree not found: {worktree_path}")
        return 1

    print(f"\n{'='*70}")
    print(f"VERIFYING: {episode_id}")
    print(f"{'='*70}")
    print(f"Task: {task.title}")
    print(f"Worktree: {worktree_path}")
    print()

    # Get start time (approximate from episode creation)
    start_time = time.time()  # Not accurate for total time in manual mode

    # Run verifier
    print(f"Running verifier: {task.verifier_cmd}")
    print("(timeout: 600s)")
    verify_result = run_verifier(
        worktree_path,
        task.verifier_cmd,
        episode_dir,
        flip_rate=knobs.verify_flip_rate,
        timeout=600,
    )

    print(f"\nVerifier exit code: {verify_result.exit_code}")
    print(f"Real pass: {verify_result.real_pass}")
    if verify_result.injected_flip:
        print(f"FLIP INJECTED: final_pass={verify_result.final_pass}")

    # Finalize episode
    result = finalize_episode(
        episode_dir,
        worktree_path,
        task,
        knobs,
        verify_result,
        episode_id,
        start_time,
    )

    # Record to CSV
    record_episode_result(run_dir, result)

    print(f"\n{'='*70}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    print(f"{'='*70}")
    print(f"Reason: {result.reason}")
    print(f"Diff lines: {result.diff_lines}")
    print(f"Files touched: {result.files_touched}")
    print(f"Result saved to: {episode_dir / 'result.json'}")

    # Cleanup worktree unless --keep-worktree
    if not args.keep_worktree:
        print(f"\nRemoving worktree: {worktree_path}")
        remove_worktree(repo_path, worktree_path)
    else:
        print(f"\nKeeping worktree: {worktree_path}")

    return 0 if result.passed else 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Run a batch of episodes."""
    print("Batch mode not yet implemented.")
    print("Use 'episode' + 'verify' for now.")
    return 1


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run a smoke test."""
    repo_path = Path(args.repo).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()

    print(f"\n{'='*70}")
    print("SMOKE TEST")
    print(f"{'='*70}")

    # Initialize
    if not run_dir.exists():
        print("Initializing run directory...")
        meta = init_run_dir(repo_path, run_dir)
    else:
        meta = load_run_meta(run_dir)

    # Pick easiest task
    task = QUICK_START_TASKS[0]  # rename_provider_status
    knobs = KnobSettings(spec_level=0)  # Most precise prompt, no noise

    print(f"\nTask: {task.task_id} ({task.title})")
    print(f"Spec Level: 0 (surgical)")
    print(f"Knobs: default (no noise, no flips)")

    # Create episode
    episode_id, episode_dir = create_episode_dir(run_dir, task, knobs)
    print(f"\nEpisode: {episode_id}")
    print(f"Episode Dir: {episode_dir}")

    # Create worktree
    worktree_path = Path(meta.worktree_base) / episode_id
    print(f"Creating worktree: {worktree_path}")
    create_worktree(Path(meta.repo_path), worktree_path, meta.base_commit)

    with open(episode_dir / "worktree_path.txt", "w") as f:
        f.write(str(worktree_path))

    # Check artifacts exist
    print("\nChecking artifacts...")
    required = ["knobs.json", "task.json", "prompt.txt", "agent_notes.md"]
    for fname in required:
        path = episode_dir / fname
        if path.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} MISSING")
            return 1

    print(f"""
{'='*70}
SMOKE TEST SETUP COMPLETE
{'='*70}

To complete the smoke test:

1. Make the refactor changes in:
   {worktree_path}

2. Run verification:
   python -m agent_bench.refactor_harness_v1 verify \\
       --run-dir {run_dir} \\
       --episode {episode_id}

Or to just test the verifier on unchanged code (will fail):
   python -m agent_bench.refactor_harness_v1 verify \\
       --run-dir {run_dir} \\
       --episode {episode_id} \\
       --keep-worktree

{'='*70}
""")

    return 0


def cmd_list_tasks(args: argparse.Namespace) -> int:
    """List available tasks."""
    print(f"\n{'='*70}")
    print("AVAILABLE TASKS")
    print(f"{'='*70}\n")

    task_set = args.task_set.upper() if args.task_set else "ALL"

    if task_set == "QUICK_START":
        tasks = QUICK_START_TASKS
    elif task_set == "BOUNDARY":
        tasks = BOUNDARY_TASKS
    else:
        tasks = ALL_TASKS

    print(f"Task Set: {task_set} ({len(tasks)} tasks)\n")

    for task in tasks:
        print(f"  {task.task_id}")
        print(f"    Title: {task.title}")
        print(f"    Category: {task.category.value}")
        print(f"    Difficulty: {task.difficulty}/5")
        print(f"    Files: {len(task.target_files)}")
        print()

    return 0


def cmd_show_task(args: argparse.Namespace) -> int:
    """Show details of a specific task."""
    try:
        task = get_task(args.task_id)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\n{'='*70}")
    print(f"TASK: {task.task_id}")
    print(f"{'='*70}\n")

    print(f"Title: {task.title}")
    print(f"Category: {task.category.value}")
    print(f"Difficulty: {task.difficulty}/5")
    print(f"Max Diff Lines: {task.max_diff_lines}")
    print(f"Max Files Touched: {task.max_files_touched}")
    print(f"Knob Sensitive: {', '.join(task.knob_sensitive)}")

    print(f"\nTarget Files:")
    for f in task.target_files:
        print(f"  - {f}")

    print(f"\nAcceptance Criteria:")
    for c in task.acceptance_criteria:
        print(f"  - {c}")

    print(f"\nVerifier Command:")
    print(f"  {task.verifier_cmd}")

    spec_level = args.spec_level if hasattr(args, 'spec_level') else 0
    print(f"\nPrompt (spec_level={spec_level}):")
    print("-" * 40)
    print(get_prompt(task, spec_level))
    print("-" * 40)

    return 0


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AgentEvalBench Refactor Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a run directory")
    p_init.add_argument("--repo", required=True, help="Path to the target repo")
    p_init.add_argument("--run-dir", required=True, help="Path to the run directory")

    # episode
    p_episode = subparsers.add_parser("episode", help="Run a single episode")
    p_episode.add_argument("--run-dir", required=True, help="Path to the run directory")
    p_episode.add_argument("--task-id", required=True, help="Task ID to run")
    p_episode.add_argument("--spec-level", type=int, default=0, help="Spec level (0-3)")
    p_episode.add_argument("--knobs", help="JSON string of knob settings")
    p_episode.add_argument("--agent-cmd", help="Command to run agent (optional)")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify and finalize an episode")
    p_verify.add_argument("--run-dir", required=True, help="Path to the run directory")
    p_verify.add_argument("--episode", required=True, help="Episode ID to verify")
    p_verify.add_argument("--keep-worktree", action="store_true", help="Don't remove worktree")

    # batch
    p_batch = subparsers.add_parser("batch", help="Run a batch of episodes")
    p_batch.add_argument("--run-dir", required=True, help="Path to the run directory")
    p_batch.add_argument("--tasks", default="QUICK_START", help="Task set to run")
    p_batch.add_argument("--spec-level", type=int, default=0, help="Spec level (0-3)")
    p_batch.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    p_batch.add_argument("--knobs", help="JSON string of knob settings")

    # smoke
    p_smoke = subparsers.add_parser("smoke", help="Run a smoke test")
    p_smoke.add_argument("--repo", required=True, help="Path to the target repo")
    p_smoke.add_argument("--run-dir", required=True, help="Path to the run directory")

    # list-tasks
    p_list = subparsers.add_parser("list-tasks", help="List available tasks")
    p_list.add_argument("--task-set", help="Task set: ALL, QUICK_START, BOUNDARY")

    # show-task
    p_show = subparsers.add_parser("show-task", help="Show details of a task")
    p_show.add_argument("task_id", help="Task ID to show")
    p_show.add_argument("--spec-level", type=int, default=0, help="Spec level for prompt")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "episode": cmd_episode,
        "verify": cmd_verify,
        "batch": cmd_batch,
        "smoke": cmd_smoke,
        "list-tasks": cmd_list_tasks,
        "show-task": cmd_show_task,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
