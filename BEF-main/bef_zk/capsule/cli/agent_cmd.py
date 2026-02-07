"""CLI commands for agent execution with tandem risk gating."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@click.group("agent")
def agent_group() -> None:
    """Run and inspect proof-carrying agent executions.

    \b
    Commands:
        run      Execute an agent with tandem risk gating
        inspect  View what happened in an agent run
    """
    pass


@agent_group.command("run")
@click.argument("task", type=str)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory (default: .capseal/runs/<timestamp>)",
)
@click.option(
    "--posteriors",
    "-p",
    type=click.Path(exists=True),
    help="Path to learned model (default: .capseal/models/beta_posteriors.npz)",
)
@click.option(
    "--max-retries",
    "-r",
    default=3,
    type=int,
    help="Max adaptation attempts per action (default: 3)",
)
@click.option("--prove/--no-prove", default=True, help="Generate cryptographic proof")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def agent_run_command(
    task: str,
    output_dir: str | None,
    posteriors: str | None,
    max_retries: int,
    prove: bool,
    verbose: bool,
) -> None:
    """Execute an agent with tandem risk gating.

    The agent proposes actions, CapSeal evaluates risk, and the agent adapts
    based on feedback. Every decision is recorded and proof-carrying.

    \b
    Examples:
        capseal agent run "list the python files"
        capseal agent run "fix the SQL injection in auth.py" --prove
        capseal agent run "refactor database layer" --max-retries 5
    """
    from bef_zk.capsule.agent_loop import AgentLoop
    from bef_zk.capsule.agents.simple_agent import SimpleAgent

    # Determine output directory
    if output_dir:
        run_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = Path(".capseal/runs") / f"{timestamp}-agent"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine posteriors path
    posteriors_path = Path(posteriors) if posteriors else None

    click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
    click.echo(f"{CYAN}  AGENT RUN{RESET}")
    click.echo(f"{CYAN}{'═' * 65}{RESET}")
    click.echo(f"  Task: {task}")
    click.echo(f"  Output: {run_dir}")
    click.echo(f"  Max retries: {max_retries}")
    click.echo(f"  Prove: {prove}")
    click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    # Create a simple mock LLM for demo purposes
    # In production, this would be replaced with actual LLM calls
    action_count = [0]  # Track how many actions we've taken

    def mock_llm(prompt: str) -> str:
        """Mock LLM that demonstrates the agent loop."""
        action_count[0] += 1

        # Check if we've already taken an action (PREVIOUS ACTIONS in prompt)
        if "PREVIOUS ACTIONS" in prompt or action_count[0] > 1:
            return "DONE"

        # Check if we're in adaptation mode
        if "was flagged as risky" in prompt:
            # Return a simpler action
            return """ACTION_TYPE: tool_call
DESCRIPTION: List files (simplified approach)
INSTRUCTION: List files in current directory
INPUTS: {"command": "ls -la", "tool_name": "bash"}"""

        # Check for specific task patterns
        if "list" in task.lower() and "python" in task.lower():
            return """ACTION_TYPE: tool_call
DESCRIPTION: Find Python files
INSTRUCTION: Find all Python files in the project
INPUTS: {"command": "find . -maxdepth 2 -name '*.py' -type f | head -20", "tool_name": "bash"}"""

        if "sql injection" in task.lower() or "fix" in task.lower():
            return """ACTION_TYPE: code_gen
DESCRIPTION: Fix SQL injection vulnerability
INSTRUCTION: Update query to use parameterized queries
INPUTS: {"file": "auth.py", "line": 42}
DIFF:
-    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))"""

        # Default: acknowledge task completion
        return "DONE"

    # Create mock tools
    def mock_bash(command: str) -> str:
        """Mock bash execution."""
        import subprocess

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "(command timed out)"
        except Exception as e:
            return f"(error: {e})"

    # Build the agent
    agent = SimpleAgent(
        llm_fn=mock_llm,
        tools={"bash": mock_bash},
    )

    # Build the loop
    loop = AgentLoop(
        agent=agent,
        output_dir=run_dir,
        posteriors_path=posteriors_path,
        max_retries=max_retries,
        prove=prove,
    )

    # Run
    click.echo(f"{DIM}Running agent...{RESET}\n")

    try:
        result = loop.run(task)

        # Print results
        click.echo(f"\n{CYAN}{'─' * 65}{RESET}")
        click.echo(f"{BOLD}Results:{RESET}")
        click.echo(f"  Actions executed: {result.total_actions}")
        click.echo(f"  Adaptations: {result.total_adaptations}")
        click.echo(f"  Success rate: {result.success_rate:.0%}")

        if verbose:
            click.echo(f"\n{BOLD}Risk Log:{RESET}")
            for entry in result.risk_log:
                decision = entry.get("decision", "unknown")
                color = GREEN if decision == "pass" else (YELLOW if decision == "human_review" else RED)
                click.echo(
                    f"  {color}[{decision.upper()}]{RESET} "
                    f"{entry.get('action_type')}: {entry.get('description')}"
                )
                if entry.get("risk_score") is not None:
                    click.echo(f"    Risk: {entry['risk_score']:.2f} | {entry.get('suggestion', '')}")

        if result.capsule_hash:
            click.echo(f"\n{GREEN}Capsule:{RESET} {result.capsule_hash[:16]}...")
            click.echo(f"{DIM}Verify with: capseal verify-capsule {run_dir}/agent_capsule.json{RESET}")

        click.echo(f"\n{CYAN}{'═' * 65}{RESET}\n")

        # Save risk log for inspection
        risk_log_path = run_dir / "risk_log.json"
        risk_log_path.write_text(json.dumps(result.risk_log, indent=2))

    except Exception as e:
        click.echo(f"\n{RED}Error: {e}{RESET}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@agent_group.command("inspect")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def agent_inspect_command(run_dir: str, json_output: bool) -> None:
    """View what happened in an agent run.

    Shows the full story: proposals, adaptations, risk scores, and final actions.

    \b
    Examples:
        capseal agent inspect .capseal/runs/latest
        capseal agent inspect .capseal/runs/20240205T123456-agent --json
    """
    run_path = Path(run_dir)

    # Load artifacts
    risk_log_path = run_path / "risk_log.json"
    capsule_path = run_path / "agent_capsule.json"

    if not risk_log_path.exists() and not capsule_path.exists():
        click.echo(f"{RED}No agent run found at: {run_path}{RESET}")
        sys.exit(1)

    risk_log = []
    if risk_log_path.exists():
        risk_log = json.loads(risk_log_path.read_text())

    capsule = None
    if capsule_path.exists():
        capsule = json.loads(capsule_path.read_text())

    if json_output:
        output = {
            "run_dir": str(run_path),
            "risk_log": risk_log,
            "capsule": capsule,
        }
        click.echo(json.dumps(output, indent=2))
        return

    # Get task from risk log
    task = risk_log[0].get("task", "Unknown task") if risk_log else "Unknown task"

    click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
    click.echo(f"{CYAN}  AGENT RUN: \"{task}\"{RESET}")
    click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    # Group by action
    actions = {}
    for entry in risk_log:
        key = entry.get("description", "unknown")
        if key not in actions:
            actions[key] = []
        actions[key].append(entry)

    action_num = 0
    executed_count = 0
    adapted_count = 0
    skipped_count = 0

    for description, attempts in actions.items():
        action_num += 1

        for i, attempt in enumerate(attempts):
            decision = attempt.get("decision", "unknown")
            risk_score = attempt.get("risk_score")
            suggestion = attempt.get("suggestion", "")
            action_type = attempt.get("action_type", "unknown")
            attempt_num = attempt.get("attempt", i + 1)

            # Format attempt label
            if len(attempts) > 1 and i > 0:
                label = f"Action {action_num} (retry {attempt_num})"
                adapted_count += 1
            else:
                label = f"Action {action_num}"

            click.echo(f"  {BOLD}{label}:{RESET} {action_type} — \"{description}\"")

            # Risk info
            if risk_score is not None:
                risk_str = f"Risk score: {risk_score:.2f}"
            else:
                risk_str = "Risk score: N/A"

            if decision == "pass":
                click.echo(f"    → {risk_str} | Decision: {GREEN}PASS{RESET}")
                click.echo(f"    → Executed: {GREEN}✓ success{RESET}")
                executed_count += 1
            elif decision == "skip":
                click.echo(f"    → {risk_str} | Decision: {RED}SKIP{RESET}")
                click.echo(f"    → Suggestion: {suggestion}")
                skipped_count += 1
            elif decision == "human_review":
                click.echo(f"    → {risk_str} | Decision: {YELLOW}HUMAN REVIEW{RESET}")
                click.echo(f"    → Suggestion: {suggestion}")

            click.echo()

    # Summary
    click.echo(f"  {CYAN}{'─' * 61}{RESET}")
    click.echo(
        f"  Actions: {executed_count} executed, {adapted_count} adapted, {skipped_count} skipped"
    )

    if capsule:
        capsule_hash = capsule.get("capsule_hash", "")[:16]
        verified = capsule.get("verification", {}).get("constraints_valid", False)
        verified_str = f"{GREEN}verified ✓{RESET}" if verified else f"{RED}unverified{RESET}"
        click.echo(f"  Capsule: {capsule_hash}... ({verified_str})")

    click.echo(f"{CYAN}{'═' * 65}{RESET}\n")


__all__ = ["agent_group"]
