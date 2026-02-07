"""SimpleAgent - A basic LLM-backed agent that adapts based on risk feedback.

This is a reference implementation showing how to implement CapSealAgent.
It wraps any LLM function and demonstrates the adaptation pattern.

Usage:
    def my_llm(prompt: str) -> str:
        # Call your LLM here
        return response

    agent = SimpleAgent(llm_fn=my_llm, tools={"bash": run_bash})
    loop = AgentLoop(agent, output_dir=Path(".capseal/runs/my-task"))
    result = loop.run("fix the SQL injection in auth.py")
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..agent_loop import (
    ActionResult,
    CapSealAgent,
    ProposedAction,
    RiskFeedback,
)


class SimpleAgent(CapSealAgent):
    """A basic LLM-backed agent that adapts based on risk feedback.

    This is a reference implementation — minimal but functional.
    Extend or replace for production use.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        tools: dict[str, Callable] | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the agent.

        Args:
            llm_fn: Function that takes a prompt string and returns a response string.
            tools: Dict mapping tool names to callable functions.
            system_prompt: Optional system prompt to prepend to all LLM calls.
        """
        self.llm_fn = llm_fn
        self.tools = tools or {}
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation: list[dict] = []

    def _default_system_prompt(self) -> str:
        return """You are a helpful coding assistant. When given a task:

1. Think about what actions you need to take
2. Propose ONE action at a time using the format below
3. If you're done with the task, respond with just: DONE

Action format (use exactly this structure):
ACTION_TYPE: tool_call | code_gen | api_request
DESCRIPTION: Brief description of what this action does
INSTRUCTION: The detailed instruction or prompt
INPUTS: JSON object with parameters
DIFF: (only for code_gen) The code diff or changes

Example:
ACTION_TYPE: tool_call
DESCRIPTION: List Python files in src directory
INSTRUCTION: List all .py files in the src/ directory
INPUTS: {"command": "find src -name '*.py'", "tool_name": "bash"}

Example code generation:
ACTION_TYPE: code_gen
DESCRIPTION: Fix SQL injection vulnerability
INSTRUCTION: Update the query to use parameterized queries
INPUTS: {"file": "auth.py", "line": 42}
DIFF:
-    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

If the task cannot be completed safely, respond with: DONE
"""

    def plan(self, task: str, context: dict) -> ProposedAction | None:
        """Propose the next action."""
        # Build prompt from task + context
        prompt = self._build_prompt(task, context)

        # Call LLM
        response = self.llm_fn(prompt)

        # Parse response
        parsed = self._parse_response(response)
        if parsed is None:
            return None

        # Record in conversation
        self.conversation.append({"role": "user", "content": prompt})
        self.conversation.append({"role": "assistant", "content": response})

        return parsed

    def adapt(
        self, proposed: ProposedAction, feedback: RiskFeedback, context: dict
    ) -> ProposedAction | None:
        """Replan based on risk feedback."""
        # Build adaptation prompt
        adaptation_prompt = self._build_adaptation_prompt(proposed, feedback)

        # Add to conversation
        self.conversation.append({"role": "system", "content": adaptation_prompt})

        # Call LLM with full conversation context
        full_prompt = self._format_conversation()
        response = self.llm_fn(full_prompt)

        # Parse response
        parsed = self._parse_response(response)

        # Record in conversation
        self.conversation.append({"role": "assistant", "content": response})

        return parsed

    def execute(self, action: ProposedAction) -> ActionResult:
        """Execute the approved action."""
        import time

        start = time.time()

        try:
            if action.action_type == "tool_call":
                tool_name = action.metadata.get("tool_name") or action.inputs.get(
                    "tool_name"
                )
                if tool_name and tool_name in self.tools:
                    tool_fn = self.tools[tool_name]
                    # Remove tool_name from inputs before calling
                    inputs = {k: v for k, v in action.inputs.items() if k != "tool_name"}
                    output = tool_fn(**inputs)
                    duration = int((time.time() - start) * 1000)
                    return ActionResult(
                        success=True,
                        outputs={"result": output},
                        duration_ms=duration,
                    )
                else:
                    return ActionResult(
                        success=False,
                        outputs={},
                        error=f"Unknown tool: {tool_name}",
                        duration_ms=int((time.time() - start) * 1000),
                    )

            elif action.action_type == "code_gen":
                # For code generation, the diff is the output
                duration = int((time.time() - start) * 1000)
                return ActionResult(
                    success=True,
                    outputs={"diff": action.diff_text, "file": action.inputs.get("file")},
                    duration_ms=duration,
                )

            elif action.action_type == "api_request":
                # Placeholder for API requests
                duration = int((time.time() - start) * 1000)
                return ActionResult(
                    success=True,
                    outputs={"request": action.inputs},
                    duration_ms=duration,
                )

            else:
                return ActionResult(
                    success=False,
                    outputs={},
                    error=f"Unknown action type: {action.action_type}",
                    duration_ms=int((time.time() - start) * 1000),
                )

        except Exception as e:
            return ActionResult(
                success=False,
                outputs={},
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    def on_result(
        self, action: ProposedAction, result: ActionResult, context: dict
    ) -> None:
        """Update conversation with execution result."""
        result_msg = f"Action executed: {'SUCCESS' if result.success else 'FAILED'}"
        if result.outputs:
            result_msg += f"\nOutputs: {json.dumps(result.outputs, indent=2)}"
        if result.error:
            result_msg += f"\nError: {result.error}"

        self.conversation.append({"role": "system", "content": result_msg})

    def _build_prompt(self, task: str, context: dict) -> str:
        """Build the prompt for planning."""
        parts = [self.system_prompt, "", f"TASK: {task}"]

        # Add context about previous actions
        actions_taken = context.get("actions_taken", [])
        if actions_taken:
            parts.append("")
            parts.append("PREVIOUS ACTIONS:")
            for i, action in enumerate(actions_taken[-5:], 1):  # Last 5 actions
                status = "SUCCESS" if action.get("success") else "FAILED"
                risk = action.get("risk_score", "N/A")
                parts.append(
                    f"  {i}. [{status}] {action.get('type')}: {action.get('description')} (risk: {risk})"
                )

        # Add risk summary
        if context.get("adapted_count", 0) > 0:
            parts.append("")
            parts.append(
                f"Note: {context['adapted_count']} action(s) were adapted due to risk feedback."
            )

        parts.append("")
        parts.append("What is your next action? (or DONE if task is complete)")

        return "\n".join(parts)

    def _build_adaptation_prompt(
        self, proposed: ProposedAction, feedback: RiskFeedback
    ) -> str:
        """Build prompt for adaptation after risk feedback."""
        return f"""Your proposed action was flagged as risky.

PROPOSED ACTION:
  Type: {proposed.action_type}
  Description: {proposed.description}

RISK FEEDBACK:
  Decision: {feedback.decision}
  Risk score: {feedback.risk_score}
  Confidence: {feedback.confidence}
  Suggestion: {feedback.suggestion}
  Attempt: {feedback.attempt_number}/3

Please propose a SAFER alternative approach that addresses the risk factors.
Consider:
- Breaking the change into smaller steps
- Adding tests or validation first
- Using a more conservative approach

If you believe this task cannot be completed safely, respond with: DONE"""

    def _format_conversation(self) -> str:
        """Format conversation history for LLM."""
        parts = []
        for msg in self.conversation:
            role = msg["role"].upper()
            content = msg["content"]
            parts.append(f"[{role}]\n{content}\n")
        return "\n".join(parts)

    def _parse_response(self, response: str) -> ProposedAction | None:
        """Parse LLM response into a ProposedAction."""
        response = response.strip()

        # Check for DONE
        if response.upper() == "DONE" or response.upper().startswith("DONE"):
            return None

        # Parse structured format
        try:
            action_type = self._extract_field(response, "ACTION_TYPE")
            description = self._extract_field(response, "DESCRIPTION")
            instruction = self._extract_field(response, "INSTRUCTION")
            inputs_str = self._extract_field(response, "INPUTS")
            diff_text = self._extract_field(response, "DIFF", default="")

            if not action_type or not description:
                # Fallback: treat whole response as a simple action
                return ProposedAction(
                    action_type="observation",
                    description="Agent response",
                    instruction=response,
                    inputs={},
                )

            # Parse inputs JSON
            inputs = {}
            if inputs_str:
                try:
                    inputs = json.loads(inputs_str)
                except json.JSONDecodeError:
                    inputs = {"raw": inputs_str}

            return ProposedAction(
                action_type=action_type.lower().strip(),
                description=description.strip(),
                instruction=instruction.strip() if instruction else description,
                inputs=inputs,
                diff_text=diff_text.strip(),
                metadata={"tool_name": inputs.get("tool_name")},
            )

        except Exception:
            # Fallback for unparseable responses
            return ProposedAction(
                action_type="observation",
                description="Agent response (unparsed)",
                instruction=response,
                inputs={},
            )

    def _extract_field(
        self, text: str, field_name: str, default: str | None = None
    ) -> str | None:
        """Extract a field value from structured text."""
        # Try exact match first
        pattern = rf"^{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default


__all__ = ["SimpleAgent"]
