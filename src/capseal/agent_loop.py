"""AgentLoop - Tandem execution where agents propose, CapSeal evaluates, and agents adapt.

This is the high-level abstraction that turns CapSeal from a passive recorder into
an active collaborator. The agent proposes actions, CapSeal scores them for risk,
feeds that risk back to the agent, and the agent adapts. Every step is recorded
and proof-carrying.

Two levels of integration:
- Level 1 (low-level): AgentRuntime — record actions yourself, get receipts and proofs
- Level 2 (high-level): AgentLoop + CapSealAgent — implement plan/adapt/execute, get
  tandem risk-aware execution with proofs for free

Both produce the same proof-carrying capsules. Both verify the same way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_runtime import AgentRuntime


@dataclass
class ProposedAction:
    """What the agent wants to do next."""

    action_type: str  # "tool_call", "code_gen", "api_request", etc.
    description: str  # Human-readable description
    instruction: str  # The full instruction/prompt
    inputs: dict  # Arguments, context, parameters
    diff_text: str = ""  # For code patches — the proposed diff
    findings: list = field(default_factory=list)  # Semgrep or other findings
    metadata: dict = field(default_factory=dict)  # Anything else


@dataclass
class RiskFeedback:
    """What CapSeal tells the agent about its proposed action."""

    decision: str  # "pass", "skip", "human_review"
    risk_score: float | None  # q(x) — higher means more likely to fail
    confidence: float | None  # How confident the model is (1 - uncertainty)
    features: dict  # The 5 feature scores that drove the decision
    grid_idx: int | None  # Where this action falls in the learned model
    suggestion: str  # Plain-language guidance
    attempt_number: int  # How many times we've tried this task (1 = first try)
    prior_attempts: list[dict] = field(default_factory=list)  # Previous feedbacks


@dataclass
class ActionResult:
    """What happened when the agent executed."""

    success: bool
    outputs: dict
    error: str | None = None
    duration_ms: int = 0


class CapSealAgent(ABC):
    """Interface for agents that work in tandem with CapSeal.

    Implement this to get proof-carrying, risk-aware execution.
    The agent proposes, CapSeal evaluates, the agent adapts.

    Methods to implement:
        plan(task, context) -> ProposedAction or None
        adapt(proposed, feedback, context) -> ProposedAction or None
        execute(action) -> ActionResult

    Optional hooks:
        on_result(action, result, context) — called after execution
        on_task_complete(task, context) — called when plan() returns None
    """

    @abstractmethod
    def plan(self, task: str, context: dict) -> ProposedAction | None:
        """Propose the next action given the current task and context.

        Args:
            task: The high-level task description
            context: Shared context including actions_taken, risk history, etc.

        Returns:
            ProposedAction to attempt, or None to signal task completion.
        """
        ...

    @abstractmethod
    def adapt(
        self, proposed: ProposedAction, feedback: RiskFeedback, context: dict
    ) -> ProposedAction | None:
        """Replan based on risk feedback.

        Called when the gate says skip or human_review (and human rejected).
        Use the feedback to propose a safer alternative.

        Args:
            proposed: The action that was rejected
            feedback: Risk feedback explaining why and what to try instead
            context: Shared context

        Returns:
            A new ProposedAction to try again, or None to give up on this task.
        """
        ...

    @abstractmethod
    def execute(self, action: ProposedAction) -> ActionResult:
        """Execute the approved action.

        Only called after the gate says pass (or human approves).

        Args:
            action: The approved action to execute

        Returns:
            ActionResult with success status, outputs, and any errors.
        """
        ...

    def on_result(
        self, action: ProposedAction, result: ActionResult, context: dict
    ) -> None:
        """Optional hook — called after execution.

        Use this to update internal state, memory, conversation history, etc.
        """
        pass

    def on_task_complete(self, task: str, context: dict) -> None:
        """Optional hook — called when plan() returns None."""
        pass


@dataclass
class LoopResult:
    """What comes back from AgentLoop.run()."""

    capsule: dict | None  # The proof-carrying capsule (None if prove=False)
    actions: list  # All recorded actions
    risk_log: list[dict]  # Every gate evaluation (including skips and adaptations)
    context: dict  # Final context state

    @property
    def success_rate(self) -> float:
        """Fraction of executed actions that succeeded."""
        executed = [a for a in self.actions if hasattr(a, "success")]
        if not executed:
            return 0.0
        successes = sum(1 for a in executed if a.success)
        return successes / len(executed)

    @property
    def total_actions(self) -> int:
        """Total number of recorded actions."""
        return len(self.actions)

    @property
    def total_adaptations(self) -> int:
        """Number of times the agent adapted based on risk feedback."""
        return self.context.get("adapted_count", 0)

    @property
    def capsule_hash(self) -> str | None:
        """The capsule hash if a proof was generated."""
        return self.capsule.get("capsule_hash") if self.capsule else None


class AgentLoop:
    """The tandem execution loop.

    Drives an agent through: propose → gate → adapt → execute → record.

    Usage:
        agent = MyAgent()
        loop = AgentLoop(agent, output_dir=Path(".capseal/runs/my-task"))
        result = loop.run("refactor the auth module to use JWT")

        # result.capsule_hash — proof of everything that happened
        # result.actions — list of all recorded actions
        # result.risk_log — every gate evaluation including skips and adaptations

    The loop:
    1. agent.plan(task, context) → ProposedAction or None (done)
    2. runtime.gate(proposed) → RiskFeedback
    3. If pass: agent.execute(proposed) → ActionResult → runtime.record()
       If skip: agent.adapt(proposed, feedback) → retry from step 2
       If human_review: call callback, then pass or skip
    4. agent.on_result() → update context
    5. Loop back to step 1
    """

    def __init__(
        self,
        agent: CapSealAgent,
        output_dir: Path,
        posteriors_path: Path | None = None,
        max_retries: int = 3,
        prove: bool = True,
        human_review_callback: Callable[[ProposedAction, RiskFeedback], bool]
        | None = None,
    ):
        """Initialize the agent loop.

        Args:
            agent: The CapSealAgent implementation to drive
            output_dir: Where to store the run artifacts
            posteriors_path: Path to learned model (defaults to .capseal/models/beta_posteriors.npz)
            max_retries: Max adapt attempts per action before giving up
            prove: Whether to generate cryptographic proof on completion
            human_review_callback: Optional function called for human_review decisions.
                                   Returns True to approve, False to reject.
        """
        self.agent = agent
        self.max_retries = max_retries
        self.prove = prove
        self.human_review_callback = human_review_callback

        # Determine posteriors path
        if posteriors_path is None:
            posteriors_path = Path(".capseal/models/beta_posteriors.npz")

        # Initialize runtime
        self.runtime = AgentRuntime(
            output_dir=output_dir,
            gate_posteriors=posteriors_path if posteriors_path.exists() else None,
        )

        # Tracking
        self.risk_log: list[dict] = []
        self.context: dict = {
            "actions_taken": [],
            "total_risk_score": 0.0,
            "skipped_count": 0,
            "adapted_count": 0,
            "task_history": [],
        }

    def run(self, task: str) -> LoopResult:
        """Run the full tandem loop for a task.

        Args:
            task: The high-level task to accomplish

        Returns:
            LoopResult with capsule, actions, risk_log, and context.
        """
        self.context["task"] = task

        with self.runtime:
            while True:
                # Step 1: Agent proposes
                proposed = self.agent.plan(task, self.context)
                if proposed is None:
                    self.agent.on_task_complete(task, self.context)
                    break

                # Step 2-3: Gate and potentially adapt
                action_recorded = self._gate_and_execute(proposed, task)

                if not action_recorded:
                    # Agent gave up after max_retries
                    self._record_skip(proposed, "max_retries_exceeded")

        return LoopResult(
            capsule=self.runtime.last_capsule,
            actions=self.runtime.actions,
            risk_log=self.risk_log,
            context=self.context,
        )

    def _gate_and_execute(self, proposed: ProposedAction, task: str) -> bool:
        """Gate → adapt loop for a single proposed action.

        Returns True if an action was executed and recorded.
        """
        import time

        for attempt in range(1, self.max_retries + 1):
            # Evaluate risk
            risk_raw = self.runtime.gate(
                diff_text=proposed.diff_text,
                findings=proposed.findings,
            )

            # Build feedback
            uncertainty = risk_raw.get("uncertainty")
            confidence = 1.0 - uncertainty if uncertainty is not None else None

            feedback = RiskFeedback(
                decision=risk_raw.get("decision", "pass"),
                risk_score=risk_raw.get("q"),
                confidence=confidence,
                features=risk_raw.get("features", {}),
                grid_idx=risk_raw.get("grid_idx"),
                suggestion=self._generate_suggestion(risk_raw, attempt),
                attempt_number=attempt,
                prior_attempts=[r for r in self.risk_log if r.get("task") == task],
            )

            # Log this evaluation
            self.risk_log.append(
                {
                    "task": task,
                    "action_type": proposed.action_type,
                    "description": proposed.description,
                    "attempt": attempt,
                    "decision": feedback.decision,
                    "risk_score": feedback.risk_score,
                    "confidence": feedback.confidence,
                    "features": feedback.features,
                    "suggestion": feedback.suggestion,
                }
            )

            if feedback.decision == "pass":
                # Execute the action
                start_time = time.time()
                result = self.agent.execute(proposed)
                if result.duration_ms == 0:
                    result.duration_ms = int((time.time() - start_time) * 1000)

                # Record in runtime
                self.runtime.record_simple(
                    action_type=proposed.action_type,
                    instruction=proposed.instruction,
                    inputs=proposed.inputs,
                    outputs=result.outputs,
                    success=result.success,
                    gate_score=feedback.risk_score,
                    gate_decision="pass",
                    duration_ms=result.duration_ms,
                )

                # Update context
                self.context["actions_taken"].append(
                    {
                        "type": proposed.action_type,
                        "description": proposed.description,
                        "success": result.success,
                        "risk_score": feedback.risk_score,
                    }
                )
                if feedback.risk_score is not None:
                    self.context["total_risk_score"] += feedback.risk_score

                self.agent.on_result(proposed, result, self.context)
                return True

            elif feedback.decision == "human_review":
                if self.human_review_callback:
                    approved = self.human_review_callback(proposed, feedback)
                    if approved:
                        # Human approved — execute
                        start_time = time.time()
                        result = self.agent.execute(proposed)
                        if result.duration_ms == 0:
                            result.duration_ms = int((time.time() - start_time) * 1000)

                        self.runtime.record_simple(
                            action_type=proposed.action_type,
                            instruction=proposed.instruction,
                            inputs=proposed.inputs,
                            outputs=result.outputs,
                            success=result.success,
                            gate_score=feedback.risk_score,
                            gate_decision="human_approved",
                            duration_ms=result.duration_ms,
                        )

                        self.context["actions_taken"].append(
                            {
                                "type": proposed.action_type,
                                "description": proposed.description,
                                "success": result.success,
                                "risk_score": feedback.risk_score,
                                "human_approved": True,
                            }
                        )

                        self.agent.on_result(proposed, result, self.context)
                        return True
                # No callback or human rejected — fall through to adapt

            # Skip or rejected human_review — ask agent to adapt
            self.context["skipped_count"] += 1
            proposed = self.agent.adapt(proposed, feedback, self.context)
            if proposed is None:
                return False  # Agent gave up
            self.context["adapted_count"] += 1

        return False  # Exhausted retries

    def _generate_suggestion(self, risk_raw: dict, attempt: int) -> str:
        """Generate plain-language guidance based on risk assessment."""
        decision = risk_raw.get("decision", "pass")
        q = risk_raw.get("q")
        features = risk_raw.get("features", {})

        if decision == "pass":
            return "Proceed — low risk."

        # Build specific guidance from features
        risk_factors = []
        if features.get("cyclomatic_complexity", 0) >= 2:
            risk_factors.append("high complexity — try decomposing into smaller changes")
        if features.get("files_touched", 0) >= 2:
            risk_factors.append("too many files — focus on one file at a time")
        if features.get("finding_severity", 0) >= 2:
            risk_factors.append("security-sensitive — add validation and tests")
        if features.get("test_coverage_delta", 0) <= 0:
            risk_factors.append("no test coverage — add tests first")
        if features.get("lines_changed", 0) >= 2:
            risk_factors.append("large change — break into incremental patches")

        if not risk_factors:
            risk_factors = ["uncertain region — try a different approach"]

        prefix = f"Risk score {q:.2f}. " if q is not None else ""
        if attempt > 1:
            prefix += f"Attempt {attempt}. "

        return prefix + " | ".join(risk_factors)

    def _record_skip(self, proposed: ProposedAction, reason: str) -> None:
        """Record a skipped action in the runtime for audit trail."""
        self.runtime.record_simple(
            action_type=proposed.action_type,
            instruction=proposed.instruction,
            inputs=proposed.inputs,
            outputs={"skipped": True, "reason": reason},
            success=False,
            gate_score=None,
            gate_decision="skipped",
        )


__all__ = [
    "ProposedAction",
    "RiskFeedback",
    "ActionResult",
    "CapSealAgent",
    "AgentLoop",
    "LoopResult",
]
