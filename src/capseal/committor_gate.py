"""Committor Gate Executor for CapSeal v0.3.

The committor gate uses learned failure probabilities from the agent bench
to filter risky patches before they reach the patching step. This creates
a feedback loop: patches that historically failed get scored higher risk.

Integration:
    plan → committor.gate → patches → verify

The gate:
1. Extracts features from each plan item's diff preview
2. Maps features to a grid index (1024 possible points)
3. Looks up q(x) = estimated p_fail from beta posteriors
4. Filters/flags items based on q and uncertainty thresholds

Decisions:
- q >= 0.3: SKIP (high failure probability, don't attempt patch)
- uncertainty > 0.15: HUMAN_REVIEW (flagged but passed through)
- Otherwise: PASS (proceed with patch)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from capseal.risk_engine import (
    COMMITTOR_REVIEW_UNCERTAINTY,
    COMMITTOR_SKIP_THRESHOLD,
    committor_decision,
    posterior_from_grid_cell,
)
from capseal.shared.features import score_plan_item

from .workflow_engine import (
    NodeExecutor,
    NodeSpec,
    NodeResult,
    AgentPacket,
    sha256_json,
    sha256_file,
)


# Gate result schema version
GATE_RESULT_SCHEMA = "committor_gate_v1"


@dataclass
class GateDecision:
    """Decision for a single plan item."""
    item_id: str
    grid_idx: int
    q: float
    uncertainty: float
    decision: str  # "pass", "skip", "human_review"
    reason: str
    features: Dict[str, Any] = field(default_factory=dict)


def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


class CommittorGateExecutor(NodeExecutor):
    """Execute committor gate filtering on plan items.

    This node:
    1. Reads plan items from the plan node's output
    2. Scores each item using beta posteriors
    3. Filters out high-risk items (decision='skip')
    4. Flags uncertain items for human review
    5. Outputs filtered plan for the patches step
    """

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        """Execute the committor gate.

        Args:
            spec: Node specification with params:
                - posteriors_path: Path to beta_posteriors.npz (optional)
                - skip_threshold: q threshold for skipping (default 0.3)
                - review_uncertainty: Uncertainty threshold (default 0.15)
                - filter_skips: Whether to actually filter skips (default True)
            context: Results from dependency nodes (expects 'plan' key)

        Returns:
            NodeResult with filtered plan and gate metadata.
        """
        start_time = time.time()

        # Get parameters
        posteriors_path = spec.params.get('posteriors_path', '')
        skip_threshold = spec.params.get('skip_threshold', COMMITTOR_SKIP_THRESHOLD)
        review_uncertainty = spec.params.get('review_uncertainty', COMMITTOR_REVIEW_UNCERTAINTY)
        filter_skips = spec.params.get('filter_skips', True)

        # Resolve posteriors path
        if posteriors_path:
            posteriors = Path(posteriors_path)
        else:
            # Default: look in run directory or project directory
            posteriors = self._find_posteriors_path()

        # Get plan from dependency
        plan_result = context.get('plan') or context.get('refactor.plan')
        if not plan_result or not plan_result.packet:
            return NodeResult(
                node_id=spec.id,
                success=False,
                error="Missing plan dependency - gate requires plan output",
            )

        # Load plan items from plan output
        plan_path = self.run_dir / plan_result.packet.output_path
        if not plan_path.exists():
            return NodeResult(
                node_id=spec.id,
                success=False,
                error=f"Plan output not found: {plan_path}",
            )

        with open(plan_path) as f:
            plan_data = json.load(f)

        plan_items = plan_data.get('items', [])
        if not plan_items:
            plan_items = plan_data.get('plan_items', [])
        if not plan_items:
            plan_items = plan_data.get('patches', [])

        # Get aggregate findings if available
        aggregate_findings = plan_data.get('findings', [])

        # Score each plan item
        decisions: List[GateDecision] = []
        filtered_items: List[Dict] = []
        review_items: List[Dict] = []
        skipped_items: List[Dict] = []

        for i, item in enumerate(plan_items):
            item_id = item.get('id', f'item_{i}')

            # Score the item
            score_result = score_plan_item(
                item,
                posteriors,
                aggregate_findings,
            )

            decision = GateDecision(
                item_id=item_id,
                grid_idx=score_result['grid_idx'],
                q=score_result['q'],
                uncertainty=score_result['uncertainty'],
                decision=score_result['decision'],
                reason=score_result['reason'],
                features=score_result['features'],
            )
            decisions.append(decision)

            # Route item based on decision
            if decision.decision == 'skip' and filter_skips:
                skipped_items.append({
                    **item,
                    'gate_decision': decision.decision,
                    'gate_reason': decision.reason,
                    'gate_q': decision.q,
                })
            elif decision.decision == 'human_review':
                review_items.append({
                    **item,
                    'gate_decision': decision.decision,
                    'gate_reason': decision.reason,
                    'gate_q': decision.q,
                    'gate_uncertainty': decision.uncertainty,
                })
                # Still pass through, just flagged
                filtered_items.append(item)
            else:
                filtered_items.append(item)

        # Build gate result
        gate_result = {
            'schema': GATE_RESULT_SCHEMA,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'posteriors_path': str(posteriors),
            'posteriors_exists': posteriors.exists() if posteriors else False,
            'thresholds': {
                'skip': skip_threshold,
                'review_uncertainty': review_uncertainty,
            },
            'summary': {
                'total_items': len(plan_items),
                'passed': len(filtered_items) - len(review_items),
                'human_review': len(review_items),
                'skipped': len(skipped_items),
            },
            'decisions': [
                {
                    'item_id': d.item_id,
                    'grid_idx': d.grid_idx,
                    'q': d.q,
                    'uncertainty': d.uncertainty,
                    'decision': d.decision,
                    'reason': d.reason,
                    'features': d.features,
                }
                for d in decisions
            ],
            'skipped_items': skipped_items,
            'review_items': [
                {'item_id': item.get('id', f'item_{i}'), 'reason': item.get('gate_reason', '')}
                for i, item in enumerate(review_items)
            ],
        }

        # Write gate result
        gate_dir = self.run_dir / 'gate'
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_result_path = gate_dir / 'gate_result.json'
        gate_result_path.write_text(json.dumps(gate_result, indent=2))

        # Write filtered plan
        filtered_plan = {
            **plan_data,
            'items': filtered_items,
            'gate_applied': True,
            'gate_summary': gate_result['summary'],
        }
        filtered_plan_path = gate_dir / 'filtered_plan.json'
        filtered_plan_path.write_text(json.dumps(filtered_plan, indent=2))

        # Compute hashes
        gate_result_hash = sha256_file(gate_result_path)
        filtered_plan_hash = sha256_file(filtered_plan_path)

        # Build input manifest
        input_manifest = {
            'plan_hash': plan_result.packet.output_hash,
            'posteriors_path': str(posteriors),
            'skip_threshold': skip_threshold,
            'review_uncertainty': review_uncertainty,
        }
        input_hash = sha256_json(input_manifest)

        duration_ms = int((time.time() - start_time) * 1000)

        # Create packet
        packet = AgentPacket(
            node_id=spec.id,
            task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_type='committor_gate',
            input_hash=input_hash,
            input_manifest=input_manifest,
            executor_id='committor_gate_v1',
            policy_hash=sha256_json({
                'skip_threshold': skip_threshold,
                'review_uncertainty': review_uncertainty,
            }),
            output_hash=filtered_plan_hash,
            output_path='gate/filtered_plan.json',
            artifacts=[
                {'path': 'gate/gate_result.json', 'hash': gate_result_hash, 'type': 'gate_result'},
                {'path': 'gate/filtered_plan.json', 'hash': filtered_plan_hash, 'type': 'filtered_plan'},
            ],
            evidence_refs=[
                {'plan_hash': plan_result.packet.output_hash},
                {'posteriors_path': str(posteriors)},
            ],
            timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
            duration_ms=duration_ms,
            determinism='deterministic',
        )

        # Log summary
        summary = gate_result['summary']
        print(f"[CommittorGate] {summary['total_items']} items: "
              f"{summary['passed']} pass, {summary['human_review']} review, "
              f"{summary['skipped']} skip")

        return NodeResult(
            node_id=spec.id,
            success=True,
            packet=packet,
        )

    def _find_posteriors_path(self) -> Path:
        """Find beta_posteriors.npz in common locations."""
        # Check run directory
        candidates = [
            self.run_dir / 'beta_posteriors.npz',
            self.run_dir.parent / 'beta_posteriors.npz',
            self.project_dir / 'beta_posteriors.npz',
            self.project_dir / '.capseal' / 'beta_posteriors.npz',
            Path.home() / '.capseal' / 'beta_posteriors.npz',
        ]

        for path in candidates:
            if path.exists():
                return path

        # Return default (may not exist)
        return self.project_dir / '.capseal' / 'beta_posteriors.npz'


def verify_gate_decision(
    gate_result_path: Path,
    posteriors_path: Path,
) -> Dict[str, Any]:
    """Verify that gate decisions can be reproduced from posteriors.

    This is the checker function that validates committor gate claims.
    It re-derives q values from stored posteriors and verifies decisions match.

    Args:
        gate_result_path: Path to gate_result.json
        posteriors_path: Path to beta_posteriors.npz

    Returns:
        Verification result dict with 'valid' bool and 'mismatches' list.
    """
    with open(gate_result_path) as f:
        gate_result = json.load(f)

    try:
        data = np.load(posteriors_path, allow_pickle=True)
        alpha = data['alpha']
        beta = data['beta']
    except FileNotFoundError:
        return {
            'valid': False,
            'error': f'Posteriors not found: {posteriors_path}',
            'mismatches': [],
        }

    thresholds = gate_result.get('thresholds', {})
    skip_threshold = thresholds.get('skip', COMMITTOR_SKIP_THRESHOLD)
    review_uncertainty = thresholds.get('review_uncertainty', COMMITTOR_REVIEW_UNCERTAINTY)

    mismatches = []
    for decision in gate_result.get('decisions', []):
        grid_idx = decision['grid_idx']
        stored_q = decision['q']
        stored_decision = decision['decision']

        # Re-derive q from posteriors
        cell = posterior_from_grid_cell(grid_idx, alpha=alpha, beta=beta)
        computed_q = cell.p_fail
        uncertainty = cell.uncertainty
        expected_decision = committor_decision(
            computed_q,
            uncertainty,
            skip_threshold=skip_threshold,
            review_uncertainty=review_uncertainty,
        )

        # Check for mismatch
        if abs(stored_q - computed_q) > 1e-6:
            mismatches.append({
                'item_id': decision['item_id'],
                'type': 'q_mismatch',
                'stored': stored_q,
                'computed': computed_q,
            })

        if stored_decision != expected_decision:
            mismatches.append({
                'item_id': decision['item_id'],
                'type': 'decision_mismatch',
                'stored': stored_decision,
                'expected': expected_decision,
            })

    return {
        'valid': len(mismatches) == 0,
        'mismatches': mismatches,
        'decisions_checked': len(gate_result.get('decisions', [])),
    }


# Need numpy for verify function
import numpy as np
