"""Patch feature extraction and committor scoring.

Maps code patches to a 5-dimensional feature space that corresponds to
the agent_bench grid structure. Features are discretized to 4 levels (0-3)
giving 1024 possible grid points.

Features:
1. lines_changed: Total lines added + removed
2. cyclomatic_complexity: Estimated from branching keywords
3. files_touched: Number of files in the diff
4. finding_severity: Max severity from Semgrep findings
5. test_coverage_delta: Whether tests are included in patch
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .scoring import lookup_posterior_at_idx


# Feature discretization thresholds
# Each feature maps to 4 levels: 0, 1, 2, 3

LINES_THRESHOLDS = [10, 50, 200]  # 0: <=10, 1: 11-50, 2: 51-200, 3: >200
COMPLEXITY_THRESHOLDS = [2, 5, 10]  # 0: <=2, 1: 3-5, 2: 6-10, 3: >10
FILES_THRESHOLDS = [1, 3, 8]  # 0: 1 file, 1: 2-3, 2: 4-8, 3: >8
SEVERITY_MAP = {"none": 0, "info": 0, "low": 1, "warning": 1, "medium": 2, "high": 3, "error": 3}
TEST_THRESHOLDS = [0, 1, 5]  # 0: no tests, 1: 1 test change, 2: 2-5, 3: >5

# Decision thresholds
SKIP_THRESHOLD = 0.3  # Skip if q >= 0.3 (high failure probability)
HUMAN_REVIEW_UNCERTAINTY = 0.15  # Flag for review if uncertainty > 0.15


def extract_patch_features(diff_text: str, findings: List[Dict]) -> Dict[str, Any]:
    """Extract raw features from a diff and Semgrep findings.

    Args:
        diff_text: Unified diff text.
        findings: List of Semgrep finding dicts with 'severity' key.

    Returns:
        Dict with raw feature values:
        - lines_changed: int
        - cyclomatic_complexity: int
        - files_touched: int
        - finding_severity: int (0-3)
        - test_coverage_delta: int
    """
    # Parse diff for line counts and files
    lines_added = 0
    lines_removed = 0
    files_touched = set()
    test_files_changed = 0
    complexity_keywords = 0

    # Branching keywords for complexity estimation
    branch_patterns = [
        r'\bif\b', r'\belif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
        r'\btry\b', r'\bexcept\b', r'\bwith\b', r'\band\b', r'\bor\b',
        r'\bcase\b', r'\bmatch\b',
    ]

    current_file = None
    for line in diff_text.split('\n'):
        # Parse file header (skip 'diff --git' lines which have both paths on one line)
        if line.startswith('--- ') or line.startswith('+++ '):
            match = re.search(r'[ab]/(.+)$', line)
            if match:
                filepath = match.group(1)
                current_file = filepath
                files_touched.add(filepath)

                # Check if it's a test file
                if _is_test_file(filepath):
                    test_files_changed += 1

        # Count added/removed lines (skip headers)
        elif line.startswith('+') and not line.startswith('+++'):
            lines_added += 1
            # Count complexity in added lines
            for pattern in branch_patterns:
                complexity_keywords += len(re.findall(pattern, line))

        elif line.startswith('-') and not line.startswith('---'):
            lines_removed += 1

    # Get max severity from findings
    max_severity = 0
    for finding in findings:
        sev = finding.get('severity', 'info').lower()
        max_severity = max(max_severity, SEVERITY_MAP.get(sev, 0))

    return {
        'lines_changed': lines_added + lines_removed,
        'cyclomatic_complexity': complexity_keywords,
        'files_touched': len(files_touched),
        'finding_severity': max_severity,
        'test_coverage_delta': test_files_changed,
    }


def _is_test_file(filepath: str) -> bool:
    """Check if a file path looks like a test file."""
    name = Path(filepath).name.lower()
    return (
        name.startswith('test_') or
        name.endswith('_test.py') or
        name.endswith('_spec.py') or
        '/tests/' in filepath or
        '/test/' in filepath or
        name == 'conftest.py'
    )


def discretize_features(raw: Dict[str, Any]) -> List[int]:
    """Discretize raw features to levels 0-3.

    Args:
        raw: Dict with raw feature values.

    Returns:
        List of 5 integers, each in [0, 3].
    """
    def discretize(value: int, thresholds: List[int]) -> int:
        for i, thresh in enumerate(thresholds):
            if value <= thresh:
                return i
        return len(thresholds)  # Max level

    levels = [
        discretize(raw['lines_changed'], LINES_THRESHOLDS),
        discretize(raw['cyclomatic_complexity'], COMPLEXITY_THRESHOLDS),
        discretize(raw['files_touched'], FILES_THRESHOLDS),
        min(raw['finding_severity'], 3),  # Already 0-3
        discretize(raw['test_coverage_delta'], TEST_THRESHOLDS),
    ]

    return levels


def features_to_grid_idx(levels: List[int]) -> int:
    """Convert discretized feature levels to grid index.

    Uses base-4 encoding: idx = sum(level_i * 4^i) for i in 0..4
    This gives 4^5 = 1024 possible grid points.

    Args:
        levels: List of 5 integers, each in [0, 3].

    Returns:
        Grid index in range [0, 1023].
    """
    if len(levels) != 5:
        raise ValueError(f"Expected 5 feature levels, got {len(levels)}")

    idx = 0
    for i, level in enumerate(levels):
        if not 0 <= level <= 3:
            raise ValueError(f"Feature level {i} out of range: {level}")
        idx += level * (4 ** i)

    return idx


def grid_idx_to_features(idx: int) -> List[int]:
    """Convert grid index back to feature levels.

    Args:
        idx: Grid index in range [0, 1023].

    Returns:
        List of 5 integers, each in [0, 3].
    """
    if not 0 <= idx <= 1023:
        raise ValueError(f"Grid index out of range: {idx}")

    levels = []
    remaining = idx
    for _ in range(5):
        levels.append(remaining % 4)
        remaining //= 4

    return levels


def score_patch(
    diff_text: str,
    findings: List[Dict],
    posteriors_path: Path,
    skip_threshold: float = SKIP_THRESHOLD,
    review_uncertainty: float = HUMAN_REVIEW_UNCERTAINTY,
) -> Dict[str, Any]:
    """End-to-end patch scoring using committor gate.

    Pipeline:
    1. Extract features from diff and findings
    2. Discretize to grid coordinates
    3. Look up q(x) = estimated p_fail from beta posteriors
    4. Make decision based on thresholds

    Decision logic:
    - If q >= skip_threshold: SKIP (too risky)
    - Elif uncertainty > review_uncertainty: HUMAN_REVIEW (uncertain)
    - Else: PASS (proceed with patch)

    Args:
        diff_text: Unified diff text.
        findings: List of Semgrep finding dicts.
        posteriors_path: Path to beta_posteriors.npz file.
        skip_threshold: q threshold for skipping (default 0.3).
        review_uncertainty: Uncertainty threshold for human review.

    Returns:
        Dict with:
        - q: float (estimated p_fail)
        - uncertainty: float (posterior std)
        - grid_idx: int
        - features: dict (raw and discretized)
        - decision: str ("pass", "skip", "human_review")
        - reason: str (explanation)
    """
    # Extract and discretize features
    raw_features = extract_patch_features(diff_text, findings)
    levels = discretize_features(raw_features)
    grid_idx = features_to_grid_idx(levels)

    # Load posteriors and look up
    try:
        data = np.load(posteriors_path, allow_pickle=True)
        alpha = data['alpha']
        beta = data['beta']
        posterior = lookup_posterior_at_idx(alpha, beta, grid_idx)
    except FileNotFoundError:
        # No posteriors available - use uninformative prior
        posterior = {
            'q': 0.5,
            'uncertainty': 0.5,
            'alpha': 1,
            'beta': 1,
            'valid': False,
        }

    q = posterior['q']
    uncertainty = posterior['uncertainty']

    # Make decision
    if q >= skip_threshold:
        decision = "skip"
        reason = f"High failure probability: q={q:.3f} >= {skip_threshold}"
    elif uncertainty > review_uncertainty:
        decision = "human_review"
        reason = f"High uncertainty: std={uncertainty:.3f} > {review_uncertainty}"
    else:
        decision = "pass"
        reason = f"Low risk: q={q:.3f}, uncertainty={uncertainty:.3f}"

    return {
        'q': q,
        'uncertainty': uncertainty,
        'grid_idx': grid_idx,
        'features': {
            'raw': raw_features,
            'levels': levels,
            'level_names': [
                'lines_changed',
                'cyclomatic_complexity',
                'files_touched',
                'finding_severity',
                'test_coverage_delta',
            ],
        },
        'posterior': posterior,
        'decision': decision,
        'reason': reason,
        'thresholds': {
            'skip': skip_threshold,
            'review_uncertainty': review_uncertainty,
        },
    }


def score_plan_item(
    plan_item: Dict[str, Any],
    posteriors_path: Path,
    aggregate_findings: List[Dict] = None,
) -> Dict[str, Any]:
    """Score a single plan item from the refactor pipeline.

    This is a convenience wrapper for score_patch that extracts
    the diff preview and findings from a plan item dict.

    Args:
        plan_item: Dict with 'diff_preview' or 'patch' and 'findings' keys.
        posteriors_path: Path to beta_posteriors.npz.
        aggregate_findings: Optional list of all findings to supplement item.

    Returns:
        Score result dict (same as score_patch output).
    """
    # Extract diff text from plan item
    diff_text = plan_item.get('diff_preview', '')
    if not diff_text:
        diff_text = plan_item.get('patch', '')
    if not diff_text:
        diff_text = plan_item.get('diff', '')

    # Extract findings for this item
    findings = plan_item.get('findings', [])

    # Supplement with aggregate findings for the same file if available
    if aggregate_findings:
        file_path = plan_item.get('file_path', '')
        for f in aggregate_findings:
            if f.get('file_path') == file_path:
                if f not in findings:
                    findings.append(f)

    return score_patch(diff_text, findings, posteriors_path)
