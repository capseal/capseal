"""Shared utilities for capseal shell and bench integration."""

from .scoring import (
    compute_acquisition_score,
    select_targets,
    compute_tube_metrics,
)
from .features import (
    extract_patch_features,
    discretize_features,
    features_to_grid_idx,
    score_patch,
)
from .receipts import (
    hash_file,
    compute_trace_spec_hash,
    compute_statement_hash,
    build_round_receipt,
    build_run_receipt,
    verify_round_receipt,
    verify_run_receipt,
    collect_round_dirs,
)

__all__ = [
    "compute_acquisition_score",
    "select_targets",
    "compute_tube_metrics",
    "extract_patch_features",
    "discretize_features",
    "features_to_grid_idx",
    "score_patch",
    "hash_file",
    "compute_trace_spec_hash",
    "compute_statement_hash",
    "build_round_receipt",
    "build_run_receipt",
    "verify_round_receipt",
    "verify_run_receipt",
    "collect_round_dirs",
]
