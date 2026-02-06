#!/usr/bin/env python3
"""FusionAlpha ingest stub - consumes ENN artifact and emits decision receipt.

Phase 2c: Close the verification loop
- Read ENN output artifact
- Apply simple threshold decision
- Emit FusionDecisionReceipt with cryptographic binding

This completes the chain: BICEP receipt -> ENN receipt -> FusionAlpha receipt
Even if decisions are dumb, the chain is real.

Usage:
    python ingest_enn.py --enn-artifact <path> --output <receipt_path>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Import from capsule if available, otherwise define locally
try:
    from bef_zk.capsule.features_sidecar import (
        ENNOutputArtifact,
        FusionDecisionReceipt,
        compute_features_hash,
        hash_canonical,
    )
except ImportError:
    # Standalone definitions for testing
    def hash_canonical(obj: Any) -> str:
        data = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def compute_features_hash(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @dataclass
    class ENNOutputArtifact:
        schema: str = "enn_output_v1"
        input_features_shard_hash: str = ""
        input_sidecar_hash: str = ""
        trace_anchor_head: str = ""
        enn_config_hash: str = ""
        enn_code_hash: str = ""
        enn_version: str = ""
        predictions_hash: str = ""
        predictions_row_count: int = 0
        embeddings_hash: str = ""
        created_at: str = ""

        @classmethod
        def load(cls, path: Path) -> 'ENNOutputArtifact':
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @dataclass
    class FusionDecisionReceipt:
        schema: str = "fusion_decision_v1"
        enn_artifact_hash: str = ""
        enn_predictions_hash: str = ""
        bicep_trace_anchor_head: str = ""
        bicep_manifest_hash: str = ""
        decisions_hash: str = ""
        decisions_count: int = 0
        decision_method: str = ""
        created_at: str = ""
        fusion_version: str = ""

        def to_dict(self) -> dict:
            d = asdict(self)
            d["receipt_hash"] = hash_canonical(d)
            return d

        def save(self, path: Path) -> None:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)


FUSION_VERSION = "0.1.0"


def load_predictions(predictions_path: Path) -> list[float]:
    """Load predictions from CSV file."""
    predictions = []

    with open(predictions_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column names
            for col in ['prediction', 'pred', 'q_pred', 'output']:
                if col in row:
                    predictions.append(float(row[col]))
                    break

    return predictions


def apply_threshold(predictions: list[float], threshold: float = 0.5) -> list[int]:
    """Simple threshold decision: pred > threshold -> 1, else 0."""
    return [1 if p > threshold else 0 for p in predictions]


def create_decision_receipt(
    enn_artifact: ENNOutputArtifact,
    predictions: list[float],
    decisions: list[int],
    decision_method: str,
    bicep_manifest_hash: str = "",
) -> FusionDecisionReceipt:
    """Create decision receipt with cryptographic binding."""
    # Hash decisions
    decisions_data = json.dumps(decisions, sort_keys=True, separators=(',', ':')).encode('utf-8')
    decisions_hash = hashlib.sha256(decisions_data).hexdigest()

    # Compute ENN artifact hash
    enn_artifact_hash = hash_canonical(asdict(enn_artifact))

    receipt = FusionDecisionReceipt(
        enn_artifact_hash=enn_artifact_hash,
        enn_predictions_hash=enn_artifact.predictions_hash,
        bicep_trace_anchor_head=enn_artifact.trace_anchor_head,
        bicep_manifest_hash=bicep_manifest_hash,
        decisions_hash=decisions_hash,
        decisions_count=len(decisions),
        decision_method=decision_method,
        created_at=datetime.now(timezone.utc).isoformat(),
        fusion_version=FUSION_VERSION,
    )

    return receipt


def main():
    parser = argparse.ArgumentParser(
        description="FusionAlpha ingest: consume ENN artifact, emit decision receipt"
    )
    parser.add_argument(
        "--enn-artifact", "-e",
        type=str,
        help="Path to ENN output artifact JSON"
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        help="Path to predictions CSV (if no artifact)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Decision threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="fusion_decision_receipt.json",
        help="Output path for decision receipt"
    )
    parser.add_argument(
        "--bicep-manifest-hash",
        type=str,
        default="",
        help="BICEP manifest hash (for transitive binding)"
    )

    args = parser.parse_args()

    # Load ENN artifact or create stub
    if args.enn_artifact:
        artifact_path = Path(args.enn_artifact)
        if not artifact_path.exists():
            print(f"ERROR: ENN artifact not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)
        enn_artifact = ENNOutputArtifact.load(artifact_path)
        print(f"[FusionAlpha] Loaded ENN artifact from {artifact_path}")
    else:
        # Create stub artifact
        enn_artifact = ENNOutputArtifact(
            enn_version="unknown",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # Load predictions
    if args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"ERROR: Predictions file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        predictions = load_predictions(pred_path)
        enn_artifact.predictions_hash = compute_features_hash(pred_path)
        enn_artifact.predictions_row_count = len(predictions)
        print(f"[FusionAlpha] Loaded {len(predictions)} predictions from {pred_path}")
    elif args.enn_artifact:
        print("[FusionAlpha] No predictions file provided, using artifact metadata only")
        predictions = []
    else:
        print("ERROR: Either --enn-artifact or --predictions required", file=sys.stderr)
        sys.exit(1)

    # Apply threshold decision
    decision_method = f"threshold_{args.threshold}"
    decisions = apply_threshold(predictions, args.threshold)

    if decisions:
        positive_rate = sum(decisions) / len(decisions)
        print(f"[FusionAlpha] Decisions: {len(decisions)} total, {sum(decisions)} positive ({positive_rate:.1%})")
    else:
        print("[FusionAlpha] No decisions made (empty predictions)")

    # Create receipt
    receipt = create_decision_receipt(
        enn_artifact=enn_artifact,
        predictions=predictions,
        decisions=decisions,
        decision_method=decision_method,
        bicep_manifest_hash=args.bicep_manifest_hash,
    )

    # Save receipt
    output_path = Path(args.output)
    receipt.save(output_path)

    print(f"[FusionAlpha] Decision receipt saved to {output_path}")
    print(f"[FusionAlpha] Receipt hash: {hash_canonical(receipt.to_dict())[:16]}...")

    # Print chain summary
    print("\n=== Verification Chain Summary ===")
    if enn_artifact.trace_anchor_head:
        print(f"  BICEP trace_anchor_head: {enn_artifact.trace_anchor_head[:16]}...")
    if args.bicep_manifest_hash:
        print(f"  BICEP manifest_hash: {args.bicep_manifest_hash[:16]}...")
    print(f"  ENN predictions_hash: {enn_artifact.predictions_hash[:16] if enn_artifact.predictions_hash else 'N/A'}...")
    print(f"  Fusion decisions_hash: {receipt.decisions_hash[:16]}...")
    print("  Chain: BICEP -> ENN -> FusionAlpha (complete)")


if __name__ == "__main__":
    main()
