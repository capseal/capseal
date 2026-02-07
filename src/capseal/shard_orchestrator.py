"""
Shard Orchestrator - Multi-agent DAG for Parallel Patch Generation

This module implements the "shard workers + deterministic reducer" pattern:

1. Plan → Shards: Split work by file (or other strategy)
2. Workers: Each shard runs independently, emits a ShardReceipt
3. Reducer: Deterministically merge shard outputs, detect conflicts
4. Verify: Run verification on merged output

The key invariant: same inputs → identical output, regardless of
worker completion order or timing.

```
plan → shard(S1) →
       shard(S2) → reduce → verify → claims → policy → rollup
       shard(S3) →
```
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────
# Shard and Receipt Types
# ─────────────────────────────────────────────────────────────────

@dataclass
class Shard:
    """A unit of work assigned to a worker."""
    shard_id: str
    file_paths: list[str]  # Files this shard is responsible for
    plan_items: list[dict]  # Plan items for these files
    parent_receipt_id: str  # Receipt ID of the plan that spawned this

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "file_paths": self.file_paths,
            "plan_items": self.plan_items,
            "parent_receipt_id": self.parent_receipt_id,
        }

    def input_hash(self) -> str:
        """Hash of shard inputs for determinism."""
        content = json.dumps({
            "shard_id": self.shard_id,
            "file_paths": sorted(self.file_paths),
            "plan_items": self.plan_items,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ShardResult:
    """Result of processing a single shard."""
    shard_id: str
    status: str  # "success", "partial", "failed"
    patches: list[dict]  # Patches produced
    no_change_proofs: list[dict]  # NO_CHANGE proofs
    errors: list[str]  # Any errors encountered

    # Hashes for verification
    input_hash: str
    output_hash: str

    # Timing
    started_at: str
    completed_at: str
    duration_ms: int

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "status": self.status,
            "patches": self.patches,
            "no_change_proofs": self.no_change_proofs,
            "errors": self.errors,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardResult":
        return cls(**d)


@dataclass
class ShardReceipt:
    """Receipt for a single shard's work."""
    receipt_id: str
    shard_id: str
    parent_receipt_id: str

    # The result
    result: ShardResult

    # Chain info
    input_hash: str
    output_hash: str
    receipt_hash: str

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "shard_id": self.shard_id,
            "parent_receipt_id": self.parent_receipt_id,
            "result": self.result.to_dict(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardReceipt":
        return cls(
            receipt_id=d["receipt_id"],
            shard_id=d["shard_id"],
            parent_receipt_id=d["parent_receipt_id"],
            result=ShardResult.from_dict(d["result"]),
            input_hash=d["input_hash"],
            output_hash=d["output_hash"],
            receipt_hash=d["receipt_hash"],
        )


# ─────────────────────────────────────────────────────────────────
# Sharding Strategies
# ─────────────────────────────────────────────────────────────────

def shard_by_file(plan_items: list[dict], parent_receipt_id: str) -> list[Shard]:
    """
    Shard plan items by file - each file gets its own shard.

    This is the simplest and most parallelizable strategy.
    """
    # Group items by file
    by_file: dict[str, list[dict]] = {}
    for item in plan_items:
        file_path = item.get("file_path", "unknown")
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(item)

    # Create shards
    shards = []
    for i, (file_path, items) in enumerate(sorted(by_file.items())):
        shard_id = f"S{i:03d}-{hashlib.sha256(file_path.encode()).hexdigest()[:8]}"
        shards.append(Shard(
            shard_id=shard_id,
            file_paths=[file_path],
            plan_items=items,
            parent_receipt_id=parent_receipt_id,
        ))

    return shards


def shard_by_count(plan_items: list[dict], num_shards: int, parent_receipt_id: str) -> list[Shard]:
    """
    Shard plan items into N roughly equal shards.

    Items are assigned round-robin after sorting by file path for determinism.
    """
    # Sort items by file path for determinism
    sorted_items = sorted(plan_items, key=lambda x: x.get("file_path", ""))

    # Distribute round-robin
    shard_items: list[list[dict]] = [[] for _ in range(num_shards)]
    for i, item in enumerate(sorted_items):
        shard_items[i % num_shards].append(item)

    # Create shards
    shards = []
    for i, items in enumerate(shard_items):
        if not items:
            continue

        file_paths = sorted(set(item.get("file_path", "") for item in items))
        shard_id = f"S{i:03d}-{hashlib.sha256(str(file_paths).encode()).hexdigest()[:8]}"
        shards.append(Shard(
            shard_id=shard_id,
            file_paths=file_paths,
            plan_items=items,
            parent_receipt_id=parent_receipt_id,
        ))

    return shards


# ─────────────────────────────────────────────────────────────────
# Shard Worker
# ─────────────────────────────────────────────────────────────────

class ShardWorker:
    """
    Worker that processes a single shard.

    Takes a shard and a patch generator function, produces a ShardResult.
    """

    def __init__(
        self,
        shard: Shard,
        patch_generator: Callable[[list[dict], Path], tuple[list[dict], list[dict]]],
        project_dir: Path,
    ):
        self.shard = shard
        self.patch_generator = patch_generator
        self.project_dir = project_dir

    def run(self) -> ShardResult:
        """Execute the shard and return result."""
        started_at = datetime.utcnow()

        try:
            # Generate patches for this shard's items
            patches, no_change_proofs = self.patch_generator(
                self.shard.plan_items,
                self.project_dir,
            )

            # Compute output hash
            output_content = json.dumps({
                "patches": patches,
                "no_change_proofs": no_change_proofs,
            }, sort_keys=True)
            output_hash = hashlib.sha256(output_content.encode()).hexdigest()

            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ShardResult(
                shard_id=self.shard.shard_id,
                status="success" if patches or no_change_proofs else "partial",
                patches=patches,
                no_change_proofs=no_change_proofs,
                errors=[],
                input_hash=self.shard.input_hash(),
                output_hash=output_hash,
                started_at=started_at.isoformat() + "Z",
                completed_at=completed_at.isoformat() + "Z",
                duration_ms=duration_ms,
            )

        except Exception as e:
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ShardResult(
                shard_id=self.shard.shard_id,
                status="failed",
                patches=[],
                no_change_proofs=[],
                errors=[str(e)],
                input_hash=self.shard.input_hash(),
                output_hash="",
                started_at=started_at.isoformat() + "Z",
                completed_at=completed_at.isoformat() + "Z",
                duration_ms=duration_ms,
            )

    def run_with_receipt(self) -> ShardReceipt:
        """Execute and produce a receipt."""
        result = self.run()

        # Generate receipt hash
        receipt_content = json.dumps({
            "shard_id": self.shard.shard_id,
            "input_hash": result.input_hash,
            "output_hash": result.output_hash,
        }, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_content.encode()).hexdigest()

        receipt_id = f"shard-{self.shard.shard_id}-{receipt_hash[:8]}"

        return ShardReceipt(
            receipt_id=receipt_id,
            shard_id=self.shard.shard_id,
            parent_receipt_id=self.shard.parent_receipt_id,
            result=result,
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            receipt_hash=receipt_hash,
        )


# ─────────────────────────────────────────────────────────────────
# Conflict Detection
# ─────────────────────────────────────────────────────────────────

@dataclass
class Conflict:
    """A conflict between two patches."""
    file_path: str
    shard_a: str
    shard_b: str
    line_range_a: tuple[int, int]
    line_range_b: tuple[int, int]
    description: str

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "shard_a": self.shard_a,
            "shard_b": self.shard_b,
            "line_range_a": list(self.line_range_a),
            "line_range_b": list(self.line_range_b),
            "description": self.description,
        }


@dataclass
class ConflictBundle:
    """
    All information needed to resolve a conflict.

    This is a first-class artifact that can be:
    - Passed to an LLM resolver
    - Manually resolved by a human
    - Stored for audit/replay
    """
    conflict_id: str
    conflict: Conflict
    base_file_content: str
    base_file_hash: str
    patch_a: dict  # Full patch from shard A
    patch_b: dict  # Full patch from shard B
    task_a_description: str  # What shard A was trying to do
    task_b_description: str  # What shard B was trying to do

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "conflict": self.conflict.to_dict(),
            "base_file_hash": self.base_file_hash,
            "patch_a": self.patch_a,
            "patch_b": self.patch_b,
            "task_a_description": self.task_a_description,
            "task_b_description": self.task_b_description,
        }


@dataclass
class ConflictResolution:
    """
    Result of resolving a conflict.

    The resolution must:
    1. Apply cleanly to the base file
    2. Preserve the intent of both original tasks
    3. Be verifiable (via witness/claim)
    """
    conflict_id: str
    status: str  # "resolved", "manual_required", "incompatible"
    merged_patch: Optional[dict]  # The merged patch if resolved

    # Verification
    merged_applies_cleanly: bool
    task_a_preserved: bool  # Did we preserve shard A's intent?
    task_b_preserved: bool  # Did we preserve shard B's intent?

    # Audit trail
    resolution_method: str  # "auto_merge", "llm", "manual"
    resolution_rationale: str

    # Hashes for verification
    input_hash: str  # H(conflict_bundle)
    output_hash: str  # H(merged_patch)

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "status": self.status,
            "merged_patch": self.merged_patch,
            "merged_applies_cleanly": self.merged_applies_cleanly,
            "task_a_preserved": self.task_a_preserved,
            "task_b_preserved": self.task_b_preserved,
            "resolution_method": self.resolution_method,
            "resolution_rationale": self.resolution_rationale,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }


@dataclass
class ConflictResolutionReceipt:
    """
    Auditable receipt for a conflict resolution.

    This makes conflict resolution a first-class DAG vertex with its own receipt.
    """
    receipt_id: str
    conflict_id: str
    parent_reduce_receipt_id: str
    resolution: ConflictResolution
    receipt_hash: str

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "conflict_id": self.conflict_id,
            "parent_reduce_receipt_id": self.parent_reduce_receipt_id,
            "resolution": self.resolution.to_dict(),
            "receipt_hash": self.receipt_hash,
        }


class ConflictResolver:
    """
    Resolve conflicts between shard patches.

    This is a DAG vertex that:
    1. Takes a ConflictBundle as input
    2. Attempts resolution (auto-merge, LLM, or mark as manual)
    3. Produces a ConflictResolutionReceipt
    """

    def __init__(self, resolver_fn: Optional[Callable] = None):
        """
        Args:
            resolver_fn: Optional custom resolver function.
                         Signature: (ConflictBundle) -> ConflictResolution
        """
        self.resolver_fn = resolver_fn or self._default_resolver

    def _default_resolver(self, bundle: ConflictBundle) -> ConflictResolution:
        """
        Default conflict resolution strategy.

        Currently: mark as manual_required (conservative).
        Future: try auto-merge, then LLM.
        """
        input_hash = hashlib.sha256(
            json.dumps(bundle.to_dict(), sort_keys=True).encode()
        ).hexdigest()

        # Check if ranges are actually disjoint (false positive)
        range_a = bundle.conflict.line_range_a
        range_b = bundle.conflict.line_range_b

        # If ranges are adjacent but not overlapping, we can auto-merge
        if range_a[1] < range_b[0] or range_b[1] < range_a[0]:
            # No actual overlap - safe to merge sequentially
            merged_patch = {
                "file_path": bundle.conflict.file_path,
                "patches": [bundle.patch_a, bundle.patch_b],
                "merge_order": "sequential",
            }
            return ConflictResolution(
                conflict_id=bundle.conflict_id,
                status="resolved",
                merged_patch=merged_patch,
                merged_applies_cleanly=True,
                task_a_preserved=True,
                task_b_preserved=True,
                resolution_method="auto_merge",
                resolution_rationale="Ranges are adjacent, not overlapping",
                input_hash=input_hash,
                output_hash=hashlib.sha256(
                    json.dumps(merged_patch, sort_keys=True).encode()
                ).hexdigest(),
            )

        # True overlap - needs manual resolution
        return ConflictResolution(
            conflict_id=bundle.conflict_id,
            status="manual_required",
            merged_patch=None,
            merged_applies_cleanly=False,
            task_a_preserved=False,
            task_b_preserved=False,
            resolution_method="none",
            resolution_rationale=f"True overlap at lines {range_a} and {range_b} - requires manual resolution",
            input_hash=input_hash,
            output_hash="",
        )

    def resolve(
        self,
        bundle: ConflictBundle,
        parent_reduce_receipt_id: str,
    ) -> ConflictResolutionReceipt:
        """
        Resolve a conflict and produce a receipt.
        """
        resolution = self.resolver_fn(bundle)

        # Compute receipt hash
        receipt_content = json.dumps({
            "conflict_id": bundle.conflict_id,
            "parent": parent_reduce_receipt_id,
            "resolution": resolution.to_dict(),
        }, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_content.encode()).hexdigest()[:16]

        return ConflictResolutionReceipt(
            receipt_id=f"conflict-{bundle.conflict_id}-{receipt_hash}",
            conflict_id=bundle.conflict_id,
            parent_reduce_receipt_id=parent_reduce_receipt_id,
            resolution=resolution,
            receipt_hash=receipt_hash,
        )

    def resolve_all(
        self,
        conflicts: list[Conflict],
        shard_results: dict[str, ShardResult],
        base_files: dict[str, str],  # file_path -> content
        parent_reduce_receipt_id: str,
    ) -> list[ConflictResolutionReceipt]:
        """
        Resolve all conflicts from a reduce operation.

        Returns receipts for each resolution attempt.
        """
        receipts = []

        for i, conflict in enumerate(conflicts):
            # Build the conflict bundle
            shard_a_result = shard_results.get(conflict.shard_a)
            shard_b_result = shard_results.get(conflict.shard_b)

            # Find the specific patches
            patch_a = next(
                (p for p in (shard_a_result.patches if shard_a_result else [])
                 if p.get("file_path") == conflict.file_path),
                {}
            )
            patch_b = next(
                (p for p in (shard_b_result.patches if shard_b_result else [])
                 if p.get("file_path") == conflict.file_path),
                {}
            )

            base_content = base_files.get(conflict.file_path, "")

            bundle = ConflictBundle(
                conflict_id=f"C{i:03d}-{conflict.shard_a}-{conflict.shard_b}",
                conflict=conflict,
                base_file_content=base_content,
                base_file_hash=hashlib.sha256(base_content.encode()).hexdigest(),
                patch_a=patch_a,
                patch_b=patch_b,
                task_a_description=patch_a.get("description", f"Task from {conflict.shard_a}"),
                task_b_description=patch_b.get("description", f"Task from {conflict.shard_b}"),
            )

            receipt = self.resolve(bundle, parent_reduce_receipt_id)
            receipts.append(receipt)

        return receipts


def detect_conflicts(shard_results: list[ShardResult]) -> list[Conflict]:
    """
    Detect conflicts between shard outputs.

    A conflict occurs when two shards modify overlapping line ranges
    in the same file.
    """
    conflicts = []

    # Collect all patch ranges by file
    patches_by_file: dict[str, list[tuple[str, dict]]] = {}  # file -> [(shard_id, patch)]

    for result in shard_results:
        for patch in result.patches:
            file_path = patch.get("file_path", "")
            if file_path not in patches_by_file:
                patches_by_file[file_path] = []
            patches_by_file[file_path].append((result.shard_id, patch))

    # Check for overlaps within each file
    for file_path, patches in patches_by_file.items():
        if len(patches) < 2:
            continue

        # Check each pair
        for i in range(len(patches)):
            shard_a, patch_a = patches[i]
            range_a = _get_patch_range(patch_a)

            for j in range(i + 1, len(patches)):
                shard_b, patch_b = patches[j]
                range_b = _get_patch_range(patch_b)

                if _ranges_overlap(range_a, range_b):
                    conflicts.append(Conflict(
                        file_path=file_path,
                        shard_a=shard_a,
                        shard_b=shard_b,
                        line_range_a=range_a,
                        line_range_b=range_b,
                        description=f"Overlapping edits: {shard_a} lines {range_a[0]}-{range_a[1]}, {shard_b} lines {range_b[0]}-{range_b[1]}",
                    ))

    return conflicts


def _get_patch_range(patch: dict) -> tuple[int, int]:
    """Extract line range from a patch."""
    start = patch.get("start_line", 0)
    end = patch.get("end_line", start)
    return (start, end)


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Check if two line ranges overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


# ─────────────────────────────────────────────────────────────────
# Deterministic Reducer
# ─────────────────────────────────────────────────────────────────

@dataclass
class ReduceResult:
    """Result of reducing multiple shard outputs."""
    status: str  # "success", "conflicts", "failed"
    merged_patches: list[dict]
    merged_no_change_proofs: list[dict]
    conflicts: list[Conflict]
    shard_order: list[str]  # Order in which shards were merged (deterministic)

    # Hashes
    input_hash: str  # Hash of all shard receipts
    output_hash: str  # Hash of merged output

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "merged_patches": self.merged_patches,
            "merged_no_change_proofs": self.merged_no_change_proofs,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "shard_order": self.shard_order,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }


@dataclass
class ReduceReceipt:
    """Receipt for the reduce operation."""
    receipt_id: str
    shard_receipts: list[str]  # Receipt IDs of input shards
    result: ReduceResult
    receipt_hash: str

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "shard_receipts": self.shard_receipts,
            "result": self.result.to_dict(),
            "receipt_hash": self.receipt_hash,
        }


class Reducer:
    """
    Deterministically merge shard outputs.

    The key invariant: same shard receipts → identical merged output,
    regardless of the order they completed or were passed in.
    """

    def reduce(self, shard_receipts: list[ShardReceipt]) -> ReduceResult:
        """
        Merge shard outputs deterministically.

        1. Sort shards by shard_id (deterministic order)
        2. Detect conflicts
        3. Merge patches in order
        4. Compute output hash
        """
        # Sort by shard_id for deterministic order
        sorted_receipts = sorted(shard_receipts, key=lambda r: r.shard_id)
        shard_order = [r.shard_id for r in sorted_receipts]

        # Compute input hash (hash of all shard receipt hashes in order)
        input_content = json.dumps({
            "shard_receipts": [r.receipt_hash for r in sorted_receipts]
        }, sort_keys=True)
        input_hash = hashlib.sha256(input_content.encode()).hexdigest()

        # Collect results
        all_results = [r.result for r in sorted_receipts]

        # Detect conflicts
        conflicts = detect_conflicts(all_results)

        # Merge patches in deterministic order
        merged_patches = []
        merged_no_change_proofs = []

        for result in all_results:
            # Sort patches by file path and line number for determinism
            sorted_patches = sorted(
                result.patches,
                key=lambda p: (p.get("file_path", ""), p.get("start_line", 0))
            )
            merged_patches.extend(sorted_patches)

            sorted_proofs = sorted(
                result.no_change_proofs,
                key=lambda p: p.get("file_path", "")
            )
            merged_no_change_proofs.extend(sorted_proofs)

        # Compute output hash
        output_content = json.dumps({
            "patches": merged_patches,
            "no_change_proofs": merged_no_change_proofs,
            "shard_order": shard_order,
        }, sort_keys=True)
        output_hash = hashlib.sha256(output_content.encode()).hexdigest()

        # Determine status
        if conflicts:
            status = "conflicts"
        elif any(r.status == "failed" for r in all_results):
            status = "failed"
        else:
            status = "success"

        return ReduceResult(
            status=status,
            merged_patches=merged_patches,
            merged_no_change_proofs=merged_no_change_proofs,
            conflicts=conflicts,
            shard_order=shard_order,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def reduce_with_receipt(self, shard_receipts: list[ShardReceipt], parent_receipt_id: str) -> ReduceReceipt:
        """Reduce and produce a receipt."""
        result = self.reduce(shard_receipts)

        # Generate receipt hash
        receipt_content = json.dumps({
            "input_hash": result.input_hash,
            "output_hash": result.output_hash,
            "shard_order": result.shard_order,
        }, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_content.encode()).hexdigest()

        receipt_id = f"reduce-{parent_receipt_id}-{receipt_hash[:8]}"

        return ReduceReceipt(
            receipt_id=receipt_id,
            shard_receipts=[r.receipt_id for r in shard_receipts],
            result=result,
            receipt_hash=receipt_hash,
        )


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

@dataclass
class OrchestrationResult:
    """Complete result of orchestrated shard execution."""
    shards: list[Shard]
    shard_receipts: list[ShardReceipt]
    reduce_receipt: ReduceReceipt

    # Aggregated stats
    total_shards: int
    successful_shards: int
    failed_shards: int
    total_patches: int
    total_no_change_proofs: int
    conflicts_detected: int
    total_duration_ms: int

    def to_dict(self) -> dict:
        return {
            "shards": [s.to_dict() for s in self.shards],
            "shard_receipts": [r.to_dict() for r in self.shard_receipts],
            "reduce_receipt": self.reduce_receipt.to_dict(),
            "stats": {
                "total_shards": self.total_shards,
                "successful_shards": self.successful_shards,
                "failed_shards": self.failed_shards,
                "total_patches": self.total_patches,
                "total_no_change_proofs": self.total_no_change_proofs,
                "conflicts_detected": self.conflicts_detected,
                "total_duration_ms": self.total_duration_ms,
            },
        }


class ShardOrchestrator:
    """
    Orchestrate parallel shard execution with deterministic reduction.

    Usage:
        orchestrator = ShardOrchestrator(
            plan_items=plan.items,
            project_dir=project_path,
            patch_generator=my_patch_fn,
            parent_receipt_id="plan-123",
        )
        result = orchestrator.run(max_workers=4, shard_strategy="by_file")
    """

    def __init__(
        self,
        plan_items: list[dict],
        project_dir: Path,
        patch_generator: Callable[[list[dict], Path], tuple[list[dict], list[dict]]],
        parent_receipt_id: str,
    ):
        self.plan_items = plan_items
        self.project_dir = project_dir
        self.patch_generator = patch_generator
        self.parent_receipt_id = parent_receipt_id

    def run(
        self,
        max_workers: int = 4,
        shard_strategy: str = "by_file",
        num_shards: int | None = None,
    ) -> OrchestrationResult:
        """
        Execute shards in parallel and reduce results.

        Args:
            max_workers: Maximum concurrent workers
            shard_strategy: "by_file" or "by_count"
            num_shards: Number of shards (only for "by_count" strategy)
        """
        started_at = datetime.utcnow()

        # Create shards
        if shard_strategy == "by_file":
            shards = shard_by_file(self.plan_items, self.parent_receipt_id)
        elif shard_strategy == "by_count":
            n = num_shards or max_workers
            shards = shard_by_count(self.plan_items, n, self.parent_receipt_id)
        else:
            raise ValueError(f"Unknown shard strategy: {shard_strategy}")

        # Execute shards in parallel
        shard_receipts: list[ShardReceipt] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all workers
            futures = {}
            for shard in shards:
                worker = ShardWorker(shard, self.patch_generator, self.project_dir)
                future = executor.submit(worker.run_with_receipt)
                futures[future] = shard.shard_id

            # Collect results as they complete
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    receipt = future.result()
                    shard_receipts.append(receipt)
                except Exception as e:
                    # Create failed receipt
                    shard = next(s for s in shards if s.shard_id == shard_id)
                    failed_result = ShardResult(
                        shard_id=shard_id,
                        status="failed",
                        patches=[],
                        no_change_proofs=[],
                        errors=[str(e)],
                        input_hash=shard.input_hash(),
                        output_hash="",
                        started_at=datetime.utcnow().isoformat() + "Z",
                        completed_at=datetime.utcnow().isoformat() + "Z",
                        duration_ms=0,
                    )
                    receipt_hash = hashlib.sha256(f"failed-{shard_id}".encode()).hexdigest()
                    shard_receipts.append(ShardReceipt(
                        receipt_id=f"shard-{shard_id}-failed",
                        shard_id=shard_id,
                        parent_receipt_id=self.parent_receipt_id,
                        result=failed_result,
                        input_hash=shard.input_hash(),
                        output_hash="",
                        receipt_hash=receipt_hash,
                    ))

        # Reduce results
        reducer = Reducer()
        reduce_receipt = reducer.reduce_with_receipt(shard_receipts, self.parent_receipt_id)

        completed_at = datetime.utcnow()
        total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Compute stats
        successful = sum(1 for r in shard_receipts if r.result.status == "success")
        failed = sum(1 for r in shard_receipts if r.result.status == "failed")

        return OrchestrationResult(
            shards=shards,
            shard_receipts=shard_receipts,
            reduce_receipt=reduce_receipt,
            total_shards=len(shards),
            successful_shards=successful,
            failed_shards=failed,
            total_patches=len(reduce_receipt.result.merged_patches),
            total_no_change_proofs=len(reduce_receipt.result.merged_no_change_proofs),
            conflicts_detected=len(reduce_receipt.result.conflicts),
            total_duration_ms=total_duration_ms,
        )


# ─────────────────────────────────────────────────────────────────
# Replay Support
# ─────────────────────────────────────────────────────────────────

def replay_from_shard_receipts(
    shard_receipts: list[ShardReceipt],
    parent_receipt_id: str,
) -> ReduceReceipt:
    """
    Replay reduction from cached shard receipts.

    This allows re-running the reduce step without re-executing workers.
    """
    reducer = Reducer()
    return reducer.reduce_with_receipt(shard_receipts, parent_receipt_id)


def verify_reduce_determinism(
    shard_receipts: list[ShardReceipt],
    expected_output_hash: str,
) -> bool:
    """
    Verify that reducing shard receipts produces the expected output hash.

    This is the key invariant: same inputs → same output.
    """
    reducer = Reducer()
    result = reducer.reduce(shard_receipts)
    return result.output_hash == expected_output_hash
