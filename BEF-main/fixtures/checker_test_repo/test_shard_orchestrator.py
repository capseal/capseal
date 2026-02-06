"""Test shard orchestrator for multi-agent DAG."""

import sys
import time
import random
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bef_zk.capsule.shard_orchestrator import (
    Shard, ShardResult, ShardReceipt, ShardWorker,
    Reducer, ReduceResult, ReduceReceipt,
    ShardOrchestrator, OrchestrationResult,
    shard_by_file, shard_by_count,
    detect_conflicts, Conflict,
    replay_from_shard_receipts, verify_reduce_determinism,
)


def mock_patch_generator(plan_items: list[dict], project_dir: Path) -> tuple[list[dict], list[dict]]:
    """Mock patch generator for testing."""
    patches = []
    no_change_proofs = []

    for item in plan_items:
        file_path = item.get("file_path", "unknown")
        # Simulate some items producing patches, some producing no-change proofs
        if "skip" in item.get("description", "").lower():
            no_change_proofs.append({
                "file_path": file_path,
                "disposition": "already_mitigated",
                "justification": "Already handled",
            })
        else:
            patches.append({
                "file_path": file_path,
                "patch_id": f"P-{hash(file_path) % 10000:04d}",
                "start_line": item.get("start_line", 1),
                "end_line": item.get("end_line", 10),
                "diff": f"--- {file_path}\n+++ {file_path}\n@@ -1,3 +1,3 @@\n-old\n+new",
            })

    return patches, no_change_proofs


def test_shard_by_file():
    """Test sharding by file."""
    plan_items = [
        {"file_path": "src/a.py", "description": "Fix a"},
        {"file_path": "src/b.py", "description": "Fix b1"},
        {"file_path": "src/b.py", "description": "Fix b2"},
        {"file_path": "src/c.py", "description": "Fix c"},
    ]

    shards = shard_by_file(plan_items, "parent-123")

    print(f"\n[TEST] Shard by file")
    print(f"  Input: {len(plan_items)} items, {len(set(i['file_path'] for i in plan_items))} files")
    print(f"  Output: {len(shards)} shards")

    for s in shards:
        print(f"    {s.shard_id}: {s.file_paths} ({len(s.plan_items)} items)")

    # Should have 3 shards (one per file)
    if len(shards) == 3:
        # b.py shard should have 2 items
        b_shard = next(s for s in shards if "src/b.py" in s.file_paths)
        if len(b_shard.plan_items) == 2:
            print(f"  ✓ PASSED: Correct sharding")
            return True
        else:
            print(f"  ✗ FAILED: b.py shard should have 2 items")
            return False
    else:
        print(f"  ✗ FAILED: Expected 3 shards, got {len(shards)}")
        return False


def test_shard_by_count():
    """Test sharding by count."""
    plan_items = [
        {"file_path": "src/a.py", "description": "Fix a"},
        {"file_path": "src/b.py", "description": "Fix b"},
        {"file_path": "src/c.py", "description": "Fix c"},
        {"file_path": "src/d.py", "description": "Fix d"},
        {"file_path": "src/e.py", "description": "Fix e"},
    ]

    shards = shard_by_count(plan_items, 2, "parent-123")

    print(f"\n[TEST] Shard by count")
    print(f"  Input: {len(plan_items)} items")
    print(f"  Output: {len(shards)} shards")

    for s in shards:
        print(f"    {s.shard_id}: {len(s.plan_items)} items")

    # Should have 2 shards with roughly equal items
    if len(shards) == 2:
        counts = [len(s.plan_items) for s in shards]
        if counts == [3, 2] or counts == [2, 3]:
            print(f"  ✓ PASSED: Items distributed correctly")
            return True
        else:
            print(f"  ✗ FAILED: Expected [3,2] or [2,3], got {counts}")
            return False
    else:
        print(f"  ✗ FAILED: Expected 2 shards")
        return False


def test_conflict_detection():
    """Test that conflicts are detected when shards edit overlapping lines."""
    results = [
        ShardResult(
            shard_id="S001",
            status="success",
            patches=[{
                "file_path": "src/shared.py",
                "start_line": 10,
                "end_line": 20,
            }],
            no_change_proofs=[],
            errors=[],
            input_hash="a",
            output_hash="b",
            started_at="",
            completed_at="",
            duration_ms=0,
        ),
        ShardResult(
            shard_id="S002",
            status="success",
            patches=[{
                "file_path": "src/shared.py",
                "start_line": 15,  # Overlaps with S001
                "end_line": 25,
            }],
            no_change_proofs=[],
            errors=[],
            input_hash="c",
            output_hash="d",
            started_at="",
            completed_at="",
            duration_ms=0,
        ),
    ]

    conflicts = detect_conflicts(results)

    print(f"\n[TEST] Conflict detection")
    print(f"  Shard 1: lines 10-20")
    print(f"  Shard 2: lines 15-25")
    print(f"  Conflicts: {len(conflicts)}")

    if len(conflicts) == 1:
        c = conflicts[0]
        print(f"  Conflict: {c.shard_a} vs {c.shard_b} in {c.file_path}")
        print(f"  ✓ PASSED: Conflict detected")
        return True
    else:
        print(f"  ✗ FAILED: Expected 1 conflict")
        return False


def test_no_conflict_when_non_overlapping():
    """Test that non-overlapping edits don't cause conflicts."""
    results = [
        ShardResult(
            shard_id="S001",
            status="success",
            patches=[{
                "file_path": "src/shared.py",
                "start_line": 10,
                "end_line": 20,
            }],
            no_change_proofs=[],
            errors=[],
            input_hash="a",
            output_hash="b",
            started_at="",
            completed_at="",
            duration_ms=0,
        ),
        ShardResult(
            shard_id="S002",
            status="success",
            patches=[{
                "file_path": "src/shared.py",
                "start_line": 30,  # No overlap
                "end_line": 40,
            }],
            no_change_proofs=[],
            errors=[],
            input_hash="c",
            output_hash="d",
            started_at="",
            completed_at="",
            duration_ms=0,
        ),
    ]

    conflicts = detect_conflicts(results)

    print(f"\n[TEST] No conflict for non-overlapping")
    print(f"  Shard 1: lines 10-20")
    print(f"  Shard 2: lines 30-40")
    print(f"  Conflicts: {len(conflicts)}")

    if len(conflicts) == 0:
        print(f"  ✓ PASSED: No false conflicts")
        return True
    else:
        print(f"  ✗ FAILED: Unexpected conflict detected")
        return False


def test_reducer_determinism():
    """Test that reducer produces same output regardless of input order."""
    # Create some mock shard receipts
    def make_receipt(shard_id: str, patches: list[dict]) -> ShardReceipt:
        result = ShardResult(
            shard_id=shard_id,
            status="success",
            patches=patches,
            no_change_proofs=[],
            errors=[],
            input_hash=f"input-{shard_id}",
            output_hash=f"output-{shard_id}",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
            duration_ms=1000,
        )
        return ShardReceipt(
            receipt_id=f"receipt-{shard_id}",
            shard_id=shard_id,
            parent_receipt_id="parent-123",
            result=result,
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            receipt_hash=f"hash-{shard_id}",
        )

    receipts = [
        make_receipt("S001", [{"file_path": "a.py", "start_line": 1, "end_line": 5}]),
        make_receipt("S002", [{"file_path": "b.py", "start_line": 1, "end_line": 5}]),
        make_receipt("S003", [{"file_path": "c.py", "start_line": 1, "end_line": 5}]),
    ]

    reducer = Reducer()

    # Reduce in original order
    result1 = reducer.reduce(receipts)

    # Shuffle and reduce again
    shuffled = receipts.copy()
    random.shuffle(shuffled)
    result2 = reducer.reduce(shuffled)

    # Reverse and reduce
    reversed_receipts = receipts[::-1]
    result3 = reducer.reduce(reversed_receipts)

    print(f"\n[TEST] Reducer determinism")
    print(f"  Order 1 output_hash: {result1.output_hash[:16]}...")
    print(f"  Order 2 output_hash: {result2.output_hash[:16]}...")
    print(f"  Order 3 output_hash: {result3.output_hash[:16]}...")
    print(f"  Shard order (all): {result1.shard_order}")

    if result1.output_hash == result2.output_hash == result3.output_hash:
        print(f"  ✓ PASSED: Same output regardless of input order")
        return True
    else:
        print(f"  ✗ FAILED: Different outputs for different orders")
        return False


def test_orchestrator_end_to_end():
    """Test the full orchestrator flow."""
    plan_items = [
        {"file_path": "src/a.py", "description": "Fix a", "start_line": 1, "end_line": 10},
        {"file_path": "src/b.py", "description": "Fix b", "start_line": 1, "end_line": 10},
        {"file_path": "src/c.py", "description": "Skip c - already fixed", "start_line": 1, "end_line": 10},
    ]

    orchestrator = ShardOrchestrator(
        plan_items=plan_items,
        project_dir=Path("."),
        patch_generator=mock_patch_generator,
        parent_receipt_id="plan-123",
    )

    result = orchestrator.run(max_workers=2, shard_strategy="by_file")

    print(f"\n[TEST] Orchestrator end-to-end")
    print(f"  Total shards: {result.total_shards}")
    print(f"  Successful: {result.successful_shards}")
    print(f"  Failed: {result.failed_shards}")
    print(f"  Total patches: {result.total_patches}")
    print(f"  No-change proofs: {result.total_no_change_proofs}")
    print(f"  Conflicts: {result.conflicts_detected}")
    print(f"  Duration: {result.total_duration_ms}ms")
    print(f"  Reduce receipt: {result.reduce_receipt.receipt_id}")

    if (
        result.total_shards == 3 and
        result.total_patches == 2 and  # a.py and b.py
        result.total_no_change_proofs == 1 and  # c.py (skip)
        result.conflicts_detected == 0
    ):
        print(f"  ✓ PASSED: Orchestrator works correctly")
        return True
    else:
        print(f"  ✗ FAILED: Unexpected results")
        return False


def test_replay_from_shard_receipts():
    """Test replaying reduction from cached shard receipts."""
    def make_receipt(shard_id: str) -> ShardReceipt:
        result = ShardResult(
            shard_id=shard_id,
            status="success",
            patches=[{"file_path": f"{shard_id}.py", "start_line": 1, "end_line": 5}],
            no_change_proofs=[],
            errors=[],
            input_hash=f"input-{shard_id}",
            output_hash=f"output-{shard_id}",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
            duration_ms=1000,
        )
        return ShardReceipt(
            receipt_id=f"receipt-{shard_id}",
            shard_id=shard_id,
            parent_receipt_id="parent-123",
            result=result,
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            receipt_hash=f"hash-{shard_id}",
        )

    receipts = [make_receipt("S001"), make_receipt("S002")]

    # First reduction
    reduce_receipt1 = replay_from_shard_receipts(receipts, "parent-123")

    # Replay
    reduce_receipt2 = replay_from_shard_receipts(receipts, "parent-123")

    print(f"\n[TEST] Replay from shard receipts")
    print(f"  Original reduce: {reduce_receipt1.receipt_id}")
    print(f"  Replayed reduce: {reduce_receipt2.receipt_id}")
    print(f"  Output hash match: {reduce_receipt1.result.output_hash == reduce_receipt2.result.output_hash}")

    if reduce_receipt1.result.output_hash == reduce_receipt2.result.output_hash:
        print(f"  ✓ PASSED: Replay produces identical result")
        return True
    else:
        print(f"  ✗ FAILED: Replay produced different result")
        return False


def test_verify_reduce_determinism():
    """Test verification of reduce determinism."""
    def make_receipt(shard_id: str) -> ShardReceipt:
        result = ShardResult(
            shard_id=shard_id,
            status="success",
            patches=[{"file_path": f"{shard_id}.py", "start_line": 1, "end_line": 5}],
            no_change_proofs=[],
            errors=[],
            input_hash=f"input-{shard_id}",
            output_hash=f"output-{shard_id}",
            started_at="",
            completed_at="",
            duration_ms=0,
        )
        return ShardReceipt(
            receipt_id=f"receipt-{shard_id}",
            shard_id=shard_id,
            parent_receipt_id="parent-123",
            result=result,
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            receipt_hash=f"hash-{shard_id}",
        )

    receipts = [make_receipt("S001"), make_receipt("S002")]

    # Get expected hash
    reducer = Reducer()
    expected_result = reducer.reduce(receipts)
    expected_hash = expected_result.output_hash

    # Verify with correct hash
    valid = verify_reduce_determinism(receipts, expected_hash)

    # Verify with wrong hash
    invalid = verify_reduce_determinism(receipts, "wrong-hash")

    print(f"\n[TEST] Verify reduce determinism")
    print(f"  Expected hash: {expected_hash[:16]}...")
    print(f"  Verify with correct hash: {valid}")
    print(f"  Verify with wrong hash: {invalid}")

    if valid and not invalid:
        print(f"  ✓ PASSED: Verification works correctly")
        return True
    else:
        print(f"  ✗ FAILED: Verification logic broken")
        return False


def main():
    print("=" * 60)
    print("SHARD ORCHESTRATOR TESTS")
    print("=" * 60)

    results = []
    results.append(("Shard by file", test_shard_by_file()))
    results.append(("Shard by count", test_shard_by_count()))
    results.append(("Conflict detection", test_conflict_detection()))
    results.append(("No conflict for non-overlapping", test_no_conflict_when_non_overlapping()))
    results.append(("Reducer determinism", test_reducer_determinism()))
    results.append(("Orchestrator end-to-end", test_orchestrator_end_to_end()))
    results.append(("Replay from shard receipts", test_replay_from_shard_receipts()))
    results.append(("Verify reduce determinism", test_verify_reduce_determinism()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
