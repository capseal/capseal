"""Integration test for review command with multi-agent support.

Tests the wiring of --agents N and --shard into the review command.
"""

import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_review_help_shows_agents_option():
    """Test that review command shows --agents option in help."""
    from bef_zk.capsule.cli.shell import CapsealShell
    import io

    shell = CapsealShell()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        shell.do_review("")  # Empty arg triggers help
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    print("\n[TEST] Review help shows --agents option")
    print(f"  Help mentions '--agents': {'--agents' in output}")

    if "--agents" in output:
        print("  ✓ PASSED: Help includes --agents option")
        return True
    else:
        print("  ✗ FAILED: Help missing --agents option")
        return False


def test_shard_orchestrator_import():
    """Test that shard orchestrator imports correctly in review."""
    print("\n[TEST] Shard orchestrator import in review context")

    try:
        from bef_zk.capsule.shard_orchestrator import (
            ShardOrchestrator, shard_by_file, shard_by_count,
        )
        from bef_zk.capsule.refactor_engine import (
            generate_refactor_plan,
            run_multi_agent_patches,
        )

        print("  ShardOrchestrator: imported")
        print("  shard_by_file: imported")
        print("  shard_by_count: imported")
        print("  ✓ PASSED: All required imports available")
        return True
    except ImportError as e:
        print(f"  ✗ FAILED: Import error: {e}")
        return False


def test_orchestrator_with_mock_llm():
    """Test orchestrator with mock LLM patch generator."""
    from bef_zk.capsule.shard_orchestrator import ShardOrchestrator

    print("\n[TEST] Orchestrator with mock LLM generator")

    # Mock patch generator that simulates LLM behavior
    def mock_llm_generator(items: list[dict], project_dir: Path):
        patches = []
        no_change_proofs = []

        for item in items:
            file_path = item.get("file_path", "unknown.py")
            # Simulate some files getting patches, others not needing changes
            if "skip" in item.get("description", "").lower():
                no_change_proofs.append({
                    "file_path": file_path,
                    "disposition": "already_mitigated",
                    "justification": "Code already follows best practices",
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

    plan_items = [
        {"file_path": "src/auth.py", "description": "Fix SQL injection", "start_line": 10, "end_line": 20},
        {"file_path": "src/api.py", "description": "Fix command injection", "start_line": 5, "end_line": 15},
        {"file_path": "src/utils.py", "description": "Skip - already secure", "start_line": 1, "end_line": 5},
        {"file_path": "tests/test_auth.py", "description": "Add security tests", "start_line": 1, "end_line": 10},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        orchestrator = ShardOrchestrator(
            plan_items=plan_items,
            project_dir=project_dir,
            patch_generator=mock_llm_generator,
            parent_receipt_id="test-review-123",
        )

        # Test with 2 agents, by_file sharding
        result = orchestrator.run(max_workers=2, shard_strategy="by_file")

        print(f"  Total shards: {result.total_shards}")
        print(f"  Successful: {result.successful_shards}")
        print(f"  Total patches: {result.total_patches}")
        print(f"  No-change proofs: {result.total_no_change_proofs}")
        print(f"  Conflicts: {result.conflicts_detected}")

        # Validate results
        expected_shards = 4  # 4 unique files
        expected_patches = 3  # auth, api, tests
        expected_ncp = 1  # utils (skip)

        if (
            result.total_shards == expected_shards and
            result.total_patches == expected_patches and
            result.total_no_change_proofs == expected_ncp and
            result.reduce_receipt is not None
        ):
            print("  ✓ PASSED: Orchestrator produces expected results")
            return True
        else:
            print(f"  ✗ FAILED: Expected {expected_shards} shards, {expected_patches} patches, {expected_ncp} ncps")
            return False


def test_receipt_chain_integrity():
    """Test that shard receipts chain to parent correctly."""
    from bef_zk.capsule.shard_orchestrator import ShardOrchestrator

    print("\n[TEST] Receipt chain integrity")

    def simple_generator(items, project_dir):
        return [{"file_path": i["file_path"], "start_line": 1, "end_line": 5} for i in items], []

    plan_items = [
        {"file_path": "a.py", "description": "Fix a"},
        {"file_path": "b.py", "description": "Fix b"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        parent_id = "parent-abc123"

        orchestrator = ShardOrchestrator(
            plan_items=plan_items,
            project_dir=project_dir,
            patch_generator=simple_generator,
            parent_receipt_id=parent_id,
        )

        result = orchestrator.run(max_workers=2)

        # Check all shard receipts reference parent
        all_reference_parent = all(
            r.parent_receipt_id == parent_id
            for r in result.shard_receipts
        )

        # Check reduce receipt references shard IDs
        reduce_has_shard_order = len(result.reduce_receipt.result.shard_order) == len(result.shard_receipts)

        print(f"  Shard receipts reference parent: {all_reference_parent}")
        print(f"  Reduce receipt has shard order: {reduce_has_shard_order}")

        if all_reference_parent and reduce_has_shard_order:
            print("  ✓ PASSED: Receipt chain is correct")
            return True
        else:
            print("  ✗ FAILED: Receipt chain broken")
            return False


def test_determinism_across_runs():
    """Test that same inputs produce same outputs across runs."""
    from bef_zk.capsule.shard_orchestrator import ShardOrchestrator

    print("\n[TEST] Determinism across runs")

    def deterministic_generator(items, project_dir):
        return [
            {"file_path": i["file_path"], "patch_id": f"P-{i['file_path']}", "start_line": 1, "end_line": 5}
            for i in items
        ], []

    plan_items = [
        {"file_path": "x.py", "description": "Fix x"},
        {"file_path": "y.py", "description": "Fix y"},
        {"file_path": "z.py", "description": "Fix z"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Run 1
        orch1 = ShardOrchestrator(
            plan_items=plan_items,
            project_dir=project_dir,
            patch_generator=deterministic_generator,
            parent_receipt_id="run-1",
        )
        result1 = orch1.run(max_workers=3)

        # Run 2 (same inputs)
        orch2 = ShardOrchestrator(
            plan_items=plan_items,
            project_dir=project_dir,
            patch_generator=deterministic_generator,
            parent_receipt_id="run-1",  # Same parent
        )
        result2 = orch2.run(max_workers=3)

        # Compare reduce outputs
        hash1 = result1.reduce_receipt.result.output_hash
        hash2 = result2.reduce_receipt.result.output_hash

        print(f"  Run 1 output hash: {hash1[:16]}...")
        print(f"  Run 2 output hash: {hash2[:16]}...")

        if hash1 == hash2:
            print("  ✓ PASSED: Deterministic outputs")
            return True
        else:
            print("  ✗ FAILED: Non-deterministic outputs")
            return False


def main():
    print("=" * 60)
    print("REVIEW COMMAND INTEGRATION TESTS")
    print("=" * 60)

    results = []
    results.append(("Review help shows --agents", test_review_help_shows_agents_option()))
    results.append(("Shard orchestrator import", test_shard_orchestrator_import()))
    results.append(("Orchestrator with mock LLM", test_orchestrator_with_mock_llm()))
    results.append(("Receipt chain integrity", test_receipt_chain_integrity()))
    results.append(("Determinism across runs", test_determinism_across_runs()))

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
