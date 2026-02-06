"""Test claim cache for incremental proof reuse."""

import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bef_zk.capsule.claim_cache import (
    ClaimCache, compute_cache_key, CachedVerdict,
    check_claim_with_cache, get_cache_for_project,
)
from bef_zk.capsule.claims import (
    Claim, ClaimType, Verdict, Scope, create_claim, CHECKER_REGISTRY,
)
# Import checkers to register them
import bef_zk.capsule.checkers  # noqa: F401


def test_cache_key_determinism():
    """Test that same inputs produce same cache key."""
    key1 = compute_cache_key(
        policy_hash="abc123",
        checker_id="ast",
        checker_version="1.0.0",
        claim_type=ClaimType.NO_SHELL_INJECTION,
        scope_hash="def456",
    )

    key2 = compute_cache_key(
        policy_hash="abc123",
        checker_id="ast",
        checker_version="1.0.0",
        claim_type=ClaimType.NO_SHELL_INJECTION,
        scope_hash="def456",
    )

    print(f"\n[TEST] Cache key determinism")
    print(f"  Key 1: {key1}")
    print(f"  Key 2: {key2}")

    if key1 == key2:
        print(f"  ✓ PASSED: Same inputs produce same key")
        return True
    else:
        print(f"  ✗ FAILED: Keys should be identical")
        return False


def test_cache_key_changes_on_version_bump():
    """Test that checker version change invalidates cache."""
    key1 = compute_cache_key(
        policy_hash="abc123",
        checker_id="ast",
        checker_version="1.0.0",
        claim_type=ClaimType.NO_SHELL_INJECTION,
        scope_hash="def456",
    )

    key2 = compute_cache_key(
        policy_hash="abc123",
        checker_id="ast",
        checker_version="1.0.1",  # Version bumped
        claim_type=ClaimType.NO_SHELL_INJECTION,
        scope_hash="def456",
    )

    print(f"\n[TEST] Cache key changes on version bump")
    print(f"  Key v1.0.0: {key1}")
    print(f"  Key v1.0.1: {key2}")

    if key1 != key2:
        print(f"  ✓ PASSED: Version bump invalidates cache")
        return True
    else:
        print(f"  ✗ FAILED: Keys should be different")
        return False


def test_cache_store_and_lookup():
    """Test storing and retrieving from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "claims_cache.json"
        cache = ClaimCache(cache_path)

        # Create a test claim
        file_content = "def foo(): pass"
        claim = create_claim(
            claim_type=ClaimType.NO_SHELL_INJECTION,
            file_path="/test/file.py",
            file_content=file_content,
            description="Test claim",
        )

        # Store verdict
        cache.store(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim=claim,
            verdict=Verdict.PASS,
            witness=None,
        )

        # Lookup
        cached = cache.lookup(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim_type=ClaimType.NO_SHELL_INJECTION,
            scope_hash=claim.scope.file_hash,
        )

        print(f"\n[TEST] Cache store and lookup")
        print(f"  Stored: verdict=PASS")

        if cached and cached.verdict == "pass":
            print(f"  Lookup: found, verdict={cached.verdict}")
            print(f"  ✓ PASSED: Cache hit with correct verdict")
            return True
        else:
            print(f"  ✗ FAILED: Cache miss or wrong verdict")
            return False


def test_cache_miss_on_scope_change():
    """Test that scope change causes cache miss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "claims_cache.json"
        cache = ClaimCache(cache_path)

        # Create and store original claim
        file_content_v1 = "def foo(): pass"
        claim = create_claim(
            claim_type=ClaimType.NO_SHELL_INJECTION,
            file_path="/test/file.py",
            file_content=file_content_v1,
            description="Test claim",
        )

        cache.store(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim=claim,
            verdict=Verdict.PASS,
        )

        # Try to lookup with different file content (different scope_hash)
        file_content_v2 = "def foo(): return 42"  # Modified
        import hashlib
        new_scope_hash = hashlib.sha256(file_content_v2.encode()).hexdigest()

        cached = cache.lookup(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim_type=ClaimType.NO_SHELL_INJECTION,
            scope_hash=new_scope_hash,
        )

        print(f"\n[TEST] Cache miss on scope change")
        print(f"  Original scope_hash: {claim.scope.file_hash[:16]}...")
        print(f"  Changed scope_hash:  {new_scope_hash[:16]}...")

        if cached is None:
            print(f"  ✓ PASSED: Cache miss (scope changed)")
            return True
        else:
            print(f"  ✗ FAILED: Should be cache miss")
            return False


def test_cache_persistence():
    """Test that cache persists to disk and reloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "claims_cache.json"

        # Create cache and store
        cache1 = ClaimCache(cache_path)
        file_content = "def foo(): pass"
        claim = create_claim(
            claim_type=ClaimType.NO_SHELL_INJECTION,
            file_path="/test/file.py",
            file_content=file_content,
            description="Test claim",
        )
        cache1.store(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim=claim,
            verdict=Verdict.PASS,
        )
        cache1.save()

        # Create new cache instance (simulate new session)
        cache2 = ClaimCache(cache_path)

        cached = cache2.lookup(
            policy_hash="policy123",
            checker_id="ast",
            checker_version="1.0.0",
            claim_type=ClaimType.NO_SHELL_INJECTION,
            scope_hash=claim.scope.file_hash,
        )

        print(f"\n[TEST] Cache persistence")
        print(f"  Saved to:  {cache_path}")
        print(f"  File exists: {cache_path.exists()}")

        if cached and cached.verdict == "pass":
            print(f"  Reload lookup: found, verdict={cached.verdict}")
            print(f"  ✓ PASSED: Cache persisted and reloaded")
            return True
        else:
            print(f"  ✗ FAILED: Cache not persisted")
            return False


def test_check_claim_with_cache():
    """Test the cache-aware checker execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "claims_cache.json"
        cache = ClaimCache(cache_path)

        file_content = """
def safe_func():
    # No shell=True here
    import subprocess
    subprocess.run(['ls', '-la'])
"""

        claim = create_claim(
            claim_type=ClaimType.NO_SHELL_INJECTION,
            file_path="/test/safe.py",
            file_content=file_content,
            description="Test NO_SHELL_INJECTION",
        )

        policy_hash = "test_policy_123"

        # First call - should miss cache
        result1 = check_claim_with_cache(
            claim=claim,
            file_content=file_content,
            checker_id="ast",
            policy_hash=policy_hash,
            cache=cache,
            checker_registry=CHECKER_REGISTRY,
        )

        # Second call - should hit cache
        result2 = check_claim_with_cache(
            claim=claim,
            file_content=file_content,
            checker_id="ast",
            policy_hash=policy_hash,
            cache=cache,
            checker_registry=CHECKER_REGISTRY,
        )

        print(f"\n[TEST] Cache-aware checker execution")
        print(f"  First call:  from_cache={result1.from_cache}, verdict={result1.verdict.value}")
        print(f"  Second call: from_cache={result2.from_cache}, verdict={result2.verdict.value}")

        if not result1.from_cache and result2.from_cache:
            if result1.verdict == result2.verdict == Verdict.PASS:
                print(f"  ✓ PASSED: First miss, second hit, verdicts match")
                return True
            else:
                print(f"  ✗ FAILED: Verdicts don't match")
                return False
        else:
            print(f"  ✗ FAILED: Cache behavior wrong")
            return False


def main():
    print("=" * 60)
    print("CLAIM CACHE TESTS")
    print("=" * 60)

    results = []
    results.append(("Cache key determinism", test_cache_key_determinism()))
    results.append(("Cache key version invalidation", test_cache_key_changes_on_version_bump()))
    results.append(("Cache store and lookup", test_cache_store_and_lookup()))
    results.append(("Cache miss on scope change", test_cache_miss_on_scope_change()))
    results.append(("Cache persistence", test_cache_persistence()))
    results.append(("Cache-aware checker", test_check_claim_with_cache()))

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
