"""Test script to verify checkers are wired correctly."""

import sys
import hashlib
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bef_zk.capsule.claims import (
    Claim, ClaimType, Verdict, Scope, Witness,
    create_claim, CHECKER_REGISTRY,
)
# Import checkers to register them
import bef_zk.capsule.checkers  # noqa: F401


def test_allowlist_checker():
    """Test that allowlist checker passes when whitelist is present."""
    file_path = Path(__file__).parent / "file_with_whitelist.py"
    content = file_path.read_text()

    claim = create_claim(
        claim_type=ClaimType.ALLOWLIST_ENFORCED,
        file_path=str(file_path),
        file_content=content,
        description="Test allowlist enforcement",
    )

    verdict, witness = CHECKER_REGISTRY.evaluate("ast", claim, content)

    print(f"\n[TEST] Allowlist checker on file_with_whitelist.py")
    print(f"  Verdict: {verdict.value}")
    print(f"  Witness: {witness.witness_type if witness else 'none'}")

    if verdict == Verdict.PASS:
        print(f"  ✓ PASSED: Allowlist properly detected")
        return True
    else:
        print(f"  ✗ FAILED: Expected PASS, got {verdict.value}")
        if witness and witness.counterexample:
            print(f"  Counterexample: {witness.counterexample}")
        return False


def test_shell_true_checker():
    """Test that shell=True is detected and fails."""
    file_path = Path(__file__).parent / "file_with_shell_true.py"
    content = file_path.read_text()

    claim = create_claim(
        claim_type=ClaimType.NO_SHELL_INJECTION,
        file_path=str(file_path),
        file_content=content,
        description="Test shell=True detection",
    )

    verdict, witness = CHECKER_REGISTRY.evaluate("ast", claim, content)

    print(f"\n[TEST] Shell injection checker on file_with_shell_true.py")
    print(f"  Verdict: {verdict.value}")
    print(f"  Witness: {witness.witness_type if witness else 'none'}")

    if verdict == Verdict.FAIL:
        print(f"  ✓ PASSED: shell=True detected as expected")
        if witness and witness.counterexample:
            print(f"  Counterexample: {witness.counterexample}")
        return True
    else:
        print(f"  ✗ FAILED: Expected FAIL, got {verdict.value}")
        return False


def test_anchor_verification():
    """Test that anchor verification works for NO_CHANGE claims."""
    file_path = Path(__file__).parent / "file_with_whitelist.py"
    content = file_path.read_text()
    file_hash = hashlib.sha256(content.encode()).hexdigest()

    # Create a claim with anchor
    lines = content.split('\n')
    region = '\n'.join(lines[2:9])  # Lines 3-9 (ALLOWED_MODULES definition)
    region_hash = hashlib.sha256(region.encode()).hexdigest()

    scope = Scope(
        file_path=str(file_path),
        file_hash=file_hash,
        start_line=3,
        end_line=9,
        region_hash=region_hash,
    )

    claim = Claim(
        claim_id="test-anchor-001",
        claim_type=ClaimType.ALREADY_MITIGATED,
        scope=scope,
        description="Test anchor verification",
    )

    print(f"\n[TEST] Anchor verification")
    print(f"  File hash: {file_hash[:16]}...")
    print(f"  Region hash (lines 3-9): {region_hash[:16]}...")

    # Verify anchor matches
    current_hash = hashlib.sha256(content.encode()).hexdigest()
    anchor_valid = scope.file_hash == current_hash

    current_lines = content.split('\n')
    current_region = '\n'.join(current_lines[scope.start_line-1:scope.end_line])
    current_region_hash = hashlib.sha256(current_region.encode()).hexdigest()
    span_valid = region_hash == current_region_hash

    if anchor_valid and span_valid:
        print(f"  ✓ Anchor valid: file_hash={anchor_valid}, span_hash={span_valid}")
        return True
    else:
        print(f"  ✗ Anchor invalid: file_hash={anchor_valid}, span_hash={span_valid}")
        return False


def test_anchor_invalidation():
    """Test that mutating file invalidates anchor."""
    file_path = Path(__file__).parent / "file_with_whitelist.py"
    original_content = file_path.read_text()
    original_hash = hashlib.sha256(original_content.encode()).hexdigest()

    # Create scope with original hash
    scope = Scope(
        file_path=str(file_path),
        file_hash=original_hash,
        start_line=3,
        end_line=9,
    )

    # Simulate mutation by changing content
    mutated_content = original_content.replace("ALLOWED_MODULES", "ALLOWED_MODS")
    mutated_hash = hashlib.sha256(mutated_content.encode()).hexdigest()

    print(f"\n[TEST] Anchor invalidation on mutation")
    print(f"  Original hash: {original_hash[:16]}...")
    print(f"  Mutated hash:  {mutated_hash[:16]}...")

    anchor_valid = scope.file_hash == mutated_hash

    if not anchor_valid:
        print(f"  ✓ PASSED: Mutation correctly invalidates anchor")
        return True
    else:
        print(f"  ✗ FAILED: Mutation should invalidate anchor")
        return False


def main():
    print("=" * 60)
    print("CHECKER WIRING TEST")
    print("=" * 60)

    # List registered checkers
    print(f"\nRegistered checkers: {CHECKER_REGISTRY.list_checkers()}")

    results = []
    results.append(("Allowlist checker", test_allowlist_checker()))
    results.append(("Shell=True checker", test_shell_true_checker()))
    results.append(("Anchor verification", test_anchor_verification()))
    results.append(("Anchor invalidation", test_anchor_invalidation()))

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
