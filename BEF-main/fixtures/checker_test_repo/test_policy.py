"""Test policy loading, canonicalization, and scope filtering."""

import sys
import json
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bef_zk.capsule.policy import (
    Policy, PolicyProfile, PolicySource,
    canonical_json, load_policy, resolve_policy_for_review,
    DEFAULT_SECURITY_POLICY, BUILTIN_POLICIES,
)
from bef_zk.capsule.claims import ClaimType, Verdict, ProofObligation


def test_canonical_json_stable_hash():
    """Test that reordering dict keys produces identical canonical JSON."""
    # Original order
    d1 = {"z": 1, "a": 2, "m": {"x": 3, "b": 4}}
    # Different order, same content
    d2 = {"a": 2, "m": {"b": 4, "x": 3}, "z": 1}

    canonical1 = canonical_json(d1)
    canonical2 = canonical_json(d2)

    print(f"\n[TEST] Canonical JSON stability")
    print(f"  d1: {d1}")
    print(f"  d2: {d2}")
    print(f"  canonical1: {canonical1}")
    print(f"  canonical2: {canonical2}")

    if canonical1 == canonical2:
        print(f"  ✓ PASSED: Same canonical JSON despite key order")
        return True
    else:
        print(f"  ✗ FAILED: Different canonical JSON")
        return False


def test_policy_hash_stability():
    """Test that policy hash is stable across whitespace/order changes."""
    yaml1 = """
version: "1.0"
name: "test_policy"
obligations:
  - id: check_a
    claim_type: no_sql_injection
    file_pattern: "**/*.py"
profiles:
  default:
    obligations: [check_a]
"""

    yaml2 = """
name: "test_policy"
version: "1.0"
profiles:
  default:
    obligations:
      - check_a
obligations:
  - id: check_a
    file_pattern: "**/*.py"
    claim_type: no_sql_injection
"""

    policy1 = Policy.from_yaml(yaml1)
    policy2 = Policy.from_yaml(yaml2)

    hash1 = policy1.hash()
    hash2 = policy2.hash()

    print(f"\n[TEST] Policy hash stability (reordered YAML)")
    print(f"  Policy 1 hash: {hash1[:16]}...")
    print(f"  Policy 2 hash: {hash2[:16]}...")

    if hash1 == hash2:
        print(f"  ✓ PASSED: Same hash despite YAML key order")
        return True
    else:
        print(f"  ✗ FAILED: Different hashes")
        return False


def test_policy_mutation_changes_hash():
    """Test that changing an obligation changes the hash."""
    yaml_base = """
version: "1.0"
name: "test_policy"
obligations:
  - id: check_a
    claim_type: no_sql_injection
    file_pattern: "**/*.py"
"""

    yaml_mutated = """
version: "1.0"
name: "test_policy"
obligations:
  - id: check_a
    claim_type: no_shell_injection
    file_pattern: "**/*.py"
"""

    policy_base = Policy.from_yaml(yaml_base)
    policy_mutated = Policy.from_yaml(yaml_mutated)

    hash_base = policy_base.hash()
    hash_mutated = policy_mutated.hash()

    print(f"\n[TEST] Policy mutation changes hash")
    print(f"  Base hash:    {hash_base[:16]}...")
    print(f"  Mutated hash: {hash_mutated[:16]}...")

    if hash_base != hash_mutated:
        print(f"  ✓ PASSED: Mutation changes hash")
        return True
    else:
        print(f"  ✗ FAILED: Hash should be different")
        return False


def test_scope_filtering_src_vs_tests():
    """Test that obligations with src/** don't match tests/**."""
    policy = Policy(
        version="1.0",
        name="scope_test",
        obligations=[
            ProofObligation(
                obligation_id="src_only",
                claim_type=ClaimType.NO_SQL_INJECTION,
                description="Only for src files",
                file_pattern="src/**/*.py",
            ),
            ProofObligation(
                obligation_id="tests_only",
                claim_type=ClaimType.REFACTOR_EQUIVALENCE,
                description="Only for test files",
                file_pattern="tests/**/*.py",
            ),
            ProofObligation(
                obligation_id="all_py",
                claim_type=ClaimType.NO_SHELL_INJECTION,
                description="All Python files",
                file_pattern="**/*.py",
            ),
        ],
        profiles={
            "default": PolicyProfile(
                name="default",
                obligations=["src_only", "tests_only", "all_py"],
            ),
        },
    )

    # Test src file
    src_obligations = policy.get_obligations_for_file("src/main.py", "default")
    src_ids = {o.obligation_id for o in src_obligations}

    # Test test file
    test_obligations = policy.get_obligations_for_file("tests/test_main.py", "default")
    test_ids = {o.obligation_id for o in test_obligations}

    print(f"\n[TEST] Scope filtering (src/** vs tests/**)")
    print(f"  src/main.py obligations: {src_ids}")
    print(f"  tests/test_main.py obligations: {test_ids}")

    src_expected = {"src_only", "all_py"}
    test_expected = {"tests_only", "all_py"}

    src_ok = src_ids == src_expected
    test_ok = test_ids == test_expected

    if src_ok:
        print(f"  ✓ src/main.py correctly matched: {src_ids}")
    else:
        print(f"  ✗ src/main.py wrong: got {src_ids}, expected {src_expected}")

    if test_ok:
        print(f"  ✓ tests/test_main.py correctly matched: {test_ids}")
    else:
        print(f"  ✗ tests/test_main.py wrong: got {test_ids}, expected {test_expected}")

    return src_ok and test_ok


def test_policy_discovery_order():
    """Test policy discovery precedence: explicit > project > builtin."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create project policy
        capseal_dir = tmpdir / ".capseal"
        capseal_dir.mkdir()
        project_policy = capseal_dir / "policy.yaml"
        project_policy.write_text("""
version: "1.0"
name: "project_policy"
obligations: []
profiles:
  default:
    obligations: []
""")

        # Create explicit policy
        explicit_policy = tmpdir / "custom_policy.yaml"
        explicit_policy.write_text("""
version: "1.0"
name: "explicit_policy"
obligations: []
profiles:
  default:
    obligations: []
""")

        print(f"\n[TEST] Policy discovery order")

        # Test 1: Explicit wins
        result = load_policy(explicit_path=explicit_policy, project_root=tmpdir)
        explicit_ok = result.policy.name == "explicit_policy"
        print(f"  Explicit policy wins: {result.resolved_from}")
        if explicit_ok:
            print(f"  ✓ Explicit policy loaded: {result.policy.name}")
        else:
            print(f"  ✗ Wrong policy: {result.policy.name}")

        # Test 2: Project if no explicit
        result = load_policy(project_root=tmpdir)
        project_ok = result.policy.name == "project_policy"
        print(f"  Project policy: {result.resolved_from}")
        if project_ok:
            print(f"  ✓ Project policy loaded: {result.policy.name}")
        else:
            print(f"  ✗ Wrong policy: {result.policy.name}")

        # Test 3: Builtin if no project
        empty_dir = tmpdir / "empty"
        empty_dir.mkdir()
        result = load_policy(project_root=empty_dir)
        builtin_ok = result.policy.name == "security_strict"
        print(f"  Builtin fallback: {result.resolved_from}")
        if builtin_ok:
            print(f"  ✓ Builtin policy loaded: {result.policy.name}")
        else:
            print(f"  ✗ Wrong policy: {result.policy.name}")

        return explicit_ok and project_ok and builtin_ok


def test_policy_source_tracking():
    """Test that policy source is tracked correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        policy_file = tmpdir / "policy.yaml"
        policy_file.write_text("""
version: "1.0"
name: "tracked_policy"
obligations: []
profiles:
  default:
    obligations: []
""")

        policy = Policy.from_file(policy_file)

        print(f"\n[TEST] Policy source tracking")
        print(f"  Source type: {policy.source.source_type}")
        print(f"  Source path: {policy.source.source_path}")
        print(f"  Content hash: {policy.source.content_hash[:16] if policy.source.content_hash else 'none'}...")

        source_ok = (
            policy.source.source_type == "file" and
            policy.source.source_path == str(policy_file) and
            policy.source.content_hash is not None
        )

        if source_ok:
            print(f"  ✓ PASSED: Source correctly tracked")
            return True
        else:
            print(f"  ✗ FAILED: Source not tracked correctly")
            return False


def main():
    print("=" * 60)
    print("POLICY LOADING & CANONICALIZATION TESTS")
    print("=" * 60)

    results = []
    results.append(("Canonical JSON stability", test_canonical_json_stable_hash()))
    results.append(("Policy hash stability", test_policy_hash_stability()))
    results.append(("Policy mutation changes hash", test_policy_mutation_changes_hash()))
    results.append(("Scope filtering (src vs tests)", test_scope_filtering_src_vs_tests()))
    results.append(("Policy discovery order", test_policy_discovery_order()))
    results.append(("Policy source tracking", test_policy_source_tracking()))

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
