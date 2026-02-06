"""Intelligent bi-directional merge - combines both repos intelligently.

Workflow:
1. Scan both source and target directories
2. Classify files: UNIQUE_SOURCE, UNIQUE_TARGET, BOTH_SAME, BOTH_DIFFER
3. Load review synthesis for risk assessment
4. Auto-apply safe changes (unique files, identical files)
5. Flag conflicts for manual review (files that differ)
6. Run tests
7. Generate merge receipt

This is a TRUE MERGE, not a one-way sync:
- Files unique to source → copy to target (ADD)
- Files unique to target → keep in target (KEEP)
- Files identical in both → keep (no change)
- Files that differ → flag for manual merge (CONFLICT)
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from .redact import redact_secrets
from .git_utils import tracked_files


# Directories to skip during merge (exact match on any path component)
MERGE_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", ".tox", ".nox",
    "dist", "build", "target",  # Build outputs
    ".cache", ".tmp", "tmp", "temp",
    ".idea", ".vscode", ".vs",
    "coverage", "htmlcov", ".coverage",
    ".eggs", "*.egg-info",
    ".capseal",
    # Rust specific
    "debug", "release",  # inside target/
    ".fingerprint",
    # Node specific
    ".next", ".nuxt", ".output",
    # Python specific
    ".tox", "site-packages",
}

# File extensions to skip
MERGE_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".o", ".a", ".lib", ".obj",
    ".class", ".jar",
    ".log", ".tmp", ".bak", ".swp", ".swo",
    ".rlib", ".rmeta", ".d",  # Rust build artifacts
}

# File names to skip
MERGE_SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
    ".gitignore", ".dockerignore",
}


def should_skip(path: Path) -> bool:
    """Check if path should be skipped during merge."""
    # Check directory names
    for part in path.parts:
        if part in MERGE_SKIP_DIRS:
            return True
        # Handle patterns like *.egg-info
        if part.endswith(".egg-info"):
            return True

    # Check file extension
    if path.suffix.lower() in MERGE_SKIP_EXTENSIONS:
        return True

    # Check file name
    if path.name in MERGE_SKIP_FILES:
        return True

    return False


@dataclass
class FileChange:
    """Represents a change to apply."""
    path: str
    status: str  # A=add from source, K=keep target, M=modified/conflict, D=delete
    source_path: Path | None = None
    target_path: Path | None = None

    # Classification
    risk_level: str = "UNKNOWN"  # SAFE, RISKY, CONFLICT
    risk_reasons: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)  # From review

    # Result
    applied: bool = False
    needs_manual: bool = False
    error: str | None = None


@dataclass
class MergeResult:
    """Result of a merge operation."""
    success: bool
    target_dir: Path
    changes_applied: int
    changes_skipped: int
    conflicts: list[FileChange]
    test_passed: bool | None
    test_output: str = ""
    receipt_hash: str = ""


def scan_directories(source_dir: Path, target_dir: Path) -> list[FileChange]:
    """Scan both directories and classify all files.

    Returns list of FileChange with status:
    - A: Add (unique to source, copy to target)
    - K: Keep (unique to target, already there)
    - S: Same (identical in both)
    - M: Modified (exists in both but differs - CONFLICT)
    """
    changes = []

    def collect_files(base_dir: Path) -> set[str]:
        """Collect all files under base_dir, skipping build artifacts and symlinks.

        Uses git ls-files when available (fast), falls back to os.walk.
        """
        files = set()

        # Try git ls-files first (much faster for git repos)
        git_dir = base_dir / ".git"
        if git_dir.exists():
            git_files = tracked_files(base_dir)
            for rel_path in git_files:
                if not should_skip(Path(rel_path)):
                    files.add(rel_path)
            return files

        # Fallback: os.walk with safety filters
        for root, dirs, filenames in os.walk(base_dir):
            # Filter directories IN PLACE: skip build dirs + symlinks (security)
            dirs[:] = [
                d for d in dirs
                if d not in MERGE_SKIP_DIRS
                and not d.endswith(".egg-info")
                and not os.path.islink(os.path.join(root, d))
            ]

            for f in filenames:
                full_path = Path(root) / f

                # Skip symlinks (security: avoid following links outside tree)
                if full_path.is_symlink():
                    continue

                rel_path = full_path.relative_to(base_dir)

                # Skip if any part of path matches skip patterns
                if should_skip(rel_path):
                    continue

                files.add(str(rel_path))
        return files

    # Get all files in source
    source_files = collect_files(source_dir)

    # Get all files in target
    target_files = collect_files(target_dir)

    # Classify each file
    all_files = source_files | target_files

    for rel_path in sorted(all_files):
        src_path = source_dir / rel_path
        tgt_path = target_dir / rel_path

        in_source = rel_path in source_files
        in_target = rel_path in target_files

        if in_source and not in_target:
            # Unique to source - ADD
            changes.append(FileChange(
                path=rel_path,
                status="A",
                source_path=src_path,
                target_path=tgt_path,
                risk_level="SAFE",
            ))
        elif in_target and not in_source:
            # Unique to target - KEEP (no action needed)
            changes.append(FileChange(
                path=rel_path,
                status="K",
                source_path=None,
                target_path=tgt_path,
                risk_level="SAFE",
            ))
        else:
            # In both - check if same or different
            try:
                if filecmp.cmp(src_path, tgt_path, shallow=False):
                    # Identical
                    changes.append(FileChange(
                        path=rel_path,
                        status="S",
                        source_path=src_path,
                        target_path=tgt_path,
                        risk_level="SAFE",
                    ))
                else:
                    # Different - CONFLICT
                    changes.append(FileChange(
                        path=rel_path,
                        status="M",
                        source_path=src_path,
                        target_path=tgt_path,
                        risk_level="CONFLICT",
                        risk_reasons=["File differs between source and target"],
                        needs_manual=True,
                    ))
            except (OSError, IOError) as e:
                changes.append(FileChange(
                    path=rel_path,
                    status="M",
                    source_path=src_path,
                    target_path=tgt_path,
                    risk_level="CONFLICT",
                    risk_reasons=[f"Could not compare: {e}"],
                    needs_manual=True,
                ))

    return changes


def classify_change(change: FileChange, findings: list[dict]) -> FileChange:
    """Further classify a change based on review findings.

    Rules:
    - HIGH severity finding on file → RISKY
    - File already marked CONFLICT → stays CONFLICT
    - Add of new file with no HIGH findings → SAFE
    """
    file_findings = [f for f in findings if f.get("file") == change.path]
    change.findings = file_findings

    high_findings = [f for f in file_findings if f.get("severity") == "HIGH"]
    med_findings = [f for f in file_findings if f.get("severity") == "MEDIUM"]

    # Don't downgrade CONFLICT
    if change.risk_level == "CONFLICT":
        if high_findings:
            change.risk_reasons.extend([f.get("issue", "HIGH severity issue") for f in high_findings])
        return change

    if high_findings:
        change.risk_level = "RISKY"
        change.risk_reasons = [f.get("issue", "HIGH severity issue") for f in high_findings]
        change.needs_manual = True
    elif len(med_findings) > 2:
        change.risk_level = "RISKY"
        change.risk_reasons = [f"Multiple medium issues ({len(med_findings)})"]
        change.needs_manual = True

    return change


def apply_change(change: FileChange) -> FileChange:
    """Apply a single file change based on its status.

    Status meanings:
    - A: Add from source to target (copy)
    - K: Keep target (no action)
    - S: Same in both (no action)
    - M: Modified/conflict (skip unless forced)
    """
    try:
        if change.status == "A":
            # Add: copy from source to target
            if change.source_path and change.source_path.exists():
                change.target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(change.source_path, change.target_path)
                change.applied = True
            else:
                change.error = f"Source file not found: {change.source_path}"

        elif change.status in ("K", "S"):
            # Keep/Same: no action needed
            change.applied = True

        elif change.status == "M":
            # Modified: conflict - don't auto-apply
            change.needs_manual = True
            change.applied = False

    except Exception as e:
        change.error = str(e)
        change.applied = False

    return change


def create_conflict_file(change: FileChange, output_dir: Path) -> Path | None:
    """Create a conflict file showing both versions for manual merge.

    Returns path to conflict file.
    """
    if change.status != "M" or not change.source_path or not change.target_path:
        return None

    conflict_dir = output_dir / ".capseal" / "conflicts"
    conflict_dir.mkdir(parents=True, exist_ok=True)

    # Create conflict marker file
    safe_name = change.path.replace("/", "_").replace("\\", "_")
    conflict_file = conflict_dir / f"{safe_name}.conflict"

    try:
        source_content = change.source_path.read_text() if change.source_path.exists() else "(file not found)"
        target_content = change.target_path.read_text() if change.target_path.exists() else "(file not found)"

        conflict_content = f"""{'='*60}
CONFLICT: {change.path}
{'='*60}

<<<<<<< SOURCE ({change.source_path})
{source_content}
=======
{target_content}
>>>>>>> TARGET ({change.target_path})

RESOLUTION INSTRUCTIONS:
1. Review both versions above
2. Create the merged version
3. Save to: {change.target_path}
4. Delete this conflict file when resolved
"""
        conflict_file.write_text(conflict_content)
        return conflict_file
    except Exception:
        return None


def run_tests(target_dir: Path, test_cmd: str | None = None) -> tuple[bool, str]:
    """Run tests in the target directory.

    Returns (passed, output).
    """
    if not test_cmd:
        # Auto-detect test command
        if (target_dir / "pytest.ini").exists() or (target_dir / "pyproject.toml").exists():
            test_cmd = "python -m pytest -x --tb=short"
        elif (target_dir / "package.json").exists():
            test_cmd = "npm test"
        elif (target_dir / "Cargo.toml").exists():
            test_cmd = "cargo test"
        else:
            return None, "No test command detected"

    try:
        result = subprocess.run(
            test_cmd,
            shell=True,
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output[:5000]  # Truncate
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 5 minutes"
    except Exception as e:
        return False, f"Test error: {e}"


def generate_merge_receipt(
    context_hash: str,
    review_hash: str | None,
    changes: list[FileChange],
    test_result: tuple[bool, str] | None,
) -> dict:
    """Generate a receipt for the merge operation."""
    applied = [c for c in changes if c.applied]
    conflicts = [c for c in changes if c.needs_manual]

    receipt = {
        "version": "1.0",
        "type": "merge_receipt",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context_hash": context_hash,
        "review_hash": review_hash,
        "summary": {
            "total_changes": len(changes),
            "applied": len(applied),
            "conflicts": len(conflicts),
            "test_passed": test_result[0] if test_result else None,
        },
        "changes": [
            {
                "path": c.path,
                "status": c.status,
                "risk_level": c.risk_level,
                "applied": c.applied,
                "needs_manual": c.needs_manual,
                "findings_count": len(c.findings),
            }
            for c in changes
        ],
        "conflicts": [
            {
                "path": c.path,
                "reasons": c.risk_reasons,
                "findings": c.findings,
            }
            for c in conflicts
        ],
    }

    # Compute receipt hash
    content = json.dumps(receipt, sort_keys=True)
    receipt["receipt_hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]

    return receipt


# CLI Command
@click.command("merge")
@click.option("--source", "-s", type=click.Path(exists=True), required=True,
              help="Source directory (merge FROM - adds unique files)")
@click.option("--target", "-t", type=click.Path(exists=True), required=True,
              help="Target directory (merge INTO - keeps its unique files)")
@click.option("--dry-run", is_flag=True,
              help="Show what would be done without applying")
@click.option("--no-test", is_flag=True,
              help="Skip running tests after merge")
@click.option("--force", is_flag=True,
              help="Apply even conflicting files (overwrites target with source)")
@click.option("--test-cmd", default=None,
              help="Custom test command to run")
def merge_command(source, target, dry_run, no_test, force, test_cmd):
    """Bi-directional merge: combine both repos intelligently.

    This is a TRUE MERGE that keeps the best of both:
    - Files unique to SOURCE → copied to target (ADD)
    - Files unique to TARGET → kept as-is (KEEP)
    - Files identical in both → no change (SAME)
    - Files that DIFFER → flagged as CONFLICT (manual review)

    Examples:
        capseal merge -s ~/CapsuleTech -t ~/BEF-main --dry-run
        capseal merge -s ~/fork -t ~/main
        capseal merge -s ~/new -t ~/old --force  # overwrites conflicts
    """
    source_dir = Path(source).resolve()
    target_dir = Path(target).resolve()

    click.echo(f"\n{'='*60}")
    click.echo("CAPSEAL BI-DIRECTIONAL MERGE")
    click.echo(f"{'='*60}")
    click.echo(f"Source: {source_dir} (merge FROM)")
    click.echo(f"Target: {target_dir} (merge INTO)")
    click.echo(f"Mode:   {'DRY RUN' if dry_run else 'APPLY'}")
    click.echo(f"{'='*60}\n")

    # Step 1: Scan both directories
    click.echo("[1/4] Scanning directories...")
    changes = scan_directories(source_dir, target_dir)

    # Categorize
    adds = [c for c in changes if c.status == "A"]
    keeps = [c for c in changes if c.status == "K"]
    same = [c for c in changes if c.status == "S"]
    conflicts = [c for c in changes if c.status == "M"]

    click.echo(f"  Files unique to source (ADD):    {len(adds)}")
    click.echo(f"  Files unique to target (KEEP):   {len(keeps)}")
    click.echo(f"  Files identical (SAME):          {len(same)}")
    click.echo(f"  Files that differ (CONFLICT):    {len(conflicts)}")
    click.echo(f"  Total files scanned:             {len(changes)}")

    # Step 2: Load review findings if available
    click.echo("\n[2/4] Loading review synthesis...")
    findings = []
    review_hash = None

    try:
        from bef_zk.capsule.mcp_server import RESULTS_DIR
        if RESULTS_DIR.exists():
            for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
                try:
                    data = json.loads(f.read_text())
                    if data.get("result_type") == "review_synthesis":
                        content = json.loads(data.get("content", "{}"))
                        findings = content.get("findings", [])
                        review_hash = content.get("findings_hash")
                        click.echo(f"  ✓ Loaded review: {len(findings)} findings")
                        break
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Skip malformed review files, log in debug mode
                    if os.environ.get("CAPSEAL_DEBUG"):
                        click.echo(f"  [debug] Skipping {f.name}: {e}", err=True)
                    continue
                except OSError as e:
                    # File read error (permissions, etc)
                    if os.environ.get("CAPSEAL_DEBUG"):
                        click.echo(f"  [debug] Cannot read {f.name}: {e}", err=True)
                    continue
    except ImportError:
        pass

    if findings:
        # Apply review findings to classify further
        for change in changes:
            change = classify_change(change, findings)
    else:
        click.echo("  ⚠ No review found")

    if dry_run:
        click.echo(f"\n[DRY RUN] Would do the following:\n")

        if adds:
            click.echo(f"\033[92mADD ({len(adds)} files from source):\033[0m")
            for c in adds[:20]:
                click.echo(f"  + {c.path}")
            if len(adds) > 20:
                click.echo(f"  ... and {len(adds) - 20} more")

        if keeps:
            click.echo(f"\n\033[94mKEEP ({len(keeps)} files unique to target):\033[0m")
            for c in keeps[:10]:
                click.echo(f"  = {c.path}")
            if len(keeps) > 10:
                click.echo(f"  ... and {len(keeps) - 10} more")

        if same:
            click.echo(f"\n\033[90mSAME ({len(same)} files identical - no action):\033[0m")

        if conflicts:
            click.echo(f"\n\033[93mCONFLICT ({len(conflicts)} files differ - need manual merge):\033[0m")
            for c in conflicts:
                click.echo(f"  ! {c.path}")
                for reason in c.risk_reasons:
                    click.echo(f"    → {reason}")

        click.echo(f"\n[DRY RUN] No changes applied.")
        return

    # Step 3: Apply changes
    click.echo("\n[3/4] Applying changes...")

    # Apply ADDs (safe)
    for change in adds:
        change = apply_change(change)
        status = "✓" if change.applied else "✗"
        click.echo(f"  {status} ADD {change.path}")
        if change.error:
            click.echo(f"      Error: {change.error}")

    # KEEPs and SAMEs need no action
    for change in keeps + same:
        change.applied = True

    # Handle CONFLICTs
    if conflicts:
        if force:
            click.echo(f"\n  Forcing {len(conflicts)} conflicts (overwriting target)...")
            for change in conflicts:
                # Force copy from source
                try:
                    shutil.copy2(change.source_path, change.target_path)
                    change.applied = True
                    click.echo(f"  ✓ FORCE {change.path}")
                except Exception as e:
                    change.error = str(e)
                    click.echo(f"  ✗ FORCE {change.path}: {e}")
        else:
            click.echo(f"\n  Creating conflict files for {len(conflicts)} conflicts...")
            for change in conflicts:
                conflict_file = create_conflict_file(change, target_dir)
                if conflict_file:
                    click.echo(f"  ! {change.path} → .capseal/conflicts/")

    applied_count = len([c for c in changes if c.applied])
    click.echo(f"\n  Applied: {applied_count}/{len(changes)}")

    # Step 4: Run tests
    test_result = None
    if not no_test:
        click.echo("\n[4/4] Running tests...")
        test_result = run_tests(target_dir, test_cmd)
        if test_result[0] is None:
            click.echo(f"  ⚠ {test_result[1]}")
        elif test_result[0]:
            click.echo("  ✓ Tests passed!")
        else:
            click.echo("  ✗ Tests failed!")
            click.echo(f"\n{test_result[1][:1000]}")
    else:
        click.echo("\n[4/4] Skipping tests (--no-test)")

    # Generate receipt
    # Use directory names as context hash since we're doing bi-directional scan
    context_hash = hashlib.sha256(f"{source_dir}:{target_dir}".encode()).hexdigest()[:8]
    receipt = generate_merge_receipt(context_hash, review_hash, changes, test_result)

    # Save receipt
    receipt_path = target_dir / ".capseal" / "merge_receipts"
    receipt_path.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_path / f"merge_{receipt['receipt_hash']}.json"
    receipt_file.write_text(json.dumps(receipt, indent=2))

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("MERGE COMPLETE")
    click.echo(f"{'='*60}")
    click.echo(f"Added:     {len(adds)} files from source")
    click.echo(f"Kept:      {len(keeps)} files unique to target")
    click.echo(f"Same:      {len(same)} files identical")
    click.echo(f"Conflicts: {len(conflicts)} need manual merge")
    click.echo(f"Tests:     {'PASSED' if test_result and test_result[0] else 'FAILED' if test_result else 'SKIPPED'}")
    click.echo(f"Receipt:   {receipt['receipt_hash']}")
    click.echo(f"{'='*60}")

    if conflicts and not force:
        click.echo("\n\033[93mCONFLICT FILES (need manual merge):\033[0m")
        click.echo(f"  Conflict files saved to: {target_dir}/.capseal/conflicts/")
        for c in conflicts:
            click.echo(f"  • {c.path}")
        click.echo(f"\n  To resolve: edit the files, then delete the .conflict markers")
        click.echo(f"\n  TIP: Run 'capseal conflict-bundle -s {source_dir} -t {target_dir}'")
        click.echo(f"       to create a minimal bundle for Greptile review")


@click.command("conflict-bundle")
@click.option("--source", "-s", type=click.Path(exists=True), required=True,
              help="Source directory")
@click.option("--target", "-t", type=click.Path(exists=True), required=True,
              help="Target directory")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output bundle directory (default: /tmp/conflict_bundle)")
@click.option("--push-greptile", is_flag=True,
              help="Push the bundle to Greptile ephemeral automatically")
def conflict_bundle_command(source, target, output, push_greptile):
    """Create minimal bundle with ONLY conflict files for fast Greptile indexing.

    Instead of pushing 4000+ files to Greptile (slow/fails), this extracts
    just the ~86 files that actually differ between repos. Perfect for
    AI-assisted merge review.

    Structure:
      bundle/
        source/        ← conflict files from source repo
        target/        ← conflict files from target repo
        manifest.json  ← metadata about the conflicts

    Examples:
        capseal conflict-bundle -s ~/CapsuleTech -t ~/BEF-main
        capseal conflict-bundle -s ~/fork -t ~/main --push-greptile
        capseal conflict-bundle -s ~/fork -t ~/main -o ~/review_bundle
    """
    source_dir = Path(source).resolve()
    target_dir = Path(target).resolve()
    output_dir = Path(output) if output else Path("/tmp/conflict_bundle")

    click.echo(f"\n{'='*60}")
    click.echo("CAPSEAL CONFLICT BUNDLE")
    click.echo(f"{'='*60}")
    click.echo(f"Source: {source_dir}")
    click.echo(f"Target: {target_dir}")
    click.echo(f"Output: {output_dir}")

    # Step 1: Scan for conflicts
    click.echo("\n[1/3] Scanning for conflicts...")
    changes = scan_directories(source_dir, target_dir)

    conflicts = [c for c in changes if c.status == "M"]
    click.echo(f"  Found {len(conflicts)} conflicting files")

    if not conflicts:
        click.echo("  No conflicts found! Nothing to bundle.")
        return

    # Step 2: Create bundle directory
    click.echo("\n[2/3] Creating bundle...")

    # Clean and create output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "source").mkdir()
    (output_dir / "target").mkdir()

    # Copy conflict files
    copied = 0
    manifest_files = []
    for change in conflicts:
        rel_path = change.path

        # Create subdirs
        (output_dir / "source" / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "target" / rel_path).parent.mkdir(parents=True, exist_ok=True)

        # Copy source version
        try:
            if change.source_path and change.source_path.exists():
                shutil.copy2(change.source_path, output_dir / "source" / rel_path)
        except Exception as e:
            click.echo(f"  ⚠ Source copy failed: {rel_path}: {e}")

        # Copy target version
        try:
            if change.target_path and change.target_path.exists():
                shutil.copy2(change.target_path, output_dir / "target" / rel_path)
        except Exception as e:
            click.echo(f"  ⚠ Target copy failed: {rel_path}: {e}")

        copied += 1
        manifest_files.append({
            "path": rel_path,
            "source_exists": change.source_path.exists() if change.source_path else False,
            "target_exists": change.target_path.exists() if change.target_path else False,
        })

        if copied <= 10:
            click.echo(f"  + {rel_path}")
    if copied > 10:
        click.echo(f"  ... and {copied - 10} more")

    # Create manifest
    manifest = {
        "type": "conflict_bundle",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_repo": str(source_dir),
        "target_repo": str(target_dir),
        "conflict_count": len(conflicts),
        "files": manifest_files,
        "review_prompt": """You are reviewing files that DIFFER between two repo versions.

For each file in this bundle:
- source/ contains the version from the source repo
- target/ contains the version from the target repo

Analyze:
1. What changed between versions?
2. Are there any security concerns?
3. Which version should be kept, or should they be merged?
4. Any breaking changes or regressions?

Focus on: security, correctness, API compatibility.""",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "README.md").write_text(f"""# Conflict Bundle

This bundle contains {len(conflicts)} files that differ between:
- **Source**: {source_dir}
- **Target**: {target_dir}

## Structure

```
source/    ← files from source repo
target/    ← files from target repo
manifest.json
```

## Review Instructions

Compare each file pair and determine:
1. Which version is correct?
2. Should they be merged?
3. Are there security concerns?

Generated by `capseal conflict-bundle` at {time.strftime("%Y-%m-%d %H:%M")}
""")

    click.echo(f"\n  Bundle created: {output_dir}")
    click.echo(f"  Total files: {copied * 2} ({copied} pairs)")

    # Step 3: Push to Greptile if requested
    if push_greptile:
        click.echo("\n[3/3] Pushing to Greptile ephemeral...")

        try:
            # Initialize git repo in bundle
            subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=output_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Conflict bundle for review"],
                cwd=output_dir, capture_output=True
            )

            # Push to Greptile
            result = subprocess.run(
                ["greptile", "ephemeral", "--path", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            click.echo(redact_secrets(result.stdout))
            if result.returncode != 0:
                click.echo(f"  ⚠ Greptile error: {redact_secrets(result.stderr)}")
        except FileNotFoundError:
            click.echo("  ⚠ Greptile CLI not found. Install with: pip install greptile")
        except subprocess.TimeoutExpired:
            click.echo("  ⚠ Greptile timed out")
        except Exception as e:
            click.echo(f"  ⚠ Error: {e}")
    else:
        click.echo("\n[3/3] Skipping Greptile push (use --push-greptile to enable)")

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("BUNDLE READY")
    click.echo(f"{'='*60}")
    click.echo(f"Conflicts: {len(conflicts)} files")
    click.echo(f"Bundle:    {output_dir}")
    click.echo(f"Size:      {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) // 1024} KB")
    click.echo(f"\nTo review with Greptile:")
    click.echo(f"  greptile ephemeral --path {output_dir}")
    click.echo(f"\nOr review manually:")
    click.echo(f"  diff {output_dir}/source {output_dir}/target")


@click.command("merge-apply")
@click.argument("merged_dir", type=click.Path(exists=True))
@click.argument("target_dir", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be applied without doing it")
@click.option("--backup", is_flag=True, help="Create backups before overwriting")
def merge_apply_command(merged_dir, target_dir, dry_run, backup):
    """Apply merged files from agent output to target repo.

    Takes the output from the merge pipeline (/tmp/merged_output/)
    and applies each merged file to the correct location in the target repo.

    File naming convention: path__to__file.py → path/to/file.py

    Examples:
        capseal merge-apply /tmp/merged_output ~/BEF-main
        capseal merge-apply /tmp/merged_output ~/BEF-main --dry-run
        capseal merge-apply /tmp/merged_output ~/BEF-main --backup
    """
    merged_path = Path(merged_dir).resolve()
    target_path = Path(target_dir).resolve()

    click.echo(f"\n{'='*60}")
    click.echo("CAPSEAL MERGE APPLY")
    click.echo(f"{'='*60}")
    click.echo(f"Merged files: {merged_path}")
    click.echo(f"Target repo:  {target_path}")
    click.echo(f"Mode:         {'DRY RUN' if dry_run else 'APPLY'}")
    if backup:
        click.echo(f"Backup:       enabled")
    click.echo()

    # Find all merged files (exclude manifest.json)
    merged_files = [f for f in merged_path.iterdir()
                    if f.is_file() and f.name != 'manifest.json']

    if not merged_files:
        click.echo("No merged files found!")
        return

    click.echo(f"Found {len(merged_files)} merged files")
    click.echo()

    applied = 0
    skipped = 0
    errors = []

    for merged_file in sorted(merged_files):
        # Convert filename back to path: foo__bar__baz.py → foo/bar/baz.py
        # Handle special case: ____init__ → /__init__ (Python dunder files)
        rel_path = merged_file.name
        # First handle Python dunder filenames (4 underscores = separator + dunder)
        rel_path = rel_path.replace('____init__', '/##INIT##')
        rel_path = rel_path.replace('____main__', '/##MAIN##')
        # Convert remaining path separators
        rel_path = rel_path.replace('__', '/')
        # Restore dunder names
        rel_path = rel_path.replace('##INIT##', '__init__')
        rel_path = rel_path.replace('##MAIN##', '__main__')
        target_file = target_path / rel_path

        if dry_run:
            exists = "overwrite" if target_file.exists() else "create"
            click.echo(f"  [{exists}] {rel_path}")
            applied += 1
            continue

        try:
            # Create backup if requested
            if backup and target_file.exists():
                backup_file = target_file.with_suffix(target_file.suffix + '.bak')
                shutil.copy2(target_file, backup_file)

            # Ensure parent directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy merged file to target
            shutil.copy2(merged_file, target_file)
            click.echo(f"  ✓ {rel_path}")
            applied += 1

        except Exception as e:
            click.echo(f"  ✗ {rel_path}: {e}")
            errors.append((rel_path, str(e)))

    click.echo()
    click.echo(f"{'='*60}")
    if dry_run:
        click.echo(f"DRY RUN: Would apply {applied} files")
    else:
        click.echo(f"Applied: {applied} files")
        if errors:
            click.echo(f"Errors:  {len(errors)}")
            for path, err in errors:
                click.echo(f"  - {path}: {err}")
    click.echo(f"{'='*60}")
