"""Minimal Working Set (MWS) builder - deterministic context pack for agents.

This module builds a compact, focused context pack that contains:
- Changed files from the diff
- Related files via symbol references
- Necessary snippets (not full files)

The MWS enforces hard limits to control token usage:
- max_files: 25 files
- max_snippets: 80 snippets
- max_total_chars: 120,000 chars (~30-40k tokens)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Hard limits to prevent token explosion
MWS_MAX_FILES = int(os.environ.get("MWS_MAX_FILES", "25"))
MWS_MAX_SNIPPETS = int(os.environ.get("MWS_MAX_SNIPPETS", "80"))
MWS_MAX_TOTAL_CHARS = int(os.environ.get("MWS_MAX_TOTAL_CHARS", "120000"))
MWS_SNIPPET_CONTEXT_LINES = 30  # Lines before/after a reference


@dataclass
class Snippet:
    """A code snippet with location info."""
    file: str
    start_line: int
    end_line: int
    text: str
    reason: str  # Why included: "diff", "reference", "import"

    def char_count(self) -> int:
        return len(self.text)


@dataclass
class MWSConfig:
    """Configuration for MWS building."""
    max_files: int = MWS_MAX_FILES
    max_snippets: int = MWS_MAX_SNIPPETS
    max_total_chars: int = MWS_MAX_TOTAL_CHARS
    snippet_context_lines: int = MWS_SNIPPET_CONTEXT_LINES
    include_imports: bool = True
    include_tests: bool = False


@dataclass
class MWS:
    """Minimal Working Set - compact context for agents."""
    diff_summary: str
    files: list[str]
    snippets: list[Snippet]
    symbols: list[str]
    constraints: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(s.char_count() for s in self.snippets)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def snippet_count(self) -> int:
        return len(self.snippets)

    def to_dict(self) -> dict:
        return {
            "diff_summary": self.diff_summary,
            "files": self.files,
            "snippets": [
                {
                    "file": s.file,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "text": s.text,
                    "reason": s.reason,
                }
                for s in self.snippets
            ],
            "symbols": self.symbols,
            "constraints": self.constraints,
            "metadata": {
                **self.metadata,
                "total_chars": self.total_chars,
                "file_count": self.file_count,
                "snippet_count": self.snippet_count,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "MWS":
        return cls(
            diff_summary=data.get("diff_summary", ""),
            files=data.get("files", []),
            snippets=[
                Snippet(
                    file=s["file"],
                    start_line=s["start_line"],
                    end_line=s["end_line"],
                    text=s["text"],
                    reason=s.get("reason", "unknown"),
                )
                for s in data.get("snippets", [])
            ],
            symbols=data.get("symbols", []),
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
        )


class MWSBuilder:
    """Builds Minimal Working Sets from diffs.

    Algorithm:
    1. Start with changed files from diff
    2. Parse diff for touched identifiers
    3. Use local ripgrep to find references
    4. Extract only necessary ranges
    5. Enforce hard limits
    """

    def __init__(self, repo_path: str | Path, config: MWSConfig | None = None):
        self.repo_path = Path(repo_path)
        self.config = config or MWSConfig()

    def build(
        self,
        base_ref: str = "HEAD~1",
        head_ref: str = "HEAD",
        constraints: list[str] | None = None,
    ) -> MWS:
        """Build MWS from a diff.

        Args:
            base_ref: Base git reference
            head_ref: Head git reference
            constraints: Optional constraints for agents

        Returns:
            MWS object
        """
        from .git_utils import (
            diff_files,
            extract_symbols_from_diff,
            find_references_local,
        )

        # Step 1: Get changed files
        changed_files = diff_files(self.repo_path, base_ref, head_ref)

        # Step 2: Get the unified diff
        diff_text = self._get_unified_diff(base_ref, head_ref)

        # Step 3: Extract symbols from diff
        symbols = extract_symbols_from_diff(diff_text)

        # Step 4: Find local references
        references = find_references_local(
            self.repo_path,
            symbols,
            max_files=self.config.max_files,
        )

        # Step 5: Collect related files
        related_files = set()
        for symbol, matches in references.items():
            for m in matches:
                related_files.add(m["file"])

        # Step 6: Build file list (changed first, then related)
        files = list(changed_files)
        for f in related_files:
            if f not in files and len(files) < self.config.max_files:
                files.append(f)

        # Step 7: Extract snippets
        snippets = []

        # Add diff hunks as snippets
        for hunk in self._parse_diff_hunks(diff_text):
            if len(snippets) >= self.config.max_snippets:
                break
            snippets.append(hunk)

        # Add reference contexts
        total_chars = sum(s.char_count() for s in snippets)
        for symbol, matches in references.items():
            if len(snippets) >= self.config.max_snippets:
                break
            if total_chars >= self.config.max_total_chars:
                break

            for m in matches[:3]:  # Limit matches per symbol
                snippet = self._extract_snippet(
                    m["file"],
                    m["line"],
                    reason=f"reference:{symbol}",
                )
                if snippet:
                    if total_chars + snippet.char_count() > self.config.max_total_chars:
                        continue
                    snippets.append(snippet)
                    total_chars += snippet.char_count()

        # Step 8: Build diff summary
        diff_summary = self._build_diff_summary(changed_files, diff_text)

        return MWS(
            diff_summary=diff_summary,
            files=files[:self.config.max_files],
            snippets=snippets[:self.config.max_snippets],
            symbols=symbols,
            constraints=constraints or [
                "Do not propose new systems; extend existing primitives",
                "Focus on the specific changes in this diff",
                "Cite file paths and line numbers in findings",
            ],
            metadata={
                "repo_path": str(self.repo_path),
                "base_ref": base_ref,
                "head_ref": head_ref,
                "mws_hash": self._compute_hash(diff_text),
            },
        )

    def build_from_context(self, context: dict) -> MWS:
        """Build MWS from an existing context checkpoint.

        Args:
            context: Context dict from context.py

        Returns:
            MWS object
        """
        files = [f["path"] for f in context.get("files", [])]
        diffs = context.get("diffs", [])

        # Build snippets from diffs
        snippets = []
        total_chars = 0

        for diff_obj in diffs:
            if len(snippets) >= self.config.max_snippets:
                break
            if total_chars >= self.config.max_total_chars:
                break

            filepath = diff_obj.get("path", "unknown")
            patch = diff_obj.get("patch", "")

            if patch:
                # Truncate if needed
                if total_chars + len(patch) > self.config.max_total_chars:
                    patch = patch[:self.config.max_total_chars - total_chars]

                snippets.append(Snippet(
                    file=filepath,
                    start_line=0,
                    end_line=0,
                    text=patch,
                    reason="diff",
                ))
                total_chars += len(patch)

        # Add working tree changes
        working_tree = context.get("working_tree") or {}
        for f in working_tree.get("files", []):
            if len(snippets) >= self.config.max_snippets:
                break
            if total_chars >= self.config.max_total_chars:
                break

            diff_text = f.get("diff", "")
            if diff_text:
                if total_chars + len(diff_text) > self.config.max_total_chars:
                    diff_text = diff_text[:self.config.max_total_chars - total_chars]

                snippets.append(Snippet(
                    file=f.get("path", "unknown"),
                    start_line=0,
                    end_line=0,
                    text=diff_text,
                    reason="uncommitted",
                ))
                total_chars += len(diff_text)

        # Extract symbols
        all_diffs = "\n".join(d.get("patch", "") for d in diffs)
        from .git_utils import extract_symbols_from_diff
        symbols = extract_symbols_from_diff(all_diffs)

        summary = context.get("summary", {})
        diff_summary = f"Comparison: {summary.get('comparison', 'unknown')}\n"
        diff_summary += f"Files changed: {summary.get('total_files', 0)}\n"
        diff_summary += f"Uncommitted: {summary.get('uncommitted_files', 0)}"

        return MWS(
            diff_summary=diff_summary,
            files=files[:self.config.max_files],
            snippets=snippets[:self.config.max_snippets],
            symbols=symbols,
            constraints=[
                "Do not propose new systems; extend existing primitives",
                "Focus on the specific changes in this diff",
                "Cite file paths and line numbers in findings",
            ],
            metadata={
                "context_id": context.get("checkpoint_id", ""),
                "mws_hash": hashlib.sha256(all_diffs.encode()).hexdigest()[:16],
            },
        )

    def _get_unified_diff(self, base_ref: str, head_ref: str) -> str:
        """Get unified diff between two refs."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_ref}..{head_ref}"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    def _parse_diff_hunks(self, diff_text: str) -> list[Snippet]:
        """Parse diff into individual hunks."""
        import re

        snippets = []
        current_file = ""
        current_hunk = []
        current_start = 0

        for line in diff_text.split("\n"):
            # New file header
            if line.startswith("+++ b/"):
                # Save previous hunk
                if current_hunk and current_file:
                    snippets.append(Snippet(
                        file=current_file,
                        start_line=current_start,
                        end_line=current_start + len(current_hunk),
                        text="\n".join(current_hunk),
                        reason="diff",
                    ))
                current_file = line[6:]
                current_hunk = []

            # Hunk header
            elif line.startswith("@@"):
                # Save previous hunk
                if current_hunk and current_file:
                    snippets.append(Snippet(
                        file=current_file,
                        start_line=current_start,
                        end_line=current_start + len(current_hunk),
                        text="\n".join(current_hunk),
                        reason="diff",
                    ))
                current_hunk = [line]

                # Extract line number
                match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
                if match:
                    current_start = int(match.group(1))

            # Diff content
            elif current_file and (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                current_hunk.append(line)

        # Save last hunk
        if current_hunk and current_file:
            snippets.append(Snippet(
                file=current_file,
                start_line=current_start,
                end_line=current_start + len(current_hunk),
                text="\n".join(current_hunk),
                reason="diff",
            ))

        return snippets

    def _extract_snippet(
        self,
        filepath: str,
        line: int,
        reason: str,
    ) -> Snippet | None:
        """Extract a snippet around a specific line."""
        try:
            full_path = self.repo_path / filepath
            if not full_path.exists():
                return None

            content = full_path.read_text(errors="ignore")
            lines = content.split("\n")

            start = max(0, line - self.config.snippet_context_lines)
            end = min(len(lines), line + self.config.snippet_context_lines)

            snippet_text = "\n".join(lines[start:end])

            # Truncate if too long
            if len(snippet_text) > 3000:
                snippet_text = snippet_text[:3000] + "\n... (truncated)"

            return Snippet(
                file=filepath,
                start_line=start + 1,
                end_line=end,
                text=snippet_text,
                reason=reason,
            )
        except Exception:
            return None

    def _build_diff_summary(self, changed_files: list[str], diff_text: str) -> str:
        """Build a machine-readable diff summary."""
        lines_added = diff_text.count("\n+") - diff_text.count("\n+++")
        lines_removed = diff_text.count("\n-") - diff_text.count("\n---")

        summary = f"Files changed: {len(changed_files)}\n"
        summary += f"Lines added: {lines_added}\n"
        summary += f"Lines removed: {lines_removed}\n"
        summary += f"\nChanged files:\n"
        for f in changed_files[:20]:
            summary += f"  - {f}\n"
        if len(changed_files) > 20:
            summary += f"  ... and {len(changed_files) - 20} more\n"

        return summary

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def build_mws(
    repo_path: str | Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    config: MWSConfig | None = None,
) -> MWS:
    """Convenience function to build MWS.

    Args:
        repo_path: Path to git repository
        base_ref: Base git reference
        head_ref: Head git reference
        config: Optional MWS configuration

    Returns:
        MWS object
    """
    builder = MWSBuilder(repo_path, config)
    return builder.build(base_ref, head_ref)


def build_mws_from_context(context: dict, config: MWSConfig | None = None) -> MWS:
    """Build MWS from context checkpoint.

    Args:
        context: Context dict from context.py
        config: Optional MWS configuration

    Returns:
        MWS object
    """
    repo_path = context.get("summary", {}).get("repo", ".")
    builder = MWSBuilder(repo_path, config)
    return builder.build_from_context(context)
