"""
Deterministic Checkers - The Judges

These are first-class vertices in the verification DAG.
Each checker:
1. Takes a Claim and file content
2. Returns a Verdict and Witness
3. Is deterministic given the same inputs

The LLM can propose; these are the judges.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .claims import (
    Claim, ClaimType, Verdict, Witness, CheckerInfo, CheckerRegistry,
    CHECKER_REGISTRY,
)


# ─────────────────────────────────────────────────────────────────
# Semgrep Checker
# ─────────────────────────────────────────────────────────────────

def semgrep_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Check security claims using Semgrep.

    Supports:
    - NO_SQL_INJECTION
    - NO_SHELL_INJECTION
    - NO_XSS
    - NO_HARDCODED_SECRETS

    Returns:
    - PASS: no relevant findings
    - FAIL: found concrete counterexample (relevant finding)
    - NOT_APPLICABLE: file doesn't contain relevant sinks (returned as PASS with note)
    """
    # Map claim types to semgrep rule patterns that indicate this issue
    # Note: We run auto rules but filter results to relevant rule_ids
    rule_patterns = {
        ClaimType.NO_SQL_INJECTION: [
            "sql", "sqli", "cursor.execute", "raw-query", "database",
            "sqlalchemy", "django.db", "psycopg", "mysql", "sqlite"
        ],
        ClaimType.NO_SHELL_INJECTION: [
            "subprocess", "shell-true", "os.system", "os.popen",
            "command-injection", "shell-injection"
        ],
        ClaimType.NO_XSS: [
            "xss", "cross-site", "innerHTML", "document.write",
            "eval", "dangerouslySetInnerHTML"
        ],
        ClaimType.NO_HARDCODED_SECRETS: [
            "secret", "credential", "password", "api-key", "apikey",
            "token", "private-key", "auth"
        ],
    }

    def is_safe_parameterized_query(file_content: str, finding_line: int) -> bool:
        """
        Check if a SQL "injection" finding is actually safe parameterized query.

        Safe pattern:
        - f-string only interpolates placeholders like ?,?,? or %s,%s
        - Actual data passed as second argument to execute()

        Example (SAFE):
            placeholders = ",".join("?" for _ in keys)
            query = f"SELECT ... WHERE key IN ({placeholders})"
            cursor.execute(query, keys)  # keys passed as params, not interpolated
        """
        lines = file_content.split('\n')
        if finding_line < 1 or finding_line > len(lines):
            return False

        # Look at context: the flagged line and a few before/after
        start = max(0, finding_line - 10)
        end = min(len(lines), finding_line + 3)
        context = '\n'.join(lines[start:end])

        # Pattern 1: placeholders variable is just "?" repeated
        # e.g., placeholders = ",".join("?" for _ in keys)
        placeholder_pattern = r'placeholders?\s*=\s*["\',]\.join\s*\(\s*["\'][\?\%s:]["\']'
        has_placeholder_builder = bool(re.search(placeholder_pattern, context, re.IGNORECASE))

        # Pattern 2: execute() has a second argument (the params)
        # e.g., execute(query, keys) or execute(query, (*keys,))
        execute_with_params = r'\.execute\s*\(\s*\w+\s*,\s*[(\[]'
        has_params = bool(re.search(execute_with_params, context))

        # Pattern 3: The f-string only interpolates placeholder-like content
        # e.g., f"... IN ({placeholders})"
        fstring_only_placeholders = r'f["\'].*\{\s*placeholders?\s*\}.*["\']'
        has_safe_fstring = bool(re.search(fstring_only_placeholders, context, re.IGNORECASE))

        # If all conditions met, this is a safe parameterized query
        if has_placeholder_builder and has_params:
            return True
        if has_safe_fstring and has_params:
            return True

        return False

    # Sink patterns - if file doesn't contain these, the obligation is N/A
    sink_patterns = {
        ClaimType.NO_SQL_INJECTION: [
            r'cursor\.execute', r'\.raw\s*\(', r'execute\s*\(',
            r'sqlalchemy', r'django\.db', r'psycopg', r'mysql\.connector',
            r'sqlite3'
        ],
        ClaimType.NO_SHELL_INJECTION: [
            r'subprocess', r'os\.system', r'os\.popen', r'shell\s*=\s*True',
            r'Popen', r'call\s*\('
        ],
        ClaimType.NO_XSS: [
            r'innerHTML', r'document\.write', r'eval\s*\(', r'dangerouslySetInnerHTML'
        ],
        ClaimType.NO_HARDCODED_SECRETS: None,  # Always applicable
    }

    if claim.claim_type not in rule_patterns:
        return Verdict.UNKNOWN, Witness(
            witness_type="semgrep_skip",
            artifact_hash="",
            artifact_inline=f"No semgrep rules for claim type: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="semgrep",
        )

    # Check if obligation is applicable (file contains relevant sinks)
    sinks = sink_patterns.get(claim.claim_type)
    if sinks is not None:
        has_relevant_sink = any(re.search(pattern, file_content, re.IGNORECASE) for pattern in sinks)
        if not has_relevant_sink:
            # N/A - no relevant sinks in file, treat as PASS with note
            return Verdict.PASS, Witness(
                witness_type="semgrep_not_applicable",
                artifact_hash=hashlib.sha256(b"no_relevant_sinks").hexdigest(),
                artifact_inline=json.dumps({
                    "status": "not_applicable",
                    "reason": f"File contains no {claim.claim_type.value} sinks",
                    "checked_patterns": sinks,
                }),
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="semgrep",
            )

    # Write file to temp
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        # Run semgrep with auto config
        result = subprocess.run(
            ["semgrep", "--config", "auto", "--json", temp_path],
            capture_output=True,
            timeout=60,
        )

        output = json.loads(result.stdout.decode()) if result.stdout else {"results": []}
        all_findings = output.get("results", [])

        # Filter to findings relevant to this claim type
        patterns = rule_patterns.get(claim.claim_type, [])
        findings = []
        for f in all_findings:
            rule_id = f.get("check_id", "").lower()
            message = f.get("extra", {}).get("message", "").lower()
            # Check if finding is relevant to this claim type
            if any(p in rule_id or p in message for p in patterns):
                findings.append(f)

        # Filter to relevant scope
        relevant_findings = []
        filtered_as_safe = []
        for f in findings:
            finding_line = f.get("start", {}).get("line", 0)

            # Scope filter
            if claim.scope.start_line and claim.scope.end_line:
                if not (claim.scope.start_line <= finding_line <= claim.scope.end_line):
                    continue

            # False positive filter for SQL injection: safe parameterized queries
            if claim.claim_type == ClaimType.NO_SQL_INJECTION:
                if is_safe_parameterized_query(file_content, finding_line):
                    filtered_as_safe.append({
                        "line": finding_line,
                        "reason": "safe parameterized query (placeholders interpolated, data passed as params)",
                    })
                    continue

            relevant_findings.append(f)

        artifact_hash = hashlib.sha256(json.dumps(relevant_findings).encode()).hexdigest()

        if not relevant_findings:
            inline_data = {
                "findings": 0,
                "total_findings": len(all_findings),
                "relevant_patterns": patterns,
            }
            if filtered_as_safe:
                inline_data["filtered_false_positives"] = filtered_as_safe
            return Verdict.PASS, Witness(
                witness_type="semgrep_output",
                artifact_hash=artifact_hash,
                artifact_inline=json.dumps(inline_data),
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="semgrep",
            )
        else:
            return Verdict.FAIL, Witness(
                witness_type="semgrep_output",
                artifact_hash=artifact_hash,
                artifact_inline=json.dumps(relevant_findings[:3]),  # First 3 findings
                counterexample=f"Found {len(relevant_findings)} issues",
                counterexample_trace=[f.get("extra", {}).get("message", "") for f in relevant_findings[:3]],
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="semgrep",
            )

    except subprocess.TimeoutExpired:
        return Verdict.TIMEOUT, None
    except Exception as e:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline=str(e),
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="semgrep",
        )
    finally:
        os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────
# AST Checker (for structural claims)
# ─────────────────────────────────────────────────────────────────

def ast_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Check structural claims using AST analysis.

    Supports:
    - ALLOWLIST_ENFORCED
    - TYPE_SAFE (basic)
    - NO_SHELL_INJECTION (shell=True detection)
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        return Verdict.ERROR, Witness(
            witness_type="ast_error",
            artifact_hash="",
            artifact_inline=f"Syntax error: {e}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    if claim.claim_type == ClaimType.ALLOWLIST_ENFORCED:
        return _check_allowlist_enforced(tree, file_content, claim)
    elif claim.claim_type == ClaimType.ALLOWLIST_CONSTANT:
        return _check_allowlist_constant(tree, file_content, claim)
    elif claim.claim_type == ClaimType.NO_SHELL_INJECTION:
        return _check_no_shell_true(tree, file_content, claim)
    else:
        return Verdict.UNKNOWN, Witness(
            witness_type="ast_skip",
            artifact_hash="",
            artifact_inline=f"No AST check for claim type: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )


def _check_allowlist_constant(tree: ast.AST, file_content: str, claim: Claim) -> tuple[Verdict, Witness | None]:
    """
    Check that allowlist/whitelist mappings are constant (no mutation sites).

    Verifies:
    - No .update() calls on the mapping
    - No [key] = value assignments
    - No .pop(), .popitem(), .clear() calls
    - Mapping is not built from user input
    - No setdefault() with non-constant values
    """
    # Find allowlist definitions
    allowlist_defs = []  # (name, line, is_local)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name_upper = target.id.upper()
                    if any(x in name_upper for x in ["ALLOW", "WHITE", "VALID", "SAFE", "PERMIT", "REGISTRY", "MAPPING", "CLASS_MODULE", "FUNC_MODULE"]):
                        # Check if it's a dict/set literal (constant)
                        is_constant = isinstance(node.value, (ast.Dict, ast.Set, ast.List, ast.Tuple))
                        allowlist_defs.append({
                            "name": target.id,
                            "line": node.lineno,
                            "is_constant_literal": is_constant,
                        })

    if not allowlist_defs:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
            artifact_inline=json.dumps({"error": "No allowlist mapping found"}),
            counterexample="No allowlist/whitelist/registry definition found",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    allowlist_names = {d["name"] for d in allowlist_defs}
    mutations = []

    # Find mutation sites
    for node in ast.walk(tree):
        # Check for .update(), .pop(), .clear(), .popitem(), .setdefault()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("update", "pop", "clear", "popitem"):
                    # Check if the object is an allowlist
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in allowlist_names:
                        mutations.append({
                            "line": node.lineno,
                            "type": f".{node.func.attr}() call",
                            "target": node.func.value.id,
                        })
                elif node.func.attr == "setdefault":
                    # setdefault with function calls or variables is suspicious
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in allowlist_names:
                        if len(node.args) > 1:
                            # Check if the default value is a constant
                            default_val = node.args[1]
                            if not isinstance(default_val, (ast.Constant, ast.Str, ast.Num)):
                                mutations.append({
                                    "line": node.lineno,
                                    "type": ".setdefault() with non-constant value",
                                    "target": node.func.value.id,
                                })

        # Check for subscript assignment: mapping[key] = value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name) and target.value.id in allowlist_names:
                        mutations.append({
                            "line": node.lineno,
                            "type": "subscript assignment [key] = value",
                            "target": target.value.id,
                        })

        # Check for del mapping[key]
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name) and target.value.id in allowlist_names:
                        mutations.append({
                            "line": node.lineno,
                            "type": "del mapping[key]",
                            "target": target.value.id,
                        })

    if mutations:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(json.dumps(mutations).encode()).hexdigest()[:16],
            artifact_inline=json.dumps({"mutations": mutations, "allowlists": list(allowlist_names)}),
            counterexample=f"Found {len(mutations)} mutation sites in allowlist mappings",
            counterexample_trace=[f"Line {m['line']}: {m['type']} on {m['target']}" for m in mutations],
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    # Check if definitions are constant literals
    non_constant = [d for d in allowlist_defs if not d["is_constant_literal"]]
    if non_constant:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
            artifact_inline=json.dumps({
                "error": "Allowlist not defined as constant literal",
                "non_constant_defs": non_constant,
            }),
            counterexample=f"Allowlist(s) {[d['name'] for d in non_constant]} not defined as literal dict/set/list",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    return Verdict.PASS, Witness(
        witness_type="ast_analysis",
        artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
        artifact_inline=json.dumps({
            "allowlists_verified": [d["name"] for d in allowlist_defs],
            "mutation_sites_found": 0,
            "all_constant_literals": True,
        }),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="ast_checker",
    )


def _check_allowlist_enforced(tree: ast.AST, file_content: str, claim: Claim) -> tuple[Verdict, Witness | None]:
    """Check that dynamic operations are guarded by allowlists.

    Returns:
    - PASS if no dangerous patterns exist (N/A case)
    - PASS if dangerous patterns are guarded by allowlists
    - FAIL if dangerous patterns exist without allowlist guards

    Smart getattr detection - only flag if:
    - Attribute name is NOT a literal string (user input)
    - Result is immediately called: getattr(obj, name)(...)
    - Target is a module/globals/builtins (not a config object)
    """
    # Build parent map for detecting "immediately called" pattern
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    # Track imported module names
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    def is_literal_str(n) -> bool:
        return isinstance(n, ast.Constant) and isinstance(n.value, str)

    def is_immediately_called(call_node) -> bool:
        """Check if result of this call is immediately invoked: getattr(...)()"""
        p = parent.get(call_node)
        return isinstance(p, ast.Call) and p.func is call_node

    def is_module_like_target(obj) -> bool:
        """Check if target is a module, globals(), locals(), or builtins"""
        if isinstance(obj, ast.Name):
            if obj.id in imported_names:
                return True
            if obj.id in ('globals', 'locals', 'builtins', '__builtins__'):
                return True
        # globals() or locals() call
        if isinstance(obj, ast.Call) and isinstance(obj.func, ast.Name):
            if obj.func.id in ('globals', 'locals'):
                return True
        return False

    def is_dangerous_getattr(call_node) -> tuple[bool, str]:
        """Check if getattr call is actually dangerous."""
        if not isinstance(call_node.func, ast.Name) or call_node.func.id != 'getattr':
            return False, ""
        if len(call_node.args) < 2:
            return False, ""

        obj = call_node.args[0]
        attr = call_node.args[1]

        # Check dangerous conditions
        if not is_literal_str(attr):
            return True, "non-literal attribute name"
        if is_immediately_called(call_node):
            return True, "result is immediately called"
        if is_module_like_target(obj):
            return True, "target is module/globals"

        # Safe: literal attr name + not called + not module target
        return False, ""

    # Real sinks that require allowlist protection
    real_sinks = [
        ("importlib.import_module", "dynamic import"),
        ("eval", "eval"),
        ("exec", "exec"),
        ("__import__", "dynamic import"),
    ]

    # Check for dangerous patterns
    dangerous_findings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            # Check real sinks (always dangerous)
            for pattern, desc in real_sinks:
                if pattern.endswith(func_name) or func_name == pattern.split('.')[-1]:
                    dangerous_findings.append({
                        "line": node.lineno,
                        "pattern": pattern,
                        "description": desc,
                    })

            # Check getattr (only dangerous in specific forms)
            if func_name == 'getattr':
                is_dangerous, reason = is_dangerous_getattr(node)
                if is_dangerous:
                    dangerous_findings.append({
                        "line": node.lineno,
                        "pattern": "getattr",
                        "description": f"dynamic attribute access ({reason})",
                    })

    if not dangerous_findings:
        return Verdict.PASS, Witness(
            witness_type="ast_analysis_not_applicable",
            artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
            artifact_inline=json.dumps({
                "status": "not_applicable",
                "reason": "No dangerous patterns (dynamic imports, eval, exec, risky getattr) found",
                "checked_patterns": [p[0] for p in real_sinks] + ["getattr (risky forms only)"],
            }),
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    # Find allowlist definitions (expanded patterns)
    allowlist_patterns = [
        "ALLOW", "WHITE", "VALID", "SAFE", "PERMIT",
        "_CLASS_MODULES", "_FUNC_MODULES", "_SUBMODULES", "_PROVIDERS",
        "MODULES", "REGISTRY",  # Common mapping names
    ]
    allowlist_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name_upper = target.id.upper()
                    if any(x in name_upper for x in allowlist_patterns):
                        allowlist_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name_upper = node.target.id.upper()
            if any(x in name_upper for x in allowlist_patterns):
                allowlist_names.add(node.target.id)

    if not allowlist_names:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
            artifact_inline=json.dumps({
                "error": "Dangerous patterns found without allowlist",
                "dangerous_findings": dangerous_findings,
                "searched_for": allowlist_patterns,
            }),
            counterexample="File contains dangerous patterns but no allowlist definition found",
            counterexample_trace=[f"Line {f['line']}: {f['description']}" for f in dangerous_findings],
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    # Check that dangerous calls are guarded by allowlist references nearby
    violations = []
    lines = file_content.split('\n')

    for finding in dangerous_findings:
        line_num = finding["line"]

        # Check if this is within scope
        if claim.scope.start_line and claim.scope.end_line:
            if not (claim.scope.start_line <= line_num <= claim.scope.end_line):
                continue

        # Check if any allowlist is referenced nearby (simple heuristic)
        context_start = max(1, line_num - 10)
        context_end = line_num
        context = '\n'.join(lines[context_start-1:context_end])

        allowlist_used = any(name in context for name in allowlist_names)
        if not allowlist_used:
            violations.append(finding)

    if violations:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(json.dumps(violations).encode()).hexdigest()[:16],
            artifact_inline=json.dumps(violations),
            counterexample=f"Found {len(violations)} unguarded dangerous calls",
            counterexample_trace=[f"Line {v['line']}: {v['description']}" for v in violations],
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    return Verdict.PASS, Witness(
        witness_type="ast_analysis",
        artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
        artifact_inline=json.dumps({
            "allowlists_found": list(allowlist_names),
            "dangerous_patterns_checked": [p[0] for p in real_sinks] + ["getattr (risky forms only)"],
        }),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="ast_checker",
    )


def _check_no_shell_true(tree: ast.AST, file_content: str, claim: Claim) -> tuple[Verdict, Witness | None]:
    """
    Check for shell injection vulnerabilities.

    Detects:
    - os.system() - always dangerous (shell execution)
    - os.popen() - shell execution
    - subprocess.run/Popen/call with shell=True
    - commands.getoutput() (deprecated but still dangerous)
    """
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check scope
            if claim.scope.start_line and claim.scope.end_line:
                if not (claim.scope.start_line <= node.lineno <= claim.scope.end_line):
                    continue

            # Check for os.system(), os.popen() - ALWAYS dangerous
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    if node.func.attr in ("system", "popen"):
                        violations.append({
                            "line": node.lineno,
                            "issue": f"os.{node.func.attr}() is a shell injection risk",
                            "severity": "high",
                        })

                # Check for subprocess.run, subprocess.Popen, etc. with shell=True
                if node.func.attr in ("run", "Popen", "call", "check_call", "check_output"):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                        for kw in node.keywords:
                            if kw.arg == "shell":
                                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    violations.append({
                                        "line": node.lineno,
                                        "issue": f"subprocess.{node.func.attr}() with shell=True",
                                        "severity": "high",
                                    })
                                elif isinstance(kw.value, ast.Name):
                                    # shell=variable - could be True at runtime
                                    violations.append({
                                        "line": node.lineno,
                                        "issue": f"subprocess.{node.func.attr}() with shell=<variable> (could be True)",
                                        "severity": "medium",
                                    })

                # Check for commands.getoutput (deprecated but dangerous)
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "commands":
                    if node.func.attr in ("getoutput", "getstatusoutput"):
                        violations.append({
                            "line": node.lineno,
                            "issue": f"commands.{node.func.attr}() is a shell injection risk",
                            "severity": "high",
                        })

            # Also check for bare eval() with string input (could be shell-like)
            if isinstance(node.func, ast.Name) and node.func.id == "eval":
                # eval() is dangerous if argument could be user-controlled
                violations.append({
                    "line": node.lineno,
                    "issue": "eval() can execute arbitrary code",
                    "severity": "high",
                })

    if violations:
        return Verdict.FAIL, Witness(
            witness_type="ast_analysis",
            artifact_hash=hashlib.sha256(json.dumps(violations).encode()).hexdigest()[:16],
            artifact_inline=json.dumps(violations),
            counterexample=f"Found {len(violations)} shell injection risk(s)",
            counterexample_trace=[f"Line {v['line']}: {v['issue']}" for v in violations],
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="ast_checker",
        )

    return Verdict.PASS, Witness(
        witness_type="ast_analysis",
        artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
        artifact_inline=json.dumps({
            "checked": [
                "os.system()",
                "os.popen()",
                "subprocess.*(shell=True)",
                "commands.getoutput()",
                "eval()",
            ],
            "violations": 0,
        }),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="ast_checker",
    )


# ─────────────────────────────────────────────────────────────────
# Property Test Checker (Hypothesis-style)
# ─────────────────────────────────────────────────────────────────

def property_test_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Run property-based tests as witnesses.

    Requires tests to be defined and discoverable.
    Seeds are saved as witnesses for reproducibility.
    """
    if claim.claim_type not in (ClaimType.REFACTOR_EQUIVALENCE, ClaimType.PURE_FUNCTION, ClaimType.IDEMPOTENT):
        return Verdict.UNKNOWN, Witness(
            witness_type="property_test_skip",
            artifact_hash="",
            artifact_inline=f"No property tests for claim type: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="hypothesis",
        )

    # Try to find and run hypothesis tests
    try:
        # Look for test file
        file_path = Path(claim.scope.file_path)
        test_path = file_path.parent / f"test_{file_path.name}"

        if not test_path.exists():
            test_path = file_path.parent.parent / "tests" / f"test_{file_path.name}"

        if not test_path.exists():
            return Verdict.UNKNOWN, Witness(
                witness_type="property_test_skip",
                artifact_hash="",
                artifact_inline=f"No test file found for {file_path.name}",
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="hypothesis",
            )

        # Run pytest with hypothesis
        pytest_cmd = _find_executable("pytest")
        result = subprocess.run(
            [pytest_cmd, str(test_path), "--hypothesis-seed=42", "-v", "--tb=short"],
            capture_output=True,
            timeout=120,
        )

        output = result.stdout.decode() + result.stderr.decode()
        artifact_hash = hashlib.sha256(output.encode()).hexdigest()

        if result.returncode == 0:
            return Verdict.PASS, Witness(
                witness_type="property_test_output",
                artifact_hash=artifact_hash,
                artifact_inline=output[-2000:] if len(output) > 2000 else output,
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="hypothesis",
            )
        else:
            # Extract failing example
            counterexample = None
            for line in output.split('\n'):
                if "Falsifying example" in line or "AssertionError" in line:
                    counterexample = line
                    break

            return Verdict.FAIL, Witness(
                witness_type="property_test_output",
                artifact_hash=artifact_hash,
                artifact_inline=output[-2000:] if len(output) > 2000 else output,
                counterexample=counterexample,
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="hypothesis",
            )

    except subprocess.TimeoutExpired:
        return Verdict.TIMEOUT, None
    except Exception as e:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline=str(e),
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="hypothesis",
        )


# ─────────────────────────────────────────────────────────────────
# Contract Checker (pre/post conditions)
# ─────────────────────────────────────────────────────────────────

def contract_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Check runtime contracts (pre/post conditions).

    Looks for @contract, @requires, @ensures decorators
    or inline assert statements.
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        return Verdict.ERROR, Witness(
            witness_type="contract_error",
            artifact_hash="",
            artifact_inline=f"Syntax error: {e}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="contract_checker",
        )

    contracts_found = []
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if in scope
            if claim.scope.function_name and node.name != claim.scope.function_name:
                continue
            if claim.scope.start_line and claim.scope.end_line:
                if not (claim.scope.start_line <= node.lineno <= claim.scope.end_line):
                    continue

            # Check for contract decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id in ("contract", "requires", "ensures", "invariant"):
                        contracts_found.append({
                            "function": node.name,
                            "line": decorator.lineno,
                            "type": decorator.id,
                        })
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        if decorator.func.id in ("contract", "requires", "ensures", "invariant"):
                            contracts_found.append({
                                "function": node.name,
                                "line": decorator.lineno,
                                "type": decorator.func.id,
                            })

            # Check for assert statements (simple contracts)
            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    contracts_found.append({
                        "function": node.name,
                        "line": child.lineno,
                        "type": "assert",
                    })

    if not contracts_found and claim.claim_type in (ClaimType.INPUT_VALIDATED, ClaimType.STATE_MACHINE_INVARIANT):
        return Verdict.FAIL, Witness(
            witness_type="contract_analysis",
            artifact_hash=hashlib.sha256(file_content.encode()).hexdigest()[:16],
            artifact_inline=json.dumps({"error": "No contracts/asserts found"}),
            counterexample="No contracts found in specified scope",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="contract_checker",
        )

    return Verdict.PASS if contracts_found else Verdict.UNKNOWN, Witness(
        witness_type="contract_analysis",
        artifact_hash=hashlib.sha256(json.dumps(contracts_found).encode()).hexdigest()[:16],
        artifact_inline=json.dumps({"contracts_found": contracts_found}),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="contract_checker",
    )


# ─────────────────────────────────────────────────────────────────
# Mypy Checker (Type Safety)
# ─────────────────────────────────────────────────────────────────

def mypy_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Check type safety claims using mypy.

    This is a Layer 1 "semantic" checker that goes beyond security lint.
    Mypy provides:
    - Static type checking
    - Deterministic analysis (same file → same result)
    - Machine-verifiable witness (mypy JSON output)

    Supports:
    - TYPE_SAFE: General type correctness
    - PURE_FUNCTION: Functions without side effects (via --strict)
    """
    if claim.claim_type not in (ClaimType.TYPE_SAFE,):
        return Verdict.UNKNOWN, Witness(
            witness_type="mypy_skip",
            artifact_hash="",
            artifact_inline=f"No mypy check for claim type: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="mypy",
        )

    # Write file to temp
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        # Run mypy with JSON output
        mypy_cmd = _find_executable("mypy")
        result = subprocess.run(
            [
                mypy_cmd,
                "-O", "json",  # JSON output format
                "--no-error-summary",
                "--ignore-missing-imports",  # Don't fail on missing third-party types
                temp_path,
            ],
            capture_output=True,
            timeout=60,
        )

        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        # Parse JSON output
        errors = []
        for line in stdout.strip().split('\n'):
            if line:
                try:
                    error = json.loads(line)
                    # Filter to scope if specified
                    if claim.scope.start_line and claim.scope.end_line:
                        error_line = error.get("line", 0)
                        if not (claim.scope.start_line <= error_line <= claim.scope.end_line):
                            continue
                    errors.append(error)
                except json.JSONDecodeError:
                    pass

        # Create witness
        artifact_content = {
            "errors": errors,
            "error_count": len(errors),
            "mypy_return_code": result.returncode,
        }
        artifact_hash = hashlib.sha256(json.dumps(artifact_content, sort_keys=True).encode()).hexdigest()

        if not errors and result.returncode == 0:
            return Verdict.PASS, Witness(
                witness_type="mypy_output",
                artifact_hash=artifact_hash,
                artifact_inline=json.dumps({
                    "type_safe": True,
                    "errors": 0,
                    "checked_lines": claim.scope.end_line - claim.scope.start_line + 1 if claim.scope.start_line else "all",
                }),
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="mypy",
            )
        else:
            # Extract counterexamples
            counterexample_trace = []
            for err in errors[:5]:  # First 5 errors
                msg = f"Line {err.get('line', '?')}: {err.get('message', '?')}"
                counterexample_trace.append(msg)

            return Verdict.FAIL, Witness(
                witness_type="mypy_output",
                artifact_hash=artifact_hash,
                artifact_inline=json.dumps(errors[:10]),  # First 10 errors
                counterexample=f"Found {len(errors)} type error(s)",
                counterexample_trace=counterexample_trace,
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="mypy",
            )

    except subprocess.TimeoutExpired:
        return Verdict.TIMEOUT, None
    except FileNotFoundError:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline="mypy not found - install with: pip install mypy",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="mypy",
        )
    except Exception as e:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline=str(e),
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="mypy",
        )
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _find_executable(name: str) -> str:
    """Find an executable, checking venv first."""
    import sys
    import shutil

    # Check if we're in a venv and look there first
    venv_path = Path(sys.prefix) / "bin" / name
    if venv_path.exists():
        return str(venv_path)

    # Fall back to system PATH
    found = shutil.which(name)
    return found if found else name


def get_mypy_version() -> str:
    """Get mypy version for cache keying."""
    try:
        mypy_cmd = _find_executable("mypy")
        result = subprocess.run([mypy_cmd, "--version"], capture_output=True, text=True, timeout=5)
        # Output is like "mypy 1.5.1 (compiled: yes)"
        version = result.stdout.split()[1] if result.stdout else "unknown"
        return version
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────
# Pytest Checker (Test Coverage as Witness)
# ─────────────────────────────────────────────────────────────────

def pytest_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Run pytest on touched files as a witness.

    This provides:
    - Test execution as verification
    - Coverage data as witness
    - Deterministic with fixed random seed

    For REFACTOR_EQUIVALENCE claims, runs tests to verify behavior preservation.
    """
    if claim.claim_type not in (ClaimType.REFACTOR_EQUIVALENCE, ClaimType.TEST_COVERAGE):
        return Verdict.UNKNOWN, Witness(
            witness_type="pytest_skip",
            artifact_hash="",
            artifact_inline=f"No pytest check for claim type: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="pytest",
        )

    # Find test file
    file_path = Path(claim.scope.file_path)
    test_candidates = [
        file_path.parent / f"test_{file_path.name}",
        file_path.parent / "tests" / f"test_{file_path.name}",
        file_path.parent.parent / "tests" / f"test_{file_path.name}",
    ]

    test_path = None
    for candidate in test_candidates:
        if candidate.exists():
            test_path = candidate
            break

    if not test_path:
        return Verdict.UNKNOWN, Witness(
            witness_type="pytest_skip",
            artifact_hash="",
            artifact_inline=f"No test file found for {file_path.name}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="pytest",
        )

    try:
        # Run pytest with deterministic settings
        pytest_cmd = _find_executable("pytest")
        result = subprocess.run(
            [
                pytest_cmd,
                str(test_path),
                "-v",
                "--tb=short",
                "-p", "no:randomly",  # Disable random ordering
                "--randomly-seed=42",  # Fixed seed if plugin exists
            ],
            capture_output=True,
            timeout=120,
            env={**os.environ, "PYTHONHASHSEED": "42"},
        )

        output = result.stdout.decode() + result.stderr.decode()
        artifact_hash = hashlib.sha256(output.encode()).hexdigest()

        # Parse results
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        errors = output.count(" ERROR")

        if result.returncode == 0:
            return Verdict.PASS, Witness(
                witness_type="pytest_output",
                artifact_hash=artifact_hash,
                artifact_inline=json.dumps({
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "test_file": str(test_path),
                }),
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="pytest",
            )
        else:
            # Extract failure info
            counterexample = None
            for line in output.split('\n'):
                if "FAILED" in line or "AssertionError" in line:
                    counterexample = line[:200]
                    break

            return Verdict.FAIL, Witness(
                witness_type="pytest_output",
                artifact_hash=artifact_hash,
                artifact_inline=output[-2000:] if len(output) > 2000 else output,
                counterexample=counterexample or f"{failed} test(s) failed",
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer="pytest",
            )

    except subprocess.TimeoutExpired:
        return Verdict.TIMEOUT, None
    except FileNotFoundError:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline="pytest not found - install with: pip install pytest",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="pytest",
        )
    except Exception as e:
        return Verdict.ERROR, Witness(
            witness_type="error",
            artifact_hash="",
            artifact_inline=str(e),
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="pytest",
        )


def get_pytest_version() -> str:
    """Get pytest version for cache keying."""
    try:
        pytest_cmd = _find_executable("pytest")
        result = subprocess.run([pytest_cmd, "--version"], capture_output=True, text=True, timeout=5)
        # Output is like "pytest 7.4.0"
        for word in result.stdout.split():
            if word[0].isdigit():
                return word
        return "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────
# Committor Gate Checker
# ─────────────────────────────────────────────────────────────────

def committor_gate_checker(claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
    """
    Verify committor gate decisions by re-deriving from posteriors.

    This checker validates that gate decisions are reproducible:
    1. Load the gate_result.json from the claim's artifact
    2. Load the beta posteriors
    3. Re-compute q values for each decision
    4. Verify decisions match the stored ones

    The claim's scope.file_path should point to gate_result.json.
    """
    if claim.claim_type != ClaimType.COMMITTOR_GATE:
        return Verdict.UNKNOWN, Witness(
            witness_type="committor_gate_skip",
            artifact_hash="",
            artifact_inline=f"Not a committor gate claim: {claim.claim_type.value}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    try:
        # The file_content is actually the gate_result.json content
        gate_result = json.loads(file_content)
    except json.JSONDecodeError as e:
        return Verdict.ERROR, Witness(
            witness_type="committor_gate_error",
            artifact_hash="",
            artifact_inline=f"Invalid gate result JSON: {e}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    # Get posteriors path from gate result
    posteriors_path = gate_result.get('posteriors_path', '')
    if not posteriors_path:
        return Verdict.UNKNOWN, Witness(
            witness_type="committor_gate_skip",
            artifact_hash="",
            artifact_inline="No posteriors path in gate result",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    posteriors_path = Path(posteriors_path)
    if not posteriors_path.exists():
        # Can't verify without posteriors - not an error, just unknown
        return Verdict.UNKNOWN, Witness(
            witness_type="committor_gate_skip",
            artifact_hash="",
            artifact_inline=f"Posteriors file not found: {posteriors_path}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    # Import numpy and verification function
    try:
        import numpy as np
        from bef_zk.shared.scoring import lookup_posterior_at_idx
        from bef_zk.shared.features import SKIP_THRESHOLD, HUMAN_REVIEW_UNCERTAINTY
    except ImportError as e:
        return Verdict.ERROR, Witness(
            witness_type="committor_gate_error",
            artifact_hash="",
            artifact_inline=f"Missing dependency: {e}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    # Load posteriors
    try:
        data = np.load(posteriors_path, allow_pickle=True)
        alpha = data['alpha']
        beta = data['beta']
    except Exception as e:
        return Verdict.ERROR, Witness(
            witness_type="committor_gate_error",
            artifact_hash="",
            artifact_inline=f"Failed to load posteriors: {e}",
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    # Get thresholds
    thresholds = gate_result.get('thresholds', {})
    skip_threshold = thresholds.get('skip', SKIP_THRESHOLD)
    review_uncertainty = thresholds.get('review_uncertainty', HUMAN_REVIEW_UNCERTAINTY)

    # Verify each decision
    mismatches = []
    for decision in gate_result.get('decisions', []):
        grid_idx = decision['grid_idx']
        stored_q = decision['q']
        stored_decision = decision['decision']

        # Re-derive q from posteriors
        posterior = lookup_posterior_at_idx(alpha, beta, grid_idx)
        computed_q = posterior['q']
        uncertainty = posterior['uncertainty']

        # Compute expected decision
        if computed_q >= skip_threshold:
            expected_decision = 'skip'
        elif uncertainty > review_uncertainty:
            expected_decision = 'human_review'
        else:
            expected_decision = 'pass'

        # Check for mismatch (allow small float tolerance)
        if abs(stored_q - computed_q) > 1e-6:
            mismatches.append({
                'item_id': decision['item_id'],
                'type': 'q_mismatch',
                'stored': stored_q,
                'computed': computed_q,
            })

        if stored_decision != expected_decision:
            mismatches.append({
                'item_id': decision['item_id'],
                'type': 'decision_mismatch',
                'stored': stored_decision,
                'expected': expected_decision,
            })

    artifact_hash = hashlib.sha256(json.dumps({
        'mismatches': mismatches,
        'decisions_checked': len(gate_result.get('decisions', [])),
    }).encode()).hexdigest()

    if mismatches:
        return Verdict.FAIL, Witness(
            witness_type="committor_gate_verification",
            artifact_hash=artifact_hash,
            artifact_inline=json.dumps(mismatches[:5]),  # First 5 mismatches
            counterexample=f"Found {len(mismatches)} decision mismatch(es)",
            counterexample_trace=[
                f"{m['item_id']}: {m['type']}" for m in mismatches[:5]
            ],
            produced_at=datetime.utcnow().isoformat() + "Z",
            producer="committor_gate",
        )

    return Verdict.PASS, Witness(
        witness_type="committor_gate_verification",
        artifact_hash=artifact_hash,
        artifact_inline=json.dumps({
            'decisions_verified': len(gate_result.get('decisions', [])),
            'posteriors_used': str(posteriors_path),
        }),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="committor_gate",
    )


# ─────────────────────────────────────────────────────────────────
# Register all checkers
# ─────────────────────────────────────────────────────────────────

def get_semgrep_version() -> str:
    try:
        result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:20]
    except Exception:
        return "unknown"


def get_env_key() -> str:
    """Generate environment fingerprint."""
    import sys
    parts = [
        f"py{sys.version_info.major}.{sys.version_info.minor}",
        os.uname().sysname,
    ]
    return "-".join(parts)


def register_default_checkers():
    """Register all default checkers."""
    env_key = get_env_key()

    # Semgrep checker
    CHECKER_REGISTRY.register(
        "semgrep",
        semgrep_checker,
        CheckerInfo(
            checker_id="semgrep",
            checker_version=get_semgrep_version(),
            env_key=env_key,
        )
    )

    # AST checker
    CHECKER_REGISTRY.register(
        "ast",
        ast_checker,
        CheckerInfo(
            checker_id="ast",
            checker_version="1.1.0",  # Bumped: now catches os.system, os.popen, eval
            env_key=env_key,
        )
    )

    # Property test checker
    CHECKER_REGISTRY.register(
        "hypothesis",
        property_test_checker,
        CheckerInfo(
            checker_id="hypothesis",
            checker_version="6.0.0",  # TODO: detect actual version
            env_key=env_key,
        )
    )

    # Contract checker
    CHECKER_REGISTRY.register(
        "contracts",
        contract_checker,
        CheckerInfo(
            checker_id="contracts",
            checker_version="1.0.0",
            env_key=env_key,
        )
    )

    # Mypy checker (type safety)
    CHECKER_REGISTRY.register(
        "mypy",
        mypy_checker,
        CheckerInfo(
            checker_id="mypy",
            checker_version=get_mypy_version(),
            env_key=env_key,
        )
    )

    # Pytest checker (test coverage / refactor equivalence)
    CHECKER_REGISTRY.register(
        "pytest",
        pytest_checker,
        CheckerInfo(
            checker_id="pytest",
            checker_version=get_pytest_version(),
            env_key=env_key,
        )
    )

    # Committor gate checker (verifies gate decisions from posteriors)
    CHECKER_REGISTRY.register(
        "committor_gate",
        committor_gate_checker,
        CheckerInfo(
            checker_id="committor_gate",
            checker_version="1.0.0",
            env_key=env_key,
        )
    )


# Auto-register on import
register_default_checkers()
