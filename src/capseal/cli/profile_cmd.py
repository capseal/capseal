"""Profile extraction and conformance checking for CapSeal.

This module implements the "ladder of evidence" pattern:
- profile.json: Deterministic extraction of "what it is"
- capseal.intent.json: Explicit declaration of "what it should be"
- conformance.json: Drift detection between profile and intent

All outputs are committed artifacts with receipts.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import click


# =============================================================================
# Profile Schema
# =============================================================================

PROFILE_SCHEMA_VERSION = "profile_v1"
INTENT_SCHEMA_VERSION = "intent_v1"
CONFORMANCE_SCHEMA_VERSION = "conformance_v1"


def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))


# =============================================================================
# README / Documentation Extraction
# =============================================================================

def extract_readme_purpose(project_dir: Path) -> dict[str, Any]:
    """Extract purpose and structure from README files."""
    result = {
        "found": False,
        "file": None,
        "title": None,
        "first_paragraph": None,
        "headings": [],
    }

    readme_names = ["README.md", "README.rst", "README.txt", "README", "readme.md"]

    for name in readme_names:
        readme_path = project_dir / name
        if readme_path.exists():
            result["found"] = True
            result["file"] = name
            content = readme_path.read_text(errors="replace")

            # Extract title (first # heading or first line)
            lines = content.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped.startswith("# "):
                        result["title"] = stripped[2:].strip()
                    else:
                        result["title"] = stripped[:100]
                    break

            # Extract first paragraph (non-heading, non-empty block)
            in_paragraph = False
            paragraph_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if in_paragraph and paragraph_lines:
                        break
                    continue
                if stripped.startswith("#") or stripped.startswith("```"):
                    if in_paragraph and paragraph_lines:
                        break
                    continue
                in_paragraph = True
                paragraph_lines.append(stripped)
                if len(" ".join(paragraph_lines)) > 500:
                    break

            if paragraph_lines:
                result["first_paragraph"] = " ".join(paragraph_lines)[:500]

            # Extract all headings
            for line in lines:
                if line.strip().startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    text = line.strip("#").strip()
                    if text:
                        result["headings"].append({"level": level, "text": text})

            break

    return result


# =============================================================================
# Module Docstring Extraction
# =============================================================================

def extract_module_docstrings(project_dir: Path) -> list[dict[str, Any]]:
    """Extract top-level docstrings from Python modules."""
    docstrings = []

    for py_file in project_dir.rglob("*.py"):
        # Skip hidden dirs, venvs, caches
        parts = py_file.relative_to(project_dir).parts
        if any(p.startswith(".") or p in ("venv", ".venv", "__pycache__", "node_modules") for p in parts):
            continue

        try:
            content = py_file.read_text(errors="replace")
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                rel_path = str(py_file.relative_to(project_dir))
                docstrings.append({
                    "path": rel_path,
                    "docstring": docstring[:500],  # Truncate long docstrings
                    "docstring_hash": sha256_str(docstring),
                })
        except (SyntaxError, UnicodeDecodeError):
            continue

    return docstrings


# =============================================================================
# Entrypoint Detection
# =============================================================================

def extract_entrypoints(project_dir: Path) -> dict[str, Any]:
    """Extract declared entrypoints from pyproject.toml, setup.cfg, __main__.py."""
    result = {
        "pyproject_scripts": [],
        "setup_cfg_scripts": [],
        "main_modules": [],
        "cli_groups": [],
    }

    # pyproject.toml
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            try:
                content = pyproject_path.read_text()
                data = tomllib.loads(content)

                # [project.scripts]
                scripts = data.get("project", {}).get("scripts", {})
                for name, target in scripts.items():
                    result["pyproject_scripts"].append({"name": name, "target": target})

                # [project.gui-scripts]
                gui_scripts = data.get("project", {}).get("gui-scripts", {})
                for name, target in gui_scripts.items():
                    result["pyproject_scripts"].append({"name": name, "target": target, "gui": True})

                # [tool.poetry.scripts]
                poetry_scripts = data.get("tool", {}).get("poetry", {}).get("scripts", {})
                for name, target in poetry_scripts.items():
                    result["pyproject_scripts"].append({"name": name, "target": target})
            except Exception:
                pass

    # setup.cfg
    setup_cfg_path = project_dir / "setup.cfg"
    if setup_cfg_path.exists():
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(setup_cfg_path)

            if "options.entry_points" in config:
                for key, value in config["options.entry_points"].items():
                    if key == "console_scripts":
                        for line in value.strip().split("\n"):
                            line = line.strip()
                            if "=" in line:
                                name, target = line.split("=", 1)
                                result["setup_cfg_scripts"].append({
                                    "name": name.strip(),
                                    "target": target.strip(),
                                })
        except Exception:
            pass

    # __main__.py files
    for main_file in project_dir.rglob("__main__.py"):
        parts = main_file.relative_to(project_dir).parts
        if any(p.startswith(".") or p in ("venv", ".venv", "__pycache__") for p in parts):
            continue
        result["main_modules"].append(str(main_file.relative_to(project_dir)))

    # Click groups (look for @click.group or @click.command)
    for py_file in project_dir.rglob("*.py"):
        parts = py_file.relative_to(project_dir).parts
        if any(p.startswith(".") or p in ("venv", ".venv", "__pycache__", "node_modules") for p in parts):
            continue
        try:
            content = py_file.read_text(errors="replace")
            if "@click.group" in content or "click.Group" in content:
                result["cli_groups"].append(str(py_file.relative_to(project_dir)))
        except Exception:
            continue

    return result


# =============================================================================
# Dependency Extraction
# =============================================================================

def extract_dependencies(project_dir: Path) -> dict[str, Any]:
    """Extract dependencies from requirements.txt, pyproject.toml, setup.cfg."""
    result = {
        "requirements_txt": [],
        "pyproject_deps": [],
        "setup_cfg_deps": [],
        "imports_detected": [],
    }

    # requirements.txt
    req_path = project_dir / "requirements.txt"
    if req_path.exists():
        try:
            for line in req_path.read_text().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Extract package name (before ==, >=, etc.)
                    match = re.match(r"^([a-zA-Z0-9_-]+)", line)
                    if match:
                        result["requirements_txt"].append(match.group(1).lower())
        except Exception:
            pass

    # pyproject.toml dependencies
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            try:
                data = tomllib.loads(pyproject_path.read_text())
                deps = data.get("project", {}).get("dependencies", [])
                for dep in deps:
                    match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
                    if match:
                        result["pyproject_deps"].append(match.group(1).lower())

                # Poetry deps
                poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                for name in poetry_deps:
                    if name.lower() != "python":
                        result["pyproject_deps"].append(name.lower())
            except Exception:
                pass

    # Deduplicate
    result["requirements_txt"] = sorted(set(result["requirements_txt"]))
    result["pyproject_deps"] = sorted(set(result["pyproject_deps"]))

    return result


# =============================================================================
# Hot Callsite Detection
# =============================================================================

HOT_PATTERNS = {
    "sql": [
        r"\.execute\s*\(",
        r"\.executemany\s*\(",
        r"\.raw\s*\(",
        r"text\s*\(",
        r"RawSQL\s*\(",
    ],
    "subprocess": [
        r"subprocess\.(run|call|Popen|check_output|check_call)\s*\(",
        r"os\.system\s*\(",
        r"os\.popen\s*\(",
        r"commands\.(getoutput|getstatusoutput)\s*\(",
    ],
    "network": [
        r"requests\.(get|post|put|delete|patch|head|options)\s*\(",
        r"urllib\.request\.urlopen\s*\(",
        r"httpx\.",
        r"aiohttp\.",
        r"socket\.socket\s*\(",
    ],
    "dynamic_import": [
        r"importlib\.import_module\s*\(",
        r"__import__\s*\(",
        r"exec\s*\(",
        r"eval\s*\(",
    ],
    "file_ops": [
        r"open\s*\([^)]*['\"][wa]",  # open with write/append mode
        r"shutil\.(copy|move|rmtree)\s*\(",
        r"os\.(remove|unlink|rmdir)\s*\(",
    ],
    "shell": [
        r"shell\s*=\s*True",
    ],
}


def extract_hot_callsites(project_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Find security-relevant callsites with file/line anchors."""
    result = {category: [] for category in HOT_PATTERNS}

    for py_file in project_dir.rglob("*.py"):
        parts = py_file.relative_to(project_dir).parts
        if any(p.startswith(".") or p in ("venv", ".venv", "__pycache__", "node_modules", "test", "tests") for p in parts):
            continue

        try:
            content = py_file.read_text(errors="replace")
            lines = content.split("\n")
            rel_path = str(py_file.relative_to(project_dir))

            for category, patterns in HOT_PATTERNS.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            result[category].append({
                                "file": rel_path,
                                "line": line_num,
                                "pattern": pattern,
                                "snippet": line.strip()[:100],
                            })
        except Exception:
            continue

    return result


# =============================================================================
# Full Profile Extraction
# =============================================================================

def extract_profile(project_dir: Path) -> dict[str, Any]:
    """Extract complete project profile."""
    project_dir = Path(project_dir).resolve()

    profile = {
        "schema": PROFILE_SCHEMA_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "project_path": str(project_dir),
        "project_name": project_dir.name,
        "readme": extract_readme_purpose(project_dir),
        "docstrings": extract_module_docstrings(project_dir),
        "entrypoints": extract_entrypoints(project_dir),
        "dependencies": extract_dependencies(project_dir),
        "hot_callsites": extract_hot_callsites(project_dir),
    }

    # Compute summary
    profile["summary"] = {
        "has_readme": profile["readme"]["found"],
        "readme_title": profile["readme"].get("title"),
        "module_count": len(profile["docstrings"]),
        "entrypoint_count": (
            len(profile["entrypoints"]["pyproject_scripts"]) +
            len(profile["entrypoints"]["setup_cfg_scripts"]) +
            len(profile["entrypoints"]["main_modules"])
        ),
        "dependency_count": len(set(
            profile["dependencies"]["requirements_txt"] +
            profile["dependencies"]["pyproject_deps"]
        )),
        "hot_callsite_counts": {
            category: len(sites) for category, sites in profile["hot_callsites"].items()
        },
    }

    # Compute profile hash
    profile["profile_hash"] = sha256_json(profile)

    return profile


# =============================================================================
# Intent Schema
# =============================================================================

DEFAULT_INTENT = {
    "schema": INTENT_SCHEMA_VERSION,
    "purpose": "",
    "allowed_providers": [],
    "security_invariants": {
        "no_raw_sql": False,
        "no_shell_true": False,
        "no_dynamic_import": False,
        "no_eval_exec": False,
        "import_allowlist": [],
    },
    "network_policy": "any",  # any, outbound_only, none
    "file_policy": "any",  # any, read_only, none
}


def load_intent(project_dir: Path) -> dict[str, Any] | None:
    """Load capseal.intent.json from project directory."""
    intent_paths = [
        project_dir / "capseal.intent.json",
        project_dir / ".capseal" / "intent.json",
        project_dir / ".capseal.intent.json",
    ]

    for path in intent_paths:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                continue

    return None


# =============================================================================
# Conformance Checking
# =============================================================================

def check_conformance(profile: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    """Check profile against intent and return violations."""
    result = {
        "schema": CONFORMANCE_SCHEMA_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "profile_hash": profile.get("profile_hash", ""),
        "intent_hash": sha256_json(intent),
        "violations": [],
        "warnings": [],
        "gate_pass": True,
    }

    invariants = intent.get("security_invariants", {})
    hot = profile.get("hot_callsites", {})

    # Check: no_raw_sql
    if invariants.get("no_raw_sql") and hot.get("sql"):
        result["violations"].append({
            "invariant": "no_raw_sql",
            "message": f"Found {len(hot['sql'])} SQL callsites",
            "occurrences": hot["sql"][:10],  # First 10
        })
        result["gate_pass"] = False

    # Check: no_shell_true
    if invariants.get("no_shell_true") and hot.get("shell"):
        result["violations"].append({
            "invariant": "no_shell_true",
            "message": f"Found {len(hot['shell'])} shell=True callsites",
            "occurrences": hot["shell"][:10],
        })
        result["gate_pass"] = False

    # Check: no_dynamic_import
    if invariants.get("no_dynamic_import"):
        dynamic_imports = [
            s for s in hot.get("dynamic_import", [])
            if "import_module" in s.get("pattern", "") or "__import__" in s.get("pattern", "")
        ]
        if dynamic_imports:
            result["violations"].append({
                "invariant": "no_dynamic_import",
                "message": f"Found {len(dynamic_imports)} dynamic import callsites",
                "occurrences": dynamic_imports[:10],
            })
            result["gate_pass"] = False

    # Check: no_eval_exec
    if invariants.get("no_eval_exec"):
        eval_exec = [
            s for s in hot.get("dynamic_import", [])
            if "eval" in s.get("pattern", "") or "exec" in s.get("pattern", "")
        ]
        if eval_exec:
            result["violations"].append({
                "invariant": "no_eval_exec",
                "message": f"Found {len(eval_exec)} eval/exec callsites",
                "occurrences": eval_exec[:10],
            })
            result["gate_pass"] = False

    # Check: import_allowlist
    allowlist = invariants.get("import_allowlist", [])
    if allowlist:
        # This would require more sophisticated import analysis
        # For now, just note it as a warning
        result["warnings"].append({
            "check": "import_allowlist",
            "message": "Import allowlist checking not yet implemented",
        })

    # Check: network_policy
    network_policy = intent.get("network_policy", "any")
    if network_policy == "none" and hot.get("network"):
        result["violations"].append({
            "invariant": "network_policy",
            "message": f"Network policy is 'none' but found {len(hot['network'])} network callsites",
            "occurrences": hot["network"][:10],
        })
        result["gate_pass"] = False

    # Check: file_policy
    file_policy = intent.get("file_policy", "any")
    if file_policy in ("read_only", "none") and hot.get("file_ops"):
        result["violations"].append({
            "invariant": "file_policy",
            "message": f"File policy is '{file_policy}' but found {len(hot['file_ops'])} write operations",
            "occurrences": hot["file_ops"][:10],
        })
        result["gate_pass"] = False

    # Compute conformance hash
    result["conformance_hash"] = sha256_json(result)

    return result


# =============================================================================
# CLI Commands
# =============================================================================

@click.command("profile")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--out", "-o", type=click.Path(), help="Output directory (default: project_dir/.capseal)")
@click.option("--format", "fmt", type=click.Choice(["json", "summary"]), default="summary")
def profile_command(project_dir: str, out: str | None, fmt: str) -> None:
    """Extract deterministic profile from a project.

    Produces profile.json with:
    - README purpose/headings
    - Module docstrings
    - Declared entrypoints
    - Dependencies
    - Hot callsites (SQL, subprocess, network, dynamic imports)
    """
    project_path = Path(project_dir).resolve()

    if out:
        out_dir = Path(out)
    else:
        out_dir = project_path / ".capseal"

    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Extracting profile: {project_path}")
    profile = extract_profile(project_path)

    # Write profile.json
    profile_path = out_dir / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))

    if fmt == "json":
        click.echo(json.dumps(profile, indent=2))
    else:
        # Summary output
        click.echo()
        click.echo(f"Profile: {profile_path}")
        click.echo(f"  hash: {profile['profile_hash'][:32]}...")
        click.echo()

        readme = profile["readme"]
        if readme["found"]:
            click.echo(f"  README: {readme['file']}")
            if readme["title"]:
                click.echo(f"    Title: {readme['title']}")
            if readme["first_paragraph"]:
                click.echo(f"    Purpose: {readme['first_paragraph'][:100]}...")
        else:
            click.echo("  README: not found")

        click.echo()
        click.echo(f"  Modules with docstrings: {len(profile['docstrings'])}")
        for doc in profile["docstrings"][:5]:
            click.echo(f"    - {doc['path']}")
        if len(profile["docstrings"]) > 5:
            click.echo(f"    ... and {len(profile['docstrings']) - 5} more")

        click.echo()
        entrypoints = profile["entrypoints"]
        total_eps = (
            len(entrypoints["pyproject_scripts"]) +
            len(entrypoints["setup_cfg_scripts"]) +
            len(entrypoints["main_modules"])
        )
        click.echo(f"  Entrypoints: {total_eps}")
        for ep in entrypoints["pyproject_scripts"][:3]:
            click.echo(f"    - {ep['name']} -> {ep['target']}")
        for mm in entrypoints["main_modules"][:3]:
            click.echo(f"    - {mm}")

        click.echo()
        deps = profile["dependencies"]
        all_deps = sorted(set(deps["requirements_txt"] + deps["pyproject_deps"]))
        click.echo(f"  Dependencies: {len(all_deps)}")
        for dep in all_deps[:10]:
            click.echo(f"    - {dep}")
        if len(all_deps) > 10:
            click.echo(f"    ... and {len(all_deps) - 10} more")

        click.echo()
        click.echo("  Hot callsites:")
        for category, sites in profile["hot_callsites"].items():
            if sites:
                click.echo(f"    {category}: {len(sites)}")
                for site in sites[:3]:
                    click.echo(f"      - {site['file']}:{site['line']}")


@click.command("conformance")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--profile", "profile_path", type=click.Path(exists=True), help="Path to profile.json (default: extract fresh)")
@click.option("--intent", "intent_path", type=click.Path(exists=True), help="Path to intent.json (default: look in project)")
@click.option("--out", "-o", type=click.Path(), help="Output directory")
@click.option("--fail-on-violation", is_flag=True, help="Exit with code 1 if violations found")
def conformance_command(
    project_dir: str,
    profile_path: str | None,
    intent_path: str | None,
    out: str | None,
    fail_on_violation: bool,
) -> None:
    """Check project conformance against declared intent.

    Compares profile.json against capseal.intent.json and reports violations.
    """
    project_path = Path(project_dir).resolve()

    # Load or extract profile
    if profile_path:
        profile = json.loads(Path(profile_path).read_text())
    else:
        click.echo(f"Extracting profile: {project_path}")
        profile = extract_profile(project_path)

    # Load intent
    if intent_path:
        intent = json.loads(Path(intent_path).read_text())
    else:
        intent = load_intent(project_path)

    if not intent:
        click.echo("No intent file found. Create capseal.intent.json to define expectations.")
        click.echo()
        click.echo("Example capseal.intent.json:")
        click.echo(json.dumps({
            "schema": INTENT_SCHEMA_VERSION,
            "purpose": "Your project purpose here",
            "security_invariants": {
                "no_raw_sql": True,
                "no_shell_true": True,
                "no_eval_exec": True,
            },
            "network_policy": "outbound_only",
        }, indent=2))
        sys.exit(0)

    # Check conformance
    conformance = check_conformance(profile, intent)

    # Determine output path
    if out:
        out_dir = Path(out)
    else:
        out_dir = project_path / ".capseal"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write conformance.json
    conformance_path = out_dir / "conformance.json"
    conformance_path.write_text(json.dumps(conformance, indent=2))

    # Display results
    click.echo()
    click.echo(f"Conformance: {conformance_path}")
    click.echo(f"  profile_hash: {conformance['profile_hash'][:32]}...")
    click.echo(f"  intent_hash:  {conformance['intent_hash'][:32]}...")
    click.echo()

    if conformance["violations"]:
        click.echo(click.style("VIOLATIONS:", fg="red", bold=True))
        for v in conformance["violations"]:
            click.echo(f"  [{v['invariant']}] {v['message']}")
            for occ in v.get("occurrences", [])[:3]:
                click.echo(f"    - {occ['file']}:{occ['line']}: {occ.get('snippet', '')[:60]}")
        click.echo()

    if conformance["warnings"]:
        click.echo(click.style("WARNINGS:", fg="yellow"))
        for w in conformance["warnings"]:
            click.echo(f"  [{w['check']}] {w['message']}")
        click.echo()

    if conformance["gate_pass"]:
        click.echo(click.style("CONFORMANCE PASSED", fg="green", bold=True))
    else:
        click.echo(click.style("CONFORMANCE FAILED", fg="red", bold=True))
        if fail_on_violation:
            sys.exit(1)


@click.command("init-intent")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--purpose", "-p", help="Project purpose statement")
@click.option("--strict", is_flag=True, help="Enable all security invariants")
def init_intent_command(project_dir: str, purpose: str | None, strict: bool) -> None:
    """Initialize a capseal.intent.json file for a project."""
    project_path = Path(project_dir).resolve()
    intent_path = project_path / "capseal.intent.json"

    if intent_path.exists():
        click.echo(f"Intent file already exists: {intent_path}")
        if not click.confirm("Overwrite?"):
            return

    # Extract profile to suggest values
    profile = extract_profile(project_path)

    intent = {
        "schema": INTENT_SCHEMA_VERSION,
        "purpose": purpose or profile["readme"].get("first_paragraph", "")[:200] or "Describe project purpose",
        "allowed_providers": [],
        "security_invariants": {
            "no_raw_sql": strict,
            "no_shell_true": strict,
            "no_dynamic_import": False,  # Often legitimate
            "no_eval_exec": strict,
            "import_allowlist": [],
        },
        "network_policy": "outbound_only",
        "file_policy": "any",
    }

    # Add comments about detected patterns
    hot = profile["hot_callsites"]
    if hot.get("sql"):
        intent["_note_sql"] = f"Detected {len(hot['sql'])} SQL callsites - review before enabling no_raw_sql"
    if hot.get("shell"):
        intent["_note_shell"] = f"Detected {len(hot['shell'])} shell=True callsites - review before enabling no_shell_true"
    if hot.get("dynamic_import"):
        intent["_note_imports"] = f"Detected {len(hot['dynamic_import'])} dynamic imports"

    intent_path.write_text(json.dumps(intent, indent=2))
    click.echo(f"Created: {intent_path}")
    click.echo()
    click.echo("Edit this file to define your project's expected behavior.")
    click.echo("Then run: capseal conformance " + str(project_path))
