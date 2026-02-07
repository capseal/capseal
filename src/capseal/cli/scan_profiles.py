"""Scan profiles — map profile names to semgrep rule sets.

Profiles:
    security  - OWASP, injection, deserialization, secrets
    quality   - Style, complexity, refactoring opportunities
    bugs      - Type errors, null checks, dead code, race conditions
    all       - Everything above
    custom    - User-provided rules path
"""
from __future__ import annotations

from pathlib import Path

# Profile name → list of semgrep --config values
SCAN_PROFILES: dict[str, list[str]] = {
    "security": [
        "p/security-audit",
        "p/owasp-top-ten",
    ],
    "quality": [
        "p/python",
        "p/refactoring",
    ],
    "bugs": [
        "p/python-bugs",
        "p/deadcode",
    ],
}

# "all" is the union of all built-in profiles
SCAN_PROFILES["all"] = sorted(
    set(rule for rules in SCAN_PROFILES.values() for rule in rules)
)

PROFILE_NAMES = list(SCAN_PROFILES.keys()) + ["custom"]

PROFILE_DISPLAY = {
    "security": "Security issues",
    "quality": "Code quality",
    "bugs": "Bug detection",
    "all": "Everything",
    "custom": "Custom rules",
}


def resolve_semgrep_configs(
    profile: str | None = None,
    custom_rules: str | None = None,
    config_json: dict | None = None,
) -> list[str]:
    """Resolve profile + custom_rules into a list of --config values.

    Priority:
        1. Explicit custom_rules path (if profile == "custom")
        2. Explicit profile name
        3. Profile from config.json
        4. Default: "auto" (semgrep's built-in)
    """
    # Determine effective profile
    effective = profile
    if not effective and config_json:
        effective = config_json.get("scan_profile")

    if not effective:
        return ["auto"]

    if effective == "custom":
        if custom_rules:
            return [custom_rules]
        # Check config for custom_rules path
        if config_json and config_json.get("scan_rules"):
            return [config_json["scan_rules"]]
        return ["auto"]

    configs = SCAN_PROFILES.get(effective)
    if configs:
        return list(configs)

    # Unknown profile name — treat as a semgrep config path
    return [effective]


def build_semgrep_args(
    target_path: str | Path,
    profile: str | None = None,
    custom_rules: str | None = None,
    config_json: dict | None = None,
) -> list[str]:
    """Build the full semgrep command line for a scan.

    Returns the complete argv list (including 'semgrep').
    """
    configs = resolve_semgrep_configs(profile, custom_rules, config_json)

    cmd = ["semgrep"]
    for cfg in configs:
        cmd.extend(["--config", cfg])
    cmd.extend([
        "--json",
        "--exclude", "node_modules",
        "--exclude", ".venv",
        "--exclude", "vendor",
        str(target_path),
    ])
    return cmd


__all__ = [
    "SCAN_PROFILES",
    "PROFILE_NAMES",
    "PROFILE_DISPLAY",
    "resolve_semgrep_configs",
    "build_semgrep_args",
]
