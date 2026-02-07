"""CapSeal Contract Lock v0.1 - Stable interfaces for receipts, policies, and CLI.

This module defines the stable contracts that MUST remain backward-compatible
across minor versions. Breaking changes require major version bumps.

Contract Categories:
- RECEIPT_SCHEMA: Receipt JSON structure and version compatibility
- POLICY_SCHEMA: Policy document structure
- CLI_SURFACE: Command names, flags, and exit codes
- WORKSPACE_LAYOUT: File paths and environment variables
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# =============================================================================
# VERSION CONTRACTS
# =============================================================================

CAPSEAL_VERSION = "0.2.0"
RECEIPT_SCHEMA_VERSION = "v2"
POLICY_SCHEMA_VERSION = "v1"
PROOF_FORMAT_VERSION = "fri_v1"

# Compatibility window: verifier supports receipts from these schema versions
SUPPORTED_RECEIPT_SCHEMAS = ["v1", "v2"]
SUPPORTED_POLICY_SCHEMAS = ["v1"]


# =============================================================================
# RECEIPT SCHEMA CONTRACT
# =============================================================================

@dataclass
class ReceiptContract:
    """
    Receipt JSON contract - fields that MUST be present in all receipts.

    A valid receipt MUST contain:
    - schema_version: string (e.g., "v2")
    - capseal_version: string (e.g., "0.2.0")
    - header: CapsuleHeader
    - payload: CapsulePayload
    - signature: optional signature block

    The verifier MUST accept receipts with schema_version in SUPPORTED_RECEIPT_SCHEMAS.
    """
    # Required top-level fields
    REQUIRED_FIELDS = frozenset([
        "schema",           # "capsule_receipt_v2"
        "capsule_id",       # SHA256 hash of (header_commit_hash, payload_hash)
        "header",           # Full header object
        "payload",          # Proof and trace data
    ])

    # Header required fields
    HEADER_REQUIRED = frozenset([
        "schema",           # "capsule_header_v2"
        "vm_id",            # VM identifier
        "backend_id",       # Backend identifier
        "trace_id",         # Unique trace ID
        "trace_spec_hash",  # Hash of trace specification
        "statement_hash",   # Hash of public statement
        "row_commitment",   # Merkle commitment to trace rows
        "policy_ref",       # Policy reference
    ])

    # Payload required fields
    PAYLOAD_REQUIRED = frozenset([
        "proof_format",     # "fri_v1", "nova_v1", etc.
        "proof_data",       # Serialized proof bytes (base64 or hex)
    ])


# =============================================================================
# POLICY SCHEMA CONTRACT
# =============================================================================

@dataclass
class PolicyContract:
    """
    Policy JSON contract - structure for benchmark policies.

    A valid policy MUST contain:
    - schema: "bef_benchmark_policy_v1"
    - policy_id: unique identifier
    - policy_version: semver string
    - tracks: list of track definitions
    """
    REQUIRED_FIELDS = frozenset([
        "schema",
        "policy_id",
        "policy_version",
        "tracks",
    ])

    # Track required fields
    TRACK_REQUIRED = frozenset([
        "track_id",
        "rules",
    ])

    # Known rule keys (not all required)
    KNOWN_RULES = frozenset([
        "forbid_gpu",
        "require_deterministic_build",
        "required_public_outputs",
        "allowed_data_sources",
        "max_runtime_seconds",
        "network_policy",
    ])


# =============================================================================
# CLI SURFACE CONTRACT
# =============================================================================

class ExitCode(Enum):
    """
    Stable exit codes for CI integration.

    These codes MUST NOT change between minor versions.
    New codes may be added, but existing codes are immutable.
    """
    VERIFIED = 0            # Receipt verified successfully
    PROOF_INVALID = 10      # Cryptographic proof failed verification
    POLICY_MISMATCH = 11    # Policy validation failed
    COMMITMENT_FAILED = 12  # Merkle commitment verification failed
    DA_AUDIT_FAILED = 13    # Data availability audit failed
    REPLAY_DIVERGED = 14    # Replay produced different results
    MALFORMED = 20          # Parse error or malformed input
    RUNTIME_ERROR = 30      # Sandbox/runtime failure
    NETWORK_ERROR = 31      # Network fetch failed


@dataclass
class CLIContract:
    """
    CLI command contract - stable command names and core flags.

    Commands in STABLE_COMMANDS must maintain backward compatibility.
    Flags can be added but not removed from stable commands.
    """
    # Commands that MUST exist and maintain backward compatibility
    STABLE_COMMANDS = frozenset([
        "init",     # Initialize workspace
        "fetch",    # Fetch datasets with network governance
        "run",      # Generate proof
        "verify",   # Verify receipt
        "replay",   # Semantic replay verification
        "doctor",   # Environment diagnostics
        "demo",     # Self-contained demo (offline)
        "inspect",  # Display receipt metadata
        "explain",  # Human-readable verification report
    ])

    # Global flags available on all commands
    GLOBAL_FLAGS = frozenset([
        "--help",
        "--version",
        "--json",     # JSON output mode
        "--verbose",  # Verbose output
        "--quiet",    # Suppress non-essential output
    ])

    # verify command required flags
    VERIFY_FLAGS = frozenset([
        # Positional: receipt path
        "--policy",   # Optional policy override
        "--strict",   # Fail on warnings
    ])

    # run command required flags
    RUN_FLAGS = frozenset([
        "-p", "--policy",      # Policy file (required)
        "--policy-id",         # Policy ID to use
        "-d", "--data",        # Data directory
        "-o", "--output",      # Output directory
        "--backend",           # Force sandbox backend
    ])


# =============================================================================
# WORKSPACE LAYOUT CONTRACT
# =============================================================================

@dataclass
class WorkspaceContract:
    """
    Workspace layout contract - standard directories and files.

    CAPSEAL_HOME defaults to ~/.capseal or $CAPSEAL_HOME if set.
    Project workspace is .capseal/ in project root.
    """
    # Environment variables
    ENV_CAPSEAL_HOME = "CAPSEAL_HOME"
    ENV_CAPSEAL_WORKSPACE = "CAPSEAL_WORKSPACE"
    ENV_CAPSEAL_BACKEND = "CAPSEAL_BACKEND"
    ENV_CAPSEAL_POLICY = "CAPSEAL_POLICY"

    # Default workspace paths (relative to project root)
    PROJECT_WORKSPACE = ".capseal"

    # Workspace subdirectories
    RUNS_DIR = "runs"           # runs/<run_id>/
    DATASETS_DIR = "datasets"   # datasets/<dataset_id>/
    POLICIES_DIR = "policies"   # policies/
    RECEIPTS_DIR = "receipts"   # receipts/

    # Standard files in workspace
    CONFIG_FILE = "config.json"
    EVENTS_LOG = "mcp_events.jsonl"

    # Run directory structure
    RUN_MANIFEST = "manifest.json"
    RUN_RECEIPT = "receipt.json"
    RUN_EVENTS = "events.jsonl"
    RUN_ARTIFACTS = "artifacts/"

    # Dataset directory structure
    DATASET_ROOT = "root.json"
    DATASET_NETLOG = "netlog.jsonl"


# =============================================================================
# RUNTIME BACKEND CONTRACT
# =============================================================================

class RuntimeBackend(Enum):
    """
    Sandbox backend identifiers.

    The runtime MUST report which backend was used in the receipt.
    """
    BUBBLEWRAP = "bwrap"
    FIREJAIL = "firejail"
    NSJAIL = "nsjail"
    DOCKER = "docker"
    SANDBOX_EXEC = "sandbox-exec"  # macOS
    NONE = "none"                   # No sandbox (development only)


@dataclass
class RuntimeContract:
    """
    Runtime backend contract - interface that all backends must implement.
    """
    # Methods every backend must provide
    REQUIRED_METHODS = frozenset([
        "run",              # Execute command in sandbox
        "is_available",     # Check if backend is available
        "get_info",         # Get backend version/config info
    ])

    # Result fields from run()
    RESULT_FIELDS = frozenset([
        "returncode",
        "stdout",
        "stderr",
        "backend",
        "resource_usage",
    ])


# =============================================================================
# VERIFICATION PROFILES
# =============================================================================

class VerificationProfile(Enum):
    """
    Verification profiles - define what checks are performed.

    Profiles allow for different verification strictness levels.
    """
    MINIMAL = "minimal"       # Proof + commitment only
    STANDARD = "standard"     # + policy + authorship
    STRICT = "strict"         # + DA audit + replay
    AUDIT = "audit"           # Full forensic verification


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_schema_supported(schema: str, category: Literal["receipt", "policy"]) -> bool:
    """Check if a schema version is supported by this verifier."""
    if category == "receipt":
        # Extract version from "capsule_receipt_v2" -> "v2"
        if schema.startswith("capsule_receipt_"):
            version = schema.replace("capsule_receipt_", "")
            return version in SUPPORTED_RECEIPT_SCHEMAS
        if schema.startswith("capsule_header_"):
            version = schema.replace("capsule_header_", "")
            return version in SUPPORTED_RECEIPT_SCHEMAS
    elif category == "policy":
        if schema.startswith("bef_benchmark_policy_"):
            version = schema.replace("bef_benchmark_policy_", "")
            return version in SUPPORTED_POLICY_SCHEMAS
    return False


def get_workspace_path(name: str = None) -> str:
    """Get the workspace path for a given component."""
    import os
    from pathlib import Path

    # Check for explicit workspace env var
    if os.environ.get(WorkspaceContract.ENV_CAPSEAL_WORKSPACE):
        base = Path(os.environ[WorkspaceContract.ENV_CAPSEAL_WORKSPACE])
    else:
        # Default to .capseal in current directory
        base = Path.cwd() / WorkspaceContract.PROJECT_WORKSPACE

    if name is None:
        return str(base)

    paths = {
        "runs": base / WorkspaceContract.RUNS_DIR,
        "datasets": base / WorkspaceContract.DATASETS_DIR,
        "policies": base / WorkspaceContract.POLICIES_DIR,
        "receipts": base / WorkspaceContract.RECEIPTS_DIR,
        "config": base / WorkspaceContract.CONFIG_FILE,
        "events": base / WorkspaceContract.EVENTS_LOG,
    }

    return str(paths.get(name, base / name))


__all__ = [
    "CAPSEAL_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "PROOF_FORMAT_VERSION",
    "SUPPORTED_RECEIPT_SCHEMAS",
    "SUPPORTED_POLICY_SCHEMAS",
    "ReceiptContract",
    "PolicyContract",
    "CLIContract",
    "WorkspaceContract",
    "RuntimeContract",
    "RuntimeBackend",
    "ExitCode",
    "VerificationProfile",
    "is_schema_supported",
    "get_workspace_path",
]
