"""Workflow Engine for CapSeal v0.3.

This module implements DAG-of-agents where every subagent emits a verifiable packet.
The head agent orchestrates but never becomes a trust bottleneck - every output is
independently hashed, provenance-bound, and verifiable.

Architecture:
- Agent = tool invocation that produces artifacts with receipts
- Node = (inputs, policy, budget, executor, outputs, receipt)
- Edges = content-addressed pointers (hashes, not blobs)
- Workflow = DAG of nodes executed in topological order
- Rollup = cryptographic commitment to entire DAG + all receipts

Key invariants:
- Same inputs + policy + executor → same output (memoizable)
- Every claim points to evidence by hash
- Compaction preserves provenance via evidence_index
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from capseal.explain_pipeline import run_explain_pipeline


# =============================================================================
# Schema Versions
# =============================================================================

WORKFLOW_SCHEMA_VERSION = "workflow_v1"
AGENT_PACKET_SCHEMA_VERSION = "agent_packet_v1"
NODE_RECEIPT_SCHEMA_VERSION = "node_receipt_v1"
COMPACTION_SCHEMA_VERSION = "compaction_v1"
WORKFLOW_ROLLUP_SCHEMA_VERSION = "workflow_rollup_v1"


# =============================================================================
# Hashing Utilities
# =============================================================================

def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON bytes for hashing.

    Rules for canonicalization:
    1. Sort all object keys alphabetically
    2. No whitespace (use separators=(",", ":"))
    3. Recursively sort nested objects/lists containing objects
    4. Convert sets to sorted lists
    5. Use consistent float formatting

    This ensures identical objects produce identical hashes.
    """
    def _canonicalize(val: Any) -> Any:
        if isinstance(val, dict):
            return {k: _canonicalize(v) for k, v in sorted(val.items())}
        elif isinstance(val, (list, tuple)):
            return [_canonicalize(v) for v in val]
        elif isinstance(val, set):
            return sorted([_canonicalize(v) for v in val])
        elif isinstance(val, float):
            # Consistent float representation
            if val == int(val):
                return int(val)
            return val
        return val

    return json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    """Canonical JSON hash."""
    return sha256_str(canonical_json(obj))


def sha256_file(path: Path) -> str:
    """Hash file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_finding_key(finding: dict) -> tuple:
    """Generate canonical sort key for a finding.

    Order by: (rule_id, file_path, start_line, fingerprint)
    This ensures deterministic ordering across runs.
    """
    return (
        finding.get("rule_id", ""),
        finding.get("file_path", ""),
        finding.get("line_range", [0])[0] if finding.get("line_range") else 0,
        finding.get("finding_fingerprint", ""),
    )


def sort_findings(findings: list[dict]) -> list[dict]:
    """Sort findings canonically for stable hashes."""
    return sorted(findings, key=canonical_finding_key)


def sort_shards(shards: list[dict]) -> list[dict]:
    """Sort shards by shard_id for stable ordering."""
    return sorted(shards, key=lambda s: s.get("shard_id", 0))


# =============================================================================
# Environment Fingerprint (Measured)
# =============================================================================

def _find_lockfile(start_dir: Path = None) -> Optional[tuple[str, Path]]:
    """Find the nearest lockfile/requirements file.

    Search order: uv.lock, poetry.lock, Pipfile.lock, requirements.txt
    Returns (lockfile_type, path) or None.
    """
    start = start_dir or Path.cwd()
    lockfiles = [
        ("uv.lock", "uv.lock"),
        ("poetry.lock", "poetry.lock"),
        ("Pipfile.lock", "Pipfile.lock"),
        ("requirements.txt", "requirements.txt"),
    ]

    for dir_path in [start] + list(start.parents)[:5]:  # Check up to 5 levels
        for lock_type, filename in lockfiles:
            candidate = dir_path / filename
            if candidate.exists():
                return (lock_type, candidate)
    return None


def _get_pip_freeze_hash() -> str:
    """Get hash of pip freeze output (installed packages)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Sort lines for stability (pip freeze order can vary)
            lines = sorted(result.stdout.strip().split("\n"))
            return sha256_str("\n".join(lines))
    except Exception:
        pass
    return ""


def _get_container_digest() -> str:
    """Get container image digest if running in a container.

    Checks:
    1. CONTAINER_IMAGE_DIGEST env var (set by CI)
    2. /.dockerenv presence + /proc/1/cgroup
    """
    # First check if explicitly set
    digest = os.environ.get("CONTAINER_IMAGE_DIGEST", "")
    if digest:
        return digest

    # Check if running in container
    if not Path("/.dockerenv").exists():
        return ""

    # Try to get image info from cgroup
    try:
        cgroup_path = Path("/proc/1/cgroup")
        if cgroup_path.exists():
            content = cgroup_path.read_text()
            # Look for docker/podman container ID
            for line in content.split("\n"):
                if "docker" in line or "containerd" in line:
                    # Return the container ID as pseudo-digest
                    parts = line.split("/")
                    if parts:
                        return f"container:{parts[-1][:12]}"
    except Exception:
        pass

    return "container:unknown"


def get_environment_fingerprint(
    project_dir: Path = None,
    include_pip_freeze: bool = True,
    include_lockfile: bool = True,
) -> dict[str, Any]:
    """Capture measured environment fingerprint.

    This creates verifiable claims about the execution environment:
    - Declared: version strings (could be faked)
    - Measured: hashes of actual files/outputs (harder to fake)

    The measured hashes prevent silent dependency drift.

    Args:
        project_dir: Directory to search for lockfiles (default: cwd)
        include_pip_freeze: Include hash of pip freeze output
        include_lockfile: Include hash of lockfile contents
    """
    import platform

    fingerprint: dict[str, Any] = {
        # Declared (for human readability)
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "capseal_version": "0.3.0",
        # Measurement timestamps
        "measured_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # Measured: lockfile hash
    if include_lockfile:
        lockfile = _find_lockfile(project_dir)
        if lockfile:
            lock_type, lock_path = lockfile
            fingerprint["lockfile_type"] = lock_type
            fingerprint["lockfile_path"] = str(lock_path)
            fingerprint["lockfile_hash"] = sha256_file(lock_path)

    # Measured: pip freeze hash
    if include_pip_freeze:
        pip_hash = _get_pip_freeze_hash()
        if pip_hash:
            fingerprint["pip_freeze_hash"] = pip_hash

    # Measured: container digest (if in container)
    container = _get_container_digest()
    if container:
        fingerprint["container_digest"] = container

    # Tool versions (declared but useful)
    try:
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            fingerprint["semgrep_version"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return fingerprint


def get_env_fingerprint_hash(project_dir: Path = None) -> str:
    """Get hash of environment fingerprint.

    Note: This hash is for memoization/comparison. The fingerprint
    includes measured_at timestamp, so the hash will change between
    runs. Use specific field hashes for memoization.
    """
    fp = get_environment_fingerprint(project_dir)
    # Exclude timestamp for hash stability
    hashable = {k: v for k, v in fp.items() if k != "measured_at"}
    return sha256_json(hashable)


def get_env_memoization_key(project_dir: Path = None) -> str:
    """Get a stable key for memoization based on measured environment.

    This key changes when:
    - Python version changes
    - Lockfile changes
    - Installed packages change
    - Running in different container

    This prevents serving cached results from stale environments.
    """
    fp = get_environment_fingerprint(project_dir)
    key_parts = [
        fp.get("python_version", ""),
        fp.get("lockfile_hash", ""),
        fp.get("pip_freeze_hash", ""),
        fp.get("container_digest", ""),
    ]
    return sha256_str("|".join(key_parts))


# =============================================================================
# Agent Packet Schema
# =============================================================================

@dataclass
class AgentPacket:
    """Output from any agent node - the atomic unit of verifiable work.

    Every agent (semgrep shard, LLM explain, profile extract, etc.) must emit
    this structure. The receipt binds inputs → outputs cryptographically.
    """
    schema: str = AGENT_PACKET_SCHEMA_VERSION
    node_id: str = ""
    task_id: str = ""
    agent_type: str = ""  # e.g., "semgrep", "llm_explain", "profile_extract"

    # Input provenance
    input_hash: str = ""  # Hash of exact inputs (trace_root + config + selected chunks)
    input_manifest: dict = field(default_factory=dict)  # Detailed input breakdown

    # Execution context
    executor_id: str = ""  # e.g., "semgrep:auto@1.149.0", "openai:gpt-4o-mini"
    policy_hash: str = ""  # Hash of policy constraints
    budget: dict = field(default_factory=dict)  # Token/time/compute limits

    # Outputs
    output_hash: str = ""  # Hash of primary output artifact
    output_path: str = ""  # Relative path within run directory
    artifacts: list = field(default_factory=list)  # [{path, hash, size, type}]

    # LLM-specific (optional)
    prompt_hash: str = ""
    raw_response_hash: str = ""

    # Evidence references
    evidence_refs: list = field(default_factory=list)  # Pointers to trace chunks used

    # Metadata
    timestamp: str = ""
    duration_ms: int = 0
    tool_version: str = "capseal-0.3.0"

    # Determinism classification
    determinism: str = "deterministic"  # deterministic, non_deterministic, external
    env_fingerprint: str = ""  # Hash of execution environment

    def compute_receipt_hash(self) -> str:
        """Compute deterministic receipt hash."""
        receipt_data = {
            "schema": self.schema,
            "node_id": self.node_id,
            "agent_type": self.agent_type,
            "input_hash": self.input_hash,
            "executor_id": self.executor_id,
            "policy_hash": self.policy_hash,
            "output_hash": self.output_hash,
            "artifacts": sorted(self.artifacts, key=lambda a: a.get("path", "")),
            "determinism": self.determinism,
            "env_fingerprint": self.env_fingerprint,
        }
        return sha256_json(receipt_data)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "node_id": self.node_id,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "input_hash": self.input_hash,
            "input_manifest": self.input_manifest,
            "executor_id": self.executor_id,
            "policy_hash": self.policy_hash,
            "budget": self.budget,
            "output_hash": self.output_hash,
            "output_path": self.output_path,
            "artifacts": self.artifacts,
            "prompt_hash": self.prompt_hash,
            "raw_response_hash": self.raw_response_hash,
            "evidence_refs": self.evidence_refs,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "tool_version": self.tool_version,
            "determinism": self.determinism,
            "env_fingerprint": self.env_fingerprint,
            "receipt_hash": self.compute_receipt_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPacket":
        packet = cls()
        for key, value in data.items():
            if key != "receipt_hash" and hasattr(packet, key):
                setattr(packet, key, value)
        return packet


# =============================================================================
# Node Definition
# =============================================================================

class NodeKind(Enum):
    """Built-in node types."""
    TRACE = "trace"
    PROFILE_EXTRACT = "agent.profile_extract"
    INTENT_CHECK = "agent.intent_check"
    SEMGREP_REVIEW = "review.semgrep"
    LLM_REVIEW = "review.llm"
    EXPLAIN_LLM = "explain_llm"
    REVIEW_DIFF = "review_diff"
    COMPACTION = "compaction"
    RUN_TESTS = "run_tests"
    PATCH_SUGGEST = "patch_suggest"
    CUSTOM = "custom"


class Determinism(Enum):
    """Node determinism classification.

    DETERMINISTIC: Same inputs always produce same outputs (semgrep, profile extract)
    NON_DETERMINISTIC: Outputs may vary (LLM calls) - receipt makes specific output verifiable
    EXTERNAL: Depends on external state (network, time) - receipt captures point-in-time
    """
    DETERMINISTIC = "deterministic"
    NON_DETERMINISTIC = "non_deterministic"
    EXTERNAL = "external"


class RequiredMode(Enum):
    """Node requirement policy.

    REQUIRED: Must PASS for workflow to succeed
    OPTIONAL: SKIP allowed without failing workflow
    REQUIRED_IF_PRESENT: Must PASS if inputs exist, otherwise SKIP is ok
    ADVISORY: Runs if possible, never fails workflow (informational only)
    """
    REQUIRED = "required"
    OPTIONAL = "optional"
    REQUIRED_IF_PRESENT = "required_if_present"
    ADVISORY = "advisory"


class SkipReason(Enum):
    """Why a node was skipped.

    MISSING_INPUT: Required input dependency unavailable
    NOT_APPLICABLE: Node's preconditions not met (e.g., no intent file)
    DISABLED_BY_POLICY: Explicitly disabled in workflow config
    DEPENDENCY_FAILED: A required dependency failed
    BUDGET_EXHAUSTED: Would exceed resource budget
    """
    MISSING_INPUT = "missing_input"
    NOT_APPLICABLE = "not_applicable"
    DISABLED_BY_POLICY = "disabled_by_policy"
    DEPENDENCY_FAILED = "dependency_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class NodeStatus(Enum):
    """Terminal node status."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


# Map node kinds to their determinism
NODE_DETERMINISM = {
    "trace": Determinism.DETERMINISTIC,
    "agent.profile_extract": Determinism.DETERMINISTIC,
    "agent.intent_check": Determinism.DETERMINISTIC,
    "review.semgrep": Determinism.DETERMINISTIC,
    "review.llm": Determinism.NON_DETERMINISTIC,
    "explain_llm": Determinism.NON_DETERMINISTIC,
    "review_diff": Determinism.DETERMINISTIC,
    "run_tests": Determinism.EXTERNAL,
    "patch_suggest": Determinism.NON_DETERMINISTIC,
    "committor.gate": Determinism.DETERMINISTIC,
}


@dataclass
class NodeSpec:
    """Specification for a workflow node."""
    id: str
    kind: str
    needs: list = field(default_factory=list)  # Dependencies (node IDs)
    params: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    required_mode: str = "required"  # required, optional, required_if_present, advisory
    determinism: str = ""  # Override default determinism if needed

    # Backward compatibility
    @property
    def required(self) -> bool:
        return self.required_mode in (RequiredMode.REQUIRED.value, "required", True)

    @property
    def is_deterministic(self) -> bool:
        if self.determinism:
            return self.determinism == Determinism.DETERMINISTIC.value
        default = NODE_DETERMINISM.get(self.kind, Determinism.DETERMINISTIC)
        return default == Determinism.DETERMINISTIC

    def get_required_mode(self) -> RequiredMode:
        """Get RequiredMode enum value."""
        try:
            return RequiredMode(self.required_mode)
        except ValueError:
            # Backward compatibility for boolean
            return RequiredMode.REQUIRED if self.required_mode else RequiredMode.OPTIONAL

    @classmethod
    def from_dict(cls, data: dict) -> "NodeSpec":
        # Handle backward compatibility: convert boolean 'required' to required_mode
        required_mode = data.get("required_mode", "")
        if not required_mode:
            required_val = data.get("required", True)
            if isinstance(required_val, bool):
                required_mode = "required" if required_val else "optional"
            else:
                required_mode = str(required_val)

        return cls(
            id=data.get("id", ""),
            kind=data.get("kind", ""),
            needs=data.get("needs", []),
            determinism=data.get("determinism", ""),
            params=data.get("params", {}),
            policy=data.get("policy", {}),
            budget=data.get("budget", {}),
            required_mode=required_mode,
        )


@dataclass
class StatusDetail:
    """Detailed status information for a node result."""
    skip_reason: Optional[str] = None  # SkipReason value if skipped
    missing_inputs: list = field(default_factory=list)  # IDs of missing deps
    required_mode: str = "required"  # RequiredMode value
    failure_severity: str = "error"  # error, warning, info
    failure_code: str = ""  # Machine-readable error code

    def to_dict(self) -> dict:
        return {
            "skip_reason": self.skip_reason,
            "missing_inputs": self.missing_inputs,
            "required_mode": self.required_mode,
            "failure_severity": self.failure_severity,
            "failure_code": self.failure_code,
        }


@dataclass
class NodeResult:
    """Result of executing a node."""
    node_id: str
    success: bool
    packet: Optional[AgentPacket] = None
    error: str = ""
    skipped: bool = False
    cached: bool = False
    reason: str = ""
    status: str = field(init=False)
    status_detail: StatusDetail = field(default_factory=StatusDetail)

    def __post_init__(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        if self.skipped:
            self.status = NodeStatus.SKIP.value
        elif self.success:
            self.status = NodeStatus.PASS.value
        else:
            self.status = NodeStatus.FAIL.value

        if not self.reason:
            if self.error and self.status != NodeStatus.PASS.value:
                self.reason = self.error
            elif self.skipped:
                if self.status_detail.skip_reason:
                    self.reason = f"Skipped: {self.status_detail.skip_reason}"
                else:
                    self.reason = "Skipped"

    @classmethod
    def skip(
        cls,
        node_id: str,
        skip_reason: SkipReason,
        reason: str = "",
        missing_inputs: list = None,
        required_mode: str = "optional",
    ) -> "NodeResult":
        """Create a SKIP result with proper detail."""
        result = cls(
            node_id=node_id,
            success=False,
            skipped=True,
            reason=reason,
            status_detail=StatusDetail(
                skip_reason=skip_reason.value,
                missing_inputs=missing_inputs or [],
                required_mode=required_mode,
            ),
        )
        return result

    @classmethod
    def fail(
        cls,
        node_id: str,
        error: str,
        failure_code: str = "",
        failure_severity: str = "error",
        required_mode: str = "required",
    ) -> "NodeResult":
        """Create a FAIL result with proper detail."""
        return cls(
            node_id=node_id,
            success=False,
            error=error,
            status_detail=StatusDetail(
                required_mode=required_mode,
                failure_severity=failure_severity,
                failure_code=failure_code,
            ),
        )


# =============================================================================
# Workflow Definition
# =============================================================================

@dataclass
class WorkflowSpec:
    """Complete workflow specification."""
    schema: str = WORKFLOW_SCHEMA_VERSION
    name: str = ""
    description: str = ""
    inputs: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)  # List of NodeSpec

    @classmethod
    def from_yaml(cls, path: Path) -> "WorkflowSpec":
        """Load workflow from YAML file."""
        content = path.read_text()
        data = yaml.safe_load(content)

        spec = cls(
            schema=data.get("schema", WORKFLOW_SCHEMA_VERSION),
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            inputs=data.get("inputs", {}),
        )

        for task_data in data.get("tasks", []):
            spec.tasks.append(NodeSpec.from_dict(task_data))

        return spec

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowSpec":
        spec = cls(
            schema=data.get("schema", WORKFLOW_SCHEMA_VERSION),
            name=data.get("name", ""),
            description=data.get("description", ""),
            inputs=data.get("inputs", {}),
        )
        for task_data in data.get("tasks", []):
            spec.tasks.append(NodeSpec.from_dict(task_data))
        return spec

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "tasks": [
                {
                    "id": t.id,
                    "kind": t.kind,
                    "needs": t.needs,
                    "params": t.params,
                    "policy": t.policy,
                    "budget": t.budget,
                    "required_mode": t.required_mode,
                }
                for t in self.tasks
            ],
        }

    def topological_order(self) -> list[str]:
        """Return task IDs in topological order."""
        # Build adjacency
        graph = {t.id: set(t.needs) for t in self.tasks}

        # Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(order) != len(graph):
            raise ValueError("Workflow has cycles")

        return order


# =============================================================================
# Memoization Cache
# =============================================================================

class MemoizationCache:
    """Content-addressed cache for node outputs.

    Key insight: if input_hash + executor_id + policy_hash are identical,
    we can reuse the output without re-executing.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = cache_dir / "memo_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def compute_cache_key(
        self,
        node_id: str,
        input_hash: str,
        executor_id: str,
        policy_hash: str,
        env_key: str = "",
    ) -> str:
        """Compute deterministic cache key.

        The key includes environment fingerprint to prevent serving
        cached results from different environments.
        """
        # If no env_key provided, compute it
        if not env_key:
            env_key = get_env_memoization_key()
        key_data = f"{node_id}|{input_hash}|{executor_id}|{policy_hash}|{env_key}"
        return sha256_str(key_data)

    def get(self, cache_key: str) -> Optional[AgentPacket]:
        """Retrieve cached packet if available."""
        if cache_key not in self.index:
            return None

        entry = self.index[cache_key]
        packet_path = self.cache_dir / entry["packet_path"]

        if not packet_path.exists():
            del self.index[cache_key]
            self._save_index()
            return None

        try:
            data = json.loads(packet_path.read_text())
            return AgentPacket.from_dict(data)
        except Exception:
            return None

    def put(self, cache_key: str, packet: AgentPacket) -> None:
        """Store packet in cache."""
        packet_filename = f"{cache_key[:16]}_packet.json"
        packet_path = self.cache_dir / packet_filename

        packet_path.write_text(json.dumps(packet.to_dict(), indent=2))

        self.index[cache_key] = {
            "packet_path": packet_filename,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "node_id": packet.node_id,
            "input_hash": packet.input_hash,
        }
        self._save_index()


# =============================================================================
# Compaction Node
# =============================================================================

@dataclass
class CompactionResult:
    """Result of compaction: summary + evidence index.

    This is the key to preventing post-compaction fragmentation.
    The summary is what goes in-context; the evidence_index provides
    hash-addressed pointers to retrieve details on demand.
    """
    schema: str = COMPACTION_SCHEMA_VERSION
    timestamp: str = ""

    # What goes in-context (fixed schema, bounded size)
    state_summary: dict = field(default_factory=dict)

    # Hash-addressed pointers to evidence (for retrieval)
    evidence_index: dict = field(default_factory=dict)

    # Source receipts
    source_receipt_hashes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "timestamp": self.timestamp,
            "state_summary": self.state_summary,
            "evidence_index": self.evidence_index,
            "source_receipt_hashes": self.source_receipt_hashes,
            "compaction_hash": sha256_json({
                "state_summary": self.state_summary,
                "evidence_index": self.evidence_index,
                "source_receipt_hashes": self.source_receipt_hashes,
            }),
        }


def build_compaction(
    node_results: dict[str, NodeResult],
    findings: list[dict],
    gate_status: str,
    run_dir: Path,
) -> CompactionResult:
    """Build compaction from node results.

    This creates a fixed-schema summary that can be carried in-context,
    plus an evidence_index that maps claims to hash-addressed artifacts.
    """
    result = CompactionResult(
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )

    # Collect receipt hashes
    for node_id, node_result in node_results.items():
        if node_result.packet:
            result.source_receipt_hashes.append(node_result.packet.compute_receipt_hash())

    # Build state summary (fixed schema, bounded)
    top_risks = []
    risk_by_rule = {}
    for f in findings:
        rule = f.get("rule_id", "unknown")
        severity = f.get("severity", "info")
        if rule not in risk_by_rule:
            risk_by_rule[rule] = {"rule": rule, "severity": severity, "count": 0, "fingerprints": []}
        risk_by_rule[rule]["count"] += 1
        risk_by_rule[rule]["fingerprints"].append(f.get("finding_fingerprint", ""))

    # Sort by severity then count
    severity_order = {"error": 0, "warning": 1, "info": 2}
    top_risks = sorted(
        risk_by_rule.values(),
        key=lambda r: (severity_order.get(r["severity"], 99), -r["count"]),
    )[:10]  # Top 10

    # Remove fingerprints from summary (they go in evidence_index)
    summary_risks = []
    for risk in top_risks:
        summary_risks.append({
            "rule": risk["rule"],
            "severity": risk["severity"],
            "count": risk["count"],
        })

    result.state_summary = {
        "gate_status": gate_status,
        "total_findings": len(findings),
        "findings_by_severity": {
            "error": sum(1 for f in findings if f.get("severity") == "error"),
            "warning": sum(1 for f in findings if f.get("severity") == "warning"),
            "info": sum(1 for f in findings if f.get("severity") == "info"),
        },
        "top_risks": summary_risks,
        "nodes_executed": len(node_results),
        "nodes_succeeded": sum(1 for r in node_results.values() if r.success),
    }

    status_counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for node_result in node_results.values():
        status_counts[node_result.status] = status_counts.get(node_result.status, 0) + 1
    result.state_summary["node_status_counts"] = status_counts

    # Build evidence index (maps summary claims to hash pointers)
    for i, risk in enumerate(top_risks):
        key = f"top_risks[{i}]"
        result.evidence_index[key] = {
            "rule_id": risk["rule"],
            "fingerprints": risk["fingerprints"][:20],  # Cap at 20
            "aggregate_path": "reviews/aggregate.json",
        }

    # Add node receipt pointers
    for node_id, node_result in node_results.items():
        if node_result.packet:
            result.evidence_index[f"node:{node_id}"] = {
                "receipt_hash": node_result.packet.compute_receipt_hash(),
                "output_path": node_result.packet.output_path,
                "artifacts": node_result.packet.artifacts,
            }

    return result


# =============================================================================
# Node Executors
# =============================================================================

class NodeExecutor:
    """Base class for node executors."""

    def __init__(self, run_dir: Path, project_dir: Path, cache: MemoizationCache):
        self.run_dir = run_dir
        self.project_dir = project_dir
        self.cache = cache

    def execute(
        self,
        spec: NodeSpec,
        context: dict,  # Results from dependency nodes
    ) -> NodeResult:
        raise NotImplementedError


class TraceExecutor(NodeExecutor):
    """Execute trace node."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        from capseal.project_trace_emitter import emit_project_trace

        start_time = time.time()

        policy_id = spec.params.get("policy_id", "review_v1")
        num_shards = spec.params.get("num_shards", 4)

        # Compute input hash
        input_manifest = {
            "project_dir": str(self.project_dir),
            "policy_id": policy_id,
            "num_shards": num_shards,
        }
        input_hash = sha256_json(input_manifest)

        try:
            manifest = emit_project_trace(
                self.project_dir,
                self.run_dir,
                policy_id=policy_id,
                num_shards=num_shards,
            )

            # Read commitments
            with open(self.run_dir / "commitments.json") as f:
                commitments = json.load(f)

            duration_ms = int((time.time() - start_time) * 1000)

            packet = AgentPacket(
                node_id=spec.id,
                task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_type="trace",
                input_hash=input_hash,
                input_manifest=input_manifest,
                executor_id=f"project_trace_v1",
                policy_hash=sha256_str(policy_id),
                output_hash=commitments["head_T"],
                output_path="trace.jsonl",
                artifacts=[
                    {"path": "trace.jsonl", "hash": commitments["head_T"], "type": "trace"},
                    {"path": "manifest.json", "hash": sha256_file(self.run_dir / "manifest.json"), "type": "manifest"},
                    {"path": "commitments.json", "hash": sha256_file(self.run_dir / "commitments.json"), "type": "commitments"},
                    {"path": "fold/shards.json", "hash": sha256_file(self.run_dir / "fold" / "shards.json"), "type": "shards"},
                ],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
            )

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            return NodeResult(node_id=spec.id, success=False, error=str(e))


class ProfileExtractExecutor(NodeExecutor):
    """Execute profile extraction node."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        from capseal.cli.profile_cmd import extract_profile

        start_time = time.time()

        # Get trace root from dependency
        trace_result = context.get("trace")
        trace_root = ""
        if trace_result and trace_result.packet:
            trace_root = trace_result.packet.output_hash

        # Compute input hash
        input_manifest = {
            "project_dir": str(self.project_dir),
            "trace_root": trace_root,
        }
        input_hash = sha256_json(input_manifest)

        # Check cache (include env key for honest memoization)
        env_key = get_env_memoization_key(self.project_dir)
        cache_key = self.cache.compute_cache_key(
            spec.id, input_hash, "profile_extract_v1", "", env_key
        )
        cached = self.cache.get(cache_key)
        if cached:
            cached.node_id = spec.id  # Update node ID
            return NodeResult(node_id=spec.id, success=True, packet=cached, cached=True)

        try:
            profile = extract_profile(self.project_dir)

            # Write output
            out_dir = self.run_dir / "profile"
            out_dir.mkdir(parents=True, exist_ok=True)
            profile_path = out_dir / "profile.json"
            profile_path.write_text(json.dumps(profile, indent=2))

            output_hash = sha256_file(profile_path)
            duration_ms = int((time.time() - start_time) * 1000)

            packet = AgentPacket(
                node_id=spec.id,
                task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_type="profile_extract",
                input_hash=input_hash,
                input_manifest=input_manifest,
                executor_id="profile_extract_v1",
                policy_hash="",
                output_hash=output_hash,
                output_path="profile/profile.json",
                artifacts=[
                    {"path": "profile/profile.json", "hash": output_hash, "type": "profile"},
                ],
                evidence_refs=[{"trace_root": trace_root}] if trace_root else [],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
            )

            # Cache result
            self.cache.put(cache_key, packet)

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            return NodeResult(node_id=spec.id, success=False, error=str(e))


class IntentCheckExecutor(NodeExecutor):
    """Execute intent conformance check node."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        from capseal.cli.profile_cmd import (
            extract_profile, load_intent, check_conformance
        )

        start_time = time.time()

        # Get profile from dependency
        profile_result = context.get("profile")
        profile_hash = ""
        if profile_result and profile_result.packet:
            profile_hash = profile_result.packet.output_hash
            # Load the profile
            profile_path = self.run_dir / profile_result.packet.output_path
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
            else:
                profile = extract_profile(self.project_dir)
        else:
            profile = extract_profile(self.project_dir)

        # Load intent
        intent_file = spec.params.get("intent_file", "capseal.intent.json")
        intent = load_intent(self.project_dir)

        if not intent:
            reason = f"No intent file found ({intent_file})" if intent_file else "No intent file found"
            req_mode = spec.get_required_mode()

            # Handle based on required_mode
            if req_mode == RequiredMode.REQUIRED:
                return NodeResult.fail(
                    node_id=spec.id,
                    error=reason,
                    failure_code="MISSING_INTENT_FILE",
                    required_mode=spec.required_mode,
                )
            elif req_mode == RequiredMode.REQUIRED_IF_PRESENT:
                # required_if_present: profile exists but intent doesn't = SKIP (allowed)
                return NodeResult.skip(
                    node_id=spec.id,
                    skip_reason=SkipReason.NOT_APPLICABLE,
                    reason=reason,
                    required_mode=spec.required_mode,
                )
            else:
                # optional or advisory: SKIP is fine
                return NodeResult.skip(
                    node_id=spec.id,
                    skip_reason=SkipReason.NOT_APPLICABLE,
                    reason=reason,
                    required_mode=spec.required_mode,
                )

        intent_hash = sha256_json(intent)

        # Compute input hash
        input_manifest = {
            "profile_hash": profile_hash or profile.get("profile_hash", ""),
            "intent_hash": intent_hash,
        }
        input_hash = sha256_json(input_manifest)

        # Check cache (include env key for honest memoization)
        env_key = get_env_memoization_key(self.project_dir)
        cache_key = self.cache.compute_cache_key(
            spec.id, input_hash, "intent_check_v1", "", env_key
        )
        cached = self.cache.get(cache_key)
        if cached:
            cached.node_id = spec.id
            return NodeResult(node_id=spec.id, success=True, packet=cached, cached=True)

        try:
            conformance = check_conformance(profile, intent)

            # Write output
            out_dir = self.run_dir / "conformance"
            out_dir.mkdir(parents=True, exist_ok=True)
            conf_path = out_dir / "conformance.json"
            conf_path.write_text(json.dumps(conformance, indent=2))

            output_hash = sha256_file(conf_path)
            duration_ms = int((time.time() - start_time) * 1000)

            packet = AgentPacket(
                node_id=spec.id,
                task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_type="intent_check",
                input_hash=input_hash,
                input_manifest=input_manifest,
                executor_id="intent_check_v1",
                policy_hash="",
                output_hash=output_hash,
                output_path="conformance/conformance.json",
                artifacts=[
                    {"path": "conformance/conformance.json", "hash": output_hash, "type": "conformance"},
                ],
                evidence_refs=[
                    {"profile_hash": profile_hash},
                    {"intent_hash": intent_hash},
                ],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
            )

            # Cache result
            self.cache.put(cache_key, packet)

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            return NodeResult(node_id=spec.id, success=False, error=str(e))


class SemgrepReviewExecutor(NodeExecutor):
    """Execute semgrep review node (sharded)."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        start_time = time.time()

        # Get trace root from dependency
        trace_result = context.get("trace")
        if not trace_result or not trace_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Missing trace dependency")

        trace_root = trace_result.packet.output_hash

        num_shards = spec.params.get("shards", 4)
        num_agents = spec.params.get("agents", 4)

        # Compute input hash
        input_manifest = {
            "trace_root": trace_root,
            "num_shards": num_shards,
            "backend": "semgrep",
        }
        input_hash = sha256_json(input_manifest)

        try:
            # Run semgrep directly
            aggregate_path, backend_id = self._run_semgrep_review(trace_root, num_agents)

            if not aggregate_path.exists():
                return NodeResult(node_id=spec.id, success=False, error="No aggregate.json produced")

            aggregate_hash = sha256_file(aggregate_path)
            duration_ms = int((time.time() - start_time) * 1000)

            # Collect shard artifacts
            artifacts = [
                {"path": "reviews/aggregate.json", "hash": aggregate_hash, "type": "aggregate"},
            ]
            for shard_file in (self.run_dir / "reviews").glob("review_shard_*.json"):
                artifacts.append({
                    "path": f"reviews/{shard_file.name}",
                    "hash": sha256_file(shard_file),
                    "type": "shard",
                })

            packet = AgentPacket(
                node_id=spec.id,
                task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_type="semgrep_review",
                input_hash=input_hash,
                input_manifest=input_manifest,
                executor_id=backend_id,
                policy_hash="",
                output_hash=aggregate_hash,
                output_path="reviews/aggregate.json",
                artifacts=artifacts,
                evidence_refs=[{"trace_root": trace_root}],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
            )

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            import traceback
            return NodeResult(node_id=spec.id, success=False, error=f"{str(e)}\n{traceback.format_exc()}")

    def _run_semgrep_review(self, trace_root: str, num_agents: int) -> tuple[Path, str]:
        """Run semgrep and produce review packets."""
        import concurrent.futures

        # Load shards
        with open(self.run_dir / "fold" / "shards.json") as f:
            shards_data = json.load(f)
        with open(self.run_dir / "manifest.json") as f:
            manifest = json.load(f)

        shards = shards_data.get("shards", [])
        policy_id = manifest.get("policy_id", "unknown")
        policy_version = manifest.get("policy_version", "unknown")
        review_rules = manifest.get("review_rules", {})

        # Run semgrep globally
        semgrep_by_file, chunk_map, file_hash_map, backend_id = self._run_semgrep_global()

        # Create reviews directory
        reviews_dir = self.run_dir / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)

        all_findings = []
        review_paths = []

        def run_shard(shard):
            sid = shard["shard_id"]
            out_path = reviews_dir / f"review_shard_{sid}.json"

            findings = self._distribute_semgrep_to_shard(shard, semgrep_by_file, chunk_map, file_hash_map)
            # Canonical sort for stable hashes
            findings = sort_findings(findings)

            review = {
                "schema": "review_packet_v1",
                "trace_root": trace_root,
                "shard_id": sid,
                "backend_id": backend_id,
                "findings": findings,
                "policy_id": policy_id,
                "policy_version": policy_version,
                "review_rules": review_rules,
            }
            # Use canonical JSON for stable output
            out_path.write_text(canonical_json(review))
            return out_path, findings

        # Execute shards in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = {executor.submit(run_shard, s): s for s in shards}
            for future in concurrent.futures.as_completed(futures):
                path, findings = future.result()
                review_paths.append(path)
                all_findings.extend(findings)

        # Canonical sort of all_findings for stable aggregate hash
        all_findings = sort_findings(all_findings)

        # Write aggregate
        aggregate = {
            "schema": "review_aggregate_v1",
            "trace_root": trace_root,
            "backend_id": backend_id,
            "total_findings": len(all_findings),
            "findings": all_findings,
            "shard_count": len(shards),
            "policy_id": policy_id,
            "policy_version": policy_version,
            "review_rules": review_rules,
        }
        aggregate_path = reviews_dir / "aggregate.json"
        # Use canonical JSON for stable hash
        aggregate_path.write_text(canonical_json(aggregate))

        return aggregate_path, backend_id

    def _run_semgrep_global(self) -> tuple:
        """Run Semgrep once on the whole project."""
        # Get semgrep version
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True, text=True,
        )
        version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        backend_id = f"semgrep:auto@{version}"

        # Build chunk lookup from trace
        chunk_map: dict[str, list[dict]] = {}
        file_hash_map: dict[str, str] = {}
        with open(self.run_dir / "trace.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("row_type") == "chunk_entry":
                    chunk_map.setdefault(row["path"], []).append({
                        "offset": row["offset"],
                        "length": row["length"],
                        "chunk_hash": row["chunk_hash"],
                    })
                elif row.get("row_type") == "file_entry" and row.get("content_hash"):
                    file_hash_map[row["path"]] = row["content_hash"]

        # Run semgrep
        result = subprocess.run(
            ["semgrep", "scan", "--config", "auto", "--json", str(self.project_dir)],
            capture_output=True, text=True,
        )

        # Parse results
        by_file: dict[str, list] = {}
        if result.returncode == 0 or result.stdout:
            try:
                data = json.loads(result.stdout)
                for res in data.get("results", []):
                    path = res.get("path", "")
                    try:
                        rel = str(Path(path).relative_to(self.project_dir))
                    except ValueError:
                        rel = path
                    by_file.setdefault(rel, []).append(res)
            except json.JSONDecodeError:
                pass

        return by_file, chunk_map, file_hash_map, backend_id

    def _distribute_semgrep_to_shard(
        self,
        shard: dict,
        semgrep_by_file: dict,
        chunk_map: dict,
        file_hash_map: dict,
    ) -> list:
        """Distribute semgrep findings to a shard's files."""
        from capseal.finding_utils import (
            compute_finding_fingerprint, FINDING_NORM_VERSION
        )

        findings = []
        shard_files = {f["path"] for f in shard.get("files", [])}

        for fpath in shard_files:
            results = semgrep_by_file.get(fpath, [])
            for res in results:
                start_line = res.get("start", {}).get("line", 0)
                end_line = res.get("end", {}).get("line", start_line)

                # Find overlapping chunks
                chunks = chunk_map.get(fpath, [])
                overlapping = []
                if chunks:
                    # Rough line-to-byte mapping (assume ~80 chars/line)
                    approx_offset = (start_line - 1) * 80
                    for c in chunks:
                        if c["offset"] <= approx_offset < c["offset"] + c["length"]:
                            overlapping.append(c["chunk_hash"])

                snippet = res.get("extra", {}).get("lines", "")[:200]
                snippet_hash = sha256_str(snippet.lower().strip()) if snippet else ""

                finding = {
                    "rule_id": res.get("check_id", "unknown"),
                    "message": res.get("extra", {}).get("message", ""),
                    "severity": res.get("extra", {}).get("severity", "warning").lower(),
                    "file_path": fpath,
                    "file_hash": file_hash_map.get(fpath, ""),
                    "line_range": [start_line, end_line],
                    "chunk_hashes": overlapping or [""],
                    "snippet": "requires login" if snippet else "",  # Redact by default
                    "snippet_hash": snippet_hash,
                    "finding_norm_version": FINDING_NORM_VERSION,
                }
                finding["finding_fingerprint"] = compute_finding_fingerprint(finding, backend_id="semgrep")
                findings.append(finding)

        return findings


class ExplainLLMExecutor(NodeExecutor):
    """Execute LLM explanation node."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        start_time = time.time()

        # Get review result from dependency
        review_result = context.get("semgrep") or context.get("review")
        if not review_result or not review_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Missing review dependency")

        aggregate_hash = review_result.packet.output_hash

        # Get params
        provider = spec.params.get("provider", "openai")
        model = spec.params.get("model", "gpt-4o-mini")
        fmt = spec.params.get("format", "markdown")
        min_severity = spec.params.get("min_severity", "warning")
        max_findings = spec.params.get("max_findings", 20)
        report_top = spec.params.get("report_top", 5)
        temperature = spec.params.get("temperature", 0.0)
        diff_param = spec.params.get("diff")
        diff_path = None
        if diff_param:
            diff_candidate = Path(diff_param)
            if not diff_candidate.is_absolute():
                diff_candidate = self.run_dir / diff_candidate
            diff_path = diff_candidate

        # Budget constraints
        budget = spec.budget or {}
        max_tokens = budget.get("max_output_tokens", 1500)

        # Compute input hash
        input_manifest = {
            "aggregate_hash": aggregate_hash,
            "provider": provider,
            "model": model,
            "min_severity": min_severity,
            "max_findings": max_findings,
            "format": fmt,
            "report_top": report_top,
            "temperature": temperature,
            "diff_path": str(diff_path) if diff_path else "",
        }
        input_hash = sha256_json(input_manifest)

        # Check cache (include env key for honest memoization)
        policy_hash = sha256_json({"provider": provider, "model": model})
        env_key = get_env_memoization_key(self.project_dir)
        cache_key = self.cache.compute_cache_key(
            spec.id, input_hash, f"{provider}:{model}", policy_hash, env_key
        )
        cached = self.cache.get(cache_key)
        if cached:
            cached.node_id = spec.id
            return NodeResult(node_id=spec.id, success=True, packet=cached, cached=True)

        try:
            result = run_explain_pipeline(
                self.run_dir,
                provider,
                model,
                temperature,
                max_tokens,
                max_findings,
                min_severity,
                diff_path,
                fmt,
                report_top,
                force=False,
                out=None,
            )

            explain_subdir = result.summary_path.parent
            if not explain_subdir.exists():
                return NodeResult(node_id=spec.id, success=False, error="No explain output produced")

            # Collect artifacts
            artifacts = []
            output_hash = sha256_file(result.summary_path)
            for f in explain_subdir.iterdir():
                if f.is_file():
                    fhash = sha256_file(f)
                    artifacts.append({
                        "path": str(f.relative_to(self.run_dir)),
                        "hash": fhash,
                        "type": f.suffix.lstrip(".") or "unknown",
                    })

            duration_ms = int((time.time() - start_time) * 1000)

            receipt_data = json.loads(result.receipt_path.read_text())
            prompt_hash = receipt_data.get("prompt_text_hash", "")
            raw_hash = receipt_data.get("raw_hash", "")

            packet = AgentPacket(
                node_id=spec.id,
                task_id=f"{spec.id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                agent_type="explain_llm",
                input_hash=input_hash,
                input_manifest=input_manifest,
                executor_id=f"{provider}:{model}",
                policy_hash=policy_hash,
                budget=budget,
                output_hash=output_hash,
                output_path=str(result.summary_path.relative_to(self.run_dir)),
                artifacts=artifacts,
                prompt_hash=prompt_hash,
                raw_response_hash=raw_hash,
                evidence_refs=[{"aggregate_hash": aggregate_hash}],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
            )

            # Cache result
            self.cache.put(cache_key, packet)

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            import traceback
            return NodeResult(node_id=spec.id, success=False, error=f"{str(e)}\n{traceback.format_exc()}")


# =============================================================================
# Workflow Runner
# =============================================================================

EXECUTORS = {
    "trace": TraceExecutor,
    "agent.profile_extract": ProfileExtractExecutor,
    "agent.intent_check": IntentCheckExecutor,
    "review.semgrep": SemgrepReviewExecutor,
    "explain_llm": ExplainLLMExecutor,
}

# Import and register committor gate executor
try:
    from capseal.committor_gate import CommittorGateExecutor
    EXECUTORS["committor.gate"] = CommittorGateExecutor
except ImportError:
    pass  # committor_gate not available

# Import and register refactor executors
try:
    from capseal.refactor_engine import REFACTOR_EXECUTORS
    EXECUTORS.update(REFACTOR_EXECUTORS)
except ImportError:
    pass  # refactor_engine not available


class WorkflowRunner:
    """Execute a workflow DAG with verifiable packets.

    The runner:
    1. Parses workflow spec
    2. Executes nodes in topological order
    3. Records node outputs as artifacts
    4. Emits node receipts
    5. Builds workflow rollup that commits to DAG + receipts
    """

    def __init__(
        self,
        project_dir: Path,
        run_dir: Path,
        cache_dir: Optional[Path] = None,
        proof_carrying: bool = False,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir or (self.run_dir.parent / ".capseal_cache")
        self.cache = MemoizationCache(self.cache_dir)
        self.proof_carrying = proof_carrying

        self.results: dict[str, NodeResult] = {}
        self.task_map: dict[str, NodeSpec] = {}
        self.workflow_spec: Optional[WorkflowSpec] = None
        self.topo_order: list[str] = []

        # Proof artifacts (populated if proof_carrying=True)
        self.capsule: Optional[dict[str, Any]] = None
        self.proof_artifacts: Optional[Any] = None

    def run(self, spec: WorkflowSpec) -> dict[str, Any]:
        """Execute workflow and return rollup."""
        self.workflow_spec = spec
        self.results = {}
        self.task_map = {task.id: task for task in spec.tasks}

        # Get topological order
        order = spec.topological_order()
        self.topo_order = order

        # Execute each node
        for node_id in order:
            node_spec = self.task_map.get(node_id)
            if not node_spec:
                continue

            # Build context from dependencies
            context = {}
            for dep_id in node_spec.needs:
                if dep_id in self.results:
                    context[dep_id] = self.results[dep_id]

            # Get executor
            executor_cls = EXECUTORS.get(node_spec.kind)
            if not executor_cls:
                self.results[node_id] = NodeResult(
                    node_id=node_id,
                    success=False,
                    error=f"Unknown node kind: {node_spec.kind}",
                )
                continue

            # Execute
            executor = executor_cls(self.run_dir, self.project_dir, self.cache)
            result = executor.execute(node_spec, context)
            result = self._apply_requirement(node_spec, result)
            self.results[node_id] = result

            # Write packet receipt
            if result.packet:
                self._write_packet_receipt(result.packet)

        # Generate proof if proof_carrying=True
        if self.proof_carrying:
            self._generate_workflow_proof()

        # Build workflow rollup
        return self._build_rollup(spec)

    def _generate_workflow_proof(self) -> None:
        """Generate FRI proof over the workflow execution.

        This creates a WorkflowAIR proof that the DAG execution is valid:
        - Chain constraint: receipt hashes form an unbroken chain
        - Ordering constraint: nodes executed in topological order
        - Boundary constraints: first node has no prev_receipt, last matches dag_root
        """
        from capseal.workflow_adapter import (
            WorkflowAdapter,
            build_workflow_capsule,
            save_workflow_capsule,
        )

        # Collect results with packets in topo order
        ordered_results = []
        for nid in self.topo_order:
            if nid in self.results:
                ordered_results.append(self.results[nid])

        # Filter to only results with packets
        results_with_packets = [r for r in ordered_results if r.packet is not None]

        if not results_with_packets:
            # No packets to prove - skip proof generation
            return

        adapter = WorkflowAdapter()

        # Phase A: Simulate trace (convert node results to AIR rows)
        trace_artifacts = adapter.simulate_trace(results_with_packets)

        # Phase B: Commit to trace (Merkle tree over rows)
        row_archive_dir = self.run_dir / "row_archive"
        commitment = adapter.commit_to_trace(
            trace_artifacts,
            row_archive_dir=row_archive_dir,
        )

        # Phase C: Generate proof
        proof_artifacts = adapter.generate_proof(
            trace_artifacts,
            commitment,
            encoding_id="workflow_air_v1",
            trace_path=self.run_dir / "trace",
        )

        # Phase D: Build and save capsule
        capsule = build_workflow_capsule(
            trace_artifacts,
            commitment,
            proof_artifacts,
            self.run_dir,
        )

        # Save capsule
        capsule_path = self.run_dir / "workflow_capsule.json"
        save_workflow_capsule(capsule, capsule_path)

        # Store for rollup integration
        self.capsule = capsule
        self.proof_artifacts = proof_artifacts

        # Also save the proof JSON separately
        proof_dir = self.run_dir / "proof"
        proof_dir.mkdir(parents=True, exist_ok=True)
        (proof_dir / "workflow_proof.json").write_text(proof_artifacts.proof_json)

    def _write_packet_receipt(self, packet: AgentPacket) -> None:
        """Write packet receipt to run directory."""
        receipts_dir = self.run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        receipt_path = receipts_dir / f"{packet.node_id}_receipt.json"
        receipt_path.write_text(json.dumps(packet.to_dict(), indent=2))

    def _apply_requirement(self, node_spec: NodeSpec, result: NodeResult) -> NodeResult:
        """Apply requirement policy to node result.

        This implements the RequiredMode semantics:
        - REQUIRED: SKIP → FAIL
        - REQUIRED_IF_PRESENT: SKIP is only ok if skip_reason is NOT_APPLICABLE/MISSING_INPUT
        - OPTIONAL: SKIP always allowed
        - ADVISORY: SKIP and FAIL both allowed (never fails workflow)
        """
        if not node_spec:
            return result

        req_mode = node_spec.get_required_mode()
        result.status_detail.required_mode = node_spec.required_mode

        if result.status == NodeStatus.SKIP.value:
            if req_mode == RequiredMode.REQUIRED:
                # REQUIRED + SKIP → FAIL
                result.skipped = False
                result.success = False
                if not result.reason:
                    result.reason = "Required node skipped"
                result.refresh_status()

            elif req_mode == RequiredMode.REQUIRED_IF_PRESENT:
                # Only ok if skip reason is NOT_APPLICABLE or MISSING_INPUT
                skip_reason = result.status_detail.skip_reason
                if skip_reason not in (
                    SkipReason.NOT_APPLICABLE.value,
                    SkipReason.MISSING_INPUT.value,
                ):
                    result.skipped = False
                    result.success = False
                    result.reason = f"Required-if-present node skipped unexpectedly: {skip_reason}"
                    result.refresh_status()

            # OPTIONAL and ADVISORY: SKIP is always allowed

        elif result.status == NodeStatus.FAIL.value:
            if req_mode == RequiredMode.ADVISORY:
                # ADVISORY + FAIL → still counts as PASS for workflow purposes
                # But we keep the status as FAIL for transparency
                result.status_detail.failure_severity = "info"

        return result

    def _get_required_mode(self, node_id: str) -> str:
        """Get required_mode for a node."""
        spec = self.task_map.get(node_id)
        return spec.required_mode if spec else "required"

    def _is_required(self, node_id: str) -> bool:
        """Check if node is required (backward compat)."""
        spec = self.task_map.get(node_id)
        return True if spec is None else spec.required

    def _build_rollup(self, spec: WorkflowSpec) -> dict[str, Any]:
        """Build workflow rollup with all receipts.

        The rollup commits to:
        - DAG topology (vertices + edges + node configs)
        - All node receipts
        - Merkle root of vertex states
        """

        # Collect all vertices (node receipts)
        vertices = {}
        for node_id, result in self.results.items():
            required_mode = self._get_required_mode(node_id)
            node_spec = self.task_map.get(node_id)
            determinism = "unknown"
            if node_spec:
                det = NODE_DETERMINISM.get(node_spec.kind, Determinism.DETERMINISTIC)
                determinism = det.value

            # Build status_detail for rollup
            status_detail = result.status_detail.to_dict()

            if result.packet:
                vertices[node_id] = {
                    "receipt_hash": result.packet.compute_receipt_hash(),
                    "output_hash": result.packet.output_hash,
                    "agent_type": result.packet.agent_type,
                    "determinism": determinism,
                    "cached": result.cached,
                    "status": result.status,
                    "reason": result.reason,
                    "required_mode": required_mode,
                    "status_detail": status_detail,
                }
            else:
                vertices[node_id] = {
                    "status": result.status,
                    "reason": result.reason,
                    "required_mode": required_mode,
                    "determinism": determinism,
                    "error": result.error,
                    "status_detail": status_detail,
                }

        # Build edges (sorted for canonical ordering)
        edges = []
        for task in spec.tasks:
            for dep in task.needs:
                edges.append({"from": dep, "to": task.id})
        edges = sorted(edges, key=lambda e: (e["from"], e["to"]))

        # Build node configs (for topology commitment)
        node_configs = {}
        for task in spec.tasks:
            node_configs[task.id] = {
                "kind": task.kind,
                "needs": sorted(task.needs),
                "params_hash": sha256_json(task.params),
                "policy_hash": sha256_json(task.policy),
                "budget_hash": sha256_json(task.budget),
                "required_mode": task.required_mode,
            }

        # Compute DAG topology hash (this is what prevents shuffling semantics)
        topology = {
            "nodes": sorted(node_configs.keys()),
            "edges": sorted([f"{e['from']}->{e['to']}" for e in edges]),
            "configs": node_configs,
        }
        topology_hash = sha256_json(topology)

        # Compute merkle root of vertices
        vertex_hashes = [
            sha256_json(v) for v in sorted(vertices.items(), key=lambda x: x[0])
        ]
        merkle_root = sha256_json(vertex_hashes)

        # Get trace root if available
        trace_root = ""
        if "trace" in self.results and self.results["trace"].packet:
            trace_root = self.results["trace"].packet.output_hash

        # Get environment fingerprint
        env_fp = get_environment_fingerprint()

        rollup = {
            "schema": WORKFLOW_ROLLUP_SCHEMA_VERSION,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "workflow_name": spec.name,
            "workflow_hash": sha256_json(spec.to_dict()),
            "topology_hash": topology_hash,  # Commits to DAG structure
            "trace_root": trace_root,
            "vertices": vertices,
            "edges": edges,
            "node_configs": node_configs,  # Full config for audit
            "merkle_root": merkle_root,
            "env_fingerprint": env_fp,
            "tool_version": "capseal-0.3.0",
        }

        # Add proof-carrying fields if proof was generated
        if self.proof_carrying and self.capsule:
            from capseal.workflow_adapter import (
                verify_workflow_capsule,
                row_merkle_proof,
            )

            rollup["capsule_hash"] = self.capsule.get("capsule_hash", "")
            rollup["statement_hash"] = self.capsule.get("statement_hash", "")
            rollup["row_commitment"] = self.capsule.get("commitment", {}).get("row_root", "")
            rollup["proof_backend"] = "workflow_air_v1"

            # Self-check: verify the capsule
            capsule_path = self.run_dir / "workflow_capsule.json"
            if capsule_path.exists():
                proof_valid, _ = verify_workflow_capsule(capsule_path)
                rollup["proof_verified"] = proof_valid
            else:
                rollup["proof_verified"] = False

            # Add row index and Merkle proof for each vertex
            row_archive_dir = self.run_dir / "row_archive"
            packet_nodes = [nid for nid in self.topo_order if self.results.get(nid) and self.results[nid].packet]

            for i, nid in enumerate(packet_nodes):
                if nid in vertices:
                    vertices[nid]["row_index"] = i
                    try:
                        merkle_proof = row_merkle_proof(row_archive_dir, i)
                        vertices[nid]["row_commitment"] = merkle_proof["merkle_path"]
                    except (FileNotFoundError, IndexError):
                        vertices[nid]["row_commitment"] = []

        rollup["rollup_hash"] = sha256_json(rollup)

        # Write rollup (use pretty print for readability, but hash is computed from canonical form)
        workflow_dir = self.run_dir / "workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)

        rollup_path = workflow_dir / "workflow_rollup.json"
        # Pretty print for human readability (hash is computed before writing)
        rollup_path.write_text(json.dumps(rollup, indent=2, sort_keys=True))

        # Also write DAG visualization
        dag_path = workflow_dir / "dag.json"
        dag_path.write_text(json.dumps({
            "vertices": sorted(vertices.keys()),
            "edges": edges,
            "order": spec.topological_order(),
        }, indent=2, sort_keys=True))

        return rollup


# =============================================================================
# Workflow Verification
# =============================================================================

def verify_workflow_rollup(
    rollup_path: Path,
    run_dir: Path,
    recompute_artifacts: bool = True,
) -> dict[str, Any]:
    """Verify a workflow rollup.

    A strong verification does:
    1. Recompute each node's input_hash from referenced artifacts/receipts
    2. Recompute output_hash from actual produced files
    3. Verify the receipt binds those hashes + metadata
    4. Recompute the workflow merkle root from node receipts
    5. Verify DAG topology commitment

    Args:
        rollup_path: Path to workflow_rollup.json
        run_dir: Path to run directory containing artifacts
        recompute_artifacts: If True, actually recompute hashes from files (not just compare)
    """
    rollup = json.loads(rollup_path.read_text())

    results = {
        "valid": True,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    # 1. Verify DAG topology hash
    node_configs = rollup.get("node_configs", {})
    edges = rollup.get("edges", [])
    if node_configs:
        topology = {
            "nodes": sorted(node_configs.keys()),
            "edges": sorted([f"{e['from']}->{e['to']}" for e in edges]),
            "configs": node_configs,
        }
        computed_topology = sha256_json(topology)
        if computed_topology != rollup.get("topology_hash"):
            results["errors"].append("Topology hash mismatch - DAG structure may have been tampered")
            results["valid"] = False
        else:
            results["checks"].append("[PASS] DAG topology hash verified")
    else:
        results["warnings"].append("No node_configs in rollup - cannot verify topology")

    # 2. Verify each vertex
    for node_id, vertex_data in rollup.get("vertices", {}).items():
        receipt_hash = vertex_data.get("receipt_hash")
        if not receipt_hash:
            # Node has no receipt - check status based on required_mode
            status = vertex_data.get("status")
            required_mode = vertex_data.get("required_mode", "")
            if not required_mode:
                required_mode = "required" if vertex_data.get("required", True) else "optional"

            # Determine if this status is acceptable
            status_ok = False
            if status == NodeStatus.PASS.value:
                status_ok = True
            elif status == NodeStatus.SKIP.value:
                if required_mode in ("optional", "advisory"):
                    status_ok = True
                elif required_mode == "required_if_present":
                    status_detail = vertex_data.get("status_detail", {})
                    skip_reason = status_detail.get("skip_reason", "")
                    if skip_reason in ("not_applicable", "missing_input"):
                        status_ok = True
            elif status == NodeStatus.FAIL.value:
                if required_mode == "advisory":
                    status_ok = True

            if not status_ok:
                reason = vertex_data.get("reason", "")
                msg = f"Node {node_id} status={status} (required_mode={required_mode})"
                if reason:
                    msg += f": {reason}"
                results["errors"].append(msg)
                results["valid"] = False
            else:
                if status != NodeStatus.PASS.value:
                    results["checks"].append(f"[PASS] {node_id}: {status} (allowed by {required_mode})")
            continue

        # Load receipt
        receipt_path = run_dir / "receipts" / f"{node_id}_receipt.json"
        if not receipt_path.exists():
            results["errors"].append(f"Missing receipt: {node_id}")
            results["valid"] = False
            continue

        receipt = json.loads(receipt_path.read_text())
        packet = AgentPacket.from_dict(receipt)

        # 2a. Verify receipt hash
        computed_receipt_hash = packet.compute_receipt_hash()
        if computed_receipt_hash != receipt_hash:
            results["errors"].append(f"Receipt hash mismatch for {node_id}")
            results["valid"] = False
            continue

        # 2b. Recompute output hash from actual file (if enabled)
        # Special case: trace node uses trace_root (chain hash) not file hash
        if recompute_artifacts and packet.output_path:
            if packet.agent_type == "trace":
                # For trace, verify against commitments.json trace_root
                commitments_path = run_dir / "commitments.json"
                if commitments_path.exists():
                    commitments = json.loads(commitments_path.read_text())
                    if commitments.get("head_T") == packet.output_hash:
                        results["checks"].append(f"[PASS] {node_id}: trace_root verified from commitments")
                    else:
                        results["errors"].append(
                            f"Trace root mismatch for {node_id}: "
                            f"receipt says {packet.output_hash[:16]}... "
                            f"but commitments says {commitments.get('head_T', '')[:16]}..."
                        )
                        results["valid"] = False
                else:
                    results["warnings"].append(f"commitments.json missing for trace verification")
            else:
                output_file = run_dir / packet.output_path
                if output_file.exists():
                    recomputed_output = sha256_file(output_file)
                    if recomputed_output != packet.output_hash:
                        results["errors"].append(
                            f"Output hash mismatch for {node_id}: "
                            f"receipt says {packet.output_hash[:16]}... "
                            f"but file is {recomputed_output[:16]}..."
                        )
                        results["valid"] = False
                    else:
                        results["checks"].append(f"[PASS] {node_id}: output recomputed and verified")
                else:
                    results["warnings"].append(f"Output file missing for {node_id}: {packet.output_path}")

        # 2c. Verify all artifact hashes (if enabled)
        if recompute_artifacts:
            for artifact in packet.artifacts:
                # Skip trace.jsonl for trace node (uses semantic hash, not file hash)
                if packet.agent_type == "trace" and artifact["type"] == "trace":
                    continue

                artifact_path = run_dir / artifact["path"]
                if artifact_path.exists():
                    recomputed = sha256_file(artifact_path)
                    if recomputed != artifact["hash"]:
                        results["errors"].append(
                            f"Artifact hash mismatch: {artifact['path']} in {node_id}"
                        )
                        results["valid"] = False
                else:
                    results["warnings"].append(f"Artifact missing: {artifact['path']}")

        if packet.output_path and not recompute_artifacts:
            results["checks"].append(f"[PASS] {node_id}: receipt verified (no recompute)")

        # 2d. Check node status based on required_mode
        status = vertex_data.get("status")
        # Backward compat: check required_mode or fall back to required
        required_mode = vertex_data.get("required_mode", "")
        if not required_mode:
            required_mode = "required" if vertex_data.get("required", True) else "optional"

        # Determine if this status is acceptable for the required_mode
        status_ok = False
        if status == NodeStatus.PASS.value:
            status_ok = True
        elif status == NodeStatus.SKIP.value:
            # SKIP is ok for optional, advisory, and required_if_present (when valid)
            if required_mode in ("optional", "advisory"):
                status_ok = True
            elif required_mode == "required_if_present":
                # Check if skip reason is acceptable
                status_detail = vertex_data.get("status_detail", {})
                skip_reason = status_detail.get("skip_reason", "")
                if skip_reason in ("not_applicable", "missing_input"):
                    status_ok = True
        elif status == NodeStatus.FAIL.value:
            # FAIL is only ok for advisory
            if required_mode == "advisory":
                status_ok = True

        if not status_ok:
            reason = vertex_data.get("reason", "")
            msg = f"Node {node_id} status={status} (required_mode={required_mode})"
            if reason:
                msg += f": {reason}"
            results["errors"].append(msg)
            results["valid"] = False
        else:
            if status != NodeStatus.PASS.value:
                results["checks"].append(f"[PASS] {node_id}: {status} (allowed by {required_mode})")

    # 3. Verify merkle root
    vertices = rollup.get("vertices", {})
    vertex_hashes = [
        sha256_json(v) for v in sorted(vertices.items(), key=lambda x: x[0])
    ]
    computed_merkle = sha256_json(vertex_hashes)

    if computed_merkle != rollup.get("merkle_root"):
        results["errors"].append("Merkle root mismatch")
        results["valid"] = False
    else:
        results["checks"].append("[PASS] Merkle root verified")

    # 4. Verify rollup hash itself
    rollup_copy = dict(rollup)
    stored_rollup_hash = rollup_copy.pop("rollup_hash", "")
    computed_rollup_hash = sha256_json(rollup_copy)
    if computed_rollup_hash != stored_rollup_hash:
        results["errors"].append("Rollup hash mismatch - rollup may have been tampered")
        results["valid"] = False
    else:
        results["checks"].append("[PASS] Rollup hash verified")

    return results
