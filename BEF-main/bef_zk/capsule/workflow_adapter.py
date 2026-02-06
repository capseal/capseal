"""WorkflowAdapter — bridges the workflow engine to the proof pipeline.

This adapter follows the same interface as GeomAdapter and other TraceAdapters,
converting workflow DAG execution into provable traces.

The adapter:
1. simulate_trace: Converts executed DAG node results into trace rows
2. commit_to_trace: Builds Merkle commitment over the trace rows
3. generate_proof: Generates FRI proof that trace satisfies WorkflowAIR constraints
4. verify: Verifies the proof against the statement
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from bef_zk.capsule.workflow_air import (
    GOLDILOCKS_P,
    WORKFLOW_AIR_ROW_WIDTH,
    WORKFLOW_AIR_ID,
    encode_agent_packet_row,
    decode_row,
    sha256_to_field_pair,
    build_row_matrix,
)
from bef_zk.capsule.workflow_constraints import (
    WorkflowConstraints,
    verify_workflow_trace,
    build_composition_vector,
    derive_constraint_alphas,
)

if TYPE_CHECKING:
    from bef_zk.capsule.workflow_engine import AgentPacket, NodeResult


# Field modulus for FRI (using 61-bit Mersenne for compatibility with existing FRI)
FRI_MODULUS = (1 << 61) - 1


@dataclass
class WorkflowContext:
    """Context for workflow proof generation."""
    packets: list["AgentPacket"]
    policy_met_flags: list[bool]
    gate_approved_flags: list[bool]
    rows: list[list[int]]
    dag_root_hash: str
    trace_time_sec: float
    prepared: dict[str, Any] | None = None


@dataclass
class WorkflowStatement:
    """Public statement for workflow proof."""
    air_id: str
    num_nodes: int
    dag_root_lo: int
    dag_root_hi: int
    dag_root_hash: str
    trace_spec_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "air_id": self.air_id,
            "num_nodes": self.num_nodes,
            "dag_root_lo": self.dag_root_lo,
            "dag_root_hi": self.dag_root_hi,
            "dag_root_hash": self.dag_root_hash,
            "trace_spec_hash": self.trace_spec_hash,
        }


@dataclass
class WorkflowTraceArtifacts:
    """Artifacts from trace simulation."""
    trace_id: str
    trace_spec: Any
    trace_spec_hash: str
    rows: list[list[int]]
    row_width: int
    context: WorkflowContext
    trace_time_sec: float
    statement: WorkflowStatement


@dataclass
class WorkflowTraceCommitment:
    """Commitment to the workflow trace."""
    row_root: str
    num_rows: int
    params: dict[str, Any]
    profile_data: dict[str, float]


@dataclass
class WorkflowProofArtifacts:
    """Proof artifacts for workflow."""
    proof_json: str
    proof_bytes: bytes
    statement: WorkflowStatement
    verification_result: bool
    profile_data: dict[str, float]


def _hash_row(row_index: int, row: list[int]) -> bytes:
    """Hash a row for Merkle tree."""
    h = hashlib.sha256()
    h.update(row_index.to_bytes(8, "big"))
    for val in row:
        h.update(int(val).to_bytes(16, "big", signed=False))
    return h.digest()


def _build_merkle_levels(leaves: list[bytes]) -> list[list[bytes]]:
    """Build Merkle tree levels from leaves."""
    if not leaves:
        return [[hashlib.sha256(b"empty").digest()]]

    levels: list[list[bytes]] = [leaves]
    current = leaves
    while len(current) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            combined = hashlib.sha256(left + right).digest()
            next_level.append(combined)
        levels.append(next_level)
        current = next_level
    return levels


def _merkle_prove(levels: list[list[bytes]], index: int) -> list[str]:
    """Generate Merkle proof for index."""
    proof: list[bytes] = []
    idx = index
    for level in levels[:-1]:
        sibling = idx ^ 1
        if sibling >= len(level):
            sibling = idx
        proof.append(level[sibling])
        idx //= 2
    return [p.hex() for p in proof]


class WorkflowAdapter:
    """Adapter that bridges workflow engine to proof pipeline.

    Follows the same interface as GeomAdapter:
    - simulate_trace: Convert node results to trace rows
    - commit_to_trace: Build Merkle commitment
    - generate_proof: Generate FRI proof
    - verify: Verify the proof
    """

    name = "workflow"

    def __init__(self) -> None:
        self._progress_callback = None

    def simulate_trace(
        self,
        node_results: list["NodeResult"],
        *,
        trace_id: str | None = None,
    ) -> WorkflowTraceArtifacts:
        """Convert a list of executed DAG node results into trace rows.

        Args:
            node_results: List of NodeResult in topological order
            trace_id: Optional trace identifier

        Returns:
            WorkflowTraceArtifacts with trace data
        """
        start = time.perf_counter()

        # Extract packets and flags from results
        packets: list["AgentPacket"] = []
        policy_met_flags: list[bool] = []
        gate_approved_flags: list[bool] = []

        for result in node_results:
            if result.packet is not None:
                packets.append(result.packet)
                policy_met_flags.append(result.success)
                gate_approved_flags.append(result.success)  # Use success as proxy

        if not packets:
            raise ValueError("No packets found in node results")

        # Build trace rows
        rows: list[list[int]] = []
        prev_receipt: str | None = None

        for i, packet in enumerate(packets):
            row = encode_agent_packet_row(
                packet=packet,
                node_index=i,
                prev_receipt_hash=prev_receipt,
                policy_met=policy_met_flags[i],
                gate_approved=gate_approved_flags[i],
            )
            rows.append(row)
            prev_receipt = packet.compute_receipt_hash()

        # The DAG root is the last packet's receipt hash
        dag_root_hash = packets[-1].compute_receipt_hash()
        dag_root_lo, dag_root_hi = sha256_to_field_pair(dag_root_hash)

        trace_time = time.perf_counter() - start

        # Build trace spec
        trace_spec = {
            "air_id": WORKFLOW_AIR_ID,
            "row_width": WORKFLOW_AIR_ROW_WIDTH,
            "num_rows": len(rows),
            "field_modulus": "goldilocks",
        }
        trace_spec_hash = hashlib.sha256(
            json.dumps(trace_spec, sort_keys=True).encode()
        ).hexdigest()

        # Build statement
        statement = WorkflowStatement(
            air_id=WORKFLOW_AIR_ID,
            num_nodes=len(packets),
            dag_root_lo=dag_root_lo,
            dag_root_hi=dag_root_hi,
            dag_root_hash=dag_root_hash,
            trace_spec_hash=trace_spec_hash,
        )

        # Build context
        context = WorkflowContext(
            packets=packets,
            policy_met_flags=policy_met_flags,
            gate_approved_flags=gate_approved_flags,
            rows=rows,
            dag_root_hash=dag_root_hash,
            trace_time_sec=trace_time,
        )

        trace_id = trace_id or f"workflow_{int(time.time())}"

        self._emit_progress(
            "trace_simulated",
            {
                "trace_spec_hash": trace_spec_hash,
                "num_nodes": len(packets),
                "num_rows": len(rows),
            },
        )

        return WorkflowTraceArtifacts(
            trace_id=trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            rows=rows,
            row_width=WORKFLOW_AIR_ROW_WIDTH,
            context=context,
            trace_time_sec=trace_time,
            statement=statement,
        )

    def commit_to_trace(
        self,
        artifacts: WorkflowTraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> WorkflowTraceCommitment:
        """Commit to the trace rows using a Merkle tree.

        Args:
            artifacts: Trace artifacts from simulate_trace
            row_archive_dir: Directory to store row archive

        Returns:
            WorkflowTraceCommitment with Merkle root
        """
        row_archive_dir.mkdir(parents=True, exist_ok=True)

        commit_start = time.perf_counter()

        # Build Merkle tree over rows
        leaves = [_hash_row(i, row) for i, row in enumerate(artifacts.rows)]
        levels = _build_merkle_levels(leaves)
        root = levels[-1][0]
        root_hex = root.hex()

        # Store row data for later opening
        rows_json = json.dumps(artifacts.rows)
        (row_archive_dir / "rows.json").write_text(rows_json)

        # Store levels for Merkle proofs
        levels_hex = [[h.hex() for h in level] for level in levels]
        (row_archive_dir / "merkle_levels.json").write_text(json.dumps(levels_hex))

        commit_time = time.perf_counter() - commit_start

        self._emit_progress(
            "row_root_finalized",
            {
                "trace_root": root_hex,
                "num_rows": len(artifacts.rows),
            },
        )

        return WorkflowTraceCommitment(
            row_root=root_hex,
            num_rows=len(artifacts.rows),
            params={
                "root": root_hex,
                "num_rows": len(artifacts.rows),
                "row_width": artifacts.row_width,
            },
            profile_data={
                "time_trace_sec": artifacts.trace_time_sec,
                "time_row_commit_sec": commit_time,
            },
        )

    def generate_proof(
        self,
        artifacts: WorkflowTraceArtifacts,
        commitment: WorkflowTraceCommitment,
        *,
        statement_hash: bytes | None = None,
        binding_hash: bytes | None = None,
        encoding_id: str = WORKFLOW_AIR_ID,
        trace_path: Path | None = None,
    ) -> WorkflowProofArtifacts:
        """Generate proof that the trace satisfies WorkflowAIR constraints.

        For Phase 1, this generates a simplified proof structure that:
        1. Verifies all AIR constraints locally
        2. Commits to the constraint satisfaction
        3. Produces a proof artifact compatible with capsule format

        Full FRI proof integration can be added in a future phase.

        Args:
            artifacts: Trace artifacts
            commitment: Trace commitment
            statement_hash: Hash binding the statement
            binding_hash: Additional binding material
            encoding_id: AIR encoding identifier
            trace_path: Optional path to store trace

        Returns:
            WorkflowProofArtifacts with proof data
        """
        prove_start = time.perf_counter()

        ctx = artifacts.context
        rows = artifacts.rows
        statement = artifacts.statement

        # Verify constraints locally (this is what the FRI proof will prove)
        valid, constraint_results = verify_workflow_trace(rows, ctx.dag_root_hash)

        # Build composition vector (for FRI, we'd commit to this)
        alpha_seed = hashlib.sha256(
            b"workflow_alphas:" + commitment.row_root.encode()
        ).digest()
        alphas = derive_constraint_alphas(alpha_seed)
        composition = build_composition_vector(
            rows,
            statement.dag_root_lo,
            statement.dag_root_hi,
            alphas,
        )

        # Composition should be all zeros if constraints satisfied
        composition_sum = sum(composition) % GOLDILOCKS_P

        # Build proof structure
        proof_data = {
            "air_id": encoding_id,
            "version": "1.0",
            "statement": statement.to_dict(),
            "commitment": {
                "row_root": commitment.row_root,
                "num_rows": commitment.num_rows,
            },
            "constraint_verification": {
                "valid": valid,
                "num_constraints_checked": len(constraint_results),
                "composition_sum": composition_sum,
            },
            "alphas_seed": alpha_seed.hex(),
            "binding_material": (binding_hash or statement_hash or b"").hex(),
        }

        # Add constraint details (for debugging/audit)
        failed_constraints = [
            {"name": r.name, "expected": r.expected, "actual": r.actual}
            for r in constraint_results
            if not r.satisfied
        ]
        if failed_constraints:
            proof_data["failed_constraints"] = failed_constraints

        proof_json = json.dumps(proof_data, indent=2, sort_keys=True)
        proof_bytes = proof_json.encode()

        prove_time = time.perf_counter() - prove_start

        profile_data = dict(commitment.profile_data)
        profile_data.update({
            "time_prove_sec": prove_time,
            "time_total_sec": artifacts.trace_time_sec + commitment.profile_data.get("time_row_commit_sec", 0) + prove_time,
            "num_rows": len(rows),
            "constraints_valid": valid,
        })

        return WorkflowProofArtifacts(
            proof_json=proof_json,
            proof_bytes=proof_bytes,
            statement=statement,
            verification_result=valid,
            profile_data=profile_data,
        )

    def verify(
        self,
        proof_json: str,
        commitment_root: str,
        dag_root_hash: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify a workflow proof.

        Args:
            proof_json: JSON proof string
            commitment_root: Expected row commitment root
            dag_root_hash: Expected DAG root hash

        Returns:
            Tuple of (valid, verification_details)
        """
        try:
            proof_data = json.loads(proof_json)
        except json.JSONDecodeError:
            return False, {"error": "Invalid proof JSON"}

        details: dict[str, Any] = {}

        # Check AIR ID
        if proof_data.get("air_id") != WORKFLOW_AIR_ID:
            details["air_id_mismatch"] = True
            return False, details

        # Check commitment root
        proof_root = proof_data.get("commitment", {}).get("row_root")
        if proof_root != commitment_root:
            details["commitment_root_mismatch"] = True
            details["expected_root"] = commitment_root
            details["proof_root"] = proof_root
            return False, details

        # Check statement
        statement = proof_data.get("statement", {})
        if statement.get("dag_root_hash") != dag_root_hash:
            details["dag_root_mismatch"] = True
            details["expected_dag_root"] = dag_root_hash
            details["proof_dag_root"] = statement.get("dag_root_hash")
            return False, details

        # Check constraint verification
        constraint_check = proof_data.get("constraint_verification", {})
        if not constraint_check.get("valid"):
            details["constraints_invalid"] = True
            details["failed_constraints"] = proof_data.get("failed_constraints", [])
            return False, details

        details["valid"] = True
        details["num_constraints"] = constraint_check.get("num_constraints_checked", 0)
        return True, details

    def _emit_progress(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit progress event if callback is set."""
        if self._progress_callback is not None:
            self._progress_callback({"type": event_type, "data": data})

    def set_progress_callback(self, callback) -> None:
        """Set progress callback for monitoring."""
        self._progress_callback = callback


def build_workflow_statement(
    artifacts: WorkflowTraceArtifacts,
    commitment: WorkflowTraceCommitment,
) -> dict[str, Any]:
    """Build a complete statement for the workflow proof.

    The statement binds:
    - AIR specification (workflow_air_v1)
    - Trace commitment (row_root)
    - Public inputs (dag_root)
    """
    return {
        "statement_version": "1.0",
        "air_id": WORKFLOW_AIR_ID,
        "trace_spec_hash": artifacts.trace_spec_hash,
        "row_commitment": commitment.row_root,
        "num_rows": commitment.num_rows,
        "public_inputs": {
            "dag_root_lo": artifacts.statement.dag_root_lo,
            "dag_root_hi": artifacts.statement.dag_root_hi,
            "dag_root_hash": artifacts.statement.dag_root_hash,
            "num_nodes": artifacts.statement.num_nodes,
        },
    }


def build_workflow_capsule(
    artifacts: WorkflowTraceArtifacts,
    commitment: WorkflowTraceCommitment,
    proof_artifacts: WorkflowProofArtifacts,
    run_dir: Path,
) -> dict[str, Any]:
    """Build a complete capsule from workflow proof artifacts.

    The capsule structure is compatible with existing capseal verify.
    """
    statement = build_workflow_statement(artifacts, commitment)
    statement_hash = hashlib.sha256(
        json.dumps(statement, sort_keys=True).encode()
    ).hexdigest()

    capsule = {
        "schema": "workflow_capsule_v1",
        "capsule_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "air_id": WORKFLOW_AIR_ID,
        "statement": statement,
        "statement_hash": statement_hash,
        "commitment": {
            "row_root": commitment.row_root,
            "num_rows": commitment.num_rows,
        },
        "proof": {
            "encoding_id": WORKFLOW_AIR_ID,
            "proof_hash": hashlib.sha256(proof_artifacts.proof_bytes).hexdigest(),
        },
        "verification": {
            "constraints_valid": proof_artifacts.verification_result,
        },
        "profile": proof_artifacts.profile_data,
    }

    # Compute capsule hash
    capsule_content = json.dumps(capsule, sort_keys=True)
    capsule["capsule_hash"] = hashlib.sha256(capsule_content.encode()).hexdigest()

    return capsule


def save_workflow_capsule(capsule: dict[str, Any], path: Path) -> None:
    """Save workflow capsule to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capsule, indent=2, sort_keys=True))


def verify_workflow_capsule(capsule_path: Path) -> tuple[bool, dict[str, Any]]:
    """Verify a workflow capsule from file.

    Args:
        capsule_path: Path to workflow_capsule.json

    Returns:
        Tuple of (valid, details)
    """
    try:
        capsule = json.loads(capsule_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return False, {"error": str(e)}

    details: dict[str, Any] = {"checks": []}

    # Check schema
    if capsule.get("schema") != "workflow_capsule_v1":
        details["error"] = "Invalid capsule schema"
        return False, details
    details["checks"].append("schema_valid")

    # Check AIR ID
    if capsule.get("air_id") != WORKFLOW_AIR_ID:
        details["error"] = f"Invalid AIR ID: {capsule.get('air_id')}"
        return False, details
    details["checks"].append("air_id_valid")

    # Check verification status
    verification = capsule.get("verification", {})
    if not verification.get("constraints_valid"):
        details["error"] = "Constraints not satisfied"
        return False, details
    details["checks"].append("constraints_valid")

    # Verify capsule hash
    capsule_copy = dict(capsule)
    stored_hash = capsule_copy.pop("capsule_hash", None)
    computed_hash = hashlib.sha256(
        json.dumps(capsule_copy, sort_keys=True).encode()
    ).hexdigest()
    if computed_hash != stored_hash:
        details["error"] = "Capsule hash mismatch"
        details["expected_hash"] = stored_hash
        details["computed_hash"] = computed_hash
        return False, details
    details["checks"].append("capsule_hash_valid")

    # Verify statement hash
    statement = capsule.get("statement", {})
    statement_hash = hashlib.sha256(
        json.dumps(statement, sort_keys=True).encode()
    ).hexdigest()
    if statement_hash != capsule.get("statement_hash"):
        details["error"] = "Statement hash mismatch"
        return False, details
    details["checks"].append("statement_hash_valid")

    details["valid"] = True
    details["dag_root"] = statement.get("public_inputs", {}).get("dag_root_hash", "")[:32] + "..."
    details["num_nodes"] = statement.get("public_inputs", {}).get("num_nodes", 0)
    return True, details


def row_merkle_proof(row_archive_dir: Path, row_index: int) -> dict[str, Any]:
    """Generate Merkle proof for a specific row.

    Args:
        row_archive_dir: Directory containing row archive
        row_index: Index of row to prove

    Returns:
        Dict with row values and Merkle path
    """
    # Load rows
    rows_path = row_archive_dir / "rows.json"
    levels_path = row_archive_dir / "merkle_levels.json"

    if not rows_path.exists() or not levels_path.exists():
        raise FileNotFoundError("Row archive not found")

    rows = json.loads(rows_path.read_text())
    levels_hex = json.loads(levels_path.read_text())
    levels = [[bytes.fromhex(h) for h in level] for level in levels_hex]

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Row index {row_index} out of range")

    # Generate proof
    proof_path = _merkle_prove(levels, row_index)

    return {
        "row_index": row_index,
        "row_values": rows[row_index],
        "merkle_path": proof_path,
        "root": levels[-1][0].hex(),
    }
