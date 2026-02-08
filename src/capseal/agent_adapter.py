"""AgentAdapter -- bridges any agent execution to the proof pipeline.

This adapter follows the same interface as WorkflowAdapter and EvalAdapter,
converting agent action sequences into provable traces.

The adapter:
1. simulate_trace: Converts agent actions into trace rows
2. commit_to_trace: Builds Merkle commitment over the trace rows
3. generate_proof: Generates FRI proof that trace satisfies AgentAIR constraints
4. verify: Verifies the proof against the statement
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from capseal.agent_air import (
    GOLDILOCKS_P,
    AGENT_AIR_ROW_WIDTH,
    AGENT_AIR_ID,
    encode_agent_action_row,
    decode_agent_action_row,
    sha256_to_field_pair,
    build_agent_row_matrix,
)
from capseal.agent_constraints import (
    AgentConstraints,
    verify_agent_trace,
    build_composition_vector,
    derive_constraint_alphas,
)

if TYPE_CHECKING:
    from capseal.agent_protocol import AgentAction

# Optional FRI proof upgrade — when bef_zk.fri is available, generate real FRI proofs
# instead of just constraint-check proofs.
try:
    from bef_zk.fri.prover import fri_prove
    from bef_zk.fri.verifier import fri_verify
    from bef_zk.fri.config import FRIConfig
    from bef_zk.stc.vc import VectorCommitment, VCCommitment
    HAS_FRI = True
except ImportError:
    HAS_FRI = False


def _generate_fri_proof(composition: list[int], row_root: str) -> dict[str, Any]:
    """Generate a FRI proof over the composition polynomial.

    Requires bef_zk.fri to be installed (HAS_FRI=True).
    The composition vector must be all zeros for a valid trace.
    """
    import math

    # Pad composition to next power of 2
    n = len(composition)
    domain_size = 1 << math.ceil(math.log2(max(n * 4, 8)))
    padded = list(composition) + [0] * (domain_size - n)

    # Build FRI config
    num_rounds = max(1, math.ceil(math.log2(domain_size)) - 2)
    num_queries = min(16, domain_size // 2)
    fri_cfg = FRIConfig(
        domain_size=domain_size,
        max_degree=n - 1,
        num_rounds=num_rounds,
        num_queries=num_queries,
    )

    # Commit to evaluations
    vc = VectorCommitment(chunk_size=64)
    base_commitment = vc.commit(padded)

    # Derive query indices via Fiat-Shamir
    seed = hashlib.sha256(
        row_root.encode() + base_commitment.root
    ).digest()
    query_indices = []
    for i in range(num_queries):
        idx_bytes = hashlib.sha256(seed + i.to_bytes(4, "little")).digest()
        idx = int.from_bytes(idx_bytes[:4], "little") % domain_size
        query_indices.append(idx)

    # Generate FRI proof
    proof = fri_prove(fri_cfg, vc, padded, base_commitment, query_indices)

    # Verify
    query_values = [padded[qi] for qi in query_indices]
    verified = fri_verify(
        fri_cfg, vc, base_commitment, proof, query_indices, query_values,
    )

    return {
        "verified": verified,
        "domain_size": domain_size,
        "num_rounds": num_rounds,
        "num_queries": num_queries,
        "commitment_root": base_commitment.root.hex() if isinstance(base_commitment.root, bytes) else str(base_commitment.root),
    }


@dataclass
class AgentContext:
    """Context for agent proof generation."""
    actions: list["AgentAction"]
    rows: list[list[int]]
    final_receipt_hash: str
    trace_time_sec: float
    prepared: dict[str, Any] | None = None


@dataclass
class AgentStatement:
    """Public statement for agent proof."""
    air_id: str
    num_actions: int
    final_receipt_lo: int
    final_receipt_hi: int
    final_receipt_hash: str
    trace_spec_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "air_id": self.air_id,
            "num_actions": self.num_actions,
            "final_receipt_lo": self.final_receipt_lo,
            "final_receipt_hi": self.final_receipt_hi,
            "final_receipt_hash": self.final_receipt_hash,
            "trace_spec_hash": self.trace_spec_hash,
        }


@dataclass
class AgentTraceArtifacts:
    """Artifacts from trace simulation."""
    trace_id: str
    trace_spec: Any
    trace_spec_hash: str
    rows: list[list[int]]
    row_width: int
    context: AgentContext
    trace_time_sec: float
    statement: AgentStatement


@dataclass
class AgentTraceCommitment:
    """Commitment to the agent trace."""
    row_root: str
    num_rows: int
    params: dict[str, Any]
    profile_data: dict[str, float]


@dataclass
class AgentProofArtifacts:
    """Proof artifacts for agent."""
    proof_json: str
    proof_bytes: bytes
    statement: AgentStatement
    verification_result: bool
    profile_data: dict[str, float]
    proof_type: str = "constraint_check"


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


class AgentAdapter:
    """Adapter that bridges agent execution to proof pipeline.

    Follows the same interface as WorkflowAdapter and EvalAdapter:
    - simulate_trace: Convert agent actions to trace rows
    - commit_to_trace: Build Merkle commitment
    - generate_proof: Generate FRI proof
    - verify: Verify the proof
    """

    name = "agent"

    def __init__(self, air_id: str = AGENT_AIR_ID) -> None:
        self.air_id = air_id
        self._progress_callback = None

    def simulate_trace(
        self,
        actions: list["AgentAction"],
        *,
        trace_id: str | None = None,
    ) -> AgentTraceArtifacts:
        """Convert a list of agent actions into trace rows.

        Args:
            actions: List of AgentActions in execution order
            trace_id: Optional trace identifier

        Returns:
            AgentTraceArtifacts with trace data
        """
        start = time.perf_counter()

        if not actions:
            raise ValueError("No actions provided")

        # Build trace rows
        rows = build_agent_row_matrix(actions)

        # The final receipt is the last action's receipt hash
        final_receipt_hash = actions[-1].compute_receipt_hash()
        final_receipt_lo, final_receipt_hi = sha256_to_field_pair(final_receipt_hash)

        trace_time = time.perf_counter() - start

        # Build trace spec
        trace_spec = {
            "air_id": self.air_id,
            "row_width": AGENT_AIR_ROW_WIDTH,
            "num_rows": len(rows),
            "field_modulus": "goldilocks",
        }
        trace_spec_hash = hashlib.sha256(
            json.dumps(trace_spec, sort_keys=True).encode()
        ).hexdigest()

        # Build statement
        statement = AgentStatement(
            air_id=self.air_id,
            num_actions=len(actions),
            final_receipt_lo=final_receipt_lo,
            final_receipt_hi=final_receipt_hi,
            final_receipt_hash=final_receipt_hash,
            trace_spec_hash=trace_spec_hash,
        )

        # Build context
        context = AgentContext(
            actions=actions,
            rows=rows,
            final_receipt_hash=final_receipt_hash,
            trace_time_sec=trace_time,
        )

        trace_id = trace_id or f"agent_{int(time.time())}"

        self._emit_progress(
            "trace_simulated",
            {
                "trace_spec_hash": trace_spec_hash,
                "num_actions": len(actions),
                "num_rows": len(rows),
            },
        )

        return AgentTraceArtifacts(
            trace_id=trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            rows=rows,
            row_width=AGENT_AIR_ROW_WIDTH,
            context=context,
            trace_time_sec=trace_time,
            statement=statement,
        )

    def commit_to_trace(
        self,
        artifacts: AgentTraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> AgentTraceCommitment:
        """Commit to the trace rows using a Merkle tree.

        Args:
            artifacts: Trace artifacts from simulate_trace
            row_archive_dir: Directory to store row archive

        Returns:
            AgentTraceCommitment with Merkle root
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

        return AgentTraceCommitment(
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
        artifacts: AgentTraceArtifacts,
        commitment: AgentTraceCommitment,
        *,
        statement_hash: bytes | None = None,
        binding_hash: bytes | None = None,
        encoding_id: str | None = None,
        trace_path: Path | None = None,
    ) -> AgentProofArtifacts:
        """Generate proof that the trace satisfies AgentAIR constraints.

        For Phase 3, this generates a simplified proof structure that:
        1. Verifies all AIR constraints locally
        2. Commits to the constraint satisfaction
        3. Produces a proof artifact compatible with capsule format

        Full FRI proof integration can be added in a future phase.

        Args:
            artifacts: Trace artifacts
            commitment: Trace commitment
            statement_hash: Hash binding the statement
            binding_hash: Additional binding material
            encoding_id: AIR encoding identifier (defaults to self.air_id)
            trace_path: Optional path to store trace

        Returns:
            AgentProofArtifacts with proof data
        """
        prove_start = time.perf_counter()

        encoding_id = encoding_id or self.air_id
        ctx = artifacts.context
        rows = artifacts.rows
        statement = artifacts.statement

        # Verify constraints locally (this is what the FRI proof will prove)
        valid, constraint_results = verify_agent_trace(rows, ctx.final_receipt_hash)

        # Build composition vector (for FRI, we'd commit to this)
        alpha_seed = hashlib.sha256(
            b"agent_alphas:" + commitment.row_root.encode()
        ).digest()
        alphas = derive_constraint_alphas(alpha_seed)
        composition = build_composition_vector(
            rows,
            statement.final_receipt_lo,
            statement.final_receipt_hi,
            alphas,
        )

        # Composition should be all zeros if constraints satisfied
        composition_sum = sum(composition) % GOLDILOCKS_P

        # Build proof structure
        proof_data = {
            "air_id": encoding_id,
            "version": "1.0",
            "proof_type": "constraint_check",
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

        # Upgrade to full FRI proof when bef_zk.fri is available
        if HAS_FRI and valid and composition:
            try:
                fri_result = _generate_fri_proof(composition, commitment.row_root)
                proof_data["proof_type"] = "fri"
                proof_data["fri_proof"] = fri_result
            except Exception:
                pass  # Fall back to constraint_check proof

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

        return AgentProofArtifacts(
            proof_json=proof_json,
            proof_bytes=proof_bytes,
            statement=statement,
            verification_result=valid,
            profile_data=profile_data,
            proof_type=proof_data.get("proof_type", "constraint_check"),
        )

    def verify(
        self,
        proof_json: str,
        commitment_root: str,
        final_receipt_hash: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify an agent proof.

        Args:
            proof_json: JSON proof string
            commitment_root: Expected row commitment root
            final_receipt_hash: Expected final receipt hash

        Returns:
            Tuple of (valid, verification_details)
        """
        try:
            proof_data = json.loads(proof_json)
        except json.JSONDecodeError:
            return False, {"error": "Invalid proof JSON"}

        details: dict[str, Any] = {}

        # Check AIR ID
        if proof_data.get("air_id") != self.air_id:
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
        if statement.get("final_receipt_hash") != final_receipt_hash:
            details["receipt_hash_mismatch"] = True
            details["expected_receipt_hash"] = final_receipt_hash
            details["proof_receipt_hash"] = statement.get("final_receipt_hash")
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

    def prove_actions(
        self,
        actions: list["AgentAction"],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Convenience method: trace -> commit -> prove -> capsule in one call.

        Args:
            actions: List of AgentActions in execution order
            output_dir: Directory to store output artifacts

        Returns:
            Dict with capsule data and file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        row_archive_dir = output_dir / "row_archive"

        # Step 1: Simulate trace
        artifacts = self.simulate_trace(actions)

        # Step 2: Commit to trace
        commitment = self.commit_to_trace(artifacts, row_archive_dir=row_archive_dir)

        # Step 3: Generate proof
        proof_artifacts = self.generate_proof(artifacts, commitment)

        # Step 4: Build and save capsule
        capsule = build_agent_capsule(artifacts, commitment, proof_artifacts, output_dir)
        capsule_path = output_dir / "agent_capsule.json"
        save_agent_capsule(capsule, capsule_path)

        # Save proof
        proof_path = output_dir / "proof.json"
        proof_path.write_text(proof_artifacts.proof_json)

        # Save actions log
        actions_log = output_dir / "actions.json"
        actions_data = [a.to_dict() for a in actions]
        actions_log.write_text(json.dumps(actions_data, indent=2))

        return {
            "capsule": capsule,
            "capsule_path": str(capsule_path),
            "proof_path": str(proof_path),
            "actions_path": str(actions_log),
            "final_receipt_hash": artifacts.context.final_receipt_hash,
            "num_actions": len(actions),
            "verified": proof_artifacts.verification_result,
            "proof_type": proof_artifacts.proof_type,
        }

    def _emit_progress(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit progress event if callback is set."""
        if self._progress_callback is not None:
            self._progress_callback({"type": event_type, "data": data})

    def set_progress_callback(self, callback) -> None:
        """Set progress callback for monitoring."""
        self._progress_callback = callback


def build_agent_statement(
    artifacts: AgentTraceArtifacts,
    commitment: AgentTraceCommitment,
) -> dict[str, Any]:
    """Build a complete statement for the agent proof.

    The statement binds:
    - AIR specification (agent_air_v1)
    - Trace commitment (row_root)
    - Public inputs (final_receipt)
    """
    return {
        "statement_version": "1.0",
        "air_id": AGENT_AIR_ID,
        "trace_spec_hash": artifacts.trace_spec_hash,
        "row_commitment": commitment.row_root,
        "num_rows": commitment.num_rows,
        "public_inputs": {
            "final_receipt_lo": artifacts.statement.final_receipt_lo,
            "final_receipt_hi": artifacts.statement.final_receipt_hi,
            "final_receipt_hash": artifacts.statement.final_receipt_hash,
            "num_actions": artifacts.statement.num_actions,
        },
    }


def build_agent_capsule(
    artifacts: AgentTraceArtifacts,
    commitment: AgentTraceCommitment,
    proof_artifacts: AgentProofArtifacts,
    run_dir: Path,
) -> dict[str, Any]:
    """Build a complete capsule from agent proof artifacts.

    The capsule structure is compatible with existing capseal verify.
    """
    statement = build_agent_statement(artifacts, commitment)
    statement_hash = hashlib.sha256(
        json.dumps(statement, sort_keys=True).encode()
    ).hexdigest()

    capsule = {
        "schema": "agent_capsule_v1",
        "capsule_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "air_id": AGENT_AIR_ID,
        "statement": statement,
        "statement_hash": statement_hash,
        "commitment": {
            "row_root": commitment.row_root,
            "num_rows": commitment.num_rows,
        },
        "proof": {
            "encoding_id": AGENT_AIR_ID,
            "proof_hash": hashlib.sha256(proof_artifacts.proof_bytes).hexdigest(),
        },
        "verification": {
            "constraints_valid": proof_artifacts.verification_result,
            "proof_type": proof_artifacts.proof_type,
        },
        "profile": proof_artifacts.profile_data,
    }

    # Compute capsule hash
    capsule_content = json.dumps(capsule, sort_keys=True)
    capsule["capsule_hash"] = hashlib.sha256(capsule_content.encode()).hexdigest()

    return capsule


def save_agent_capsule(capsule: dict[str, Any], path: Path) -> None:
    """Save agent capsule to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capsule, indent=2, sort_keys=True))


def verify_agent_capsule(capsule_path: Path) -> tuple[bool, dict[str, Any]]:
    """Verify an agent capsule from file.

    Args:
        capsule_path: Path to agent_capsule.json

    Returns:
        Tuple of (valid, details)
    """
    try:
        capsule = json.loads(capsule_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return False, {"error": str(e)}

    details: dict[str, Any] = {"checks": []}

    # Check schema
    if capsule.get("schema") != "agent_capsule_v1":
        details["error"] = "Invalid capsule schema"
        return False, details
    details["checks"].append("schema_valid")

    # Check AIR ID
    if capsule.get("air_id") != AGENT_AIR_ID:
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
    details["final_receipt"] = statement.get("public_inputs", {}).get("final_receipt_hash", "")[:32] + "..."
    details["num_actions"] = statement.get("public_inputs", {}).get("num_actions", 0)
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
