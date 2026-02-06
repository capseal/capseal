"""RISC Zero TraceAdapter implementation with portable fixture support.

Supports two modes:
1. Live execution: Requires risc0-zkvm Python bindings, runs guest and generates receipt
2. Receipt mode: Loads pre-generated receipt from directory (for fixtures/testing)

Binding semantics:
- image_id: Hash of the zkVM guest binary
- journal_digest: Hash of the public journal output
- binding_hash: H("RISC0_BIND_V1" || image_id || journal_digest || statement_hash)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bef_zk.adapter import (
    ProofArtifacts,
    TraceAdapter,
    TraceArtifacts,
    TraceCommitment,
)
from bef_zk.spec import TraceSpecV1, compute_trace_spec_hash

# Domain separation tags for binding
RISC0_BIND_TAG = b"RISC0_BIND_V1"
RISC0_CLAIM_TAG = b"RISC0_CLAIM_V1"

# Try to import RISC0 SDK (optional)
try:  # pragma: no cover - optional dependency
    from risc0_zkvm import Receipt as Risc0Receipt
    from risc0_zkvm import verify as risc0_verify_sdk
    _HAVE_RISC0_SDK = True
except ImportError:
    Risc0Receipt = None
    risc0_verify_sdk = None
    _HAVE_RISC0_SDK = False


def _sha256(data: bytes) -> bytes:
    """Compute SHA256 hash."""
    return hashlib.sha256(data).digest()


def _sha256_hex(data: bytes) -> str:
    """Compute SHA256 hash as hex string."""
    return hashlib.sha256(data).hexdigest()


def compute_binding_hash(
    image_id: str,
    journal_digest: str,
    statement_hash: bytes,
) -> bytes:
    """Compute the binding hash for RISC0 receipts.

    binding_hash = H(RISC0_BIND_TAG || image_id || journal_digest || statement_hash)
    """
    h = hashlib.sha256()
    h.update(RISC0_BIND_TAG)
    h.update(bytes.fromhex(image_id.replace("0x", "")))
    h.update(bytes.fromhex(journal_digest.replace("0x", "")))
    h.update(statement_hash)
    return h.digest()


def compute_claim_root(image_id: str, journal_bytes: bytes) -> str:
    """Compute the claim root from image_id and journal.

    claim_root = H(RISC0_CLAIM_TAG || image_id || journal)
    """
    h = hashlib.sha256()
    h.update(RISC0_CLAIM_TAG)
    h.update(bytes.fromhex(image_id.replace("0x", "")))
    h.update(journal_bytes)
    return h.hexdigest()


@dataclass
class Risc0ReceiptData:
    """Parsed RISC0 receipt data."""
    image_id: str
    journal: bytes
    journal_digest: str
    claim_root: str
    seal: bytes
    receipt_json: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Risc0ReceiptData":
        """Parse receipt from JSON representation."""
        image_id = data.get("image_id", data.get("imageId", ""))
        if isinstance(image_id, list):
            # Convert array of u32 to hex
            image_id = "".join(f"{x:08x}" for x in image_id)

        # Parse journal
        journal_data = data.get("journal", data.get("output", {}).get("journal", {}))
        if isinstance(journal_data, dict):
            journal_bytes_list = journal_data.get("bytes", [])
            journal = bytes(journal_bytes_list)
        elif isinstance(journal_data, str):
            journal = bytes.fromhex(journal_data.replace("0x", ""))
        elif isinstance(journal_data, list):
            journal = bytes(journal_data)
        else:
            journal = b""

        journal_digest = _sha256_hex(journal)
        claim_root = compute_claim_root(image_id, journal)

        # Parse seal
        seal_data = data.get("seal", data.get("inner", {}).get("seal", []))
        if isinstance(seal_data, list):
            # Seal is typically an array of u32
            seal = b"".join(x.to_bytes(4, "little") if isinstance(x, int) else bytes(x) for x in seal_data)
        elif isinstance(seal_data, str):
            seal = bytes.fromhex(seal_data.replace("0x", ""))
        else:
            seal = b""

        return cls(
            image_id=image_id,
            journal=journal,
            journal_digest=journal_digest,
            claim_root=claim_root,
            seal=seal,
            receipt_json=json.dumps(data),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Risc0ReceiptData":
        """Load receipt from JSON file."""
        return cls.from_json(json.loads(path.read_text()))


@dataclass
class Risc0TraceCommitment(TraceCommitment):
    """Trace commitment wrapper for RISC0 receipts."""
    receipt: Risc0ReceiptData | None = None
    image_id: str = ""
    journal_digest: str = ""


class Risc0TraceAdapter(TraceAdapter):
    """RISC Zero backend adapter.

    Supports two modes:
    1. --run-config: Execute guest and generate receipt (requires SDK)
    2. --receipt-dir: Load pre-generated receipt for verification
    """
    name = "risc0"

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        parser.add_argument(
            "--run-config",
            type=str,
            help="JSON blob describing ELF path and optional private inputs (live mode)",
        )
        parser.add_argument(
            "--receipt-dir",
            type=Path,
            help="Directory containing pre-generated receipt.json (fixture mode)",
        )
        parser.add_argument(
            "--image-id",
            type=str,
            help="RISC0 image ID (required for receipt-dir mode if not in receipt)",
        )

    def simulate_trace(self, args: Any) -> TraceArtifacts:
        """Load or generate trace artifacts."""
        receipt_dir = getattr(args, "receipt_dir", None)
        run_config = getattr(args, "run_config", None)

        if receipt_dir:
            return self._simulate_from_receipt(args, Path(receipt_dir))
        elif run_config:
            return self._simulate_live(args, run_config)
        else:
            raise ValueError("Either --run-config or --receipt-dir is required")

    def _simulate_from_receipt(self, args: Any, receipt_dir: Path) -> TraceArtifacts:
        """Load trace from pre-generated receipt directory."""
        receipt_path = receipt_dir / "receipt.json"
        if not receipt_path.exists():
            raise FileNotFoundError(f"Receipt not found: {receipt_path}")

        receipt = Risc0ReceiptData.from_file(receipt_path)

        # Allow image_id override from CLI
        image_id = getattr(args, "image_id", None) or receipt.image_id
        if not image_id:
            raise ValueError("image_id required (from receipt or --image-id)")

        trace_spec = TraceSpecV1(
            spec_version="1.0",
            trace_format_id="RISC0_RECEIPT_V1",
            record_schema_ref="inline",
            encoding_id=args.encoding_id,
            field_modulus_id="risc0_babybear",
        )
        trace_spec_hash = compute_trace_spec_hash(trace_spec)

        bef_trace = {
            "backend": "risc0",
            "image_id": image_id,
            "journal_digest": receipt.journal_digest,
            "claim_root": receipt.claim_root,
        }

        context = {
            "receipt": receipt,
            "image_id": image_id,
            "receipt_dir": str(receipt_dir),
        }

        return TraceArtifacts(
            trace_id=args.trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            bef_trace=bef_trace,
            row_width=0,  # RISC0 is succinct, no row trace
            context=context,
        )

    def _simulate_live(self, args: Any, run_config_str: str) -> TraceArtifacts:
        """Execute guest live and generate receipt (requires SDK)."""
        if not _HAVE_RISC0_SDK:
            raise RuntimeError(
                "RISC Zero SDK not available. Install risc0-zkvm or use --receipt-dir mode."
            )

        # Parse config
        try:
            if Path(run_config_str).exists():
                run_cfg = json.loads(Path(run_config_str).read_text())
            else:
                run_cfg = json.loads(run_config_str)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON in --run-config") from exc

        elf_path = run_cfg.get("elf_path")
        if not elf_path:
            raise ValueError("run-config must include 'elf_path'")

        trace_spec = TraceSpecV1(
            spec_version="1.0",
            trace_format_id="RISC0_RECEIPT_V1",
            record_schema_ref="inline",
            encoding_id=args.encoding_id,
            field_modulus_id="risc0_babybear",
        )
        trace_spec_hash = compute_trace_spec_hash(trace_spec)

        bef_trace = {"backend": "risc0", "elf_path": elf_path}
        context = {"run_config": run_cfg, "mode": "live"}

        return TraceArtifacts(
            trace_id=args.trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            bef_trace=bef_trace,
            row_width=0,
            context=context,
        )

    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict[str, Any]]:
        """Extract public inputs from journal."""
        ctx = artifacts.context or {}
        receipt = ctx.get("receipt")

        if receipt and isinstance(receipt, Risc0ReceiptData):
            return [
                {"name": "image_id", "value": receipt.image_id},
                {"name": "journal_digest", "value": receipt.journal_digest},
            ]
        return []

    def commit_to_trace(
        self,
        artifacts: TraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> Risc0TraceCommitment:
        """Commit to the trace using receipt claim as commitment."""
        ctx = artifacts.context or {}
        receipt = ctx.get("receipt")
        mode = ctx.get("mode", "receipt")

        if mode == "live":
            # Live execution mode - run prover here
            return self._commit_live(artifacts, row_archive_dir)

        if not isinstance(receipt, Risc0ReceiptData):
            raise RuntimeError("Receipt not loaded. Use --receipt-dir or --run-config")

        image_id = ctx.get("image_id", receipt.image_id)
        root_hex = receipt.claim_root

        # Create minimal row archive (RISC0 is succinct)
        row_archive_dir.mkdir(parents=True, exist_ok=True)

        # Write claim data as the "row archive"
        claim_path = row_archive_dir / "claim.json"
        claim_data = {
            "image_id": image_id,
            "journal_digest": receipt.journal_digest,
            "claim_root": receipt.claim_root,
        }
        claim_path.write_text(json.dumps(claim_data, indent=2))

        # Write empty chunk roots (succinct proof, no chunks)
        chunk_roots_json = row_archive_dir / "chunk_roots.json"
        chunk_roots_json.write_text("[]")
        chunk_roots_bin = row_archive_dir / "chunk_roots.bin"
        chunk_roots_bin.write_bytes(b"")
        chunk_roots_digest = _sha256_hex(b"")

        row_commitment = SimpleNamespace(
            backend="risc0_receipt_v1",
            row_width=0,
            params={
                "root": root_hex,
                "chunk_tree_arity": 1,
                "chunk_len": 0,
                "num_chunks": 0,
                "image_id": image_id,
                "journal_digest": receipt.journal_digest,
            },
        )

        chunk_handles: list[str] = []
        row_archive_artifact = {
            "mode": "INLINE",
            "abs_path": str(row_archive_dir),
            "chunk_roots_path": str(chunk_roots_json),
            "chunk_roots_bin_path": str(chunk_roots_bin),
            "chunk_roots_format": "hex_json_v1",
            "chunk_roots_bin_format": "raw32_v1",
            "chunk_roots_digest": chunk_roots_digest,
        }

        return Risc0TraceCommitment(
            row_commitment=row_commitment,
            row_archive_artifact=row_archive_artifact,
            chunk_handles=chunk_handles,
            chunk_roots_hex=[],
            chunk_roots_digest=chunk_roots_digest,
            chunk_roots_paths={"json": chunk_roots_json, "bin": chunk_roots_bin},
            profile_data={},
            extra={"journal_len": len(receipt.journal)},
            receipt=receipt,
            image_id=image_id,
            journal_digest=receipt.journal_digest,
        )

    def _commit_live(
        self,
        artifacts: TraceArtifacts,
        row_archive_dir: Path,
    ) -> Risc0TraceCommitment:
        """Commit via live execution (requires SDK)."""
        if not _HAVE_RISC0_SDK:
            raise RuntimeError("RISC Zero SDK required for live execution")

        # Import here to avoid issues when SDK not available
        from risc0_zkvm.executor import Executor
        from risc0_zkvm.prover import Prover

        ctx = artifacts.context or {}
        run_cfg = ctx.get("run_config", {})
        elf_path = run_cfg.get("elf_path")
        private_input = run_cfg.get("private_input")

        if not elf_path:
            raise ValueError("elf_path required for live execution")

        start = time.perf_counter()
        executor = Executor(elf_path)
        if private_input:
            executor.add_input(private_input)
        session = executor.run()

        prover = Prover(session)
        sdk_receipt = prover.run()
        elapsed = time.perf_counter() - start

        # Convert SDK receipt to our format
        journal_bytes = getattr(sdk_receipt, "journal", b"") or b""
        image_id = getattr(sdk_receipt, "image_id", "")
        if isinstance(image_id, list):
            image_id = "".join(f"{x:08x}" for x in image_id)

        receipt = Risc0ReceiptData(
            image_id=image_id,
            journal=journal_bytes,
            journal_digest=_sha256_hex(journal_bytes),
            claim_root=compute_claim_root(image_id, journal_bytes),
            seal=b"",  # SDK handles this internally
            receipt_json=sdk_receipt.to_json() if hasattr(sdk_receipt, "to_json") else "{}",
        )

        # Now proceed as receipt mode
        row_archive_dir.mkdir(parents=True, exist_ok=True)

        claim_path = row_archive_dir / "claim.json"
        claim_data = {
            "image_id": image_id,
            "journal_digest": receipt.journal_digest,
            "claim_root": receipt.claim_root,
        }
        claim_path.write_text(json.dumps(claim_data, indent=2))

        chunk_roots_json = row_archive_dir / "chunk_roots.json"
        chunk_roots_json.write_text("[]")
        chunk_roots_bin = row_archive_dir / "chunk_roots.bin"
        chunk_roots_bin.write_bytes(b"")
        chunk_roots_digest = _sha256_hex(b"")

        row_commitment = SimpleNamespace(
            backend="risc0_receipt_v1",
            row_width=0,
            params={
                "root": receipt.claim_root,
                "chunk_tree_arity": 1,
                "chunk_len": 0,
                "num_chunks": 0,
                "image_id": image_id,
                "journal_digest": receipt.journal_digest,
            },
        )

        return Risc0TraceCommitment(
            row_commitment=row_commitment,
            row_archive_artifact={
                "mode": "INLINE",
                "abs_path": str(row_archive_dir),
                "chunk_roots_path": str(chunk_roots_json),
                "chunk_roots_bin_path": str(chunk_roots_bin),
                "chunk_roots_format": "hex_json_v1",
                "chunk_roots_bin_format": "raw32_v1",
                "chunk_roots_digest": chunk_roots_digest,
            },
            chunk_handles=[],
            chunk_roots_hex=[],
            chunk_roots_digest=chunk_roots_digest,
            chunk_roots_paths={"json": chunk_roots_json, "bin": chunk_roots_bin},
            profile_data={"time_prove_sec": elapsed},
            extra={"journal_len": len(journal_bytes)},
            receipt=receipt,
            image_id=image_id,
            journal_digest=receipt.journal_digest,
        )

    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: Risc0TraceCommitment,
        *,
        statement_hash: bytes,
        binding_hash: bytes | None = None,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        """Generate proof artifacts from receipt.

        The pipeline provides binding_hash (computed via hash_instance_binding).
        We use that for the capsule proof, but also compute our own internal
        binding (risc0_binding) for additional verification.
        """
        if not isinstance(commitment, Risc0TraceCommitment):
            raise TypeError("generate_proof expected a Risc0TraceCommitment")
        if commitment.receipt is None:
            raise RuntimeError("Risc0TraceCommitment missing receipt data")

        receipt = commitment.receipt

        # Use pipeline-provided binding_hash or statement_hash as the binding material
        effective_binding = binding_hash or statement_hash

        # Also compute our internal risc0-specific binding for verification
        risc0_binding = compute_binding_hash(
            commitment.image_id,
            commitment.journal_digest,
            statement_hash,
        )

        # Build proof JSON with binding information
        proof_data = {
            "schema": "risc0_capsule_proof_v1",
            "image_id": commitment.image_id,
            "journal": receipt.journal.hex(),
            "journal_digest": receipt.journal_digest,
            "claim_root": receipt.claim_root,
            "binding_hash": effective_binding.hex(),
            "risc0_binding": risc0_binding.hex(),
            "statement_hash": statement_hash.hex(),
            "receipt": json.loads(receipt.receipt_json) if receipt.receipt_json else {},
        }

        proof_json = json.dumps(proof_data, indent=2)
        proof_bytes = proof_json.encode("utf-8")

        profile_data = dict(commitment.profile_data)
        chunk_stats = {
            "leaf_auth_bytes_baseline": 0,
            "leaf_auth_bytes_actual": 0,
            "leaf_auth_bytes_saved": 0,
            "leaf_auth_savings_ratio": 0,
        }

        return ProofArtifacts(
            proof_obj=proof_data,
            proof_json=proof_json,
            proof_bytes=proof_bytes,
            profile_data=profile_data,
            chunk_leaf_stats=chunk_stats,
            extra={"binding_hash": effective_binding.hex(), "risc0_binding": risc0_binding.hex()},
        )

    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
        *,
        binding_hash: bytes | None = None,
    ) -> tuple[bool, dict[str, Any], float]:
        """Verify the RISC0 proof.

        Verification checks:
        1. Parse proof and extract image_id, journal, claim
        2. Verify binding_hash = H(image_id || journal_digest || statement_hash)
        3. If SDK available, verify the actual receipt
        4. Otherwise, verify binding integrity only
        """
        start = time.perf_counter()

        try:
            proof_data = json.loads(proof_json)
        except json.JSONDecodeError:
            return False, {"error": "invalid proof JSON"}, time.perf_counter() - start

        # Extract proof fields
        image_id = proof_data.get("image_id", "")
        journal_hex = proof_data.get("journal", "")
        journal_digest = proof_data.get("journal_digest", "")
        risc0_binding_hex = proof_data.get("risc0_binding", "")

        if not image_id or not journal_digest:
            return False, {"error": "missing image_id or journal_digest"}, time.perf_counter() - start

        # Recompute and verify risc0-specific binding hash
        expected_risc0_binding = compute_binding_hash(image_id, journal_digest, statement_hash)

        if risc0_binding_hex and bytes.fromhex(risc0_binding_hex) != expected_risc0_binding:
            return False, {
                "error": "risc0_binding mismatch",
                "expected": expected_risc0_binding.hex(),
                "got": risc0_binding_hex,
            }, time.perf_counter() - start

        # Verify journal digest
        if journal_hex:
            journal_bytes = bytes.fromhex(journal_hex)
            computed_digest = _sha256_hex(journal_bytes)
            if computed_digest != journal_digest:
                return False, {
                    "error": "journal_digest mismatch",
                    "expected": computed_digest,
                    "got": journal_digest,
                }, time.perf_counter() - start

        # If SDK available and receipt present, verify cryptographically
        receipt_data = proof_data.get("receipt", {})
        if _HAVE_RISC0_SDK and receipt_data:
            try:
                sdk_receipt = Risc0Receipt.from_json(json.dumps(receipt_data))
                if not sdk_receipt.verify():
                    return False, {"error": "receipt verification failed"}, time.perf_counter() - start
            except Exception as exc:
                return False, {"error": f"SDK verification error: {exc}"}, time.perf_counter() - start

        elapsed = time.perf_counter() - start
        stats = {
            "image_id": image_id,
            "journal_len": len(bytes.fromhex(journal_hex)) if journal_hex else 0,
            "binding_verified": True,
            "sdk_verified": _HAVE_RISC0_SDK and bool(receipt_data),
        }

        return True, stats, elapsed
