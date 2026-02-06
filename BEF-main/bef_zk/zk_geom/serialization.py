"""Serialization helpers for GeomProof objects."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from ..air.geom_air import GeomAIRParams
from ..fri.config import FRIConfig
from ..fri.types import (
    FRIProof,
    FRILayerInfo,
    FRILayerBatch,
)
from bef_zk.codec import ENCODING_ID, canonical_encode, canonical_decode, FieldElement
from bef_zk.pc.stc_fri_pc import PCCommitment
from ..stc.vc import (
    VCCommitment,
    VCOpenProof,
    VCBatchProof,
    VCBatchEntry,
    ChunkLeafProof,
)
from ..stc.pc_backend import STCChunkProof, CHUNK_TREE_ARITY
from bef_zk.stc.merkle import MerkleMultiProof
from ..stc.aok_cpu import MODULUS, ROOT_SEED
from .types import GeomProof, GeomStatement
from .backend import RowCommitment, RowOpening


def _bytes_or_hex(value: bytes, binary: bool) -> bytes | str:
    return value if binary else value.hex()


def _bytes_from_rep(value: bytes | str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return bytes.fromhex(value)


FIELD_BITS = 64


def _fp(value: int, binary: bool) -> FieldElement | int:
    return FieldElement(int(value), FIELD_BITS) if binary else int(value)


def _fp_list(values: List[int], binary: bool) -> List[FieldElement | int]:
    return [FieldElement(int(v), FIELD_BITS) if binary else int(v) for v in values]


def _parse_fp(value: Any, binary: bool) -> int:
    if binary and isinstance(value, (bytes, bytearray, memoryview)):
        return int.from_bytes(value, "little")
    return int(value)


def _parse_fp_list(values: List[Any], binary: bool) -> List[int]:
    return [_parse_fp(v, binary) for v in values]


def _vc_commit_to_obj(commitment: VCCommitment, binary: bool) -> Dict[str, Any]:
    return {
        "root": _bytes_or_hex(commitment.root, binary),
        "length": commitment.length,
        "chunk_len": commitment.chunk_len,
        "num_chunks": commitment.num_chunks,
        "challenges": _fp_list(commitment.challenges, binary),
        "sketches": _fp_list([int(x) % MODULUS for x in commitment.sketches], binary),
        "powers": _fp_list([int(x) % MODULUS for x in commitment.powers], binary),
        "chain_root": _bytes_or_hex(commitment.chain_root, binary),
        "chunk_tree_arity": getattr(commitment, "chunk_tree_arity", CHUNK_TREE_ARITY),
    }


def _vc_commit_from_obj(data: Dict[str, Any], binary: bool) -> VCCommitment:
    root_raw = data["root"]
    chain_raw = data.get("chain_root", ROOT_SEED.hex())
    return VCCommitment(
        root=_bytes_from_rep(root_raw),
        length=int(data["length"]),
        chunk_len=int(data["chunk_len"]),
        num_chunks=int(data["num_chunks"]),
        challenges=_parse_fp_list(list(data.get("challenges", [])), binary),
        sketches=[_parse_fp(x, binary) % MODULUS for x in data.get("sketches", [])],
        powers=[_parse_fp(x, binary) % MODULUS for x in data.get("powers", [])],
        chain_root=_bytes_from_rep(chain_raw),
        chunk_tree_arity=int(data.get("chunk_tree_arity", CHUNK_TREE_ARITY)),
    )


def _vc_open_to_obj(proof: VCOpenProof, binary: bool) -> Dict[str, Any]:
    return {
        "index": proof.index,
        "value": _fp(proof.value, binary),
        "aux": proof.aux,
    }


def _vc_open_from_obj(data: Dict[str, Any], binary: bool) -> VCOpenProof:
    return VCOpenProof(
        index=int(data["index"]),
        value=_parse_fp(data["value"], binary),
        aux=dict(data.get("aux", {})),
    )


def _merkle_proof_to_obj(proof: MerkleMultiProof, binary: bool) -> Dict[str, Any]:
    return {
        "tree_size": proof.tree_size,
        "arity": getattr(proof, "arity", 2),
        "sibling_levels": [
            [_bytes_or_hex(hx, binary) for hx in level]
            for level in proof.sibling_levels
        ],
    }


def _merkle_proof_from_obj(data: Dict[str, Any]) -> MerkleMultiProof:
    return MerkleMultiProof(
        tree_size=int(data.get("tree_size", 0)),
        arity=int(data.get("arity", 2)),
        sibling_levels=[
            [_bytes_from_rep(x) for x in level]
            for level in data.get("sibling_levels", [])
        ],
    )


def _chunk_leaf_proof_to_obj(proof: ChunkLeafProof, binary: bool) -> Dict[str, Any]:
    return {
        "chunk_index": proof.chunk_index,
        "chunk_offset": proof.chunk_offset,
        "leaf_positions": list(proof.leaf_positions),
        "proof": _merkle_proof_to_obj(proof.proof, binary),
    }


def _chunk_leaf_proof_from_obj(data: Dict[str, Any]) -> ChunkLeafProof:
    return ChunkLeafProof(
        chunk_index=int(data["chunk_index"]),
        chunk_offset=int(data.get("chunk_offset", 0)),
        leaf_positions=[int(x) for x in data.get("leaf_positions", [])],
        proof=_merkle_proof_from_obj(data.get("proof", {})),
    )


def _vc_batch_to_obj(batch: VCBatchProof, binary: bool) -> Dict[str, Any]:
    return {
        "entries": [
            {
                "index": entry.index,
                "value": _fp(entry.value, binary),
                "chunk_index": entry.chunk_index,
                "chunk_offset": entry.chunk_offset,
                "leaf_pos": entry.leaf_pos,
                **(
                    {
                        "leaf_path": [
                            _bytes_or_hex(hx, binary) for hx in entry.leaf_path or []
                        ]
                    }
                    if entry.leaf_path
                    else {}
                ),
            }
            for entry in batch.entries
        ],
        "chunk_positions": list(batch.chunk_positions),
        "chunk_roots": [_bytes_or_hex(root, binary) for root in batch.chunk_roots],
        "chunk_proof": _merkle_proof_to_obj(batch.chunk_proof, binary),
        "chunk_leaf_proofs": [
            _chunk_leaf_proof_to_obj(clp, binary) for clp in batch.chunk_leaf_proofs
        ],
    }


def _vc_batch_from_obj(data: Dict[str, Any], binary: bool) -> VCBatchProof:
    entries = [
        VCBatchEntry(
            index=int(entry["index"]),
            value=_parse_fp(entry["value"], binary),
            chunk_index=int(entry["chunk_index"]),
            chunk_offset=int(entry.get("chunk_offset", 0)),
            leaf_pos=int(entry["leaf_pos"]),
            leaf_path=[_bytes_from_rep(x) for x in entry.get("leaf_path", [])]
            if entry.get("leaf_path")
            else None,
        )
        for entry in data.get("entries", [])
    ]
    chunk_positions = [int(x) for x in data.get("chunk_positions", [])]
    chunk_roots = [_bytes_from_rep(x) for x in data.get("chunk_roots", [])]
    chunk_proof = _merkle_proof_from_obj(data.get("chunk_proof", {}))
    chunk_leaf_data = data.get("chunk_leaf_proofs") or []
    chunk_leaf_proofs = [
        _chunk_leaf_proof_from_obj(item) for item in chunk_leaf_data
    ]
    return VCBatchProof(
        entries=entries,
        chunk_positions=chunk_positions,
        chunk_roots=chunk_roots,
        chunk_proof=chunk_proof,
        chunk_leaf_proofs=chunk_leaf_proofs,
    )


def _pc_commit_to_obj(commitment: PCCommitment, binary: bool) -> Dict[str, Any]:
    fri_cfg = commitment.fri_params
    return {
        "fri_params": {
            "field_modulus": fri_cfg.field_modulus,
            "domain_size": fri_cfg.domain_size,
            "max_degree": fri_cfg.max_degree,
            "num_rounds": fri_cfg.num_rounds,
            "num_queries": fri_cfg.num_queries,
        },
        "base_commitment": _vc_commit_to_obj(commitment.base_commitment, binary),
    }


def _pc_commit_from_obj(data: Dict[str, Any], binary: bool) -> PCCommitment:
    fri_cfg_data = data["fri_params"]
    fri_cfg = FRIConfig(
        field_modulus=int(fri_cfg_data["field_modulus"]),
        domain_size=int(fri_cfg_data["domain_size"]),
        max_degree=int(fri_cfg_data["max_degree"]),
        num_rounds=int(fri_cfg_data["num_rounds"]),
        num_queries=int(fri_cfg_data["num_queries"]),
    )
    base_commit = _vc_commit_from_obj(data["base_commitment"], binary)
    return PCCommitment(fri_params=fri_cfg, base_commitment=base_commit)


def _fri_proof_to_obj(proof: FRIProof, binary: bool) -> Dict[str, Any]:
    return {
        "layers": [
            {
                "commitment": _vc_commit_to_obj(layer.commitment, binary),
                "beta": _fp(layer.beta, binary),
                "length": layer.length,
            }
            for layer in proof.layers
        ],
        "batches": [
            {
                "layer_index": batch.layer_index,
                "proof": _vc_batch_to_obj(batch.proof, binary),
            }
            for batch in proof.batches
        ],
    }


def _fri_proof_from_obj(data: Dict[str, Any], binary: bool) -> FRIProof:
    layers = [
        FRILayerInfo(
            commitment=_vc_commit_from_obj(layer["commitment"], binary=binary),
            beta=_parse_fp(layer["beta"], binary),
            length=int(layer["length"]),
        )
        for layer in data.get("layers", [])
    ]
    batches = [
        FRILayerBatch(
            layer_index=int(batch["layer_index"]),
            proof=_vc_batch_from_obj(batch.get("proof", {}), binary),
        )
        for batch in data.get("batches", [])
    ]
    return FRIProof(layers=layers, batches=batches)


def _statement_to_obj(statement: GeomStatement, binary: bool) -> Dict[str, Any]:
    params = statement.params
    return {
        "params": {
            "steps": params.steps,
            "matrix": params.matrix,
            "r_challenges": _fp_list(params.r_challenges, binary),
            "seed": _bytes_or_hex(params.seed, binary),
        },
        "final_m11": _fp(statement.final_m11, binary),
        "final_m12": _fp(statement.final_m12, binary),
        "final_m22": _fp(statement.final_m22, binary),
        "final_cnt": statement.final_cnt,
    }


def _statement_from_obj(data: Dict[str, Any], binary: bool) -> GeomStatement:
    params_data = data["params"]
    params = GeomAIRParams(
        steps=int(params_data["steps"]),
        num_challenges=len(params_data["r_challenges"]),
        r_challenges=[_parse_fp(x, binary) for x in params_data["r_challenges"]],
        matrix=[[int(v) for v in row] for row in params_data["matrix"]],
        seed=_bytes_from_rep(params_data.get("seed", "")),
    )
    return GeomStatement(
        params=params,
        final_m11=_parse_fp(data["final_m11"], binary),
        final_m12=_parse_fp(data["final_m12"], binary),
        final_m22=_parse_fp(data["final_m22"], binary),
        final_cnt=int(data["final_cnt"]),
    )


def _row_commit_to_dict(commitment: RowCommitment | None) -> Dict[str, Any] | None:
    if commitment is None:
        return None
    return {
        "backend": commitment.backend,
        "row_width": commitment.row_width,
        "params": commitment.params,
    }


def _row_commit_from_dict(data: Dict[str, Any]) -> RowCommitment:
    if "backend" in data:
        return RowCommitment(
            backend=data["backend"],
            row_width=int(data["row_width"]),
            params=dict(data.get("params", {})),
            prover_state=None,
        )
    # Backwards compatibility with older transcripts that stored VC commitment fields directly.
    params = {
        "root": data["root"],
        "length": int(data["length"]),
        "chunk_len": int(data["chunk_len"]),
        "num_chunks": int(data["num_chunks"]),
        "chunk_tree_arity": int(data.get("chunk_tree_arity", CHUNK_TREE_ARITY)),
    }
    return RowCommitment(
        backend="geom_stc_fri",
        row_width=int(data.get("chunk_len", 0) or params["chunk_len"]),
        params=params,
        prover_state=None,
    )


def _row_opening_to_dict(opening: RowOpening, binary: bool) -> Dict[str, Any]:
    return {
        "backend": opening.backend,
        "row_index": opening.row_index,
        "row_values": _fp_list(opening.row_values, binary),
        "proof": opening.proof,
        "next_index": opening.next_index,
        "next_row_values": _fp_list(opening.next_row_values, binary)
        if opening.next_row_values is not None
        else None,
        "next_proof": opening.next_proof,
    }


def _row_opening_from_dict(data: Dict[str, Any], binary: bool) -> RowOpening:
    if "row_values" in data:
        next_vals = data.get("next_row_values")
        return RowOpening(
            backend=data.get("backend", "geom_stc_fri"),
            row_index=int(data["row_index"]),
            row_values=_parse_fp_list(data.get("row_values", []), binary),
            proof=dict(data.get("proof", {})),
            next_index=(int(data["next_index"]) if data.get("next_index") is not None else None),
            next_row_values=_parse_fp_list(next_vals, binary) if next_vals is not None else None,
            next_proof=dict(data.get("next_proof", {})) if data.get("next_proof") is not None else None,
        )

    # Legacy schema stored STC chunk proofs directly. Convert them into the new format.
    chunk = STCChunkProof.from_json(data["chunk"])
    next_chunk_data = data.get("next_chunk")
    next_chunk = STCChunkProof.from_json(next_chunk_data) if next_chunk_data else None
    next_idx = data.get("next_index")
    return RowOpening(
        backend="geom_stc_fri",
        row_index=int(data["row_index"]),
        row_values=[int(v) for v in chunk.values],
        proof=chunk.to_json(),
        next_index=(int(next_idx) if next_idx is not None else None),
        next_row_values=[int(v) for v in next_chunk.values] if next_chunk else None,
        next_proof=next_chunk.to_json() if next_chunk else None,
    )


def proof_to_obj(proof: GeomProof, binary: bool = False) -> Dict[str, Any]:
    return {
        "statement": _statement_to_obj(proof.statement, binary),
        "pc_commitment": _pc_commit_to_obj(proof.pc_commitment, binary),
        "row_commitment": _row_commit_to_dict(proof.row_commitment),
        "query_indices": list(proof.query_indices),
        "fri_proof": _fri_proof_to_obj(proof.fri_proof, binary),
        "alpha_digest": _bytes_or_hex(proof.alpha_digest, binary),
        "mask_digest": _bytes_or_hex(proof.mask_digest, binary),
        "row_openings": [_row_opening_to_dict(op, binary) for op in proof.row_openings],
    }


def proof_from_obj(data: Dict[str, Any], binary: bool = False) -> GeomProof:
    statement = _statement_from_obj(data["statement"], binary)
    pc_commitment = _pc_commit_from_obj(data["pc_commitment"], binary)
    row_commit_dict = data.get("row_commitment")
    row_commitment = _row_commit_from_dict(row_commit_dict) if row_commit_dict else None
    fri_proof = _fri_proof_from_obj(data["fri_proof"], binary)
    row_openings = [_row_opening_from_dict(op, binary) for op in data.get("row_openings", [])]
    alpha = data["alpha_digest"]
    mask = data["mask_digest"]
    return GeomProof(
        statement=statement,
        pc_commitment=pc_commitment,
        query_indices=[int(i) for i in data.get("query_indices", [])],
        fri_proof=fri_proof,
        alpha_digest=_bytes_from_rep(alpha),
        mask_digest=_bytes_from_rep(mask),
        row_commitment=row_commitment,
        row_openings=row_openings,
    )


def proof_to_dict(proof: GeomProof) -> Dict[str, Any]:
    return proof_to_obj(proof, binary=False)


def proof_from_dict(data: Dict[str, Any]) -> GeomProof:
    return proof_from_obj(data, binary=False)


def proof_to_json(proof: GeomProof) -> str:
    return json.dumps(proof_to_dict(proof), indent=2)


def proof_from_json(data: str) -> GeomProof:
    return proof_from_dict(json.loads(data))


def proof_to_bytes(proof: GeomProof, encoding_id: str | None = None) -> bytes:
    enc = encoding_id or ENCODING_ID
    return canonical_encode(proof_to_obj(proof, binary=True), encoding_id=enc)


def proof_from_bytes(blob: bytes) -> GeomProof:
    data = canonical_decode(blob)
    if not isinstance(data, dict):
        raise TypeError("decoded proof payload must be an object")
    return proof_from_obj(data, binary=True)
