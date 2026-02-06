"""Helpers to compute size/composition stats for GeomProof objects."""
from __future__ import annotations

from typing import Dict, Tuple

from bef_zk.fri.types import FRIProof
from bef_zk.stc.vc import VCOpenProof
from bef_zk.stc.pc_backend import STCChunkProof

from .types import GeomProof, RowOpening


def _vc_opening_path_lengths(opening: VCOpenProof) -> Tuple[int, int]:
    aux = opening.aux or {}
    leaf_path = aux.get("leaf_path") or []
    chunk_path = aux.get("chunk_root_path") or []
    return len(leaf_path), len(chunk_path)


def _chunk_proof_stats(proof_dict: Dict) -> Dict[str, int]:
    proof = STCChunkProof.from_json(proof_dict)
    chunk_path = proof.chunk_root_path or []
    return {
        "row_chunk_values": len(proof.values),
        "row_chunk_root_siblings": sum(len(level) for level in chunk_path),
    }


def compute_fri_stats(fri_proof: FRIProof) -> Dict[str, int]:
    stats = {
        "fri_layers": len(fri_proof.layers),
        "fri_batches": len(fri_proof.batches),
        "fri_openings": 0,
        "fri_leaf_siblings": 0,
        "fri_leaf_siblings_baseline": 0,
        "fri_chunk_siblings": 0,
    }
    for batch in fri_proof.batches:
        stats["fri_openings"] += len(batch.proof.entries)
        if batch.proof.chunk_leaf_proofs:
            for clp in batch.proof.chunk_leaf_proofs:
                per_chunk = sum(len(level) for level in clp.proof.sibling_levels)
                stats["fri_leaf_siblings"] += per_chunk
                depth = len(clp.proof.sibling_levels)
                stats["fri_leaf_siblings_baseline"] += depth * len(clp.leaf_positions)
        else:
            for entry in batch.proof.entries:
                length = len(entry.leaf_path or [])
                stats["fri_leaf_siblings"] += length
                stats["fri_leaf_siblings_baseline"] += length
        stats["fri_chunk_siblings"] += sum(len(level) for level in batch.proof.chunk_proof.sibling_levels)

    baseline = stats["fri_leaf_siblings_baseline"]
    actual = stats["fri_leaf_siblings"]
    if baseline > 0:
        saved = max(baseline - actual, 0)
        stats["fri_leaf_bytes_baseline"] = baseline * 32
        stats["fri_leaf_bytes_actual"] = actual * 32
        stats["fri_leaf_bytes_saved"] = saved * 32
        stats["fri_leaf_savings_ratio"] = saved / baseline
    else:
        stats["fri_leaf_bytes_baseline"] = 0
        stats["fri_leaf_bytes_actual"] = actual * 32
        stats["fri_leaf_bytes_saved"] = 0
        stats["fri_leaf_savings_ratio"] = 0.0
    return stats


def compute_row_stats(openings: list[RowOpening]) -> Dict[str, int]:
    totals = {
        "row_openings": len(openings),
        "row_chunk_values": 0,
        "row_chunk_root_siblings": 0,
    }
    for opening in openings:
        stat = _chunk_proof_stats(opening.proof)
        totals["row_chunk_values"] += stat["row_chunk_values"]
        totals["row_chunk_root_siblings"] += stat["row_chunk_root_siblings"]
        if opening.next_index is not None and opening.next_proof is not None:
            stat_next = _chunk_proof_stats(opening.next_proof)
            totals["row_openings"] += 1
            totals["row_chunk_values"] += stat_next["row_chunk_values"]
            totals["row_chunk_root_siblings"] += stat_next["row_chunk_root_siblings"]
    return totals


def compute_proof_stats(proof: GeomProof) -> Dict[str, int]:
    stats = {}
    stats.update(compute_fri_stats(proof.fri_proof))
    if proof.row_commitment is not None:
        stats.update(compute_row_stats(proof.row_openings))
    return stats


__all__ = ["compute_proof_stats"]
