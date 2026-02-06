"""Shared statement/proof types for geometry zk demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..fri.types import FRIProof
from bef_zk.pc.stc_fri_pc import PCCommitment
from ..air.geom_air import GeomAIRParams
from .backend import RowCommitment, RowOpening


@dataclass
class GeomStatement:
    params: GeomAIRParams
    final_m11: int
    final_m12: int
    final_m22: int
    final_cnt: int

@dataclass
class GeomProof:
    statement: GeomStatement
    pc_commitment: PCCommitment
    query_indices: List[int]
    fri_proof: FRIProof
    alpha_digest: bytes
    mask_digest: bytes
    row_commitment: RowCommitment | None = None
    row_openings: List[RowOpening] = field(default_factory=list)
