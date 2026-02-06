"""FRI proof data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bef_zk.stc.vc import VCCommitment, VCBatchProof

MODULUS = (1 << 61) - 1


@dataclass
class FRILayerInfo:
    commitment: VCCommitment
    beta: int
    length: int


@dataclass
class FRILayerBatch:
    layer_index: int
    proof: VCBatchProof


@dataclass
class FRIProof:
    layers: List[FRILayerInfo]
    batches: List[FRILayerBatch]
