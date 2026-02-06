"""Domain helpers for FRI (power-of-two)."""
from __future__ import annotations

from typing import List

try:
    from bef_zk import bef_rust
except ImportError:
    bef_rust = None

FieldElem = int


def fold_codeword(values: List[FieldElem], alpha: FieldElem, modulus: int) -> List[FieldElem]:
    if len(values) % 2 != 0:
        raise ValueError("folding expects even-length codeword")
    
    if bef_rust is not None:
        # Rust optimization
        # Note: modulus arg is unused as bef_rust hardcodes (1<<61)-1
        return bef_rust.fold_fri_layer(values, alpha)

    out: List[FieldElem] = []
    for i in range(0, len(values), 2):
        a = values[i]
        b = values[i + 1]
        out.append((a + alpha * b) % modulus)
    return out


def next_index(idx: int) -> int:
    return idx // 2

