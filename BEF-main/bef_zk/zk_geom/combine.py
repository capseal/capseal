"""Combine AIR constraints with random scalars."""
from __future__ import annotations

from typing import Dict, List


def combine_constraints(
    constraints: Dict[str, List[int]],
    challenges: Dict[str, int],
    modulus: int,
) -> List[int]:
    if not constraints:
        return []
    names = list(constraints.keys())
    n = len(constraints[names[0]])
    combined = [0] * n
    for name in names:
        alpha = challenges[name]
        vals = constraints[name]
        for i in range(n):
            combined[i] = (combined[i] + alpha * vals[i]) % modulus
    return combined

