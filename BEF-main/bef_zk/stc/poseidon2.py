"""Tiny Poseidon2 hash (ported from the Prospectors implementation)."""
from __future__ import annotations

from typing import Sequence

MODULUS = (1 << 61) - 1


def _ark(state: list[int], round_const: Sequence[int]) -> None:
    for i in range(len(state)):
        state[i] = (state[i] + round_const[i]) % MODULUS


def _sbox(state: list[int]) -> None:
    for i in range(len(state)):
        x = state[i]
        x2 = (x * x) % MODULUS
        x4 = (x2 * x2) % MODULUS
        x5 = (x4 * x) % MODULUS
        state[i] = x5


def _mix(state: list[int], mds: Sequence[Sequence[int]]) -> None:
    new = [0] * len(state)
    for i in range(len(state)):
        acc = 0
        for j in range(len(state)):
            acc = (acc + mds[i][j] * state[j]) % MODULUS
        new[i] = acc
    state[:] = new


ROUND_CONST = [1, 2, 3]
MDS = [
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
]


def poseidon2_hash(words: Sequence[int]) -> int:
    if len(words) != 3:
        raise ValueError("Poseidon2-3 expects 3 inputs")
    state = [int(w) % MODULUS for w in words]
    for _ in range(7):
        _ark(state, ROUND_CONST)
        _sbox(state)
        _mix(state, MDS)
    return state[0]


__all__ = ["poseidon2_hash", "MODULUS"]
