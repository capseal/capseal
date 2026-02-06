"""Simple Fiat–Shamir transcript helper."""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional


class Transcript:
    """Deterministic transcript for deriving challenges.

    The transcript is initialized with a label (domain separation tag). Call
    ``absorb_*`` methods to add public data, then ``challenge_field`` or
    ``challenge_bytes`` to derive pseudo-random challenges.
    """

    def __init__(self, label: str, hash_name: str = "sha256") -> None:
        if hash_name not in hashlib.algorithms_guaranteed:
            raise ValueError(f"unsupported hash function: {hash_name}")
        self._hash_name = hash_name
        self._state = hashlib.new(hash_name)
        self._state.update(b"FSv1|" + label.encode("utf-8"))

    def absorb_bytes(self, data: bytes) -> None:
        """Absorb raw bytes (length-prefixed) into the transcript."""
        length = len(data).to_bytes(8, "big", signed=False)
        self._state.update(length)
        self._state.update(data)

    def absorb_int(self, value: int, *, length: Optional[int] = None) -> None:
        """Absorb a non-negative integer."""
        if value < 0:
            raise ValueError("Transcript only supports non-negative integers")
        if length is None:
            byte_len = max(1, (value.bit_length() + 7) // 8)
        else:
            byte_len = length
        self.absorb_bytes(value.to_bytes(byte_len, "big", signed=False))

    def absorb_many_ints(
        self, values: Iterable[int], *, length: Optional[int] = None
    ) -> None:
        for v in values:
            self.absorb_int(v, length=length)

    def challenge_bytes(self, length: int = 32) -> bytes:
        """Derive ``length`` bytes and ratchet the internal state."""
        if length <= 0:
            raise ValueError("length must be positive")
        digest = self._state.digest()
        while len(digest) < length:
            self._state = hashlib.new(self._hash_name)
            self._state.update(b"FSv1|extend|" + digest)
            digest += self._state.digest()
        # ratchet for next squeeze
        self._state = hashlib.new(self._hash_name)
        self._state.update(b"FSv1|ratchet|" + digest)
        return digest[:length]

    def challenge_field(self, modulus: int) -> int:
        """Derive a value in ``[0, modulus)`` (uniform up to low bias)."""
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        digest = self.challenge_bytes(32)
        return int.from_bytes(digest, "big", signed=False) % modulus
