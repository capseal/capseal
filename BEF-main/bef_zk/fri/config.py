"""FRI configuration."""
from __future__ import annotations

from dataclasses import dataclass


MODULUS = (1 << 61) - 1


@dataclass
class FRIConfig:
    field_modulus: int = MODULUS
    domain_size: int = 0
    max_degree: int = 0
    num_rounds: int = 0
    num_queries: int = 0

    def __post_init__(self) -> None:
        if self.domain_size <= 0 or (self.domain_size & (self.domain_size - 1)) != 0:
            raise ValueError("FRI domain_size must be a power of two > 0")
        if self.num_rounds < 0 or self.num_rounds > self.domain_size.bit_length():
            raise ValueError("invalid num_rounds for given domain size")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")
