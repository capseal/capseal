#!/usr/bin/env python3
"""Compute STC parameter guidance tables."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List

MODULUS = (1 << 61) - 1
HASH_BITS = 128


@dataclass
class Params:
    n: int
    m: int
    delta: float
    samples: int


def sketch_collision_prob(n: int, m: int) -> float:
    if n <= 1:
        return 0.0
    ratio = (n - 1) / (MODULUS - 1)
    return ratio ** m


def da_failure_prob(n: int, m: int, delta: float, samples: int) -> float:
    return (1 - delta) ** samples + sketch_collision_prob(n, m) + 2 ** (-HASH_BITS)


def fmt_prob(value: float) -> str:
    if value == 0:
        return "0"
    exp = 0
    if value < 1:
        exp = int(round(-value.as_integer_ratio()[0]))
    return f"{value:.2e}"


def compute_table(ns: Iterable[int], ms: Iterable[int], delta: float, samples: int) -> List[str]:
    header = "| n | m | sketch collision | DA failure |\n|---:|---:|----------------:|------------:|"
    rows = [header]
    import math
    for n in ns:
        logn = int(round(math.log2(n))) if n > 0 else 0
        for m in ms:
            rows.append(
                f"| 2^{logn} ({n:,}) | {m} | {sketch_collision_prob(n, m):.3e} | {da_failure_prob(n, m, delta, samples):.3e} |"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="STC parameter table generator")
    parser.add_argument("--ns", default="1048576,16777216,4294967296", help="comma separated n values")
    parser.add_argument("--ms", default="2,4,8", help="comma separated m values")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", type=argparse.FileType("w"), default=None)
    args = parser.parse_args()

    ns = [int(x.strip(), 0) for x in args.ns.split(",") if x.strip()]
    ms = [int(x.strip(), 0) for x in args.ms.split(",") if x.strip()]
    rows = compute_table(ns, ms, args.delta, args.samples)
    content = "\n".join(rows)
    if args.output:
        args.output.write("## STC parameter guidance\n\n" + content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
