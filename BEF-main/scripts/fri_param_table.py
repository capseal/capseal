#!/usr/bin/env python3
"""Print a small table of FRI soundness estimates."""
from __future__ import annotations

import argparse
from typing import Iterable, List


def estimate_error(domain: int, degree: int, queries: int) -> float:
    if domain <= 0 or degree < 0 or queries <= 0:
        return 1.0
    ratio = degree / domain
    if ratio >= 1:
        return 1.0
    return ratio**queries


def format_row(domain: int, degree: int, queries: int) -> str:
    error = estimate_error(domain, degree, queries)
    return f"N={domain:>10}  d_max={degree:>10}  q={queries:>3}  error≈{error:.2e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="FRI parameter helper")
    parser.add_argument("--domain", type=int, nargs="*", default=[1 << 20], help="domain sizes (default: 2^20)")
    parser.add_argument("--degree", type=int, nargs="*", default=[1 << 18], help="degree bounds (default: 2^18)")
    parser.add_argument("--queries", type=int, nargs="*", default=[32], help="query counts (default: 32)")
    args = parser.parse_args()

    print("FRI parameter grid (error ≈ (degree/domain)^q)")
    for domain in args.domain:
        for degree in args.degree:
            for q in args.queries:
                print(format_row(domain, degree, q))


if __name__ == "__main__":
    main()
