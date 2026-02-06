#!/usr/bin/env python3
"""Capsule CLI - portable verification artifacts.

Usage:
    capsule emit --capsule <capsule.json> --out <receipt.cap> [OPTIONS]
    capsule verify <receipt.cap> [--mode proof-only|da|replay] [--json]
    capsule inspect <receipt.cap> [--json]

Exit codes (verify):
    0  - Verified
    10 - Proof invalid
    11 - Policy mismatch
    12 - Commitment/index failed
    13 - DA audit failed
    14 - Replay diverged
    20 - Malformed/parse error
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bef_zk.capsule.cli import main

if __name__ == "__main__":
    main()
