#!/usr/bin/env python3
"""CapSeal CLI - Predictive gating + cryptographic verification for AI agents.

This is the single entry point for the capseal command. It sets up the Python
path to include both BEF-main/ and otherstuff/ so all existing imports work
unchanged.
"""
from __future__ import annotations

import os
import sys

# Add both subdirectories to path so existing imports work unchanged
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BEF-main"))
sys.path.insert(0, os.path.join(ROOT, "otherstuff"))


def main():
    """Entry point for capseal CLI."""
    from bef_zk.capsule.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
