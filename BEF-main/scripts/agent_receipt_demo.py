#!/usr/bin/env python3
"""
Agent Receipt Demo - Hash-chained proof of tool invocations

This demonstrates the "interactive proof" where each agent tool call
generates a receipt that is hash-chained to the previous one.

The receipt chain is tamper-evident: modifying any receipt breaks the chain.

Usage:
    python scripts/agent_receipt_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/ryan/BEF-main")

# Import the MCP server tools directly
from bef_zk.capsule.mcp_server import (
    tool_verify,
    tool_doctor,
    tool_audit,
    tool_diff_bundle,
    EVENT_LOG_PATH,
    _log_event,
)


def clear_event_log():
    """Clear the event log for fresh demo."""
    if os.path.exists(EVENT_LOG_PATH):
        os.remove(EVENT_LOG_PATH)
    # Reset the global hash
    import bef_zk.capsule.mcp_server as mcp
    mcp._last_hash = None


def print_banner():
    print("=" * 70)
    print("  Agent Receipt Demo - Hash-Chained Tool Invocation Proofs")
    print("=" * 70)
    print()
    print("  Each tool call generates a receipt with:")
    print("    - Timestamp")
    print("    - Tool name + arguments")
    print("    - Result status")
    print("    - Hash of previous receipt (chain link)")
    print("    - Hash of this receipt")
    print()
    print("  The chain is tamper-evident: modify any receipt → chain breaks")
    print()


def print_receipt(event: dict, idx: int):
    """Pretty print a receipt."""
    ts = event.get("ts_ms", 0)
    ts_str = time.strftime("%H:%M:%S", time.localtime(ts / 1000))
    tool = event.get("tool", "?")
    ok = "✓" if event.get("result_ok") else "✗"
    dur = event.get("duration_ms", 0)
    prev = event.get("prev_hash", "")[:16] or "(genesis)"
    curr = event.get("event_hash", "")[:16]

    print(f"  ┌─ Receipt #{idx}")
    print(f"  │  Time:      {ts_str}")
    print(f"  │  Tool:      {tool}")
    print(f"  │  Status:    {ok} ({dur}ms)")
    print(f"  │  Args:      {json.dumps(event.get('args', {}))[:50]}...")
    print(f"  │  Prev hash: {prev}")
    print(f"  │  This hash: {curr}")
    print(f"  └────────────────────────────────────────")


def verify_chain(events: list[dict]) -> tuple[bool, str]:
    """Verify the hash chain is intact."""
    import hashlib

    prev_hash = ""
    for i, event in enumerate(events):
        # Verify prev_hash matches
        if event.get("prev_hash", "") != prev_hash:
            return False, f"Chain broken at receipt #{i}: prev_hash mismatch"

        # Recompute hash
        event_copy = {k: v for k, v in event.items() if k != "event_hash"}
        event_bytes = json.dumps(event_copy, sort_keys=True, ensure_ascii=False).encode()
        computed_hash = hashlib.sha256(event_bytes).hexdigest()[:32]

        if computed_hash != event.get("event_hash"):
            return False, f"Chain broken at receipt #{i}: hash mismatch"

        prev_hash = computed_hash

    return True, "Chain verified"


def main():
    print_banner()

    # Clear previous log
    clear_event_log()
    print("[Setup] Cleared event log for fresh demo")
    print()

    # Find a capsule to work with
    capsule_path = "/home/ryan/BEF-main/fixtures/golden_run_latest/capsule/strategy_capsule.json"
    if not os.path.exists(capsule_path):
        capsule_path = "/home/ryan/BEF-main/fixtures/golden_run/capsule/capsule.json"

    print("=" * 70)
    print("  PHASE 1: Agent makes tool calls (generates receipts)")
    print("=" * 70)
    print()

    # Simulate agent making several tool calls
    print(f"[Agent] Calling diff_bundle on BEF-main repo...")
    result1 = tool_diff_bundle(
        repo_path="/home/ryan/BEF-main",
        base_ref="HEAD~5",
        head_ref="HEAD",
    )
    print(f"  → {result1.get('file_count', '?')} files changed")
    print()

    print(f"[Agent] Calling verify on capsule...")
    result2 = tool_verify(capsule_path=capsule_path)
    status = "OK" if result2.get("ok") else "FAIL"
    print(f"  → Verification: {status}")
    print()

    print(f"[Agent] Calling doctor for full diagnosis...")
    result3 = tool_doctor(capsule_path=capsule_path, sample_rows=0)
    status = "OK" if result3.get("ok") else "FAIL"
    print(f"  → Doctor: {status}")
    print()

    print(f"[Agent] Calling audit for trail inspection...")
    result4 = tool_audit(capsule_path=capsule_path)
    status = "OK" if result4.get("ok") else "FAIL"
    print(f"  → Audit: {status}")
    print()

    # Load and display the receipt chain
    print("=" * 70)
    print("  PHASE 2: Examine the receipt chain")
    print("=" * 70)
    print()

    events = []
    with open(EVENT_LOG_PATH, "r") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    print(f"Generated {len(events)} receipts:\n")
    for i, event in enumerate(events):
        print_receipt(event, i)
        print()

    # Verify the chain
    print("=" * 70)
    print("  PHASE 3: Verify chain integrity")
    print("=" * 70)
    print()

    valid, msg = verify_chain(events)
    if valid:
        print(f"  ✓ {msg}")
        print()
        print("  The receipt chain proves:")
        print("    1. These exact tools were called in this order")
        print("    2. With these exact arguments")
        print("    3. Producing these results")
        print("    4. No receipts were modified or reordered")
    else:
        print(f"  ✗ {msg}")
    print()

    # Show tamper detection
    print("=" * 70)
    print("  PHASE 4: Tamper detection demo")
    print("=" * 70)
    print()

    # Tamper with a receipt
    tampered_events = [dict(e) for e in events]
    if len(tampered_events) >= 2:
        print("  Simulating tampering: modifying result_ok in receipt #1...")
        tampered_events[1]["result_ok"] = not tampered_events[1].get("result_ok", False)

        valid, msg = verify_chain(tampered_events)
        print(f"  Verification after tampering: {'✓' if valid else '✗'} {msg}")
        print()

    print("=" * 70)
    print("  Receipt chain is the INTERACTIVE PROOF")
    print("  - Verifier can check any subset of receipts")
    print("  - Hash chain ensures ordering and integrity")
    print("  - Each receipt links tool call → result → cryptographic proof")
    print("=" * 70)
    print()
    print(f"Event log: {EVENT_LOG_PATH}")


if __name__ == "__main__":
    main()
