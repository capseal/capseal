# CapSeal — Every AI Agent Action, Verified

CapSeal is a trust layer for AI coding agents. It learns which code changes fail on your specific codebase, gates risky actions before execution, and seals every action into a cryptographic receipt.

## Quick Start (30 seconds)

```bash
pip install capseal
cd your-project
capseal autopilot .
# Autopilot runs in dry-run mode by default — shows what it WOULD fix.
# Add --apply to write changes. Your code is not modified without explicit consent.
```

That's it. CapSeal scans your code, attempts fixes, gates risky changes, and produces a tamper-evident receipt. The risk model improves with each run — the more you use it, the sharper the gating gets.

**Requirements:** Python 3.10+, an LLM API key (Anthropic, OpenAI, or Google), and [Semgrep](https://semgrep.dev) for scanning. CapSeal uses the LLM to generate patches — no API key means no patch generation (scanning and gating still work). Code stays local. Only minimal patch context (the finding + ~20-60 surrounding lines needed to generate a fix) is sent to your chosen LLM provider for patch generation. No code is sent to CapSeal servers — there are no CapSeal servers.

## Example Run

```
$ capseal autopilot .

[1/4] Initializing workspace...           ✓
[2/4] Learning risk model (3 rounds, each: generate patches → validate → record pass/fail)...
      Round 1: ✓ 2 success  ✗ 1 fail
      Round 2: ✓ 3 success  ✗ 0 fail
      Round 3: ✓ 2 success  ✗ 1 fail
[3/4] Scanning and gating...
      Found 8 issues (semgrep security-audit)
      Approved: 5  Flagged: 2  Gated: 1
      Gated: "non-literal-import refactor" (p_fail=0.86, n=7)
[4/4] Applying 5 approved patches...      ✓

Receipt: .capseal/runs/20260207-autopilot.cap
Actions: 5 applied, 1 blocked, 2 flagged for review
Chain:   intact (5/5 hashes valid)
```

## Protect Your AI Agents (2 minutes)

```bash
pip install capseal
cd your-project
capseal init
```

Select your agents → CapSeal auto-configures → every future session is gated and sealed.

Supported agents:
- Claude Code
- OpenClaw
- Cursor
- Windsurf
- Cline
- Any MCP-compatible client (MCP is the Model Context Protocol — it lets AI agents call CapSeal's gate/record/seal tools during a coding session)

## How It Works

1. **Learn** — CapSeal generates candidate patches for each finding, applies them, then runs validation (semgrep re-scan, syntax checks, and optionally your test suite via `capseal learn --test-cmd "pytest"`). Patches that pass validation are successes; patches that break things are failures. Over multiple rounds, the model builds a statistical profile of which change patterns succeed or fail on your specific codebase.

2. **Gate** — Before any AI agent makes a change, CapSeal checks the learned model. High predicted failure? Blocked. Uncertain? Flagged for review. Safe? Approved.

3. **Seal** — Every action (gate decisions, edits, verifications) is hash-chained into a `.cap` receipt file. Tamper with any action and the chain breaks.

4. **Verify** — Anyone can verify a `.cap` file: `capseal verify .capseal/runs/latest.cap`

## What CapSeal Guarantees / What It Doesn't

**Guarantees:**
- Every AI action is recorded in a tamper-evident hash chain
- If anyone modifies a receipt after the fact, verification fails
- The risk model is trained on YOUR codebase, not generic rules
- Gated changes are blocked before execution, not flagged after

**Does not guarantee:**
- That approved changes are bug-free (gating reduces risk, doesn't eliminate it)
- That the risk model is perfect on the first run (it improves with more data)
- Signer identity (receipts prove what happened, not who did it — signing is on the roadmap)

## Four Ways to Use CapSeal

| Mode | Command | Who it's for |
|------|---------|-------------|
| Autopilot | `capseal autopilot .` | Developers who want one-command guardrails |
| Step by step | `init → learn → fix → verify` | Developers who want full control |
| Agent wrapper | `capseal init` (select agents) | Anyone using AI coding agents |
| CI/CD | `capseal autopilot . --ci` | Automated pipelines |

## Agent Wrapper Flow

```
capseal init                          # Pick your agents
   ↓
Open Claude Code / Cursor / etc.      # Start any coding session
   ↓
Agent calls capseal_status            # Checks session state
Agent calls capseal_gate              # Before every change
Agent calls capseal_record            # After every change
Agent calls capseal_seal              # End of session
   ↓
capseal verify .capseal/runs/latest.cap   # Verify anytime
```

## Why Not Just Static Rules?

Every other security tool uses pattern matching: "block rm -rf", "flag eval()".

CapSeal tracks change patterns — import refactors, dependency updates, security fixes, formatting changes, test edits — and measures their success rate against your validation pipeline. After 3 rounds on a FastAPI project, it learned that import-reorganization patches fail 86% of the time (circular dependency issues), while security-header additions succeed 93% of the time.

## The Cryptographic Receipt

Every CapSeal session produces a `.cap` file:

```
capsule_hash: a287f7e44a75...
actions: 6
chain: intact (6/6 hashes valid)
validations_passed: true
```

Validation checks confirm that patches passed semgrep re-scan, syntax verification, and any configured test commands before being applied.

Each `.cap` file contains a manifest (session metadata, timestamps, action count), a hash chain (each action's SHA-256 chains to the previous), and validation checks. `capseal verify` recomputes every hash in the chain — if any action was modified, added, or removed after sealing, verification fails. Receipts are tamper-evident but not identity-signed (signing support is planned).

## CLI Reference

| Command | Description |
|---------|-------------|
| `capseal autopilot .` | Full pipeline, zero config |
| `capseal init` | Interactive setup (pick agents, provider, model) |
| `capseal scan .` | Find issues with Semgrep |
| `capseal learn .` | Build risk model from patch outcomes |
| `capseal fix .` | Generate verified patches, gated by learned model |
| `capseal fix . --dry-run` | Preview without generating |
| `capseal fix . --apply` | Generate and apply patches |
| `capseal verify <file.cap>` | Verify sealed receipt |
| `capseal doctor` | Check everything is wired up correctly |
| `capseal mcp-serve` | Start MCP server for agent integration |
