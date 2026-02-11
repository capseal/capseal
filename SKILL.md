---
name: capseal
description: >
  Learned risk gating and cryptographic verification for code changes.
  Learns which patches fail on YOUR codebase, blocks risky changes before
  they execute, and seals every action into a tamper-proof .cap receipt.
  Use before any code modification, refactor, or security fix.
emoji: 🔐
metadata:
  openclaw:
    requires:
      bins: ["capseal", "semgrep"]
    install:
      - pip install semgrep
---

# CapSeal — Learned Risk Gating for Code Changes

## What this does

CapSeal learns which types of code changes fail on a specific codebase by
running patches over multiple rounds and tracking success/failure rates across
a 5-dimensional feature grid. It then uses that learned model to gate future
changes — blocking patches with high predicted failure probability and
approving safe ones. Every decision is sealed into a cryptographic .cap receipt.

This is NOT static rule matching. CapSeal builds a statistical model of what
works and what breaks on THIS codebase specifically.

## Deep Integration via MCP

For maximum control, CapSeal also exposes an MCP server that any agent framework
can call directly. The OpenClaw skill is a wrapper; MCP is the universal path.

```json
{
  "capseal": {
    "command": "capseal",
    "args": ["mcp-serve"],
    "transport": "stdio"
  }
}
```

MCP tools exposed by the server:
- `capseal_gate` — gate a proposed action before execution
- `capseal_record` — record the result of an executed action
- `capseal_seal` — seal the session into a `.cap` receipt
- `capseal_status` — recover current session/project state
- `capseal_context` — file-level history across sessions

Works with Claude Code, Cursor, LangChain, and any MCP-compatible client.

## CRITICAL RULES

1. **NEVER make code changes without running capseal first.** Before editing
   any source file, modifying code, applying patches, or running refactors,
   you MUST use capseal to gate the change.
2. **If capseal gates a finding (marks it high-risk), DO NOT attempt to fix it
   manually or work around the gate.** Tell the user it was flagged as
   high-risk, explain the predicted failure rate, and ask if they want to
   proceed anyway.
3. **Always report verification status** after making changes.
4. **If no learned model exists yet**, tell the user and run `capseal learn`
   first. Do not skip this step.

## First-time setup (once per project)

When working on a codebase for the first time, or when the user says
"set up capseal" or "make this project safe":

```bash
cd /path/to/project
capseal init
capseal learn . --rounds 5 --budget 5
```

This costs ~$0.50-2.00 and takes 1-2 minutes. It builds a risk model by
running AI-generated patches and tracking which succeed and fail. The model
is saved to `.capseal/models/beta_posteriors.npz`.

Tell the user what was learned: which finding types are high-risk, which are
low-risk, and the overall success rate.

## Before making ANY code change

Before writing, editing, or patching any source file, run:

```bash
capseal fix . --dry-run
```

This shows what capseal would fix and what it would skip. Review the output:
- **Approved** findings: safe to fix, the model predicts these patches succeed
- **Gated** findings: high predicted failure rate, do NOT attempt these
- **Flagged** findings: uncertain, ask the user before proceeding

Report the dry-run results to the user before proceeding.

## Generating verified patches

When the user confirms they want to proceed with the approved changes:

```bash
capseal fix .
```

This generates patches ONLY for approved findings, skipping gated ones. The
entire run is sealed into a `.cap` receipt file.

After completion, report:
- How many findings were fixed vs skipped
- The .cap receipt file location
- Verification status

## Verifying past runs

When the user asks "what did you change" or "prove what happened" or wants
to audit past changes:

```bash
capseal verify .capseal/runs/latest.cap
capseal report .capseal/runs/latest
```

## Lightweight CI check (no patches, just risk assessment)

When the user wants a quick risk scan without generating patches:

```bash
capseal review . --gate
```

## Quick reference

| Command | What it does | When to use |
|---------|-------------|-------------|
| `capseal init` | Set up workspace | First time per project |
| `capseal learn . --rounds 5` | Build risk model (~$1) | First time, or after major changes |
| `capseal scan .` | Find issues (no patches) | Quick scan |
| `capseal review . --gate` | Risk assessment only | CI checks, quick triage |
| `capseal fix . --dry-run` | Preview gated patches | Before any code change |
| `capseal fix .` | Generate verified patches | When user confirms changes |
| `capseal verify <file>.cap` | Verify receipt chain | Auditing, proving what happened |
| `capseal report .capseal/runs/latest` | Human-readable summary | After any run |

## Example conversation flow

User: "Fix the security issues in my project"

1. Check if `.capseal/models/beta_posteriors.npz` exists
   - If not: run `capseal init` then `capseal learn . --rounds 5 --budget 5`
2. Run `capseal fix . --dry-run`
3. Tell user: "Found 8 issues. I can safely fix 3 (low predicted failure).
   5 are gated as high-risk (71-86% predicted failure rate). Want me to
   proceed with the 3 safe fixes?"
4. On confirmation: run `capseal fix .`
5. Report: "Fixed 3 issues. Run sealed as .capseal/runs/20260206-fix.cap.
   Receipt verified. 5 high-risk issues left for manual review."
