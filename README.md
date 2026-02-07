# CapSeal — Every AI Agent Action, Verified

CapSeal is a trust layer for AI coding agents. It learns which code changes fail on your specific codebase, gates risky actions before execution, and seals every action into a cryptographic receipt.

## Quick Start (30 seconds)

```bash
pip install capseal
cd your-project
capseal autopilot .
```

That's it. CapSeal scans your code, learns what's risky, fixes what's safe, and gives you a verified receipt.

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
- Any MCP-compatible client

## How It Works

1. **Learn** — CapSeal runs your codebase through multiple rounds of AI-generated patches, tracking which succeed and which fail. It builds a statistical model (Beta posteriors) specific to YOUR code.

2. **Gate** — Before any AI agent makes a change, CapSeal checks the learned model. High predicted failure? Blocked. Uncertain? Flagged for review. Safe? Approved.

3. **Seal** — Every action (gate decisions, edits, verifications) is hash-chained into a `.cap` receipt file. Tamper with any action and the chain breaks.

4. **Verify** — Anyone can verify a `.cap` file: `capseal verify .capseal/runs/latest.cap`

## Four Ways to Use CapSeal

| Mode | Command | Who it's for |
|------|---------|-------------|
| Autopilot | `capseal autopilot .` | Developers who want one-command security |
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

CapSeal learns. It knows that non-literal-import patches fail 86% of the time on YOUR codebase. It knows that simple style fixes succeed 95% of the time. It doesn't guess — it measures.

## The Cryptographic Receipt

Every CapSeal session produces a `.cap` file:

```
capsule_hash: a287f7e44a75...
actions: 6
chain: intact (6/6 hashes valid)
constraints_valid: true
```

Each action chains to the previous one. Tamper with one and verification fails.

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

## Requirements

- Python 3.10+
- Semgrep (`pip install semgrep`)
- One of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`
