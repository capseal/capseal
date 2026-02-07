# CapSeal — Learned Risk Gating for OpenClaw

Every other security skill uses static rules. CapSeal *learns* which code changes fail on your specific codebase, then gates your agent's patches accordingly. Every decision is sealed into a cryptographic receipt.

## What it does

1. **Learns** — Runs AI-generated patches against your codebase over multiple rounds, tracking which types of changes succeed and fail. Builds a statistical model (Beta posteriors over a 1024-point feature grid).

2. **Gates** — Before your OpenClaw agent modifies any code, CapSeal scores the proposed change against the learned model. High predicted failure? Blocked. Uncertain? Flagged for your review. Safe? Approved.

3. **Proves** — Every learn round, gate decision, and patch is sealed into a `.cap` receipt with a hash chain. You can independently verify that nothing was tampered with.

## Install

```bash
pip install capseal semgrep
```

Or tell your OpenClaw agent:
> "Install the capseal skill from ClawHub"

## Usage

Tell your agent:
> "Set up capseal on this project and fix the security issues"

The agent will:
1. Initialize the workspace (`capseal init`)
2. Learn which patches work on your codebase (`capseal learn . --rounds 5`)
3. Preview what can be safely fixed (`capseal fix . --dry-run`)
4. Generate verified patches for approved findings (`capseal fix .`)
5. Seal the run and report verification status

## Why not just use static rules?

| Approach | How it works | Limitation |
|----------|-------------|------------|
| openclaw-shield | Pattern matches on dangerous commands | Can't predict patch failures |
| safe-exec | Blocks dangerous shell patterns | No learning, no receipts |
| skill-scanner | Scans skills for malware patterns | Doesn't cover code changes |
| **capseal** | **Learns failure rates from your codebase** | **Requires ~$1 learning phase** |

Static rules can tell you "rm -rf is dangerous." CapSeal can tell you "non-literal-import patches fail 86% of the time on *this specific codebase* but sqlalchemy patches succeed 100%."

## Links

- [CapSeal GitHub](https://github.com/yourusername/capseal)
- [How it works](https://capseal.dev)
