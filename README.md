# CapSeal

Predictive gating and cryptographic verification for AI code changes. CapSeal learns which patches fail on your codebase, gates risky changes before they run, and produces verifiable receipts proving what happened.

## Install

```bash
git clone <repo>
cd capseal
pip install -e .
```

## Quick start

```bash
capseal init                    # set up workspace
capseal scan .                  # find issues with semgrep
capseal learn . --rounds 5      # learn which patches succeed/fail ($0.50-2.00)
capseal fix . --dry-run         # preview what would be fixed
capseal fix .                   # generate verified patches
capseal verify .capseal/runs/latest.cap   # verify the sealed run
```

## How it works

**Scan** — Find issues in your codebase using Semgrep's rule library.

**Learn** — CapSeal runs AI-generated patches against your codebase in rounds, tracking which succeed and fail across a 5-dimensional feature grid. Each round updates Beta posteriors that model failure probability across 1024 grid regions.

**Fix** — Generate patches for issues, gated by the learned model. High-risk patches (>60% predicted failure) are skipped. Low-risk patches are generated, verified, and sealed into a cryptographic receipt.

**Verify** — Every run is sealed into a `.cap` file with a hash-chained receipt. These receipts can be independently verified, giving you a tamper-evident audit trail.

## CLI Reference

| Command | Description |
|---------|-------------|
| `capseal init` | Initialize workspace |
| `capseal scan .` | Find issues with Semgrep |
| `capseal learn . --rounds N` | Learn risk model (default 5 rounds, $5 budget) |
| `capseal fix .` | Generate verified patches, gated by learned model |
| `capseal fix . --dry-run` | Preview patches without generating |
| `capseal fix . --apply` | Generate and apply patches |
| `capseal review . --gate` | Gate-only mode (no patch generation) |
| `capseal verify <file.cap>` | Verify sealed run |
| `capseal report <run>` | Generate summary report |
| `capseal watch .` | CI integration (JSON output) |
| `capseal demo` | 30-second interactive demo |
| `capseal mcp-serve` | MCP server for agent integration |
| `capseal advanced` | Power user commands (shell, trace, merge, refactor) |

## Agent Integration (MCP)

CapSeal exposes itself as an MCP server, allowing any agent framework to use it as a trust layer:

```bash
capseal mcp-serve   # Start MCP server (stdio transport)
```

**MCP Tools:**
| Tool | When to call | What it does |
|------|-------------|--------------|
| `capseal_gate` | Before every tool call | Returns approve/deny/flag based on learned risk |
| `capseal_record` | After every tool call | Records what happened for audit trail |
| `capseal_seal` | End of session | Seals everything into a .cap receipt |

**mcporter config:**
```json
{
  "capseal": {
    "command": "capseal",
    "args": ["mcp-serve"],
    "transport": "stdio"
  }
}
```

Works with OpenClaw, Claude Code, Cursor, LangChain — anything that speaks MCP.

## Configuration

```
Learned models:    .capseal/models/beta_posteriors.npz
Run artifacts:     .capseal/runs/<timestamp>-<type>/
Sealed runs:       .capseal/runs/<timestamp>-<type>.cap
Gate threshold:    --threshold 0.6 (failure probability cutoff)
Budget:            --budget 5.0 (max dollars to spend learning)
```

## Requirements

- Python 3.10+
- Semgrep (`pip install semgrep`)
- OpenAI API key (for learning and fix commands)
