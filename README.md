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
| `capseal advanced` | Power user commands (shell, trace, merge, refactor) |

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
