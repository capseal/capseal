# CapSeal

**Proof-carrying execution for AI agents.**

CapSeal makes AI agent execution provable. It learns which actions will fail before you run them, gates risky actions, and generates cryptographic proofs of everything that happens. Verify any run in under a second.

## Try It

```bash
pip install -e .
capseal demo
```

The demo creates a sample project, learns which patches fail, gates a risky patch, generates proofs, and shows tamper detection — all in under 30 seconds.

### Real Workflow

```bash
# 1. Learn where patches fail
capseal eval src/ --rounds 10 --synthetic

# 2. Gate patches using what you learned
capseal review src/ --gate

# 3. Verify the proofs
capseal verify-capsule .capseal/runs/latest/eval_capsule.json
```

## Integrate With Your Agent

```python
from capseal import AgentRuntime
from pathlib import Path

with AgentRuntime(output_dir=Path(".capseal/runs/my-agent")) as runtime:
    runtime.record_simple(
        action_type="tool_call",
        instruction="Query database",
        inputs={"query": "SELECT * FROM users"},
        outputs={"rows": 42},
        success=True,
    )
# Proof generated automatically on exit
```

For framework adapters (LangChain, OpenAI), detailed API docs, and architecture details, see [docs/](docs/).

## License

MIT
