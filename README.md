# CapSeal

**Proof-carrying execution for AI agents.**

CapSeal is a runtime that makes any AI agent's execution cryptographically provable.

An agent does things — calls tools, writes code, makes decisions. CapSeal records each action, chains them together, encodes the chain as an algebraic trace over a Goldilocks field, and generates a FRI proof that the entire execution sequence is internally consistent and untampered. The proof verifies in milliseconds. The agent doesn't need to know any of this is happening.

On top of that, CapSeal learns where agents fail. The epistemic eval loop actively discovers the boundary between success and failure, builds a committor model, and uses it to gate future actions before they execute. The eval loop itself is proof-carrying — you can verify that the learning process was honest.

## 5-Line Integration

```python
from capseal import AgentRuntime

with AgentRuntime(output_dir=Path("my_run")) as runtime:
    runtime.record_simple(action_type="tool_call", instruction="List files",
                          inputs={"cmd": "ls"}, outputs={"files": ["a.py"]}, success=True)
    runtime.record_simple(action_type="code_gen", instruction="Write function",
                          inputs={"spec": "add(a,b)"}, outputs={"code": "..."}, success=True)
# Proof-carrying capsule generated automatically on exit
```

## Install

```bash
pip install -e '.[dev]'
```

## Quick Start

```bash
# 1. Learn where patches fail (epistemic eval with proof)
capseal eval src/ --rounds 10 --synthetic --prove

# 2. Gate patches using learned posteriors
capseal review src/ --gate

# 3. Verify the eval was honest
capseal verify-capsule .capseal/runs/latest/eval_capsule.json
```

## Three Proof Systems

CapSeal includes three specialized AIR (Algebraic Intermediate Representation) encodings:

| AIR | Purpose | CLI |
|-----|---------|-----|
| **WorkflowAIR** | DAG node execution proofs | `capseal workflow --prove` |
| **EvalAIR** | Epistemic learning round proofs | `capseal eval --prove` |
| **AgentAIR** | General agent action proofs | `AgentRuntime.finalize(prove=True)` |

All three use the same 14-element Goldilocks field encoding and FRI prover.

## How It Works

### Proof-Carrying Execution

1. **Record**: Each agent action is captured as an `AgentAction` with hashed inputs/outputs
2. **Chain**: Actions are linked via `prev_receipt_hash` — an unbroken provenance chain
3. **Encode**: The chain is encoded as trace rows over the Goldilocks field (p = 2^64 - 2^32 + 1)
4. **Commit**: Rows are Merkle-committed using STC (Sumcheck Tensor Commitment)
5. **Prove**: FRI (Fast Reed-Solomon IOP) generates a succinct proof
6. **Verify**: Anyone can verify in milliseconds that the trace is internally consistent

### Epistemic Learning

1. **Grid**: 5-dimensional feature space (lines changed, complexity, files touched, severity, test coverage) → 1024 grid points
2. **Acquisition**: Boundary-aware uncertainty sampling selects which patches to try
3. **Update**: Beta posteriors updated after each episode: α += failures, β += successes
4. **Gate**: Committor q(x) = α/(α+β) predicts failure probability; skip if q ≥ 0.3

### What the Proofs Guarantee

- **Chain integrity**: Every action's receipt links to its predecessor
- **Ordering**: Actions executed in declared sequence
- **No substitution**: You can't swap posteriors between eval rounds
- **Honest counts**: Episode success/failure counts match declared totals

## CLI Reference

```bash
# Epistemic evaluation with FRI proof
capseal eval src/ --rounds 20 --synthetic --prove

# Review with committor gate
capseal review src/ --gate --posteriors .capseal/models/beta_posteriors.npz

# Verify any capsule
capseal verify-capsule .capseal/runs/*/eval_capsule.json
capseal verify-capsule .capseal/runs/*/workflow_capsule.json
capseal verify-capsule .capseal/runs/*/agent_capsule.json

# Inspect agent runs
capseal agent inspect .capseal/runs/latest
```

## Python API

```python
from capseal import AgentRuntime, AgentAction, wrap_function

# Option 1: Context manager (recommended)
with AgentRuntime(output_dir=Path("run")) as rt:
    rt.record_simple(action_type="tool_call", ...)

# Option 2: Decorator
@wrap_function(runtime, action_type="tool_call")
def my_tool(args):
    return result

# Option 3: Full control
action = AgentAction(
    action_id="act_001",
    action_type="code_gen",
    instruction_hash=hash_str(prompt),
    input_hash=hash_json(context),
    output_hash=hash_str(code),
    success=True,
    duration_ms=1234,
    timestamp=datetime.now().isoformat(),
)
runtime.record(action)
capsule = runtime.finalize(prove=True)
```

## Architecture

```
capseal/
├── capseal_cli/              # Entry point, public API exports
├── BEF-main/bef_zk/
│   ├── capsule/
│   │   ├── agent_protocol.py # AgentAction schema
│   │   ├── agent_runtime.py  # Main integration surface
│   │   ├── agent_air.py      # 14-element row encoding
│   │   ├── agent_adapter.py  # Trace → commit → prove
│   │   ├── workflow_*.py     # Phase 1: DAG proofs
│   │   ├── eval_*.py         # Phase 2: Eval round proofs
│   │   └── adapters/         # Framework integrations
│   ├── fri/                  # FRI prover/verifier
│   ├── stc/                  # STC commitment
│   └── shared/               # Scoring, features, receipts
└── tests/                    # 61 tests
```

## Tests

```bash
python -m pytest tests/ -v
# 61 passed
```

## What Makes This Different

- **Eval frameworks** test agents but don't prove anything
- **Logging tools** record events but can't detect tampering
- **ZK systems** prove computations but don't understand agent workflows

CapSeal combines all three: epistemic learning, tamper-evident logging, and cryptographic proofs — in one CLI with a unified proof format.

## License

MIT
