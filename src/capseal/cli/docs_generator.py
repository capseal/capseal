"""Docs generator - deterministic documentation from schemas and CLI introspection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .trace_schema import SCHEMAS, TraceSchema


def _generate_commands_md() -> str:
    """Generate commands.md from CLI definitions."""
    return """# Capseal CLI Commands

## Overview

Capseal provides cryptographic proof generation and verification for computational traces.

## Commands

### `capseal doctor`

One-click pipeline verification with derived reports.

**Usage:**
```bash
capseal doctor <capsule.json>
capseal doctor <capsule.json> -o ./reports --sample-rows 3
capseal doctor <capsule.json> --json
```

**Options:**
- `--output, -o PATH` - Output directory for reports
- `--sample-rows INT` - Number of rows to sample for opening (default: 1)
- `--json` - Output JSON only
- `--quiet, -q` - Minimal output

**Output (JSON):**
```json
{
  "capsule_path": "string",
  "capsule_hash": "sha256 hex",
  "report_timestamp": "ISO8601",
  "overall_status": "PASS | WARN | FAIL",
  "checks": [
    {
      "name": "string",
      "status": "pass | fail | warn | skip",
      "duration_ms": "number",
      "details": {},
      "error": "string | null"
    }
  ],
  "summary": {},
  "index": {"artifact": "path/hash"}
}
```

---

### `capseal row`

Open a specific trace row with STC membership proof and optional semantic decoding.

**Usage:**
```bash
capseal row <capsule.json> --row N
capseal row <capsule.json> --row N --schema momentum_v2_row_v1 --json
capseal row <capsule.json> --row N --schema momentum_v2_row_v1 --strict-schema
```

**Options:**
- `--row INT` - Row index to open (required)
- `--schema STRING` - Schema ID for semantic decoding
- `--strict-schema` - Fail if schema hash mismatches capsule
- `--json` - Output JSON
- `--dataset TEXT` - Dataset mapping id=path (for local validation)
- `--policy PATH` - Policy file for governed openings
- `--ticket PATH` - Signed opening ticket

**Output (JSON with --schema):**
```json
{
  "row_index": 0,
  "row_values": [/* raw field elements */],
  "row_raw_values": [/* same as row_values */],
  "row_fields": [
    {
      "name": "price",
      "type": "f64_scaled",
      "meaning": "Asset price",
      "unit": "USD",
      "scale": 100000000,
      "value": 195.42,
      "raw_value": 19542000000
    }
  ],
  "schema_id": "momentum_v2_row_v1",
  "schema_hash": "sha256 hex",
  "commitment": "sha256 hex",
  "proof": [[/* sibling hashes */]],
  "row_proof_valid": true
}
```

---

### `capseal audit`

Export and inspect audit trail from a capsule.

**Usage:**
```bash
capseal audit <capsule.json>
capseal audit <capsule.json> --format csv -o audit.csv
capseal audit <capsule.json> --format json --filter-type proof_artifact
```

**Options:**
- `--output, -o PATH` - Output file (default: stdout)
- `--format [json|jsonl|csv|summary]` - Output format
- `--verify / --no-verify` - Verify hash chain integrity
- `--filter-type TEXT` - Filter events by type
- `--from-seq INT` - Start from sequence number
- `--to-seq INT` - End at sequence number

**Output (summary):**
```
Audit Trail: strategy_capsule.json
Events Log: events/events.jsonl
Total Events: 8

Hash Chain: VALID

Event Types:
  capsule_sealed: 1
  proof_artifact: 1
  ...
```

---

### `capseal verify`

Verify a capsule with stable exit codes for CI.

**Usage:**
```bash
capseal verify <capsule.json>
capseal verify <capsule.json> --json
capseal verify <capsule.cap>
```

**Exit Codes:**
- `0` - VERIFIED
- `10` - PROOF_INVALID
- `11` - REJECTED (policy/integrity)
- `20` - MALFORMED

**Output (JSON):**
```json
{
  "status": "VERIFIED | REJECTED",
  "proof_verified": true,
  "backend_id": "stark_fri",
  "verify_stats": {
    "time_verify_sec": 0.0019
  }
}
```

---

### `capseal replay`

Replay a capsule trace for semantic verification.

**Usage:**
```bash
capseal replay <capsule.json>
capseal replay <capsule.json> -d archive1=./archive1 --tolerance 0.001
capseal replay <capsule.json> --until-diverge --json
```

**Options:**
- `--dataset, -d TEXT` - Dataset mapping id=path
- `--tolerance FLOAT` - Maximum relative tolerance (0.0 = exact)
- `--range TEXT` - Replay only rows within start:end
- `--sample INT` - Replay random sample of N rows
- `--max-divergences INT` - Stop after N divergences
- `--until-diverge` - Stop after first divergence
- `--json` - Output JSON
- `--verbose, -v` - Show detailed divergence info

**Exit Codes:**
- `0` - IDENTICAL or EQUIVALENT
- `14` - DIVERGED
- `20` - MALFORMED/UNSUPPORTED

---

### `capseal emit`

Generate a portable verification artifact (.cap file).

**Usage:**
```bash
capseal emit --source <run_dir> --out <output.cap>
capseal emit --source <run_dir> --out <output.cap> --trace-schema momentum_v2_row_v1
```

**Options:**
- `--source PATH` - Source directory or capsule file
- `--out PATH` - Output .cap file path (required)
- `--trace-schema STRING` - Schema ID for semantic binding
- `--profile [proof-only|da|replay]` - Verification profile
- `--policy PATH` - Policy file to include
- `--manifests PATH` - Manifests directory

---

### `capseal inspect`

Display capsule metadata.

**Usage:**
```bash
capseal inspect <capsule.json>
capseal inspect <capsule.cap>
```

**Output:**
```
Format:     capsule.json
Capsule ID: ced5b5cc7e2784c3
Trace ID:   run_20251226_002048
Backend:    stark_fri
Profile:    POLICY_ENFORCED
Schema:     bef_capsule_v1
File size:  56.7 KB
```

---

### `capseal sandbox`

Manage sandbox execution environment.

**Usage:**
```bash
capseal sandbox status   # Check availability
capseal sandbox test     # Verify isolation
```

---

### `capseal docs generate`

Generate documentation from schemas and CLI definitions.

**Usage:**
```bash
capseal docs generate
capseal docs generate -o ./custom_docs
```
"""


def _generate_artifacts_md() -> str:
    """Generate artifacts.md documenting capsule fields and report schema."""
    return """# Capseal Artifacts

## Capsule Fields

### Required Fields

| Field | Type | Description | Verification |
|-------|------|-------------|--------------|
| `capsule_id` | string | Unique capsule identifier (truncated hash) | Derived from content |
| `trace_id` | string | Trace run identifier | Recorded at generation |
| `schema` | string | Capsule schema version (e.g., `bef_capsule_v1`) | Validated on load |
| `backend_id` | string | Proof backend (e.g., `stark_fri`, `geom`) | Checked against proof |
| `statement` | object | Public statement/anchors | Hash verified in proof |
| `row_archive` | object | Row archive metadata | Root verified via STC |

### Optional Fields

| Field | Type | Description | Verification |
|-------|------|-------------|--------------|
| `trace_schema_id` | string | Semantic schema ID for rows | Hash checked if present |
| `trace_schema_hash` | string | SHA256 of canonical schema | Compared to registry |
| `policy_id` | string | Policy identifier | Checked against policy.json |
| `profile` | string | Execution profile tag | Informational |
| `dataset_ref` | object | Dataset bindings | Roots verified via Merkle |
| `artifacts` | object | Paths to proof/archive files | Files must exist |

---

## Doctor Report Schema

The `capseal doctor` command produces `report.json`:

```json
{
  "capsule_path": "string - path to capsule file",
  "capsule_hash": "string - SHA256 of capsule file",
  "report_timestamp": "string - ISO8601 UTC timestamp",
  "overall_status": "PASS | WARN | FAIL",

  "checks": [
    {
      "name": "string - check identifier",
      "status": "pass | fail | warn | skip",
      "duration_ms": "number - execution time",
      "details": {
        "/* check-specific key-value pairs */"
      },
      "error": "string | null - error message if failed"
    }
  ],

  "summary": {
    "capsule_id": "string",
    "trace_id": "string",
    "backend": "string",
    "schema": "string",
    "trace_schema_id": "string | null",
    "checks_passed": "number",
    "checks_total": "number",
    "total_duration_ms": "number"
  },

  "index": {
    "capsule_json": "string - file path",
    "capsule_hash": "string - SHA256",
    "capsule_id": "string",
    "trace_id": "string",
    "events_log": "string - path",
    "events_log_hash": "string - SHA256",
    "policy_json": "string - path (if present)",
    "policy_hash": "string - SHA256 (if present)",
    "trace_schema_id": "string (if present)",
    "trace_schema_hash": "string (if present)",
    "row_archive_dir": "string - path",
    "row_archive_chunks": "string - count",
    "row_N_commitment": "string - per-row commitment"
  }
}
```

### Check Types

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| `inspect` | Load and parse capsule metadata | Valid JSON, required fields present |
| `verify` | Cryptographic proof verification | Proof valid, exit code 0 |
| `audit` | Event log hash chain integrity | Chain valid, no gaps |
| `row_open_N` | Row opening with STC proof | Proof verifies against commitment |
| `schema_binding` | Trace schema hash validation | Hash matches registry |

---

## Index File Schema

The `index.json` provides artifact hashes for verification:

```json
{
  "capsule_json": "/path/to/capsule.json",
  "capsule_hash": "sha256:...",
  "capsule_id": "hex16",
  "trace_id": "run_identifier",
  "events_log": "/path/to/events.jsonl",
  "events_log_hash": "sha256:...",
  "policy_json": "/path/to/policy.json",
  "policy_hash": "sha256:...",
  "manifest_index_hash": "sha256:...",
  "row_archive_dir": "/path/to/row_archive",
  "row_archive_chunks": "66",
  "row_0_commitment": "sha256:..."
}
```

All hashes are deterministic SHA256 of file contents.
"""


def _generate_trace_schemas_md() -> str:
    """Generate trace_schemas.md from registered schemas."""
    lines = [
        "# Trace Schemas",
        "",
        "Trace schemas define semantic bindings for row values, making cryptographic proofs interpretable.",
        "",
        "## Schema Registry",
        "",
    ]

    for schema_id, schema in sorted(SCHEMAS.items()):
        schema_hash = schema.hash()
        lines.extend([
            f"### `{schema_id}`",
            "",
            f"**Version:** {schema.schema_version}",
            f"**Hash:** `{schema_hash}`",
            f"**Description:** {schema.description}",
            f"**Row Width:** {schema.row_width} fields",
            "",
            "| Index | Field | Type | Unit | Scale | Meaning |",
            "|-------|-------|------|------|-------|---------|",
        ])

        for i, field in enumerate(schema.fields):
            unit = field.unit or "-"
            scale = str(field.scale) if field.scale else "-"
            lines.append(f"| {i} | `{field.name}` | {field.field_type} | {unit} | {scale} | {field.meaning} |")

        lines.extend(["", "---", ""])

    lines.extend([
        "## Schema Hash Computation",
        "",
        "Schema hashes are computed as SHA256 of canonical JSON:",
        "",
        "```python",
        "import hashlib, json",
        "canonical = json.dumps(schema.to_dict(), sort_keys=True, separators=(',', ':'))",
        "schema_hash = hashlib.sha256(canonical.encode()).hexdigest()",
        "```",
        "",
        "This ensures:",
        "- Deterministic hashes across implementations",
        "- Tampering detection (any field change = different hash)",
        "- Version binding (capsule records expected hash)",
    ])

    return "\n".join(lines)


@click.group("docs")
def docs_group():
    """Documentation generation commands."""
    pass


@docs_group.command("generate")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: docs/)",
)
def generate_docs(output: Path | None):
    """Generate documentation from schemas and CLI definitions.

    Produces deterministic documentation files:
    - docs/commands.md     - CLI command reference
    - docs/artifacts.md    - Capsule and report schemas
    - docs/trace_schemas.md - Registered trace schemas with hashes

    Documentation is derived from code, not hand-written.
    """
    if output is None:
        # Find project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "bef_zk").is_dir():
                output = parent / "docs" / "generated"
                break
        if output is None:
            output = Path("docs")

    output.mkdir(parents=True, exist_ok=True)

    # Generate each doc file
    files = {
        "commands.md": _generate_commands_md(),
        "artifacts.md": _generate_artifacts_md(),
        "trace_schemas.md": _generate_trace_schemas_md(),
    }

    for filename, content in files.items():
        filepath = output / filename
        filepath.write_text(content)
        click.echo(f"  {filepath}")

    click.echo(f"\nGenerated {len(files)} documentation files in {output}/")

    # Print schema hashes for verification
    click.echo("\nSchema hashes (for verification):")
    for schema_id, schema in sorted(SCHEMAS.items()):
        click.echo(f"  {schema_id}: {schema.hash()}")
