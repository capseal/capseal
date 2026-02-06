# Spec 10 — `.cap` format (normative)

A `.cap` is a hermetic verification archive. It contains the capsule and the minimum artifacts required for verification, with strict extraction safety rules.

## Contents

- `capsule.json` — the full capsule descriptor
- `proof.bin.zst` — compressed proof payload (optional if proofs are small/JSON)
- `commitments.json` — root/chunk metadata (optional; convenience)
- `artifact_manifest.json` — content-addressed artifact index (optional but RECOMMENDED)
- `events/events.jsonl` — event stream (optional)
- `archive/` — row archive snapshots (optional)
- `policy.json` — policy document (optional)

## Safe extraction (required)

Implementations MUST enforce the following when extracting a `.cap`:

- Reject absolute paths, `..` traversal, and backslashes as separators.
- Reject symlinks and hardlinks.
- Allow only directories and regular files.
- Enforce a maximum member size (configurable; default ≤ 512 MiB).
- Reject duplicate normalized paths.
- Refuse to overwrite an existing path in the sandbox.

## Materialization (required)

After extraction into a sandbox directory, implementations MUST:

- Write the proof blob (if present) to the rel path recorded in the capsule’s `proofs[...].formats[*].rel_path` (or a default); validate size and `sha256_payload_hash` before use.
- Relocate `archive/` to the row archive’s recorded `rel_path` (default `row_archive/`).
- If `artifacts.row_archive.rel_path` differs, mirror or alias accordingly.
- Preserve `events/events.jsonl` and `artifact_manifest.json` so verifier can recompute anchors.

## Verification mapping

- The verifier MUST use the materialized paths within the sandbox and MUST NOT read outside the sandbox.
- Size/hash mismatches for materialized artifacts MUST produce a hard failure.
- The behavior and reason codes MUST match the canonical verifier run over the original `strategy_capsule.json`.

This spec complements Spec 06 (Protocol) and Spec 08 (Registries) by standardizing portable verification.

