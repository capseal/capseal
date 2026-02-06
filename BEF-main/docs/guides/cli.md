# Capsule CLI

Hermetic, portable verification via a single file.

## Commands

### Emit
```
capsule emit \
  --capsule out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --artifacts out/capsule_runs/<run_id>/pipeline \
  --policy out/capsule_runs/<run_id>/policy.json \
  --out /tmp/receipt.cap
```

Produces a `.cap` archive containing:
- `capsule.json` (full capsule)
- `proof.bin.zst` (compressed proof payload)
- `commitments.json` (root/chunk metadata)
- `artifact_manifest.json` (content-addressed artifact index)
- `events/events.jsonl` (event chain, optional)
- `archive/` (row archive, optional)
- `policy.json` (optional)

### Verify
```
# Raw capsule.json (requires policy/manifests on disk)
PYTHONPATH=. python scripts/verify_capsule.py <capsule.json> \
  --policy <policy.json> --manifest-root <manifests/>

# Hermetic .cap (uses embedded artifacts and safe extraction)
capsule verify /tmp/receipt.cap --json
```

Exit codes:
- 0 verified
- 10 proof invalid (E05x/E3xx)
- 11 policy mismatch (E03x/E10x)
- 12 commitment/index failed (E06x/E2xx)
- 13 DA audit failed (E07x)
- 20 malformed/parse error (E001–E004)

### Inspect
```
capsule inspect /tmp/receipt.cap --json
```

## Safety & Semantics

- Extraction is sandboxed: no symlinks/hardlinks, no traversal/absolute paths, size limits per entry.
- Materialization writes proof/archive/events to their recorded `rel_path` and validates sizes/hashes before invoking the canonical verifier.
- `.cap` verification matches raw `scripts/verify_capsule.py` behavior and reason codes.

See `docs/spec/10_cap_format.md` and `docs/spec/06_protocol.md`.

