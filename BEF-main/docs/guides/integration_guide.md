# Integration Guide

Quick links: `docs/primer.md` (intro) • `docs/spec/` (normative) • `docs/security_model.md` (predicates)

## Run
```
capsule-bench run --backend <id> --policy <path> --policy-id <id> --track-id <track> \
  --manifest-signer-id <id> --manifest-signer-key <hex|path>
```

## Verify (raw capsule.json)
```
PYTHONPATH=. python scripts/verify_capsule.py <capsule.json> \
  --policy <policy.json> --manifest-root <manifests/>
```

## Package as a portable .cap
```
# Create a hermetic verification artifact
capsule emit \
  --capsule out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --artifacts out/capsule_runs/<run_id>/pipeline \
  --policy out/capsule_runs/<run_id>/policy.json \
  --out /tmp/receipt.cap
```

## Verify a .cap (hermetic)
```
# Self-contained: extractor enforces path safety; verifier uses embedded artifacts
capsule verify /tmp/receipt.cap --json

# Or provide policy/manifests explicitly
capsule verify /tmp/receipt.cap --policy policy.json --manifests manifests/
```

Notes:
- `.cap` verification reconstructs the expected rel-path layout in a sandboxed temp dir and validates sizes/hashes before calling the canonical verifier.
- See `docs/spec/10_cap_format.md` for the archive format and safety guarantees.

See `docs/spec/07_adapter_contract.md` for backend integration and mutation tests.
