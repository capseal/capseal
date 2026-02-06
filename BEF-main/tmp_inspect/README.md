# Capsule E2E Harness Scaffold (fixtures + verify + mutation + adapter conformance)

This scaffold is designed to drop into your repo and give you a tight loop:

1) Build a fixture from an existing `out/capsule_runs/<run_id>/` directory
2) Verify it under PROOF_ONLY / POLICY_ENFORCED / FULL
3) Apply single-field mutations and assert expected reason codes
4) Run adapter conformance (binding-hash poisoning)

It is intentionally conservative: it tries to call your existing verifier entrypoint
(`scripts/verify_capsule.py`) as a subprocess so we don't guess internal APIs.

## Install (from repo root)

```bash
unzip -o capsole_e2e_harness_scaffold.zip -d .
```

## Quick start

```bash
python scripts/e2e/build_fixture.py --run-dir out/capsule_runs/<run_id> --out fixtures/golden_run
python scripts/e2e/verify_fixture.py --fixture fixtures/golden_run --profiles PROOF_ONLY POLICY_ENFORCED FULL
python scripts/e2e/mutate_fixture.py --fixture fixtures/golden_run --plan fixtures/golden_run/expected_results.json
python scripts/e2e/adapter_conformance.py --fixture fixtures/golden_run --backends geom risc0
```

## Expected verifier interface

This scaffold expects your verifier can be invoked like:

```bash
python scripts/verify_capsule.py <capsule_path> --profile <PROFILE> --json
```

and prints a single JSON object to stdout containing at least:

- `status` (e.g. "FULL", "POLICY_ENFORCED", "PROOF_ONLY", "REJECTED")
- `reason_codes` (list of strings, optional)
- optionally `capsule_hash`, `payload_hash`, `header_commit_hash`, `instance_hash`

If your verifier uses different flags or output format, edit:
`python scripts/e2e/common.py: run_verifier()`.

## Make target (optional)

Add to your Makefile:

```make
.PHONY: e2e
e2e:
	python scripts/e2e/verify_fixture.py --fixture fixtures/golden_run --profiles PROOF_ONLY POLICY_ENFORCED FULL
	python scripts/e2e/mutate_fixture.py --fixture fixtures/golden_run --plan fixtures/golden_run/expected_results.json
	python scripts/e2e/adapter_conformance.py --fixture fixtures/golden_run --backends geom risc0
```

## Notes

- Fixtures should be **offline-verifiable** (no network).
- Do NOT store private signing keys in fixtures.
- Mutation plans can start coarse (e.g., expected codes prefix "E30") and tighten later.

