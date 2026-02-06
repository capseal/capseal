# Fixtures

`fixtures/golden_run/` is the canonical offline test fixture.

Layout:

- capsule/
  - capsule.json or capsule.bin  (the exact file your verifier consumes)
  - capsule_hash.txt             (optional; populated if verifier outputs it)
  - header_commit_hash.txt       (optional)
  - payload_hash.txt             (optional)
  - instance_hash.txt            (optional)

- artifacts/
  - proof.*                      (whatever your capsule references)
  - row_archive/                 (chunked trace archive)
  - manifests/                   (anchor + signature, etc.)
  - da/                          (relay challenge, if FULL)
  - registries/                  (pinned signer/relay registries + pins)

- expected_results.json          (profiles + mutation matrix)

The builder (`scripts/e2e/build_fixture.py`) copies known subpaths from a run dir:
- pipeline/*capsule*.{json,bin}
- row_archive/
- manifests/
- da/
- registries/
and records what it found into expected_results.json as a starting point.

You then edit expected_results.json to add mutation expectations.
