# Risc0 Backend

Purpose: demonstrate backend generality by binding a Risc0 receipt to the capsule instance tuple and verifying it under the same verifier profiles.

## Binding

- `image_id` and `journal_digest` MUST bind to the capsule’s `statement_hash` via the backend’s claim.
- The capsule’s `proof_system` MUST include `scheme_id="risc0_receipt_v1"`, and `vk`/program identities as appropriate.
- The verifier recomputes `instance_hash` and ensures the receipt claim/journal binding occurs before any Fiat–Shamir challenges.

## Adapter contract

- `simulate_trace`: loads journal/public outputs; supplies a minimal `TraceSpecV1` describing the image.
- `commit_to_trace`: uses receipt claim (or a synthetic stub) to populate `row_root` and artifact pointers.
- `generate_proof`: returns the receipt as JSON/bin; the adapter must enforce binding to `statement_hash`.
- `verify`: calls the Risc0 verifier and rejects if claim/journal do not match the bound statement.

## Limitations

- No row-level archive; the commitment is a receipt claim (merkle root), not a full STC archive.
- DA sampling is N/A for Risc0-only proofs; use POLICY_ENFORCED or legacy DA profiles until a challenger-backed DA is available.

