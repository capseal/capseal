# Registries & signatures (normative, aligned to current code)

## Registry pinning

Registries (manifest signers, trusted relays) are pinned by a root hash computed over a canonical `id → pubkey` map.

- The verifier must know the expected root hash out-of-band (config pin).
- Overriding a registry requires supplying both:
  1) the new key map and
  2) the expected root for that map

If the computed root mismatches, verification MUST fail.

## Manifest anchor signing (current)

- The manifest bundle is reduced to an anchor payload and then to an `anchor_digest = sha256(Enc(anchor_payload))`.
- The manifest signature is a secp256k1 signature over the **anchor_digest bytes**.

No additional domain tag is prepended in current code. If you want a DST, you must introduce it as a breaking change and update verifiers accordingly.

## Relay challenge signing (summary)

Relay challenges are accepted only if:
- signed by a relay key id present in the pinned relay registry
- unexpired (if expiry is modeled)
- bound to the capsule commit/capsule hash as specified in `docs/spec/06_protocol.md`
