# CapSole docs patch (docs-only)

This folder contains a docs tree aligned to the *current code behavior described in your audit*:

- Canonical encoding: `dag_cbor_compact_fields_v1`
- Domain tags: BEF_CAPSULE_V1, BEF_AUDIT_SEED_V1, CAPSULE_HEADER_V2::, CAPSULE_HEADER_COMMIT_V1::, CAPSULE_ID_V2::, CAPSULE_INSTANCE_V1::
- Instance binding: extended tuple under CAPSULE_INSTANCE_V1:: (includes params_hash + chunk_meta_hash + row_tree_arity + air/fri/program/vk + statement/spec/root)
- Manifest signatures: signature over anchor_digest bytes (no DST)

## How to apply

Copy the `docs/` directory in this folder into your repo root (or merge it with your existing `docs/`), then:

- Replace/merge your existing `primer.md` and `security_model.md` with the versions here.
- If you already have `docs/spec/*`, copy the updated files over (02/03/04/05/08) to remove drift.

No code changes are included.
