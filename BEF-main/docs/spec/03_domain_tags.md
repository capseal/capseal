# Domain tags (normative)

This file is the single source of truth for hash/signature domain separation strings used by the current codebase.

## Hash prefixes / domain tags

| Name | Value (bytes) | Used for |
| --- | --- | --- |
| Payload hash | `b"BEF_CAPSULE_V1"` | `payload_hash = H(tag || Enc(payload_view))` |
| Audit seed | `b"BEF_AUDIT_SEED_V1"` | `seed = H(tag || capsule_hash || challenge_bytes)` |
| Header hash | `b"CAPSULE_HEADER_V2::"` | hashing the full header (if present) |
| Header commit hash | `b"CAPSULE_HEADER_COMMIT_V1::"` | `header_commit_hash` |
| Capsule ID hash | `b"CAPSULE_ID_V2::"` | `capsule_hash` |
| Instance binding hash | `b"CAPSULE_INSTANCE_V1::"` | `instance_hash` |

## Notes

- Version numbers in tags are not “marketing”; they are part of the security boundary.
- If you change any domain tag, you MUST treat it as a breaking format change.
