# Pipeline Verification Report

**Generated:** 2026-01-29T17:03:26.400204+00:00
**Capsule:** `fixtures/golden_run_latest/capsule/strategy_capsule.json`
**Hash:** `67c1299b203a293e...`
**Status:** ✓ PASS

## Checks

| Check | Status | Duration | Details |
|-------|--------|----------|---------|
| inspect | ✓ pass | 749.2ms | format=capsule.json, capsule_id=ced5b5cc7e2784c3, trace_id=run_20251226_002048 |
| verify | ✓ pass | 757.7ms | status=PROOF_ONLY, proof_verified=True, backend=geom_stc_fri |
| audit | ✓ pass | 738.2ms | total_events=8, hash_chain=VALID |
| row_open_0 | ✓ pass | 749.9ms | row_index=0, commitment=3e20dd87336d0c3c..., proof_length=2 |

## Summary

- **capsule_id:** ced5b5cc7e2784c3
- **trace_id:** run_20251226_002048
- **backend:** stark_fri
- **schema:** bef_capsule_v1
- **trace_schema_id:** None
- **checks_passed:** 4
- **checks_total:** 4
- **total_duration_ms:** 2995.01131400757

## Artifact Index

| Artifact | Location/Hash |
|----------|---------------|
| capsule_json | `fixtures/golden_run_latest/capsule/strategy_capsule.json` |
| capsule_hash | `67c1299b203a293e68eb6ee01f9d4f91c52495b05a14f6a706c3316c7336...` |
| capsule_id | `ced5b5cc7e2784c3` |
| trace_id | `run_20251226_002048` |
| events_log | `fixtures/golden_run_latest/capsule/events/events.jsonl` |
| events_log_hash | `5d5411ba25f44133965d7f8a96798bc70aa15c9716e05e380ada6f59aab2...` |
| row_0_commitment | `3e20dd87336d0c3cb319f85447974dec37a32a23f2e693db8026b3c34bf0...` |
| policy_json | `fixtures/golden_run_latest/capsule/policy.json` |
| policy_hash | `7b7162a296d86be2bd242dd08465a96957184769db1c7c51c83970716c5f...` |
| row_archive_dir | `fixtures/golden_run_latest/capsule/row_archive` |
| row_archive_chunks | `66` |
| proofs_dir | `fixtures/golden_run_latest/capsule/proofs` |
| manifests_dir | `fixtures/golden_run_latest/capsule/manifests` |
| manifest_index_hash | `9fd11e48cf20b6854abd310fc25b32c48f8e84405f79fc42193f96f1ca0e...` |