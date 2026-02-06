# Pipeline Verification Report

**Generated:** 2026-01-23T22:46:37.797571+00:00
**Capsule:** `ran_computations/trading/momentum_strategy_v2/strategy_capsule.json`
**Hash:** `263b9b73939bf704...`
**Status:** ✗ WARN

## Checks

| Check | Status | Duration | Details |
|-------|--------|----------|---------|
| inspect | ✓ pass | 134.5ms | format=capsule.json, capsule_id=ced5b5cc7e2784c3, trace_id=run_20251226_002048 |
| verify | ⚠ warn | 217.3ms | Policy document missing (E036) |
| audit | ✓ pass | 130.2ms | total_events=8, hash_chain=VALID |
| row_open_0 | ✓ pass | 130.4ms | row_index=0, commitment=3e20dd87336d0c3c..., proof_length=2 |

## Summary

- **capsule_id:** ced5b5cc7e2784c3
- **trace_id:** run_20251226_002048
- **backend:** stark_fri
- **schema:** bef_capsule_v1
- **trace_schema_id:** None
- **checks_passed:** 3
- **checks_total:** 4
- **total_duration_ms:** 612.3296320001828

## Artifact Index

| Artifact | Location/Hash |
|----------|---------------|
| capsule_json | `ran_computations/trading/momentum_strategy_v2/strategy_capsu...` |
| capsule_hash | `263b9b73939bf704be4559074856c4004a72457f8f713e064b5d8a9067b5...` |
| capsule_id | `ced5b5cc7e2784c3` |
| trace_id | `run_20251226_002048` |
| events_log | `ran_computations/trading/momentum_strategy_v2/events/events....` |
| events_log_hash | `5d5411ba25f44133965d7f8a96798bc70aa15c9716e05e380ada6f59aab2...` |
| row_0_commitment | `3e20dd87336d0c3cb319f85447974dec37a32a23f2e693db8026b3c34bf0...` |
| policy_json | `ran_computations/trading/momentum_strategy_v2/policy.json` |
| policy_hash | `7b7162a296d86be2bd242dd08465a96957184769db1c7c51c83970716c5f...` |
| row_archive_dir | `ran_computations/trading/momentum_strategy_v2/row_archive` |
| row_archive_chunks | `66` |
| proofs_dir | `ran_computations/trading/momentum_strategy_v2/proofs` |
| manifests_dir | `ran_computations/trading/momentum_strategy_v2/manifests` |
| manifest_index_hash | `9fd11e48cf20b6854abd310fc25b32c48f8e84405f79fc42193f96f1ca0e...` |