# Pipeline Verification Report

**Generated:** 2026-01-24T01:13:55.367001+00:00
**Capsule:** `/home/ryan/BEF-main/fixtures/golden_run/capsule/capsule.json`
**Hash:** `354e7eb2703fb2ab...`
**Status:** ✗ FAIL

## Checks

| Check | Status | Duration | Details |
|-------|--------|----------|---------|
| inspect | ✓ pass | 626.5ms | format=capsule.json, capsule_id=1a476abdba9f422d, trace_id=run_20251223_000934 |
| verify | ✗ fail | 686.7ms | REJECTED: E013_CAPSULE_HEADER_MISMATCH in 0.1ms
 |
| audit | ✓ pass | 620.2ms | total_events=8, hash_chain=VALID |

## Summary

- **capsule_id:** 1a476abdba9f422d
- **trace_id:** run_20251223_000934
- **backend:** stark_fri
- **schema:** bef_capsule_v1
- **trace_schema_id:** None
- **checks_passed:** 2
- **checks_total:** 3
- **total_duration_ms:** 1933.3073860034347

## Artifact Index

| Artifact | Location/Hash |
|----------|---------------|
| capsule_json | `/home/ryan/BEF-main/fixtures/golden_run/capsule/capsule.json` |
| capsule_hash | `354e7eb2703fb2abfd3ec4c26c6956fb1dcaeab1fd3ac2a3be9d0bed3812...` |
| capsule_id | `1a476abdba9f422d` |
| trace_id | `run_20251223_000934` |
| events_log | `/home/ryan/BEF-main/fixtures/golden_run/capsule/events/event...` |
| events_log_hash | `37a9543e90760ebd983b294e6d8682acdb646caa398a851cf11796a39a77...` |
| policy_json | `/home/ryan/BEF-main/fixtures/golden_run/capsule/policy.json` |
| policy_hash | `7b7162a296d86be2bd242dd08465a96957184769db1c7c51c83970716c5f...` |
| row_archive_dir | `/home/ryan/BEF-main/fixtures/golden_run/capsule/row_archive` |
| row_archive_chunks | `66` |
| proofs_dir | `/home/ryan/BEF-main/fixtures/golden_run/capsule/proofs` |
| manifests_dir | `/home/ryan/BEF-main/fixtures/golden_run/capsule/manifests` |
| manifest_index_hash | `5d4c174cd8575bcc125d3ff1f4580d2618d7dfca2c6d793bb694addb1895...` |