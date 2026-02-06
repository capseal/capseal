# Pipeline Verification Report

**Generated:** 2026-01-26T06:55:19.957166+00:00
**Capsule:** `/home/ryan/BEF-main/test_capseal_run/strategy_capsule.json`
**Hash:** `99f9a210735958ed...`
**Status:** ✗ FAIL

## Checks

| Check | Status | Duration | Details |
|-------|--------|----------|---------|
| inspect | ✓ pass | 166.0ms | format=capsule.json, capsule_id=fa75fca867b33f54, trace_id=capsule_run |
| verify | ✗ fail | 234.6ms | raw=VERIFIED (proof_only) in 2.0ms
 |
| audit | ✓ pass | 162.8ms | total_events=8, hash_chain=VALID |
| row_open_0 | ✓ pass | 169.6ms | row_index=0, commitment=3e20dd87336d0c3c..., proof_length=2 |
| row_open_1 | ✓ pass | 163.8ms | row_index=1, commitment=3e20dd87336d0c3c..., proof_length=2 |
| row_open_2 | ✓ pass | 164.5ms | row_index=2, commitment=3e20dd87336d0c3c..., proof_length=2 |

## Summary

- **capsule_id:** fa75fca867b33f54
- **trace_id:** capsule_run
- **backend:** stark_fri
- **schema:** bef_capsule_v1
- **trace_schema_id:** None
- **checks_passed:** 5
- **checks_total:** 6
- **total_duration_ms:** 1061.3221050298307

## Artifact Index

| Artifact | Location/Hash |
|----------|---------------|
| capsule_json | `/home/ryan/BEF-main/test_capseal_run/strategy_capsule.json` |
| capsule_hash | `99f9a210735958ede1e5dcb7a433ad0c80724b9879c1f1669ee1e3043ecc...` |
| capsule_id | `fa75fca867b33f54` |
| trace_id | `capsule_run` |
| events_log | `/home/ryan/BEF-main/test_capseal_run/events.jsonl` |
| events_log_hash | `98d88b5d8033e41dac4481468d7f85926e8d3ab6b0b4b9088fbcb46693ff...` |
| row_0_commitment | `3e20dd87336d0c3cb319f85447974dec37a32a23f2e693db8026b3c34bf0...` |
| row_1_commitment | `3e20dd87336d0c3cb319f85447974dec37a32a23f2e693db8026b3c34bf0...` |
| row_2_commitment | `3e20dd87336d0c3cb319f85447974dec37a32a23f2e693db8026b3c34bf0...` |
| row_archive_dir | `/home/ryan/BEF-main/test_capseal_run/row_archive` |
| row_archive_chunks | `66` |