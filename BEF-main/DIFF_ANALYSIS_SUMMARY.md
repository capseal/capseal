# Diff Context Analysis: ctx_cmp/main → HEAD

## Executive Summary

**Status**: ✅ **No merge conflicts exist**

The "diff context" you loaded shows files that were **added between ctx_cmp/main and HEAD** - these additions are already complete and committed. However, you have **25 uncommitted modified files** in your working directory that need attention.

---

## What the Diff Context Showed

### Files Added (ctx_cmp/main → HEAD)

The diff comparison revealed **27 new files** added since the ctx_cmp/main baseline:

#### 📁 Documentation Archive (docs/archive/)
- `BACKEND_READINESS_GAPS.md` - Gap analysis before production readiness
- `BEF-Stream-Accum-summary.md` - Formal specification of streaming accumulation
- `BEF_Architecture.html` & `.pdf` - Architecture documents
- `BEF_CURRENT_STATUS.md` - Status tracking
- `bef_compact_v1_spec.md` - Compact specification
- `bef_trace_commitment.md` - Trace commitment details
- `colab_instructions.txt` - Colab setup
- `da_profile_router.md` - DA profile routing
- `encoding.md` - Encoding specifications
- `fri_parameter_guidance.md` - FRI parameter recommendations
- `geom_backend_analysis.md` - Geom backend analysis
- `hssa_vs_kzg_bench.md` - Benchmark comparisons
- `ivc_state_R.md` - IVC state specifications
- `operator_flow.md` - Operator workflow
- `stc_formal.tex` - Formal STC definitions
- `stc_parameter_guidance.md` - STC parameter recommendations
- And more...

#### 📄 Root Level Files
- `INSTRUCTIONS.txt` (82KB) - **Critical cryptographic security guidance**
- `demo_assets/demo_private_key.hex` - Demo key material

---

## Current Working Directory Status

### 25 Modified Files (Uncommitted)

```
M  Dockerfile
M  README.md
M  backends/geom_adapter.py
M  bef_zk/adapter.py
M  bef_zk/capsule/cli/__init__.py
M  bef_zk/capsule/cli/cap_format.py
M  bef_zk/capsule/cli/emit.py
M  bef_zk/capsule/cli/exit_codes.py
M  bef_zk/capsule/cli/verify.py
M  bef_zk/capsule/header.py
M  bef_zk/capsule/payload.py
M  bef_zk/verifier_errors.py
M  capsule_bench/packing.py
M  docs/spec/03_domain_tags.md
M  policies/demo_policy_v1.json
M  pyproject.toml
M  scripts/check_binary_tamper.py
M  scripts/run_pipeline.py
M  scripts/stc_da_sample.py
M  scripts/verify_capsule.py
M  ui/package-lock.json
M  ui/package.json
M  ui/src/App.jsx
M  ui/src/main.jsx
M  ui/src/style.css
```

**Stats**: +2,557 insertions, -331 deletions

---

## Critical Findings from INSTRUCTIONS.txt

The INSTRUCTIONS.txt file (added to HEAD) contains **extensive cryptographic security guidance** that prescribes major architectural changes:

### 🔴 Security Issues Identified

#### 1. **DA Challenge Must Be Relay-Issued**
**Current Problem**: Prover generates DA challenge locally (deterministic seed)
**Required Fix**: Implement 3-phase protocol:
```
Phase A: Commit (register header with relay)
Phase B: Challenge (relay issues signed randomness)
Phase C: Finalize (include challenge in capsule)
```

#### 2. **Verification Levels Must Be Enforceable**
**Current Problem**: Status can be "OK" with failed checks
**Required Fix**: Strict status hierarchy:
- `REJECTED` - proof or header failed
- `PROOF_ONLY` - proof verified, but extras missing/failed
- `POLICY_ENFORCED` - proof + policy + ACL verified
- `FULLY_VERIFIED` - all checks passed (proof + policy + DA + events)

**Rule**: Only `FULLY_VERIFIED` gets a green check

#### 3. **Chunk Handles Must Be Content-Addressed**
**Current Problem**: Handles commit to locations (strings), not content
**Required Fix**: Each handle must include:
```json
{
  "id": 42,
  "uri": "r2://...",
  "sha256": "...",
  "size": 112
}
```

#### 4. **Capsule Hash Must Bind Proof System Parameters**
**Current Problem**: Only row_width is bound
**Required Fix**: Include in capsule hash:
- `vk_hash` (verifier key hash)
- `air_params_hash` (AIR parameters)
- `fri_params_hash` (FRI configuration)
- Full proof system specification

#### 5. **Absolute Paths Must Be Eliminated**
**Current Problem**: Verifier can be tricked into reading arbitrary files
**Required Fix**: All paths must be:
- Relative to pack root
- Content-addressed via manifest
- Sandboxed (reject `..` and absolute paths)

---

## Backend Readiness Gaps (from BACKEND_READINESS_GAPS.md)

The gap analysis identifies **5 major categories** of missing work:

### 1. Crypto/Theory Gaps
- ❌ No parameter → soundness guarantee tables
- ❌ No "recommended configs" for rollup workloads
- ❌ No AoK (Argument of Knowledge) implementation
- ❌ Trace vs arbitrary vector distinction unclear

### 2. IVC/zk Backend Gaps
- ❌ No actual step circuit (SNARK/IVC-friendly)
- ❌ No working IVC recursion loop
- ❌ No constraint-level cost analysis

### 3. Systems + Benchmarks Gaps
- ❌ KZG baseline is CPU-only (unfair comparison)
- ❌ No end-to-end zk stack benchmarks
- ❌ Missing large-N configs and visualization

### 4. DA/Protocol Gaps
- ❌ No concrete network message formats
- ❌ No DA parameter calculator tool
- ❌ No Monte Carlo simulations of DA soundness

### 5. Story/Docs Gaps
- ❌ No single backend architecture diagram
- ❌ No "when to use STC vs alternatives" positioning
- ❌ No worked paper-style example

**Quote from the document:**
> "If you knock out even a toy IVC integration + a parameter table + a short 'backend architecture' doc, you go from 'cool primitive + repo' to 'this is a coherent alternative backend someone can actually evaluate.'"

---

## Recommendations

### Option A: Commit Current Work
If your 25 modified files implement fixes from INSTRUCTIONS.txt:

```bash
# Review what you've changed
git diff --stat

# Commit if satisfied
git add -A
git commit -m "Implement security improvements from architecture review

- Add relay-issued DA challenge protocol
- Enforce strict verification levels
- Implement content-addressed chunk manifests
- Bind proof system parameters to capsule hash
- Eliminate absolute path vulnerabilities"
```

### Option B: Implement Missing Security Fixes
If you haven't yet addressed the INSTRUCTIONS.txt requirements:

**Priority 1 (Security Critical)**:
1. Implement relay-issued DA challenge endpoint
2. Make verification levels enforceable (no green check for partial verification)
3. Eliminate absolute path resolution
4. Bind proof system parameters to capsule hash

**Priority 2 (Backend Readiness)**:
1. Create parameter → soundness tables
2. Build toy IVC integration
3. Write backend architecture doc

**Priority 3 (Production Polish)**:
1. End-to-end benchmarks
2. DA parameter calculator
3. Positioning document

### Option C: Analysis & Planning
If you want help understanding what your uncommitted changes do:

```bash
# Generate detailed diff report
git diff > current_changes.diff

# Review specific high-impact files
git diff bef_zk/capsule/header.py
git diff scripts/run_pipeline.py
git diff bef_zk/capsule/cli/verify.py
```

---

## Next Steps - Choose Your Path

**Path 1**: Review and commit your current work
**Path 2**: Implement remaining security fixes (I can help)
**Path 3**: Deep dive into specific files to understand changes
**Path 4**: Create implementation roadmap for backend readiness gaps

**What would you like to do?**

---

## Quick Reference

### Git Commands
```bash
# See what changed in key files
git diff bef_zk/capsule/header.py
git diff scripts/run_pipeline.py

# See commit history since ctx_cmp/main
git log ctx_cmp/main..HEAD --oneline

# Create a backup before major changes
git branch backup-$(date +%Y%m%d)

# Commit current work
git add -A
git commit -m "Your message here"
```

### Key Files to Review
- `bef_zk/capsule/header.py` - Capsule header structure
- `scripts/run_pipeline.py` - Pipeline execution (DA challenge generation?)
- `scripts/verify_capsule.py` - Verification logic
- `bef_zk/capsule/cli/verify.py` - CLI verification interface

---

**Generated**: 2026-01-23 20:35 EST
**Context**: ctx_cmp/main → HEAD comparison analysis
