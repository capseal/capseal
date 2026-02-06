# Gap Map

## Determinism

This section documents all sources of nondeterminism found in the codebase that could affect verifiable receipt generation. Each instance is classified as 'allowed' or 'not allowed' for receipt generation, with proposed fixes and minimal tests.

### Random Number Generation

#### ALLOWED: Fiat-Shamir Randomness (Deterministic from Commitment)
- **File**: `scripts/stark_hello.py`
- **Symbol**: `random.seed(verifier_seed)` (line 117)
- **Classification**: ALLOWED
- **Reason**: Uses commitment root as seed source (Fiat-Shamir heuristic), making it deterministic
- **Fix**: None needed - already deterministic
- **Test**: Verify that same commitment root produces same query indices

#### NOT ALLOWED: SystemRandom in Test Scripts
- **File**: `scripts/stc_aok.py`
- **Symbol**: `secrets.SystemRandom(seed)` (line 363)
- **Classification**: NOT ALLOWED
- **Reason**: SystemRandom doesn't accept a seed parameter (this is a bug), and even if it did, it uses OS-provided randomness which is non-deterministic
- **Fix**: Replace with `random.Random(seed)` or use `random.seed(seed)` then `random.randrange()`
- **Test**: Run twice with same seed, verify identical output

#### ALLOWED: Seeded PRNG in Rust
- **File**: `nova_stc/src/main.rs`
- **Symbol**: `ChaCha20Rng::seed_from_u64(seed)` (line 1037)
- **Classification**: ALLOWED
- **Reason**: Cryptographic PRNG with explicit seed - deterministic
- **Fix**: None needed
- **Test**: Verify same seed produces same chunks

#### NOT ALLOWED: Math.random() for Run IDs
- **File**: `ui/src/engine/impl/flask.js`
- **Symbol**: `Math.random()` (line 732)
- **Classification**: NOT ALLOWED
- **Reason**: Non-deterministic random for run ID generation
- **Fix**: Use deterministic ID from hash of inputs (e.g., `hashlib.sha256(inputs).hexdigest()[:16]`)
- **Test**: Same inputs should produce same run ID

#### NOT ALLOWED: UUID Generation
- **File**: `bef_zk/capsule/cli/greptile.py`
- **Symbol**: `uuid.uuid4().hex[:8]` (line 660)
- **Classification**: NOT ALLOWED
- **Reason**: UUID4 is non-deterministic
- **Fix**: Use deterministic short ID from hash: `hashlib.sha256(context).hexdigest()[:8]`
- **Test**: Same context should produce same short_id

#### NOT ALLOWED: UUID Generation in MCP Server
- **File**: `bef_zk/capsule/mcp_server.py`
- **Symbol**: `uuid.uuid4().hex[:8]` (line 441)
- **Classification**: NOT ALLOWED
- **Reason**: UUID4 is non-deterministic for agent IDs
- **Fix**: Use deterministic ID from hash of inputs: `hashlib.sha256(json.dumps(inputs)).hexdigest()[:8]`
- **Test**: Same inputs should produce same agent_id

#### NOT ALLOWED: UUID Generation in DA Provider
- **File**: `bef_zk/capsule/da.py`
- **Symbol**: `uuid.uuid4()` (line 39)
- **Classification**: NOT ALLOWED
- **Reason**: UUID4 is non-deterministic for challenge IDs
- **Fix**: Use deterministic challenge ID from hash of capsule commit and payload: `hashlib.sha256(f"{commit_hash}:{payload_hash}").hexdigest()`
- **Test**: Same capsule commit and payload should produce same challenge_id

### Timestamps

#### NOT ALLOWED: Event Timestamps
- **File**: `bef_zk/capsule/event_log.py`
- **Symbol**: `timestamp: float = field(default_factory=time.time)` (line 58)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in events make receipts non-deterministic
- **Fix**: Remove timestamp from hash computation, or use deterministic sequence number
- **Test**: Two identical event sequences should produce same hash chain

#### NOT ALLOWED: Context Pack Timestamps
- **File**: `bef_zk/capsule/context_pack.py`
- **Symbol**: `created_at: float = field(default_factory=time.time)` (line 56)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamp affects context_root hash
- **Fix**: Exclude `created_at` from `context_root` computation (already done in `to_dict()`), but ensure it's not used in hash
- **Test**: Same chunks should produce same context_root regardless of creation time

#### NOT ALLOWED: OracleCall Timestamps
- **File**: `bef_zk/capsule/context_pack.py`
- **Symbol**: `started_at: float = field(default_factory=time.time)` (line 174), `completed_at = time.time()` (lines 185, 193)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in OracleCall affect receipt determinism if included in hash
- **Fix**: Exclude timing fields from hash computation, or use deterministic sequence numbers
- **Test**: Same oracle call should produce same hash regardless of timing

#### NOT ALLOWED: Orchestrator Session IDs
- **File**: `bef_zk/capsule/orchestrator.py`
- **Symbol**: `session_id or f"session_{int(time.time())}"` (line 83)
- **Classification**: NOT ALLOWED
- **Reason**: Time-based session IDs are non-deterministic
- **Fix**: Use deterministic ID from hash of inputs: `hashlib.sha256(json.dumps(inputs)).hexdigest()[:16]`
- **Test**: Same inputs should produce same session_id

#### NOT ALLOWED: Postgres Event Sequence Generation
- **File**: `server/event_store_pg.py`
- **Symbol**: `seq = int(time.time() * 1_000_000)` (lines 55, 60)
- **Classification**: NOT ALLOWED
- **Reason**: Time-based sequence numbers are non-deterministic
- **Fix**: Use deterministic sequence counter or hash-based sequence
- **Test**: Same events should produce same sequence numbers

#### NOT ALLOWED: Rust Timestamp Functions
- **File**: `bef_rust/src/capseal_core.rs`
- **Symbol**: `chrono_now_iso()` using `SystemTime::now()` (line 785)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in verification reports make them non-deterministic
- **Fix**: Use fixed timestamp or exclude from hash computation
- **Test**: Same verification should produce same report hash

#### NOT ALLOWED: Tauri Timestamp Functions
- **File**: `ui/src-tauri/src/lib.rs`
- **Symbol**: `chrono_now()` using `SystemTime::now()` (line 402)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in run summaries are non-deterministic
- **Fix**: Use deterministic timestamp or exclude from hash
- **Test**: Same run should produce same summary hash

#### NOT ALLOWED: Manifest Timestamps
- **File**: `capsule_bench/manifests.py`
- **Symbol**: `datetime.now(timezone.utc).isoformat()` (line 36)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in manifests make them non-deterministic
- **Fix**: Exclude timestamp from hash computation or use fixed value
- **Test**: Same manifest content should produce same hash

#### NOT ALLOWED: Doctor Report Timestamps
- **File**: `bef_zk/capsule/cli/doctor.py`
- **Symbol**: `datetime.now(timezone.utc).isoformat()` (line 155)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in doctor reports make them non-deterministic
- **Fix**: Exclude timestamp from hash computation or use fixed value
- **Test**: Same report content should produce same hash

#### NOT ALLOWED: Policy Openings Timestamps
- **File**: `bef_zk/capsule/policy_openings.py`
- **Symbol**: `datetime.now(tz=timezone.utc).isoformat()` (lines 129, 272), `datetime.now(tz=timezone.utc).timestamp()` (line 166)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in policy openings affect determinism
- **Fix**: Exclude timestamps from hash computation or use deterministic sequence numbers
- **Test**: Same policy opening should produce same hash regardless of time

#### NOT ALLOWED: Cap Format Timestamps
- **File**: `bef_zk/capsule/cli/cap_format.py`
- **Symbol**: `datetime.now(timezone.utc).isoformat()` (line 252)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in cap format output make it non-deterministic
- **Fix**: Exclude timestamp from hash computation or use fixed value
- **Test**: Same cap content should produce same hash

#### NOT ALLOWED: Orchestrator Event Timestamps
- **File**: `bef_zk/capsule/orchestrator.py`
- **Symbol**: `ts_ms: _now_ms()` (line 94), `time.strftime()` (line 206)
- **Classification**: NOT ALLOWED
- **Reason**: Timestamps in orchestrator events affect hash chain determinism
- **Fix**: Exclude timestamp from hash computation or use deterministic sequence numbers
- **Test**: Same event sequence should produce same hash chain

### Filesystem Ordering

#### NOT ALLOWED: Directory Listing Without Sort
- **File**: `bef_zk/capsule/cli/merge.py`
- **Symbol**: `os.walk(base_dir)` (line 153)
- **Classification**: NOT ALLOWED
- **Reason**: Filesystem ordering is platform/filesystem-dependent
- **Fix**: Sort collected files: `sorted(collect_files(base_dir))`
- **Test**: Same directory should produce same file order on different filesystems

#### NOT ALLOWED: Greptile Directory Walking
- **File**: `bef_zk/capsule/cli/greptile.py`
- **Symbol**: `os.walk(local_path)` (lines 246, 524)
- **Classification**: NOT ALLOWED
- **Reason**: Filesystem ordering is non-deterministic
- **Fix**: Sort files after collection: `sorted(os.walk(...))` or collect then sort
- **Test**: Same directory should produce same file order

#### NOT ALLOWED: Rust Directory Reading
- **File**: `bef_rust/src/capseal_core.rs`
- **Symbol**: `std::fs::read_dir(run_dir)` (line 608)
- **Classification**: NOT ALLOWED
- **Reason**: Directory entry order is non-deterministic
- **Fix**: Collect entries, sort by name: `entries.sort_by(|a, b| a.file_name().cmp(b.file_name()))`
- **Test**: Same directory should produce same artifact list order

#### NOT ALLOWED: Tauri Directory Reading
- **File**: `ui/src-tauri/src/lib.rs`
- **Symbol**: `fs::read_dir(&path)` (line 340)
- **Classification**: NOT ALLOWED
- **Reason**: Directory entry order affects run list ordering
- **Fix**: Sort entries: `runs.sort_by(|a, b| a.run_id.cmp(&b.run_id))` (already done by created_at, but should sort by run_id for determinism)
- **Test**: Same runs directory should produce same list order

#### ALLOWED: Git ls-files Fallback
- **File**: `bef_zk/capsule/cli/merge.py`
- **Symbol**: `tracked_files()` using git ls-files (line 146)
- **Classification**: ALLOWED
- **Reason**: Git provides deterministic ordering
- **Fix**: None needed
- **Test**: Verify git ls-files produces consistent order

#### ALLOWED: Sorted Directory Listing
- **File**: `server/relay.py`
- **Symbol**: `sorted(p for p in root.rglob("*"))` (line 373)
- **Classification**: ALLOWED
- **Reason**: Files are explicitly sorted, ensuring deterministic ordering
- **Fix**: None needed
- **Test**: Same directory should produce same sorted order

### Concurrency / Async Races

#### NOT ALLOWED: ThreadPoolExecutor with as_completed
- **File**: `bef_zk/capsule/cli/shell.py`
- **Symbol**: `as_completed(futures)` (line 1288)
- **Classification**: NOT ALLOWED
- **Reason**: Completion order is non-deterministic, affects result ordering
- **Fix**: Collect results in submission order: `[f.result() for f in futures]` instead of `as_completed()`
- **Test**: Same tasks should produce results in same order

#### NOT ALLOWED: Parallel Merge Orchestration
- **File**: `bef_zk/capsule/cli/merge_orchestrate.py`
- **Symbol**: `as_completed(futures)` (line 215)
- **Classification**: NOT ALLOWED
- **Reason**: Completion order affects merged file order
- **Fix**: Process results in submission order: `[f.result() for f in futures]`
- **Test**: Same batch should produce merged files in same order

#### ALLOWED: Threading for Background Tasks
- **File**: `server/flask_app/jobs.py`, `server/flask_app/storage.py`
- **Symbol**: `threading.Thread` for background GC (lines 164, 42)
- **Classification**: ALLOWED
- **Reason**: Background tasks don't affect receipt generation determinism
- **Fix**: None needed
- **Test**: N/A - background tasks

#### ALLOWED: Parallel Processing in Rust (Deterministic)
- **File**: `bef_rust/src/lib.rs`
- **Symbol**: `into_par_iter()` for chunk processing (line 270)
- **Classification**: ALLOWED
- **Reason**: Parallel processing is deterministic if order is preserved in final collection
- **Fix**: None needed - results are collected deterministically
- **Test**: Same inputs should produce same results regardless of thread count

### Network Calls During Receipt Generation

#### NOT ALLOWED: DA Challenge Fetch from Relay
- **File**: `scripts/run_pipeline.py`
- **Symbol**: `_fetch_da_challenge_from_relay()` (line 255)
- **Classification**: NOT ALLOWED
- **Reason**: Network calls during receipt generation introduce non-determinism
- **Fix**: Fetch DA challenge before receipt generation, or use deterministic challenge generation
- **Test**: Same capsule should produce same DA challenge without network call

#### NOT ALLOWED: Greptile API Calls
- **File**: `bef_zk/capsule/cli/greptile.py`
- **Symbol**: `_api_request()` (line 404)
- **Classification**: NOT ALLOWED
- **Reason**: Network calls during receipt generation are non-deterministic
- **Fix**: Use offline mode or cache results deterministically
- **Test**: Same query should produce same result without network call

#### NOT ALLOWED: MCP Server API Calls
- **File**: `bef_zk/capsule/mcp_server.py`
- **Symbol**: `_call_anthropic_api()`, `_call_gemini_api()`, `_call_openai_api()` (lines 312, 349, 383)
- **Classification**: NOT ALLOWED
- **Reason**: Network calls during receipt generation are non-deterministic
- **Fix**: Use offline mode, cache results deterministically, or record API responses in receipt
- **Test**: Same inputs should produce same receipt without network calls

#### NOT ALLOWED: Audit URL Fetching
- **File**: `bef_zk/capsule/cli/audit.py`
- **Symbol**: `urllib.request.urlopen(url)` (line 424)
- **Classification**: NOT ALLOWED
- **Reason**: Network calls during receipt generation are non-deterministic
- **Fix**: Cache results deterministically or record fetched content in receipt
- **Test**: Same URL should produce same result without network call

### Environment Variables

#### NOT ALLOWED: Environment-Dependent Behavior
- **File**: `bef_zk/capsule/cli/greptile.py`
- **Symbol**: `os.environ.get("GREPTILE_API_KEY")` (line 360)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Environment variables can change behavior between runs
- **Fix**: Record environment state in receipt, or use deterministic defaults
- **Test**: Same inputs with different env vars should produce same receipt (or record env in receipt)

#### NOT ALLOWED: Environment-Dependent Config
- **File**: `bef_zk/capsule/cli/metrics.py`
- **Symbol**: `os.environ.get("CAPSEAL_MAX_FILES", "25")` (line 29)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Environment-dependent limits affect receipt content
- **Fix**: Record environment config in receipt metadata
- **Test**: Same inputs with different env should produce same receipt content (or record env)

#### NOT ALLOWED: Environment-Dependent Verify Config
- **File**: `bef_zk/capsule/cli/verify.py`
- **Symbol**: `os.environ.get("CAP_MAX_PROOF_BYTES", ...)` (line 36)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Environment-dependent proof size limits affect receipt content
- **Fix**: Record environment config in receipt metadata
- **Test**: Same inputs with different env should produce same receipt content (or record env)

#### NOT ALLOWED: Environment-Dependent Workspace Root
- **File**: `bef_zk/capsule/cli/utils.py`
- **Symbol**: `os.environ.get("CAPSEAL_WORKSPACE_ROOT", ...)` (line 69)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Different workspace roots can affect file paths in receipts
- **Fix**: Normalize paths relative to a fixed root or record workspace root in receipt
- **Test**: Same inputs with different workspace roots should produce same receipt (or record root)

#### NOT ALLOWED: Environment-Dependent Sandbox Config
- **File**: `bef_zk/sandbox/backend.py`
- **Symbol**: `os.environ.get("PATH", ...)`, `os.environ.get("HOME", ...)` (lines 172-175), `dict(os.environ)` (line 179)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Environment variables can affect sandbox execution and receipt content
- **Fix**: Record environment state in receipt metadata or use deterministic minimal env
- **Test**: Same inputs with different env should produce same receipt (or record env)

#### NOT ALLOWED: Environment-Dependent MCP Server Config
- **File**: `bef_zk/capsule/mcp_server.py`
- **Symbol**: `os.environ.get("CAPSEAL_BIN", ...)`, `os.environ.get("CAPSEAL_WORKSPACE_ROOT", ...)` (lines 38-41)
- **Classification**: NOT ALLOWED (if affects receipt)
- **Reason**: Environment-dependent paths affect receipt content
- **Fix**: Record environment config in receipt metadata or use fixed paths
- **Test**: Same inputs with different env should produce same receipt (or record env)

#### ALLOWED: Environment for Configuration Only
- **File**: `server/flask_app/__init__.py`
- **Symbol**: Various `os.environ.get()` calls (lines 32-61)
- **Classification**: ALLOWED
- **Reason**: Server configuration doesn't affect receipt generation determinism
- **Fix**: None needed
- **Test**: N/A - server config

### Floating Point / Platform Differences

#### POTENTIALLY NOT ALLOWED: Floating Point Operations
- **Files**: Multiple files with floating point math
- **Symbols**: Various floating point operations in Rust CUDA kernels, Python numpy operations
- **Classification**: POTENTIALLY NOT ALLOWED
- **Reason**: Floating point precision can vary across platforms/architectures
- **Fix**: Use fixed-point arithmetic or record platform info in receipt
- **Test**: Same inputs should produce same floating point results across platforms

**Note**: Most floating point operations appear to be in computation backends (FusionAlpha, BICEP) which may be acceptable if they're part of the computation being proven, not the receipt generation itself. However, any floating point operations that affect receipt hashes need to be made deterministic.

### Summary

**Critical Issues (Must Fix for Receipt Determinism):**
1. Timestamps in event logs, context packs, orchestrator events, policy openings, manifests
2. Non-deterministic RNG (Math.random, uuid4, SystemRandom) in multiple locations
3. Filesystem ordering without canonical sort (os.walk, read_dir)
4. Concurrent execution result ordering (as_completed() in shell.py and merge_orchestrate.py)
5. Network calls during receipt generation (Greptile API, MCP server APIs, DA challenge fetch)

**Medium Priority:**
1. Environment variable dependencies (workspace roots, config limits, sandbox env)
2. Floating point platform differences (mostly in computation backends, acceptable if not in receipt hash)

**Low Priority / Already Handled:**
1. Fiat-Shamir randomness (deterministic from commitment)
2. Seeded PRNGs (already deterministic - ChaCha20Rng, random.Random with seed)
3. Background threading (doesn't affect receipts)
4. Deterministic parallel processing (Rust par_iter with ordered collection)
5. Sorted directory listings (already deterministic)
