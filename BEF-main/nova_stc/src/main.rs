#![allow(clippy::needless_borrow)]

use nova_stc::stc::{update_state, StcChunk, StcParams, StcState, StcStepCircuit};
use nova_stc::Fp;

type E1 = PallasEngine;
type E2 = VestaEngine;
type EE1 = ipa_pc::EvaluationEngine<E1>;
type EE2 = ipa_pc::EvaluationEngine<E2>;
type S1 = RelaxedR1CSSNARK<E1, EE1>;
type S2 = RelaxedR1CSSNARK<E2, EE2>;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Nova IVC demo + benchmark harness for the STC accumulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single proof using chunk data from disk
    Prove(ProveArgs),
    /// Run a benchmark sweep over different parameter choices / seeds
    Bench(BenchArgs),
    /// Describe circuit characteristics (constraints, variables) for a parameter choice
    Describe(DescribeArgs),
}

#[derive(Args, Debug)]
struct ProveArgs {
    /// Path to JSON file containing an array of chunk arrays
    #[arg(long)]
    chunks: PathBuf,

    /// Number of sketch challenges (m)
    #[arg(long, default_value_t = 2)]
    challenges: usize,

    /// Optional number of steps to execute (defaults to all chunks)
    #[arg(long)]
    steps: Option<usize>,

    /// Also produce a compressed SNARK and report its size/timings
    #[arg(long, default_value_t = false)]
    compressed: bool,

    /// Optional path to dump execution stats as JSON
    #[arg(long)]
    stats_out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct BenchArgs {
    /// Optional path to a JSON chunk file; otherwise random data is generated
    #[arg(long)]
    chunks: Option<PathBuf>,

    /// Number of chunks to generate when no chunk file is provided
    #[arg(long, default_value_t = 256)]
    num_chunks: usize,

    /// Chunk length to use for synthetic random traces
    #[arg(long, default_value_t = 8)]
    chunk_len: usize,

    /// Comma separated list of challenge counts to benchmark (default: 2,4,8)
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    challenges: Vec<usize>,

    /// Number of repetitions per challenge count
    #[arg(long, default_value_t = 3)]
    repeats: usize,

    /// Seed used when generating random chunks
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Optional path to dump benchmark results as JSON
    #[arg(long)]
    output: Option<PathBuf>,

    /// Also produce compressed SNARK statistics (slower)
    #[arg(long, default_value_t = false)]
    compressed: bool,
}

#[derive(Args, Debug)]
struct DescribeArgs {
    /// Number of sketch challenges (m)
    #[arg(long)]
    challenges: usize,

    /// Chunk length to describe
    #[arg(long, default_value_t = 8)]
    chunk_len: usize,
}

#[derive(Clone, Debug)]
struct ExecutionStats {
    plain_state: StcState<Fp>,
    nova_state: StcState<Fp>,
    timings: TimingSummary,
    recursive_proof_bytes: usize,
    steps: usize,
    chunk_len: usize,
    constraints: (usize, usize),
    pp_cache_hit: bool,
    compressed: Option<CompressedStats>,
    trace_id: Option<String>,
}

struct ProofBuild {
    plain_state: StcState<Fp>,
    nova_state: StcState<Fp>,
    recursive_snark: RecursiveSNARK<E1, E2, StcStepCircuit<Fp>>,
    pp: Arc<PublicParams<E1, E2, StcStepCircuit<Fp>>>,
    z0: Vec<Fp>,
    timings: TimingSummary,
    constraints: (usize, usize),
    steps: usize,
    pp_cache_hit: bool,
}

#[derive(Clone, Debug)]
struct TimingSummary {
    plain: Duration,
    pp: Duration,
    base: Duration,
    prove_total: Duration,
    verify: Duration,
}

#[derive(Clone, Debug)]
struct CompressedStats {
    setup: Duration,
    prove: Duration,
    verify: Duration,
    proof_bytes: usize,
}

#[derive(Clone)]
struct ChunkBatch {
    chunks: Vec<StcChunk<Fp>>,
    chunk_len: usize,
    trace_id: Option<String>,
}

impl ChunkBatch {
    fn from_chunks(chunks: Vec<StcChunk<Fp>>, chunk_len: usize) -> Self {
        ChunkBatch {
            chunks,
            chunk_len,
            trace_id: None,
        }
    }

    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    fn slice_prefix(&self, limit: usize) -> ChunkBatch {
        let take = limit.min(self.chunks.len());
        ChunkBatch {
            chunks: self.chunks[..take].to_vec(),
            chunk_len: self.chunk_len,
            trace_id: self.trace_id.clone(),
        }
    }
}

#[derive(Clone, Serialize)]
struct BenchRecord {
    run_id: usize,
    challenge_count: usize,
    num_chunks: usize,
    chunk_len: usize,
    seed: u64,
    timings_ms: TimingRecord,
    proof_bytes: usize,
    compressed: Option<CompressedRecord>,
    final_n: String,
    final_root: String,
    pp_cache_hit: bool,
}

#[derive(Clone, Serialize)]
struct TimingRecord {
    plain: f64,
    pp: f64,
    base: f64,
    prove_total: f64,
    prove_avg: f64,
    verify: f64,
}

#[derive(Clone, Serialize)]
struct CompressedRecord {
    setup: f64,
    prove: f64,
    verify: f64,
    proof_bytes: usize,
}

#[derive(Default, Serialize)]
struct BenchOutput {
    runs: Vec<BenchRecord>,
    summary: Vec<BenchSummary>,
}

#[derive(Clone, Serialize)]
struct BenchSummary {
    challenge_count: usize,
    num_chunks: usize,
    chunk_len: usize,
    repeats: usize,
    constraints_primary: usize,
    constraints_secondary: usize,
    plain_ms: SummaryStats,
    pp_ms: SummaryStats,
    base_ms: SummaryStats,
    prove_total_ms: SummaryStats,
    prove_avg_ms: SummaryStats,
    verify_ms: SummaryStats,
    proof_bytes: SummaryStats,
    compressed: Option<CompressedSummary>,
}

#[derive(Clone, Serialize)]
struct CompressedSummary {
    setup_ms: SummaryStats,
    prove_ms: SummaryStats,
    verify_ms: SummaryStats,
    proof_bytes: SummaryStats,
}

#[derive(Clone, Serialize)]
struct SummaryStats {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

#[derive(Default, Clone)]
struct RunningStats {
    count: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn add(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn to_summary(&self) -> SummaryStats {
        let mean = if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        };
        let variance = if self.count <= 1 {
            0.0
        } else {
            (self.sum_sq / self.count as f64) - mean * mean
        };
        SummaryStats {
            mean,
            stddev: variance.max(0.0).sqrt(),
            min: if self.count == 0 { 0.0 } else { self.min },
            max: if self.count == 0 { 0.0 } else { self.max },
        }
    }
}

#[derive(Default)]
struct SummaryAccumulator {
    challenge_count: usize,
    num_chunks: usize,
    chunk_len: usize,
    repeats: usize,
    constraints: (usize, usize),
    plain: RunningStats,
    pp: RunningStats,
    base: RunningStats,
    prove_total: RunningStats,
    prove_avg: RunningStats,
    verify: RunningStats,
    proof_bytes: RunningStats,
    compressed: Option<CompressedAccumulator>,
}

#[derive(Default, Clone)]
struct CompressedAccumulator {
    setup: RunningStats,
    prove: RunningStats,
    verify: RunningStats,
    proof_bytes: RunningStats,
}

impl SummaryAccumulator {
    fn ensure_metadata(
        &mut self,
        challenges: usize,
        num_chunks: usize,
        chunk_len: usize,
        constraints: (usize, usize),
    ) {
        if self.repeats == 0 {
            self.challenge_count = challenges;
            self.num_chunks = num_chunks;
            self.chunk_len = chunk_len;
            self.constraints = constraints;
        }
        self.repeats += 1;
    }

    fn into_summary(self) -> BenchSummary {
        BenchSummary {
            challenge_count: self.challenge_count,
            num_chunks: self.num_chunks,
            chunk_len: self.chunk_len,
            repeats: self.repeats,
            constraints_primary: self.constraints.0,
            constraints_secondary: self.constraints.1,
            plain_ms: self.plain.to_summary(),
            pp_ms: self.pp.to_summary(),
            base_ms: self.base.to_summary(),
            prove_total_ms: self.prove_total.to_summary(),
            prove_avg_ms: self.prove_avg.to_summary(),
            verify_ms: self.verify.to_summary(),
            proof_bytes: self.proof_bytes.to_summary(),
            compressed: self.compressed.map(|c| CompressedSummary {
                setup_ms: c.setup.to_summary(),
                prove_ms: c.prove.to_summary(),
                verify_ms: c.verify.to_summary(),
                proof_bytes: c.proof_bytes.to_summary(),
            }),
        }
    }
}

struct PpCache {
    entries: HashMap<(usize, usize), Arc<PublicParams<E1, E2, StcStepCircuit<Fp>>>>,
}

impl PpCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(
        &mut self,
        params: &Arc<StcParams<Fp>>,
    ) -> Result<(
        Arc<PublicParams<E1, E2, StcStepCircuit<Fp>>>,
        bool,
        Duration,
    )> {
        let key = (params.challenges_len(), params.chunk_len);
        if let Some(pp) = self.entries.get(&key) {
            return Ok((pp.clone(), true, Duration::ZERO));
        }
        let ck_hint1 = &*default_ck_hint::<E1>();
        let ck_hint2 = &*default_ck_hint::<E2>();
        let blank_circuit = StcStepCircuit::blank(params.clone());
        let start = Instant::now();
        let pp = PublicParams::<E1, E2, _>::setup(&blank_circuit, ck_hint1, ck_hint2)?;
        let elapsed = start.elapsed();
        let arc = Arc::new(pp);
        self.entries.insert(key, arc.clone());
        Ok((arc, false, elapsed))
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();
    match cli.command {
        Commands::Prove(args) => run_prove(args),
        Commands::Bench(args) => run_bench(args),
        Commands::Describe(args) => run_describe(args),
    }
}

fn run_prove(args: ProveArgs) -> Result<()> {
    let mut batch = load_chunk_batch(&args.chunks)?;
    ensure!(batch.chunk_count() > 0, "no chunks found in input file");

    if let Some(limit) = args.steps {
        ensure!(
            limit > 0 && limit <= batch.chunk_count(),
            "invalid step count"
        );
        batch = batch.slice_prefix(limit);
    }

    let params = Arc::new(StcParams::new(
        derive_challenges(args.challenges),
        batch.chunk_len,
    ));
    let mut cache = PpCache::new();
    let stats = execute_proof(params, &batch, &mut cache, args.compressed)?;
    ensure!(
        stats.nova_state == stats.plain_state,
        "Nova proof output does not match software STC update"
    );

    println!(
        "Nova proof verified: steps={}, n_final={:?}, root_final={:?}",
        stats.steps, stats.plain_state.n, stats.plain_state.root
    );
    println!("s_final={:?}", stats.plain_state.s);
    println!("pow_final={:?}", stats.plain_state.pow);
    println!(
        "timings_ms: plain={:.3}, pp={:.3}, base={:.3}, prove_total={:.3}, prove_avg={:.3}, verify={:.3}, proof_bytes={}",
        ms(stats.timings.plain),
        ms(stats.timings.pp),
        ms(stats.timings.base),
        ms(stats.timings.prove_total),
        ms(stats.timings.prove_total) / stats.steps as f64,
        ms(stats.timings.verify),
        stats.recursive_proof_bytes
    );
    if let Some(compressed) = &stats.compressed {
        println!(
            "compressed_ms: setup={:.3}, prove={:.3}, verify={:.3}, proof_bytes={}",
            ms(compressed.setup),
            ms(compressed.prove),
            ms(compressed.verify),
            compressed.proof_bytes
        );
    }

    if let Some(path) = args.stats_out {
        let file = File::create(&path)?;
        serde_json::to_writer_pretty(file, &stats_to_json(&stats))?;
        println!("Wrote stats to {}", path.display());
    }

    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<()> {
    ensure!(args.repeats > 0, "repeats must be non-zero");
    let challenge_sets = if args.challenges.is_empty() {
        DEFAULT_BENCH_CHALLENGES.to_vec()
    } else {
        args.challenges.clone()
    };

    let chunk_source = if let Some(path) = args.chunks {
        let batch = load_chunk_batch(&path)?;
        ensure!(batch.chunk_count() > 0, "chunk file has no entries");
        ChunkSource::File(batch)
    } else {
        ensure!(
            args.num_chunks > 0,
            "num_chunks must be > 0 when generating random data"
        );
        ensure!(
            args.chunk_len > 0,
            "chunk_len must be > 0 when generating random data"
        );
        ChunkSource::Random {
            num_chunks: args.num_chunks,
            chunk_len: args.chunk_len,
        }
    };

    let mut records = Vec::new();
    let mut run_id = 0usize;
    let mut cache = PpCache::new();
    let mut summary_map: HashMap<(usize, usize), SummaryAccumulator> = HashMap::new();

    for &m in &challenge_sets {
        for rep in 0..args.repeats {
            let seed = args.seed + run_id as u64 + rep as u64;
            let batch = chunk_source.materialize(seed);

            let params = Arc::new(StcParams::new(derive_challenges(m), batch.chunk_len));
            let stats = execute_proof(params, &batch, &mut cache, args.compressed)?;
            ensure!(
                stats.nova_state == stats.plain_state,
                "Nova state mismatch in bench run"
            );

            let compressed_opt = stats.compressed.clone();
            let compressed_record = compressed_opt.as_ref().map(|c| CompressedRecord {
                setup: ms(c.setup),
                prove: ms(c.prove),
                verify: ms(c.verify),
                proof_bytes: c.proof_bytes,
            });

            let record = BenchRecord {
                run_id,
                challenge_count: m,
                num_chunks: batch.chunk_count(),
                chunk_len: stats.chunk_len,
                seed,
                timings_ms: TimingRecord {
                    plain: ms(stats.timings.plain),
                    pp: ms(stats.timings.pp),
                    base: ms(stats.timings.base),
                    prove_total: ms(stats.timings.prove_total),
                    prove_avg: ms(stats.timings.prove_total) / stats.steps as f64,
                    verify: ms(stats.timings.verify),
                },
                proof_bytes: stats.recursive_proof_bytes,
                compressed: compressed_record.clone(),
                final_n: format_field(&stats.plain_state.n),
                final_root: format_field(&stats.plain_state.root),
                pp_cache_hit: stats.pp_cache_hit,
            };

            println!(
                "bench run {run_id}: m={m}, chunks={}, proof_bytes={}, plain_ms={:.2}, pp_ms={:.2}, prove_total_ms={:.2}, verify_ms={:.2}",
                batch.chunk_count(),
                stats.recursive_proof_bytes,
                record.timings_ms.plain,
                record.timings_ms.pp,
                record.timings_ms.prove_total,
                record.timings_ms.verify,
            );

            records.push(record);

            let key = (m, batch.chunk_count());
            let acc = summary_map.entry(key).or_default();
            acc.ensure_metadata(m, batch.chunk_count(), batch.chunk_len, stats.constraints);
            acc.plain.add(ms(stats.timings.plain));
            acc.pp.add(ms(stats.timings.pp));
            acc.base.add(ms(stats.timings.base));
            acc.prove_total.add(ms(stats.timings.prove_total));
            acc.prove_avg
                .add(ms(stats.timings.prove_total) / stats.steps as f64);
            acc.verify.add(ms(stats.timings.verify));
            acc.proof_bytes.add(stats.recursive_proof_bytes as f64);
            if let Some(comp) = compressed_opt {
                let entry = acc
                    .compressed
                    .get_or_insert_with(CompressedAccumulator::default);
                entry.setup.add(ms(comp.setup));
                entry.prove.add(ms(comp.prove));
                entry.verify.add(ms(comp.verify));
                entry.proof_bytes.add(comp.proof_bytes as f64);
            }

            run_id += 1;
        }
    }

    let mut summary: Vec<_> = summary_map
        .into_iter()
        .map(|(_key, acc)| acc.into_summary())
        .collect();
    summary.sort_by_key(|s| (s.challenge_count, s.num_chunks));

    if let Some(path) = args.output {
        let file = File::create(path)?;
        let output = BenchOutput {
            runs: records.clone(),
            summary: summary.clone(),
        };
        serde_json::to_writer_pretty(file, &output)?;
    }

    println!("\nBenchmark summary:");
    for entry in &summary {
        println!(
            "m={}, chunks={}, plain_mean_ms={:.2}, prove_avg_mean_ms={:.2}, verify_mean_ms={:.2}",
            entry.challenge_count,
            entry.num_chunks,
            entry.plain_ms.mean,
            entry.prove_avg_ms.mean,
            entry.verify_ms.mean
        );
    }

    Ok(())
}

fn run_describe(args: DescribeArgs) -> Result<()> {
    ensure!(args.chunk_len > 0, "chunk_len must be > 0");
    let params = Arc::new(StcParams::new(
        derive_challenges(args.challenges),
        args.chunk_len,
    ));
    let mut cache = PpCache::new();
    let (pp, cache_hit, setup_time) = cache.get(&params)?;
    let (constraints_primary, constraints_secondary) = pp.num_constraints();
    let (vars_primary, vars_secondary) = pp.num_variables();
    println!("Nova STC circuit description (m = {}):", args.challenges);
    println!(
        "  cached_pp = {} (setup {:.3} ms)",
        cache_hit,
        ms(setup_time)
    );
    println!("  constraints_primary   = {}", constraints_primary);
    println!("  constraints_secondary = {}", constraints_secondary);
    println!("  variables_primary     = {}", vars_primary);
    println!("  variables_secondary   = {}", vars_secondary);
    println!(
        "  state_arity           = {}",
        StcState::<Fp>::initial(params.challenges_len())
            .to_vec()
            .len()
    );
    Ok(())
}

fn execute_proof(
    params: Arc<StcParams<Fp>>,
    batch: &ChunkBatch,
    cache: &mut PpCache,
    compress: bool,
) -> Result<ExecutionStats> {
    ensure!(
        batch.chunk_len == params.chunk_len,
        "chunk_len mismatch in execute_proof"
    );
    let build = build_recursive_snark(params, batch, cache)?;
    let ProofBuild {
        plain_state,
        nova_state,
        recursive_snark,
        pp,
        z0,
        timings,
        constraints,
        steps,
        pp_cache_hit,
    } = build;

    let recursive_bytes = bincode::options()
        .with_fixint_encoding()
        .serialize(&recursive_snark)?
        .len();

    let compressed_stats = if compress {
        Some(compress_proof(&pp, &recursive_snark, steps, &z0)?)
    } else {
        None
    };

    Ok(ExecutionStats {
        plain_state,
        nova_state,
        timings,
        recursive_proof_bytes: recursive_bytes,
        steps,
        chunk_len: batch.chunk_len,
        constraints,
        pp_cache_hit,
        compressed: compressed_stats,
        trace_id: batch.trace_id.clone(),
    })
}

fn build_recursive_snark(
    params: Arc<StcParams<Fp>>,
    batch: &ChunkBatch,
    cache: &mut PpCache,
) -> Result<ProofBuild> {
    ensure!(batch.chunk_count() > 0, "no chunks available for execution");

    let plain_start = Instant::now();
    let mut plain_state = StcState::initial(params.challenges_len());
    for chunk in &batch.chunks {
        plain_state = update_state(&params, &plain_state, chunk);
    }
    let plain_time = plain_start.elapsed();

    let circuits: Vec<StcStepCircuit<Fp>> = batch
        .chunks
        .iter()
        .cloned()
        .map(|chunk| StcStepCircuit::new(params.clone(), chunk))
        .collect();

    let initial_state = StcState::initial(params.challenges_len());
    let z0 = initial_state.to_vec();

    let (pp, cache_hit, pp_time) = cache.get(&params)?;

    let base_start = Instant::now();
    let mut recursive_snark = RecursiveSNARK::<E1, E2, _>::new(&pp, &circuits[0], &z0)?;
    let base_time = base_start.elapsed();

    let mut prove_total = Duration::ZERO;
    for circuit in circuits.iter() {
        let step_start = Instant::now();
        recursive_snark.prove_step(&pp, circuit)?;
        prove_total += step_start.elapsed();
    }

    let verify_start = Instant::now();
    let z_final = recursive_snark.verify(&pp, circuits.len(), &z0)?;
    let verify_time = verify_start.elapsed();
    let nova_state = StcState::from_vec(&z_final).ok_or_else(|| eyre!("malformed Nova output"))?;

    let timings = TimingSummary {
        plain: plain_time,
        pp: pp_time,
        base: base_time,
        prove_total,
        verify: verify_time,
    };

    let constraints = pp.num_constraints();

    Ok(ProofBuild {
        plain_state,
        nova_state,
        recursive_snark,
        pp,
        z0,
        timings,
        constraints,
        steps: circuits.len(),
        pp_cache_hit: cache_hit,
    })
}

fn compress_proof(
    pp: &PublicParams<E1, E2, StcStepCircuit<Fp>>,
    recursive_snark: &RecursiveSNARK<E1, E2, StcStepCircuit<Fp>>,
    steps: usize,
    z0: &[Fp],
) -> Result<CompressedStats> {
    let setup_start = Instant::now();
    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(pp)?;
    let setup_time = setup_start.elapsed();

    let prove_start = Instant::now();
    let compressed = CompressedSNARK::<_, _, _, S1, S2>::prove(pp, &pk, recursive_snark)?;
    let prove_time = prove_start.elapsed();

    let verify_start = Instant::now();
    compressed.verify(&vk, steps, z0)?;
    let verify_time = verify_start.elapsed();

    let proof_bytes = bincode::options()
        .with_fixint_encoding()
        .serialize(&compressed)?
        .len();

    Ok(CompressedStats {
        setup: setup_time,
        prove: prove_time,
        verify: verify_time,
        proof_bytes,
    })
}

fn load_chunk_batch(path: &PathBuf) -> Result<ChunkBatch> {
    let reader = File::open(path)?;
    let json_val: Value = serde_json::from_reader(reader)?;
    if json_val
        .get("schema")
        .and_then(|s| s.as_str())
        .map(|s| s == "bef_trace_v1")
        .unwrap_or(false)
    {
        parse_bef_trace(json_val)
    } else {
        parse_simple_chunks(json_val)
    }
}

#[derive(Deserialize)]
struct BefTraceChunk {
    chunk_index: usize,
    offset: usize,
    values: Vec<Value>,
}

#[derive(Deserialize)]
struct BefTraceFile {
    trace_id: Option<String>,
    vector_length: usize,
    chunk_length: usize,
    chunks: Vec<BefTraceChunk>,
}

fn parse_bef_trace(val: Value) -> Result<ChunkBatch> {
    let trace: BefTraceFile = serde_json::from_value(val)?;
    ensure!(trace.chunk_length > 0, "bef_trace chunk_length must be > 0");
    let mut chunks = Vec::with_capacity(trace.chunks.len());
    let mut expected_offset = 0usize;
    for chunk in trace.chunks {
        ensure!(
            chunk.chunk_index == chunks.len(),
            "chunk_index mismatch in bef_trace"
        );
        ensure!(
            chunk.offset == expected_offset,
            "chunk offset mismatch in bef_trace"
        );
        ensure!(
            !chunk.values.is_empty(),
            "chunk {} has no values",
            chunk.chunk_index
        );
        ensure!(
            chunk.values.len() == trace.chunk_length,
            "chunk {} has len {} but expected {}",
            chunk.chunk_index,
            chunk.values.len(),
            trace.chunk_length
        );
        let mut parsed = Vec::with_capacity(chunk.values.len());
        for value in chunk.values {
            parsed.push(parse_json_field(&value)?);
        }
        expected_offset += parsed.len();
        chunks.push(StcChunk::new(parsed));
    }
    ensure!(
        expected_offset == trace.vector_length,
        "vector_length mismatch: expected {}, saw {}",
        trace.vector_length,
        expected_offset
    );
    Ok(ChunkBatch {
        chunks,
        chunk_len: trace.chunk_length,
        trace_id: trace.trace_id,
    })
}

fn parse_simple_chunks(val: Value) -> Result<ChunkBatch> {
    let arr = val
        .as_array()
        .ok_or_else(|| eyre!("expected an array of chunks"))?;
    ensure!(!arr.is_empty(), "chunk file has no entries");
    let mut chunk_len = None;
    let mut chunks = Vec::with_capacity(arr.len());
    for (index, chunk_val) in arr.iter().enumerate() {
        let values_arr = chunk_val
            .as_array()
            .ok_or_else(|| eyre!("chunk {index} must be an array"))?;
        ensure!(
            !values_arr.is_empty(),
            "chunk {index} in input file is empty"
        );
        let mut parsed = Vec::with_capacity(values_arr.len());
        for entry in values_arr.iter() {
            parsed.push(parse_json_field(entry)?);
        }
        if let Some(expected) = chunk_len {
            ensure!(
                parsed.len() == expected,
                "chunk {index} has len {} but expected {}",
                parsed.len(),
                expected
            );
        } else {
            chunk_len = Some(parsed.len());
        }
        chunks.push(StcChunk::new(parsed));
    }
    let len = chunk_len.unwrap_or(0);
    ensure!(len > 0, "chunk length must be > 0");
    Ok(ChunkBatch::from_chunks(chunks, len))
}

fn parse_field(val: &str) -> Result<Fp> {
    if let Some(hex) = val.strip_prefix("0x") {
        let raw = decode(hex).map_err(|e| eyre!("invalid hex '{val}': {e}"))?;
        ensure!(raw.len() <= 32, "hex value too large");
        let mut bytes = [0u8; 32];
        bytes[32 - raw.len()..].copy_from_slice(&raw);
        let maybe = Fp::from_bytes(&bytes);
        ensure!(bool::from(maybe.is_some()), "hex value outside field range");
        Ok(maybe.unwrap())
    } else {
        let parsed = val.parse::<u64>()?;
        Ok(Fp::from(parsed))
    }
}

fn parse_json_field(val: &Value) -> Result<Fp> {
    if let Some(s) = val.as_str() {
        parse_field(s)
    } else if let Some(u) = val.as_u64() {
        Ok(Fp::from(u))
    } else if let Some(i) = val.as_i64() {
        ensure!(i >= 0, "negative field element not supported");
        Ok(Fp::from(i as u64))
    } else {
        Err(eyre!("unsupported value type in chunk: {:?}", val))
    }
}

fn stats_to_json(stats: &ExecutionStats) -> Value {
    json!({
        "trace_id": stats.trace_id,
        "steps": stats.steps,
        "chunk_len": stats.chunk_len,
        "plain_state": stc_state_to_json(&stats.plain_state),
        "nova_state": stc_state_to_json(&stats.nova_state),
        "timings_ms": {
            "plain": ms(stats.timings.plain),
            "pp": ms(stats.timings.pp),
            "base": ms(stats.timings.base),
            "prove_total": ms(stats.timings.prove_total),
            "prove_avg": ms(stats.timings.prove_total) / stats.steps as f64,
            "verify": ms(stats.timings.verify),
        },
        "recursive_proof_bytes": stats.recursive_proof_bytes,
        "constraints": {
            "primary": stats.constraints.0,
            "secondary": stats.constraints.1,
        },
        "pp_cache_hit": stats.pp_cache_hit,
        "compressed": stats.compressed.as_ref().map(|c| compressed_to_json(c)),
    })
}

fn stc_state_to_json(state: &StcState<Fp>) -> Value {
    json!({
        "n": format_field(&state.n),
        "root": format_field(&state.root),
        "s": state.s.iter().map(format_field).collect::<Vec<_>>(),
        "pow": state.pow.iter().map(format_field).collect::<Vec<_>>(),
    })
}

fn compressed_to_json(stats: &CompressedStats) -> Value {
    json!({
        "setup": ms(stats.setup),
        "prove": ms(stats.prove),
        "verify": ms(stats.verify),
        "proof_bytes": stats.proof_bytes,
    })
}

fn derive_challenges(m: usize) -> Vec<Fp> {
    assert!(m > 0, "at least one challenge required");
    (0..m)
        .map(|i| {
            let seed = Fp::from(((i as u64) << 32) | 0xdead_beefu64);
            seed + Fp::from(17u64)
        })
        .collect()
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn format_field(value: &Fp) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_proof_matches_plain_state() {
        let chunk_len = 8;
        let params = Arc::new(StcParams::new(derive_challenges(2), chunk_len));
        let chunks = generate_random_chunks(4, chunk_len, 1234);
        let batch = ChunkBatch::from_chunks(chunks, chunk_len);
        let mut cache = PpCache::new();
        let build = build_recursive_snark(params, &batch, &mut cache).unwrap();
        assert_eq!(build.plain_state, build.nova_state);
        assert!(build
            .recursive_snark
            .verify(&build.pp, build.steps, &build.z0)
            .is_ok());
    }

    #[test]
    fn recursive_verify_rejects_wrong_initial_state() {
        let chunk_len = 8;
        let params = Arc::new(StcParams::new(derive_challenges(2), chunk_len));
        let chunks = generate_random_chunks(3, chunk_len, 77);
        let batch = ChunkBatch::from_chunks(chunks, chunk_len);
        let mut cache = PpCache::new();
        let build = build_recursive_snark(params, &batch, &mut cache).unwrap();
        let mut bad_z0 = build.z0.clone();
        bad_z0[0] = bad_z0[0] + Fp::ONE;
        assert!(build
            .recursive_snark
            .verify(&build.pp, build.steps, &bad_z0)
            .is_err());
    }
}

enum ChunkSource {
    File(ChunkBatch),
    Random { num_chunks: usize, chunk_len: usize },
}

impl ChunkSource {
    fn materialize(&self, seed: u64) -> ChunkBatch {
        match self {
            ChunkSource::File(batch) => batch.clone(),
            ChunkSource::Random {
                num_chunks,
                chunk_len,
            } => {
                let chunks = generate_random_chunks(*num_chunks, *chunk_len, seed);
                ChunkBatch::from_chunks(chunks, *chunk_len)
            }
        }
    }
}

fn generate_random_chunks(num_chunks: usize, chunk_len: usize, seed: u64) -> Vec<StcChunk<Fp>> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..num_chunks)
        .map(|_| {
            let mut values = Vec::with_capacity(chunk_len);
            for _ in 0..chunk_len {
                values.push(Fp::random(&mut rng));
            }
            StcChunk::new(values)
        })
        .collect()
}
