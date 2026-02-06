use anyhow::Result;
use clap::Parser;
use polars::prelude::*;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate parity task trajectories for ENN training"
)]
struct Args {
    #[arg(long, default_value_t = 1000)]
    sequences: usize,

    #[arg(long, default_value_t = 15)]
    seq_len: usize,

    #[arg(long, default_value_t = 1e-2)]
    dt: f64,

    #[arg(long, default_value = "runs/parity_trajectories.parquet")]
    out: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Ensembles in adaptive BICEP (controls epistemic variance)
    #[arg(long, default_value_t = 3)]
    ensembles: usize,

    /// Paths simulated per ensemble (controls aleatoric variance)
    #[arg(long, default_value_t = 8)]
    paths_per_ensemble: usize,

    /// Oscillation amplitude injected during SDE steps
    #[arg(long, default_value_t = 0.15)]
    oscillation_amp: f64,

    /// Base oscillation frequency in Hz
    #[arg(long, default_value_t = 8.0)]
    oscillation_freq: f64,

    /// Jump rate per step for ERP-like spikes
    #[arg(long, default_value_t = 0.15)]
    jump_rate: f64,

    /// Jump magnitude scale
    #[arg(long, default_value_t = 0.25)]
    jump_scale: f64,
}

fn ensemble_configs(base: f64, args: &Args) -> Vec<EnsembleConfig> {
    (0..args.ensembles.max(1))
        .map(|idx| {
            let theta = 1.5 + 0.3 * idx as f64;
            let sigma = 0.25 + 0.05 * idx as f64;
            let freq = args.oscillation_freq * (1.0 + 0.1 * idx as f64);
            EnsembleConfig {
                theta,
                mu: base,
                sigma,
                freq,
            }
        })
        .collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped = q.clamp(0.0, 1.0);
    let pos = clamped * (sorted.len() - 1) as f64;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;
    if idx + 1 >= sorted.len() {
        sorted[idx]
    } else {
        sorted[idx] + (sorted[idx + 1] - sorted[idx]) * frac
    }
}

fn simulate_single(
    base: f64,
    cfg: &EnsembleConfig,
    dt: f64,
    inner_steps: usize,
    rng: &mut rand::rngs::StdRng,
    normal: &Normal<f64>,
    amp: f64,
    jump_rate: f64,
    jump_scale: f64,
    base_time: f64,
) -> f64 {
    let mut state = base;
    let inner_dt = dt / inner_steps as f64;
    for step in 0..inner_steps {
        let drift = cfg.theta * (cfg.mu - state);
        let noise = cfg.sigma * normal.sample(rng) * inner_dt.sqrt();
        state += drift * inner_dt + noise;

        let t = base_time + step as f64 * inner_dt;
        state += amp * (TAU * cfg.freq * t).sin() * inner_dt;

        if rng.gen::<f64>() < jump_rate * inner_dt {
            state += jump_scale * normal.sample(rng);
        }
    }
    state
}

fn adaptive_stats(
    base: f64,
    args: &Args,
    seq_rng: &mut rand::rngs::StdRng,
    base_time: f64,
) -> PathStats {
    let configs = ensemble_configs(base, args);
    let mut all_paths = Vec::new();
    let mut ensemble_means = Vec::new();
    let mut aleatoric_sum = 0.0;
    let inner_steps = 5;
    let paths_per = args.paths_per_ensemble.max(1);
    let normal = Normal::new(0.0, 1.0).unwrap();

    for cfg in configs.iter() {
        let mut values = Vec::with_capacity(paths_per);
        for _ in 0..paths_per {
            values.push(simulate_single(
                base,
                cfg,
                args.dt,
                inner_steps,
                seq_rng,
                &normal,
                args.oscillation_amp,
                args.jump_rate,
                args.jump_scale,
                base_time,
            ));
        }
        let ensemble_mean = values.iter().copied().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - ensemble_mean).powi(2))
            .sum::<f64>()
            / values.len().max(1) as f64;
        aleatoric_sum += variance;
        ensemble_means.push(ensemble_mean);
        all_paths.extend(values);
    }

    let mut sorted = all_paths.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = all_paths.iter().copied().sum::<f64>() / all_paths.len().max(1) as f64;
    let variance =
        all_paths.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / all_paths.len().max(1) as f64;
    let std = variance.sqrt();

    let epistemic = if ensemble_means.len() > 1 {
        let m = ensemble_means.iter().copied().sum::<f64>() / ensemble_means.len() as f64;
        ensemble_means.iter().map(|v| (v - m).powi(2)).sum::<f64>() / ensemble_means.len() as f64
    } else {
        0.0
    };

    PathStats {
        mean,
        std,
        q10: quantile(&sorted, 0.1),
        q90: quantile(&sorted, 0.9),
        aleatoric: aleatoric_sum / configs.len().max(1) as f64,
        epistemic,
    }
}

#[derive(Clone, Copy, Debug)]
struct EnsembleConfig {
    theta: f64,
    mu: f64,
    sigma: f64,
    freq: f64,
}

#[derive(Default, Debug)]
struct PathStats {
    mean: f64,
    std: f64,
    q10: f64,
    q90: f64,
    aleatoric: f64,
    epistemic: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Generating {} parity task sequences", args.sequences);
    println!("Sequence length: {}, dt: {}", args.seq_len, args.dt);

    // Data collectors
    let mut run_ids = Vec::new();
    let mut seeds = Vec::new();
    let mut models = Vec::new();
    let mut calcs = Vec::new();
    let mut dts = Vec::new();
    let mut seq_lens = Vec::new();
    let mut sequence_ids = Vec::new();
    let mut step_nums = Vec::new();
    let mut times = Vec::new();
    let mut state_values = Vec::new();
    let mut state_std = Vec::new();
    let mut state_q10 = Vec::new();
    let mut state_q90 = Vec::new();
    let mut aleatoric_unc = Vec::new();
    let mut epistemic_unc = Vec::new();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    // Generate parity sequences
    for seq_id in 0..args.sequences {
        if seq_id % 100 == 0 {
            println!("Progress: {}/{} sequences", seq_id, args.sequences);
        }

        // Generate random binary sequence
        let binary_seq: Vec<i32> = (0..args.seq_len)
            .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
            .collect();

        // Compute XOR parity (1 if odd number of 1s, 0 if even)
        let parity = binary_seq.iter().sum::<i32>() % 2;

        // Convert to floating point inputs {-1, 1} for SDE simulation
        let float_seq: Vec<f64> = binary_seq
            .iter()
            .map(|&x| if x == 1 { 1.0 } else { -1.0 })
            .collect();

        // Simulate paths per step
        let mut seq_rng = rand::rngs::StdRng::seed_from_u64(
            args.seed ^ (seq_id as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );

        for (step, &input_val) in float_seq.iter().enumerate() {
            let base_time = step as f64 * args.dt;
            let stats = adaptive_stats(input_val, &args, &mut seq_rng, base_time);

            // Store the data
            run_ids.push("parity_task".to_string());
            seeds.push(args.seed);
            models.push("ParityTask".to_string());
            calcs.push("Ito".to_string());
            dts.push(args.dt);
            seq_lens.push(args.seq_len as u32);
            sequence_ids.push(seq_id as u64);
            step_nums.push(step as u32);
            times.push(step as f64 * args.dt);
            state_values.push(stats.mean);
            state_std.push(stats.std);
            state_q10.push(stats.q10);
            state_q90.push(stats.q90);
            aleatoric_unc.push(stats.aleatoric);
            epistemic_unc.push(stats.epistemic);
            inputs.push(input_val);

            // Target is only revealed at the end
            if step == args.seq_len - 1 {
                targets.push(parity as f64);
            } else {
                targets.push(0.0); // No target during sequence
            }
        }
    }

    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("run_id", run_ids),
        Series::new("seed", seeds),
        Series::new("model", models),
        Series::new("calc", calcs),
        Series::new("dt", dts),
        Series::new("seq_len", seq_lens),
        Series::new("sequence_id", sequence_ids),
        Series::new("step", step_nums),
        Series::new("t", times),
        Series::new("state", state_values.clone()),
        Series::new("state_std", state_std),
        Series::new("state_q10", state_q10),
        Series::new("state_q90", state_q90),
        Series::new("aleatoric_unc", aleatoric_unc),
        Series::new("epistemic_unc", epistemic_unc),
        Series::new("input", inputs),
        Series::new("target", targets),
    ])?;

    // Save to Parquet
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    let mut file = std::fs::File::create(&args.out)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;

    println!("Saved {} rows to {}", df.height(), args.out.display());

    // Print statistics
    let final_df = df
        .lazy()
        .group_by([col("sequence_id")])
        .agg([
            col("target").last().alias("final_target"),
            col("input").sum().alias("input_sum"),
        ])
        .collect()?;

    let n_positive = final_df.column("final_target")?.f64()?.sum().unwrap_or(0.0) as usize;
    let n_negative = args.sequences - n_positive;

    println!("\nParity task statistics:");
    println!(
        "Sequences with parity 1: {} ({:.1}%)",
        n_positive,
        100.0 * n_positive as f64 / args.sequences as f64
    );
    println!(
        "Sequences with parity 0: {} ({:.1}%)",
        n_negative,
        100.0 * n_negative as f64 / args.sequences as f64
    );

    Ok(())
}
