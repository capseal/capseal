use anyhow::Result;
use bicep_core::noise::NoiseGenerator;
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::GeometricBrownianMotion;
use clap::Parser;
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate GBM paths and save to Parquet")]
struct Args {
    #[arg(long, default_value_t = 200000)]
    paths: usize,

    #[arg(long, default_value_t = 2000)]
    steps: usize,

    #[arg(long, default_value_t = 1e-3)]
    dt: f64,

    #[arg(long, default_value_t = 0.2)]
    mu: f64,

    #[arg(long, default_value_t = 0.35)]
    sigma: f64,

    #[arg(long, default_value = "runs/gbm.parquet")]
    out: PathBuf,

    #[arg(long, default_value_t = 7)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create model
    let gbm = GeometricBrownianMotion::new(args.mu, args.sigma);
    let integrator = EulerMaruyama;
    let x0 = 1.0;

    println!(
        "Generating {} GBM paths with {} steps (dt={})",
        args.paths, args.steps, args.dt
    );
    println!("Parameters: μ={}, σ={}", args.mu, args.sigma);

    // Prepare data collectors
    let mut path_ids = Vec::new();
    let mut step_nums = Vec::new();
    let mut times = Vec::new();
    let mut values = Vec::new();
    let mut state_means = Vec::new();
    let mut state_stds = Vec::new();
    let mut state_q10 = Vec::new();
    let mut state_q90 = Vec::new();
    let mut aleatoric_unc = Vec::new();
    let mut epistemic_unc = Vec::new();

    // Generate paths
    for path_id in 0..args.paths {
        if path_id % 10000 == 0 {
            println!("Progress: {}/{} paths", path_id, args.paths);
        }

        let mut rng = NoiseGenerator::from_path_id(args.seed, path_id as u64);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;

        // Store initial state
        path_ids.push(path_id as u64);
        step_nums.push(0u32);
        times.push(0.0);
        values.push(x0);
        state_means.push(x0);
        state_stds.push(0.0);
        state_q10.push(x0);
        state_q90.push(x0);
        aleatoric_unc.push(0.0);
        epistemic_unc.push(0.0);

        // Simulate path
        for step in 1..=args.steps {
            let dw = rng.generate_dw(1, args.dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, args.dt, &dw, &gbm, &gbm);
            t += args.dt;

            // Store state
            let value = state.0[0];
            path_ids.push(path_id as u64);
            step_nums.push(step as u32);
            times.push(t);
            values.push(value);
            state_means.push(value);
            state_stds.push(0.0);
            state_q10.push(value);
            state_q90.push(value);
            aleatoric_unc.push(0.0);
            epistemic_unc.push(0.0);
        }
    }

    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("path_id", path_ids),
        Series::new("step", step_nums),
        Series::new("t", times),
        Series::new("x", values),
        Series::new("state_mean", state_means),
        Series::new("state_std", state_stds),
        Series::new("state_q10", state_q10),
        Series::new("state_q90", state_q90),
        Series::new("aleatoric_unc", aleatoric_unc),
        Series::new("epistemic_unc", epistemic_unc),
    ])?;

    // Save to Parquet
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    let mut file = std::fs::File::create(&args.out)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;

    println!("Saved {} rows to {}", df.height(), args.out.display());

    // Print statistics
    let final_values: Vec<f64> = df
        .lazy()
        .filter(col("step").eq(args.steps as u32))
        .select([col("x")])
        .collect()?
        .column("x")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let mean = final_values.iter().sum::<f64>() / final_values.len() as f64;
    let var = final_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (final_values.len() - 1) as f64;

    println!("\nFinal value statistics:");
    println!("Mean: {:.6}", mean);
    println!("Variance: {:.6}", var);
    println!(
        "Expected mean: {:.6}",
        x0 * (args.mu * args.steps as f64 * args.dt).exp()
    );

    Ok(())
}
