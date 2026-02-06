use anyhow::Result;
use bicep_core::integrators::EulerMaruyama;
use bicep_core::noise::NoiseGenerator;
use bicep_core::{
    seed::{SeedIdentity, SeedSpec},
    Calc, Diffusion, Drift, SdeIntegrator, State,
};
use clap::Parser;
use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate double-well paths with first-passage annotations"
)]
struct Args {
    #[arg(long, default_value_t = 500000)]
    paths: usize,

    #[arg(long, default_value_t = 20000)]
    steps: usize,

    #[arg(long, default_value_t = 1e-3)]
    dt: f64,

    #[arg(long, default_value_t = 20)]
    save_stride: usize,

    #[arg(long, default_value = "runs/dw.parquet")]
    out: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// Double-well potential: V(x) = (x²-1)²/4
struct DoubleWell {
    temperature: f64,
}

impl Drift for DoubleWell {
    fn mu(&self, _t: f64, x: &State) -> State {
        // Overdamped Langevin: dx = -∇V(x)dt + √(2T)dW
        // ∇V(x) = x³ - x
        let grad_v = x.0[0].powi(3) - x.0[0];
        State::new(vec![-grad_v])
    }
}

impl Diffusion for DoubleWell {
    fn sigma(&self, _t: f64, _x: &State) -> DMatrix<f64> {
        let sigma_val = (2.0 * self.temperature).sqrt();
        DMatrix::from_element(1, 1, sigma_val)
    }

    fn sigma_jacobian(&self, _t: f64, _x: &State) -> Option<Vec<DMatrix<f64>>> {
        Some(vec![DMatrix::from_element(1, 1, 0.0)]) // Constant diffusion
    }

    fn noise_dim(&self, _t: f64, _x: &State) -> usize {
        1
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = DoubleWell { temperature: 0.5 };
    let integrator = EulerMaruyama;
    let seed_spec = SeedSpec::new("cli-seed", seed_key_from_u64(args.seed));

    println!("Generating {} double-well paths", args.paths);
    println!(
        "Steps: {}, dt: {}, save_stride: {}",
        args.steps, args.dt, args.save_stride
    );

    // Data collectors
    let mut run_ids = Vec::new();
    let mut seeds = Vec::new();
    let mut models = Vec::new();
    let mut calcs = Vec::new();
    let mut dts = Vec::new();
    let mut steps_vec = Vec::new();
    let mut path_ids = Vec::new();
    let mut step_nums = Vec::new();
    let mut times = Vec::new();
    let mut states = Vec::new();
    let mut state_means = Vec::new();
    let mut state_stds = Vec::new();
    let mut state_q10 = Vec::new();
    let mut state_q90 = Vec::new();
    let mut aleatoric_unc = Vec::new();
    let mut epistemic_unc = Vec::new();
    let mut first_hit_as = Vec::new();
    let mut first_hit_bs = Vec::new();
    let mut first_hit_times_a = Vec::new();
    let mut first_hit_times_b = Vec::new();

    // Define basins
    let basin_a_threshold = -1.0;
    let basin_b_threshold = 1.0;

    // Generate paths
    for path_id in 0..args.paths {
        if path_id % 10000 == 0 {
            println!("Progress: {}/{} paths", path_id, args.paths);
        }

        let identity = SeedIdentity::new("double_well", "1970-01-01", 0, path_id as u64);
        let mut rng = NoiseGenerator::from_identity(&seed_spec, &identity);
        let x0 = if path_id % 2 == 0 { -0.5 } else { 0.5 }; // Start near different wells
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;
        let mut first_hit_a = false;
        let mut first_hit_b = false;
        let mut fpt_a = None;
        let mut fpt_b = None;

        // Save initial state
        if args.save_stride == 1 || 0 % args.save_stride == 0 {
            run_ids.push("double_well_run".to_string());
            seeds.push(args.seed);
            models.push("DoubleWell".to_string());
            calcs.push("Ito".to_string());
            dts.push(args.dt);
            steps_vec.push(args.steps as u32);
            path_ids.push(path_id as u64);
            step_nums.push(0u32);
            times.push(0.0);
            states.push(vec![x0]);
            state_means.push(x0);
            state_stds.push(0.0);
            state_q10.push(x0);
            state_q90.push(x0);
            aleatoric_unc.push(0.0);
            epistemic_unc.push(0.0);
            first_hit_as.push(false);
            first_hit_bs.push(false);
            first_hit_times_a.push(None::<f64>);
            first_hit_times_b.push(None::<f64>);
        }

        // Simulate path
        for step in 1..=args.steps {
            let dw = rng.generate_dw(1, args.dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, args.dt, &dw, &model, &model);
            t += args.dt;

            // Check first passages
            if !first_hit_a && state.0[0] <= basin_a_threshold {
                first_hit_a = true;
                fpt_a = Some(t);
            }
            if !first_hit_b && state.0[0] >= basin_b_threshold {
                first_hit_b = true;
                fpt_b = Some(t);
            }

            // Save state if on stride
            if step % args.save_stride == 0 {
                run_ids.push("double_well_run".to_string());
                seeds.push(args.seed);
                models.push("DoubleWell".to_string());
                calcs.push("Ito".to_string());
                dts.push(args.dt);
                steps_vec.push(args.steps as u32);
                path_ids.push(path_id as u64);
                step_nums.push(step as u32);
                times.push(t);
                let value = state.0[0];
                states.push(vec![value]);
                state_means.push(value);
                state_stds.push(0.0);
                state_q10.push(value);
                state_q90.push(value);
                aleatoric_unc.push(0.0);
                epistemic_unc.push(0.0);
                first_hit_as.push(first_hit_a);
                first_hit_bs.push(first_hit_b);
                first_hit_times_a.push(fpt_a);
                first_hit_times_b.push(fpt_b);
            }
        }
    }

    // Create DataFrame matching the specified schema
    let state_series = Series::new(
        "state",
        states
            .into_iter()
            .map(|v| Series::new("", v))
            .collect::<Vec<_>>(),
    );

    let df = DataFrame::new(vec![
        Series::new("run_id", run_ids),
        Series::new("seed", seeds),
        Series::new("model", models),
        Series::new("calc", calcs),
        Series::new("dt", dts),
        Series::new("steps", steps_vec),
        Series::new("path_id", path_ids),
        Series::new("step", step_nums),
        Series::new("t", times),
        state_series,
        Series::new("state_mean", state_means),
        Series::new("state_std", state_stds),
        Series::new("state_q10", state_q10),
        Series::new("state_q90", state_q90),
        Series::new("aleatoric_unc", aleatoric_unc),
        Series::new("epistemic_unc", epistemic_unc),
        Series::new("first_hit_a", first_hit_as),
        Series::new("first_hit_b", first_hit_bs),
        Series::new("first_hit_time_a", first_hit_times_a),
        Series::new("first_hit_time_b", first_hit_times_b),
    ])?;

    // Save to Parquet
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    let mut file = std::fs::File::create(&args.out)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;

    println!("Saved {} rows to {}", df.height(), args.out.display());

    // Print statistics
    let final_df = df
        .lazy()
        .group_by([col("path_id")])
        .agg([
            col("first_hit_a").last(),
            col("first_hit_b").last(),
            col("first_hit_time_a").last(),
            col("first_hit_time_b").last(),
        ])
        .collect()?;

    let n_hit_a = final_df.column("first_hit_a")?.bool()?.sum().unwrap_or(0);
    let n_hit_b = final_df.column("first_hit_b")?.bool()?.sum().unwrap_or(0);
    let n_hit_both = final_df
        .lazy()
        .filter(col("first_hit_a").and(col("first_hit_b")))
        .collect()?
        .height();

    println!("\nFirst-passage statistics:");
    println!(
        "Paths hitting A: {} ({:.1}%)",
        n_hit_a,
        100.0 * n_hit_a as f64 / args.paths as f64
    );
    println!(
        "Paths hitting B: {} ({:.1}%)",
        n_hit_b,
        100.0 * n_hit_b as f64 / args.paths as f64
    );
    println!(
        "Paths hitting both: {} ({:.1}%)",
        n_hit_both,
        100.0 * n_hit_both as f64 / args.paths as f64
    );

    Ok(())
}

fn seed_key_from_u64(seed: u64) -> [u8; 32] {
    let mut key = [0u8; 32];
    for i in 0..4 {
        let value = seed.wrapping_add(i as u64).to_le_bytes();
        key[i * 8..(i + 1) * 8].copy_from_slice(&value);
    }
    key
}
