use anyhow::Result;
use bicep_core::integrators::{EulerMaruyama, HeunStratonovich};
use bicep_core::noise::NoiseGenerator;
use bicep_core::{Calc, Diffusion, Drift, SdeIntegrator, State};
use clap::Parser;
use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Compare Itô vs Stratonovich methods and generate KS distance sweep"
)]
struct Args {
    #[arg(long, default_value_t = 200000)]
    paths: usize,

    #[arg(long, default_value_t = 1e-2)]
    dt: f64,

    #[arg(long, default_value_t = 100)]
    steps: usize,

    #[arg(long, default_value = "runs/ito_strat_ks.csv")]
    out: PathBuf,
}

// State-dependent diffusion model: σ(x) = 0.4 + 0.3x²
struct StateDependentDiffusion;

impl Drift for StateDependentDiffusion {
    fn mu(&self, _t: f64, _x: &State) -> State {
        State::new(vec![0.1]) // Small drift
    }
}

impl Diffusion for StateDependentDiffusion {
    fn sigma(&self, _t: f64, x: &State) -> DMatrix<f64> {
        let sigma_x = 0.4 + 0.3 * x.0[0].powi(2);
        DMatrix::from_element(1, 1, sigma_x)
    }

    fn sigma_jacobian(&self, _t: f64, x: &State) -> Option<Vec<DMatrix<f64>>> {
        // d(σ(x))/dx = 0.6x
        Some(vec![DMatrix::from_element(1, 1, 0.6 * x.0[0])])
    }

    fn noise_dim(&self, _t: f64, _x: &State) -> usize {
        1
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = StateDependentDiffusion;
    let x0 = 1.0;
    let t_final = 1.0;

    // Different dt values to test
    let dt_values = vec![1e-1, 5e-2, 2e-2, 1e-2, 5e-3];

    println!("Comparing Itô vs Stratonovich with {} paths", args.paths);
    println!("dt sweep: {:?}", dt_values);

    // Prepare CSV output
    let mut csv_content = String::from("dt,ks_distance\n");

    for &dt in &dt_values {
        let steps = (t_final / dt) as usize;
        println!("\nProcessing dt={} ({} steps)", dt, steps);

        let ks_dist = compute_ks_for_dt(&model, x0, dt, steps, args.paths);

        println!("KS distance: {:.6}", ks_dist);
        csv_content.push_str(&format!("{},{}\n", dt, ks_dist));
    }

    // Save CSV
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    let mut file = File::create(&args.out)?;
    write!(file, "{}", csv_content)?;

    println!("\nSaved results to {}", args.out.display());

    Ok(())
}

fn compute_ks_for_dt(
    model: &StateDependentDiffusion,
    x0: f64,
    dt: f64,
    steps: usize,
    n_paths: usize,
) -> f64 {
    let mut ito_finals = Vec::with_capacity(n_paths);
    let mut strat_finals = Vec::with_capacity(n_paths);

    for path_id in 0..n_paths {
        if path_id % 10000 == 0 && path_id > 0 {
            println!("  Progress: {}/{} paths", path_id, n_paths);
        }

        let seed = 42 + path_id as u64;

        // Itô with EM + drift correction
        let mut rng_ito = NoiseGenerator::new(seed);
        let mut state_ito = State::new(vec![x0]);
        let mut t = 0.0;

        for _ in 0..steps {
            let dw = rng_ito.generate_dw(1, dt.sqrt());
            state_ito = EulerMaruyama.step(Calc::Ito, t, &state_ito, dt, &dw, model, model);
            t += dt;
        }
        ito_finals.push(state_ito.0[0]);

        // Stratonovich with Heun midpoint
        let mut rng_strat = NoiseGenerator::new(seed);
        let mut state_strat = State::new(vec![x0]);
        t = 0.0;

        for _ in 0..steps {
            let dw = rng_strat.generate_dw(1, dt.sqrt());
            state_strat =
                HeunStratonovich.step(Calc::Stratonovich, t, &state_strat, dt, &dw, model, model);
            t += dt;
        }
        strat_finals.push(state_strat.0[0]);
    }

    compute_ks_distance(&ito_finals, &strat_finals)
}

fn compute_ks_distance(sample1: &[f64], sample2: &[f64]) -> f64 {
    let n1 = sample1.len();
    let n2 = sample2.len();

    // Combine and sort all values
    let mut all_values: Vec<(f64, bool)> = Vec::with_capacity(n1 + n2);
    for &x in sample1 {
        all_values.push((x, true)); // true = from sample1
    }
    for &x in sample2 {
        all_values.push((x, false)); // false = from sample2
    }
    all_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute KS statistic
    let mut d_max = 0.0;
    let mut count1 = 0;
    let mut count2 = 0;

    for &(_, from_sample1) in &all_values {
        if from_sample1 {
            count1 += 1;
        } else {
            count2 += 1;
        }

        let cdf1 = count1 as f64 / n1 as f64;
        let cdf2 = count2 as f64 / n2 as f64;
        let d = (cdf1 - cdf2).abs();

        if d > d_max {
            d_max = d;
        }
    }

    d_max
}
