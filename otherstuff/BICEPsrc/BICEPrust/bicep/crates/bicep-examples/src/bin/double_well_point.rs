use anyhow::Result;
use bicep_core::integrators::EulerMaruyama;
use bicep_core::noise::NoiseGenerator;
use bicep_core::{Calc, SdeIntegrator, State};
use bicep_models::double_well::DoubleWell;
use clap::Parser;

/// Simulate trajectories for a single start state and report hits of the right well before the left.
#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Run BICEP double-well trajectories for a single start point"
)]
struct Args {
    /// Initial x coordinate
    #[arg(long)]
    x: f64,

    /// Initial y coordinate
    #[arg(long, default_value_t = 0.0)]
    y: f64,

    /// Number of trajectories to run
    #[arg(long, default_value_t = 64)]
    paths: usize,

    /// Time step for Euler-Maruyama integrator
    #[arg(long, default_value_t = 1e-3)]
    dt: f64,

    /// Maximum simulation time
    #[arg(long, default_value_t = 10.0)]
    t_max: f64,

    /// Double-well quartic coefficient a
    #[arg(long, default_value_t = 1.0)]
    a: f64,

    /// Double-well quadratic coefficient b
    #[arg(long, default_value_t = 2.0)]
    b: f64,

    /// Temperature (β^{-1})
    #[arg(long, default_value_t = 0.5)]
    temperature: f64,

    /// Left-well threshold for declaring hit of set B
    #[arg(long, default_value_t = -0.9)]
    left_threshold: f64,

    /// Right-well threshold for declaring hit of set A
    #[arg(long, default_value_t = 0.9)]
    right_threshold: f64,

    /// Random seed base
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn simulate_hit(
    model: &DoubleWell,
    integrator: EulerMaruyama,
    start: State,
    dt: f64,
    steps: usize,
    left_threshold: f64,
    right_threshold: f64,
    rng: &mut NoiseGenerator,
) -> bool {
    let mut state = start;
    let mut t = 0.0_f64;

    for _ in 0..steps {
        let x = state.0[0];
        if x >= right_threshold {
            return true; // Hit right basin first
        }
        if x <= left_threshold {
            return false; // Hit left basin first
        }

        let dim = state.dim();
        let dw = rng.generate_dw(dim, dt.sqrt());
        state = integrator.step(Calc::Ito, t, &state, dt, &dw, model, model);
        t += dt;
    }

    // Timeout -> treat as failure (hit B) to be conservative
    false
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = DoubleWell::new(args.a, args.b, args.temperature);
    let integrator = EulerMaruyama;
    let steps = (args.t_max / args.dt).ceil() as usize;
    let start_state = State::new(vec![args.x, args.y]);

    let mut successes = 0_u64;

    for path_id in 0..args.paths {
        let mut rng = NoiseGenerator::from_path_id(args.seed, path_id as u64);
        if simulate_hit(
            &model,
            integrator,
            start_state.clone(),
            args.dt,
            steps,
            args.left_threshold,
            args.right_threshold,
            &mut rng,
        ) {
            successes += 1;
        }
    }

    println!(
        "{{\"successes\":{},\"trials\":{},\"seed\":{}}}",
        successes,
        args.paths,
        args.seed
    );

    Ok(())
}
